# inference.py
# OOP FastFlow inference for MVTec:
# - exact Resize->CenterCrop preprocessing (torchvision v2)
# - image-level: Accuracy, Precision, AUROC + F1AdaptiveThreshold for image threshold (auto direction)
# - pixel-level: AUROC (sampled) + **SELF-IMPLEMENTED AUPRO** (region overlap) on globally-normalized maps
# - contour overlay drawn on ORIGINAL image, but thresholding done in crop-space then "uncropped" back

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from anomalib.metrics.threshold import F1AdaptiveThreshold
from model import FastFlowModel


# ============================================================
# Self-implemented AUPRO (minimal, compute-only, no torchmetrics)
# ============================================================
class SelfAUPRO:
    """
    Minimal AUPRO implementation inspired by anomalib's AUPRO:
    - connected components on GT mask
    - compute PRO curve (FPR vs averaged per-region overlap)
    - integrate area up to fpr_limit and normalize by fpr_limit

    Expected:
      preds:  (B,H,W) or (B,1,H,W) float in [0,1] (higher => more anomalous)
      target: (B,H,W) or (B,1,H,W) binary {0,1} or {0,255}
    """

    def __init__(self, fpr_limit: float = 0.3):
        self.fpr_limit = float(fpr_limit)
        self._preds: List[torch.Tensor] = []
        self._target: List[torch.Tensor] = []

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.dim() == 4 and preds.size(1) == 1:
            preds = preds.squeeze(1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        # store on CPU to reduce GPU memory
        preds = preds.detach().float().cpu()
        target = target.detach().float().cpu()

        # normalize target into {0,1}
        if target.max() > 1.0:
            target = (target > 0).float()
        else:
            target = (target > 0.5).float()

        # clamp preds
        preds = preds.clamp(0.0, 1.0)

        self._preds.append(preds)
        self._target.append(target)

    @staticmethod
    def _cca_cv2(mask01: np.ndarray) -> np.ndarray:
        """Connected components for one image. mask01 is {0,1} uint8."""
        mask01 = (mask01 > 0).astype(np.uint8)
        # returns labels in [0..N]
        _, labels = cv2.connectedComponents(mask01, connectivity=8)
        return labels.astype(np.int32)

    @staticmethod
    def _make_global_region_labels(cca_bhw: torch.Tensor) -> torch.Tensor:
        """Offset connected component labels across batch to make them unique (except 0 background)."""
        cca_off = cca_bhw.clone()
        current_offset = 0
        B = int(cca_off.size(0))
        for b in range(B):
            img = cca_off[b]
            uniq = torch.unique(img)
            uniq_fg = uniq[uniq != 0]
            num_regions = int(uniq_fg.numel())
            if num_regions == 0:
                continue
            fg = img > 0
            img[fg] = img[fg] + current_offset
            cca_off[b] = img
            current_offset += num_regions
        return cca_off

    def perform_cca(self) -> torch.Tensor:
        """Return (B,H,W) integer labels; 0 is background; >0 are region IDs unique across batch."""
        target = torch.cat(self._target, dim=0)  # (B,H,W)
        if target.min() < 0 or target.max() > 1:
            raise ValueError(f"AUPRO expects target in [0,1], got [{float(target.min())},{float(target.max())}]")

        # CPU CCA via OpenCV per image
        target_np = target.numpy()
        ccas = []
        for b in range(target_np.shape[0]):
            labels = self._cca_cv2(target_np[b])
            ccas.append(labels)
        cca = torch.from_numpy(np.stack(ccas, axis=0))  # (B,H,W) int32
        cca = self._make_global_region_labels(cca)
        return cca.long()

    @staticmethod
    def _auc_trapz(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Trapezoidal integration (assumes x sorted ascending)."""
        if x.numel() < 2:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)
        dx = x[1:] - x[:-1]
        avg = 0.5 * (y[1:] + y[:-1])
        return torch.sum(dx * avg)

    def compute_pro(self, cca: torch.Tensor, preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PRO curve (FPR vs averaged per-region overlap).
        cca:   (B,H,W) int labels, 0 background
        preds: (B,H,W) float, higher => more anomalous
        """
        device = preds.device

        labels = cca.reshape(-1).long()
        preds_flat = preds.reshape(-1).float()

        background = labels == 0
        fp_change = background.float()
        num_bg = fp_change.sum()

        f_lim = float(self.fpr_limit)

        if num_bg <= 0:
            return (
                torch.tensor([0.0, f_lim], device=device),
                torch.tensor([0.0, 0.0], device=device),
            )

        max_label = int(labels.max().item())
        if max_label == 0:
            return (
                torch.tensor([0.0, f_lim], device=device),
                torch.tensor([0.0, 0.0], device=device),
            )

        region_sizes = torch.bincount(labels, minlength=max_label + 1).float()
        num_regions = (region_sizes[1:] > 0).sum()

        if num_regions <= 0:
            return (
                torch.tensor([0.0, f_lim], device=device),
                torch.tensor([0.0, 0.0], device=device),
            )

        fg_mask = labels > 0
        pro_change = torch.zeros_like(preds_flat)
        pro_change[fg_mask] = 1.0 / region_sizes[labels[fg_mask]]

        idx = torch.argsort(preds_flat, descending=True)
        fp_sorted = fp_change[idx]
        pro_sorted = pro_change[idx]
        preds_sorted = preds_flat[idx]

        fpr = torch.cumsum(fp_sorted, 0) / num_bg
        pro = torch.cumsum(pro_sorted, 0) / num_regions
        fpr = torch.clamp(fpr, max=1.0)
        pro = torch.clamp(pro, max=1.0)

        # remove duplicate thresholds
        keep = torch.ones_like(preds_sorted, dtype=torch.bool)
        keep[:-1] = preds_sorted[:-1] != preds_sorted[1:]
        fpr = fpr[keep]
        pro = pro[keep]

        # prepend zero
        fpr = torch.cat([torch.tensor([0.0], device=device), fpr])
        pro = torch.cat([torch.tensor([0.0], device=device), pro])

        # clip at fpr_limit with linear interpolation
        mask = fpr <= f_lim
        if mask.any():
            i = int(mask.nonzero(as_tuple=True)[0][-1].item())
            if fpr[i] < f_lim and i + 1 < fpr.numel():
                f1, f2 = fpr[i], fpr[i + 1]
                p1, p2 = pro[i], pro[i + 1]
                p_lim = p1 + (p2 - p1) * (f_lim - f1) / (f2 - f1 + 1e-12)
                fpr = torch.cat([fpr[: i + 1], torch.tensor([f_lim], device=device)])
                pro = torch.cat([pro[: i + 1], torch.tensor([p_lim], device=device)])
            else:
                fpr = fpr[: i + 1]
                pro = pro[: i + 1]
        else:
            fpr = torch.tensor([0.0, f_lim], device=device)
            pro = torch.tensor([0.0, 0.0], device=device)

        return fpr, pro

    def compute(self) -> torch.Tensor:
        if not self._preds:
            return torch.tensor(0.0)

        cca = self.perform_cca()               # (B,H,W) on CPU
        preds = torch.cat(self._preds, dim=0)  # (B,H,W) on CPU

        # compute PRO curve on CPU tensors
        fpr, pro = self.compute_pro(cca=cca, preds=preds)

        area = self._auc_trapz(fpr, pro)
        # normalize by fpr_limit (or by last x)
        denom = fpr[-1].clamp_min(1e-12)
        return area / denom


# -----------------------------
# Utilities: metrics (no sklearn)
# -----------------------------
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def precision_from_counts(tp: int, fp: int) -> float:
    return float(tp / (tp + fp + 1e-12))


def accuracy_from_counts(tp: int, tn: int, fp: int, fn: int) -> float:
    return float((tp + tn) / (tp + tn + fp + fn + 1e-12))


def auroc_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC via Mann–Whitney U (rank-based), tie-safe."""
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = float(ranks[pos].sum())
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    auc = u / (n_pos * n_neg)
    return float(auc)


def f1adaptive_threshold_1d(scores: np.ndarray, labels: np.ndarray) -> float:
    scores_t = torch.tensor(scores, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.int64)
    metric = F1AdaptiveThreshold()
    metric.update(scores_t, labels_t)
    thr = metric.compute()
    return float(thr.item() if hasattr(thr, "item") else thr)


# -----------------------------
# Data helpers
# -----------------------------
def iter_images_recursive(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if "ground_truth" in {x.lower() for x in p.parts}:
            continue
        yield p


def infer_gt_label_from_path(p: Path) -> Optional[int]:
    parts = [x.lower() for x in p.parts]
    if "test" not in parts:
        return None
    i = parts.index("test")
    if i + 1 >= len(parts):
        return None
    return 0 if parts[i + 1] == "good" else 1


def defect_name_from_path(p: Path) -> str:
    parts = list(p.parts)
    lower = [x.lower() for x in parts]
    if "test" in lower:
        i = lower.index("test")
        if i + 1 < len(parts):
            return parts[i + 1]
    return p.parent.name


def load_gt_mask_mvtec(img_path: Path) -> Optional[np.ndarray]:
    parts = list(img_path.parts)
    lower = [p.lower() for p in parts]
    if "test" not in lower:
        return None
    ti = lower.index("test")
    if ti + 1 >= len(parts):
        return None

    defect = parts[ti + 1]
    if defect.lower() == "good":
        return None

    cat_dir = img_path.parents[2]  # .../<category>
    gt_path = cat_dir / "ground_truth" / defect / f"{img_path.stem}_mask.png"
    if not gt_path.exists():
        gt_path = cat_dir / "ground_truth" / defect / f"{img_path.stem}.png"
        if not gt_path.exists():
            return None

    m = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return ((m > 0).astype(np.uint8) * 255)


# -----------------------------
# Core OOP runner
# -----------------------------
@dataclass
class Sample:
    path: Path
    gt_label: int
    score_raw: float
    amap_hw: np.ndarray
    orig_rgb: np.ndarray
    gt_mask_orig: Optional[np.ndarray]


class FastFlowInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        backbone: str,
        flow_steps: int,
        image_size: Tuple[int, int],
        hidden_ratio: float,
        clamp: float,
        conv3x3_only: bool,
        device: str,
        topk_ratio: float,
        crop_scale: float = 0.875,
    ):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.image_size = tuple(image_size)
        self.topk_ratio = float(topk_ratio)

        if backbone == "wide_resnet50_2":
            self.model = FastFlowModel(
                backbone_name=backbone,
                flow_steps=flow_steps,
                input_size=self.image_size,
                hidden_ratio=hidden_ratio,
                reducer_channels=(128, 192, 256),
                clamp=clamp,
                conv3x3_only=conv3x3_only,
            )
        else:
            self.model = FastFlowModel(
                backbone_name=backbone,
                flow_steps=flow_steps,
                input_size=self.image_size,
                hidden_ratio=hidden_ratio,
                clamp=clamp,
                conv3x3_only=conv3x3_only,
            )

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        from torchvision.transforms import v2 as T

        h, w = self.image_size
        pre_h = int(math.ceil(h / crop_scale))
        pre_w = int(math.ceil(w / crop_scale))

        self.h, self.w = h, w
        self.pre_h, self.pre_w = pre_h, pre_w
        self.crop_top = (pre_h - h) // 2
        self.crop_left = (pre_w - w) // 2

        self.transform = T.Compose([
            T.ToImage(),
            T.Resize((pre_h, pre_w), antialias=True),
            T.CenterCrop((h, w)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        pil = Image.open(image_path).convert("RGB")
        orig_rgb = np.array(pil)
        x = self.transform(pil).unsqueeze(0)
        return x, orig_rgb

    @torch.inference_mode()
    def get_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "get_anomaly_map"):
            return self.model.get_anomaly_map(x)
        out = self.model(x)
        if not isinstance(out, torch.Tensor):
            raise RuntimeError(f"Expected FastFlowModel to return torch.Tensor, got {type(out)}")
        return out

    @torch.inference_mode()
    def infer_one(self, image_path: str) -> Tuple[float, np.ndarray, np.ndarray]:
        x, orig_rgb = self.preprocess(image_path)
        x = x.to(self.device)

        amap = self.get_anomaly_map(x)                 # [1,1,h,w]
        flat = amap.squeeze(1).flatten(1)              # [1, h*w]
        k = max(1, int(self.topk_ratio * flat.shape[1]))
        score = flat.topk(k, dim=1).values.mean(dim=1) # [1]
        score_val = float(score.item())
        amap_np = amap[0, 0].detach().cpu().numpy().astype(np.float32)
        return score_val, amap_np, orig_rgb

    def gt_mask_to_crop(self, gt_mask_orig: np.ndarray, orig_rgb: np.ndarray) -> np.ndarray:
        H0, W0 = orig_rgb.shape[:2]
        m = gt_mask_orig
        if m.shape[:2] != (H0, W0):
            m = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST)
        m = cv2.resize(m, (self.pre_w, self.pre_h), interpolation=cv2.INTER_NEAREST)
        t, l = self.crop_top, self.crop_left
        m = m[t : t + self.h, l : l + self.w]
        return ((m > 0).astype(np.uint8) * 255)

    def uncrop_mask_to_original(self, orig_rgb: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
        H0, W0 = orig_rgb.shape[:2]
        canvas = np.zeros((self.pre_h, self.pre_w), dtype=np.uint8)
        t, l = self.crop_top, self.crop_left
        canvas[t : t + self.h, l : l + self.w] = mask_hw
        return cv2.resize(canvas, (W0, H0), interpolation=cv2.INTER_NEAREST)

    def save_contour_overlay(
        self,
        orig_rgb: np.ndarray,
        anomaly_map_hw: np.ndarray,
        save_path: Path,
        pixel_thr_norm: float,
        global_min: float,
        global_max: float,
        min_area: int,
        contour_thickness: int,
    ) -> int:
        den = (float(global_max) - float(global_min)) + 1e-12
        vis = (anomaly_map_hw - float(global_min)) / den
        vis = np.clip(vis, 0.0, 1.0)

        mask_hw = (vis >= float(pixel_thr_norm)).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        mask_hw = cv2.morphologyEx(mask_hw, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_hw = cv2.morphologyEx(mask_hw, cv2.MORPH_CLOSE, kernel, iterations=1)

        mask_orig = self.uncrop_mask_to_original(orig_rgb, mask_hw)

        cnts, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = [c for c in cnts if cv2.contourArea(c) >= float(min_area)]

        out_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        if kept:
            cv2.drawContours(out_bgr, kept, -1, (0, 0, 255), int(contour_thickness))

        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), out_bgr)
        return len(kept)


class MVTecFastFlowEvaluator:
    def __init__(
        self,
        engine: FastFlowInferenceEngine,
        out_dir: Path,
        category: str,
        min_area: int,
        thickness: int,
        fpr_limit: float,
        aupro_downsample: int,
        save_mode: str,
        pixel_sample_per_image: int = 5000,
    ):
        self.engine = engine
        self.out_dir = out_dir / category
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.category = category
        self.min_area = int(min_area)
        self.thickness = int(thickness)
        self.fpr_limit = float(fpr_limit)
        self.aupro_downsample = max(1, int(aupro_downsample))
        self.save_mode = save_mode
        self.pixel_sample_per_image = int(pixel_sample_per_image)

        self.samples: List[Sample] = []

        self.direction: float = 1.0
        self.img_thr: float = 0.0
        self.global_min: float = 0.0
        self.global_max: float = 1.0
        self.pix_thr_norm: float = 0.5

        self.image_metrics = {}
        self.pixel_metrics = {}

    def collect(self, paths: List[Path]) -> None:
        self.samples.clear()
        for p in paths:
            gt = infer_gt_label_from_path(p)
            if gt is None:
                continue
            score, amap_hw, orig_rgb = self.engine.infer_one(str(p))
            gt_mask_orig = load_gt_mask_mvtec(p)
            self.samples.append(
                Sample(
                    path=p,
                    gt_label=int(gt),
                    score_raw=float(score),
                    amap_hw=amap_hw,
                    orig_rgb=orig_rgb,
                    gt_mask_orig=gt_mask_orig,
                )
            )
        if not self.samples:
            raise RuntimeError("No images have inferable GT labels. Expected MVTec test structure.")

    def compute_image_metrics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y_true = np.array([s.gt_label for s in self.samples], dtype=np.int64)
        scores = np.array([s.score_raw for s in self.samples], dtype=np.float32)

        n = scores[y_true == 0]
        a = scores[y_true == 1]
        self.direction = 1.0
        if len(n) and len(a) and float(a.mean()) < float(n.mean()):
            self.direction = -1.0

        eff_scores = scores * self.direction
        self.img_thr = f1adaptive_threshold_1d(eff_scores, y_true)

        y_pred = (eff_scores > self.img_thr).astype(np.int64)
        tp, tn, fp, fn = confusion_counts(y_true, y_pred)
        acc = accuracy_from_counts(tp, tn, fp, fn)
        prec = precision_from_counts(tp, fp)
        auc = auroc_from_scores(y_true, eff_scores)

        self.image_metrics = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc,
            "precision": prec,
            "auroc": auc,
            "direction": self.direction,
            "threshold": self.img_thr,
        }

        print(f"[Image] direction={self.direction:+.0f}  img_thr={self.img_thr:.6f}")
        print(f"[Image] Accuracy:  {acc*100:.2f}%")
        print(f"[Image] Precision: {prec*100:.2f}%  (TP={tp}, FP={fp}, TN={tn}, FN={fn})")
        print(f"[Image] AUROC:     {auc:.6f}")

        return y_true, scores, eff_scores, y_pred

    def compute_pixel_metrics(self) -> None:
        self.global_min = min(float(s.amap_hw.min()) for s in self.samples)
        self.global_max = max(float(s.amap_hw.max()) for s in self.samples)
        den = (self.global_max - self.global_min) + 1e-12

        ds = self.aupro_downsample
        aupro_metric = SelfAUPRO(fpr_limit=self.fpr_limit)

        rng = np.random.default_rng(123)
        pix_scores_all = []
        pix_labels_all = []

        for s in self.samples:
            if s.gt_mask_orig is None:
                gt_crop_255 = np.zeros_like(s.amap_hw, dtype=np.uint8)
            else:
                gt_crop_255 = self.engine.gt_mask_to_crop(s.gt_mask_orig, s.orig_rgb)

            gt01 = (gt_crop_255 > 0).astype(np.float32)  # {0,1}

            pred01 = (s.amap_hw - self.global_min) / den
            pred01 = np.clip(pred01, 0.0, 1.0).astype(np.float32)

            # AUPRO update (downsample for memory)
            if ds > 1:
                H, W = pred01.shape
                newW, newH = max(1, W // ds), max(1, H // ds)
                pred_ds = cv2.resize(pred01, (newW, newH), interpolation=cv2.INTER_LINEAR)
                gt_ds = cv2.resize(gt01, (newW, newH), interpolation=cv2.INTER_NEAREST)
            else:
                pred_ds = pred01
                gt_ds = gt01

            aupro_metric.update(
                torch.from_numpy(pred_ds).unsqueeze(0),   # (1,H,W)
                torch.from_numpy(gt_ds).unsqueeze(0),     # (1,H,W)
            )

            # sampled pixels for AUROC + pix thr (cheap)
            flat_pred = pred01.reshape(-1)
            flat_gt = gt01.reshape(-1).astype(np.int64)
            n = flat_pred.size
            m = min(n, self.pixel_sample_per_image)
            idx = np.arange(n) if n == m else rng.choice(n, size=m, replace=False)
            pix_scores_all.append(flat_pred[idx].astype(np.float32))
            pix_labels_all.append(flat_gt[idx].astype(np.int64))

        pix_scores_all = np.concatenate(pix_scores_all, axis=0)
        pix_labels_all = np.concatenate(pix_labels_all, axis=0)

        pix_auc = auroc_from_scores(pix_labels_all, pix_scores_all)
        self.pix_thr_norm = f1adaptive_threshold_1d(pix_scores_all, pix_labels_all)

        pix_pred = (pix_scores_all > self.pix_thr_norm).astype(np.int64)
        tp, tn, fp, fn = confusion_counts(pix_labels_all, pix_pred)
        pix_prec = precision_from_counts(tp, fp)

        pixel_aupro = float(aupro_metric.compute().item())

        self.pixel_metrics = {
            "global_min": self.global_min,
            "global_max": self.global_max,
            "pix_thr_norm": self.pix_thr_norm,
            "pixel_auroc_sampled": pix_auc,
            "pixel_precision_sampled": pix_prec,
            "pixel_aupro": pixel_aupro,
            "aupro_downsample": ds,
            "fpr_limit": self.fpr_limit,
        }

        print(f"[Pixel] pix_thr_norm={self.pix_thr_norm:.6f} (sampled)  global_min={self.global_min:.6f} global_max={self.global_max:.6f}")
        print(f"[Pixel] AUROC:     {pix_auc:.6f} (sampled)")
        print(f"[Pixel] AUPRO@FPR<={self.fpr_limit:.2f}: {pixel_aupro:.6f} (downsample={ds}x)")
        print(f"[Pixel] Precision: {pix_prec*100:.2f}%  (TP={tp}, FP={fp}) (sampled)")

    def save_overlays(self, eff_scores: np.ndarray, y_pred: np.ndarray) -> None:
        saved = 0
        tp_saved = fp_saved = 0

        for idx, s in enumerate(self.samples, 1):
            pred = int(y_pred[idx - 1])
            gt = int(s.gt_label)
            eff_s = float(eff_scores[idx - 1])

            defect = defect_name_from_path(s.path)
            print(f"{idx:06d} | {s.path.name} | raw={s.score_raw:.6f} eff={eff_s:.6f} | True={gt} Pred={pred} | folder={defect}")

            do_save = False
            subdir = ""

            if self.save_mode == "pred":
                do_save = (pred == 1)
                subdir = "Pred1"
            elif self.save_mode == "tp":
                do_save = (pred == 1 and gt == 1)
                subdir = "TP"
            elif self.save_mode == "all_pred":
                do_save = (pred == 1)
                subdir = "TP" if (pred == 1 and gt == 1) else "FP"
            else:
                raise ValueError(f"Unknown save_mode: {self.save_mode}")

            if not do_save:
                continue

            save_dir = self.out_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
            save_name = f"True_{gt}_Pred_{pred}_{defect}_{s.path.name}"
            save_path = save_dir / save_name

            _ = self.engine.save_contour_overlay(
                orig_rgb=s.orig_rgb,
                anomaly_map_hw=s.amap_hw,
                save_path=save_path,
                pixel_thr_norm=self.pix_thr_norm,
                global_min=self.global_min,
                global_max=self.global_max,
                min_area=self.min_area,
                contour_thickness=self.thickness,
            )

            saved += 1
            if pred == 1 and gt == 1:
                tp_saved += 1
            if pred == 1 and gt == 0:
                fp_saved += 1

        print("\n=== Saving ===")
        print(f"Saved(pred=1 only): {saved} (TP={tp_saved}, FP={fp_saved})")
        print(f"Saved to: {self.out_dir.resolve()}")

    def run(self, paths: List[Path]) -> None:
        self.collect(paths)
        y_true, scores, eff_scores, y_pred = self.compute_image_metrics()
        self.compute_pixel_metrics()
        self.save_overlays(eff_scores, y_pred)


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)   # folder (MVTec test) or single image
    parser.add_argument("--category", type=str, required=True)

    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--flow_steps", type=int, default=8)
    parser.add_argument("--image_size", type=int, nargs=2, default=[416, 416])  # H W
    parser.add_argument("--hidden_ratio", type=float, default=1.0)
    parser.add_argument("--clamp", type=float, default=2.0)
    parser.add_argument("--conv3x3_only", action="store_true")
    parser.add_argument("--topk_ratio", type=float, default=0.01)

    parser.add_argument("--save_dir", type=str, default="./inference_results")
    parser.add_argument("--min_area", type=int, default=30)
    parser.add_argument("--thickness", type=int, default=2)

    # Pixel region metric
    parser.add_argument("--fpr_limit", type=float, default=0.3)
    parser.add_argument("--aupro_downsample", type=int, default=2, help="Downsample factor for AUPRO memory. 1=no downsample.")
    parser.add_argument("--pixel_sample_per_image", type=int, default=5000, help="Sample size per image for pixel AUROC/threshold (speed).")

    # Save behavior:
    # pred     => save all Pred=1 (TP+FP)
    # tp       => save only TP
    # all_pred => save Pred=1 into TP/ and FP/ folders
    parser.add_argument("--save_mode", type=str, default="pred", choices=["pred", "tp", "all_pred"])

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    engine = FastFlowInferenceEngine(
        checkpoint_path=args.checkpoint_path,
        backbone=args.backbone,
        flow_steps=args.flow_steps,
        image_size=tuple(args.image_size),
        hidden_ratio=args.hidden_ratio,
        clamp=args.clamp,
        conv3x3_only=args.conv3x3_only,
        device=args.device,
        topk_ratio=args.topk_ratio,
    )

    evaluator = MVTecFastFlowEvaluator(
        engine=engine,
        out_dir=Path(args.save_dir),
        category=args.category,
        min_area=args.min_area,
        thickness=args.thickness,
        fpr_limit=args.fpr_limit,
        aupro_downsample=args.aupro_downsample,
        save_mode=args.save_mode,
        pixel_sample_per_image=args.pixel_sample_per_image,
    )

    ip = Path(args.image_path)
    paths = list(iter_images_recursive(ip)) if ip.is_dir() else [ip]
    if not paths:
        print("Found 0 images.")
        return

    evaluator.run(paths)


if __name__ == "__main__":
    main()
