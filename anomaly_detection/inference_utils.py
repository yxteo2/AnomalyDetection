"""Shared metrics, calibration and MVTec file helpers for inference."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch


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


def load_calibration(checkpoint_path: str, calibration_path: Optional[str]) -> dict:
    candidates = []
    if calibration_path:
        candidates.append(Path(calibration_path))
    checkpoint = Path(checkpoint_path)
    candidates.extend(
        [checkpoint.parent / "calibration.json", checkpoint.parent.parent / "calibration.json"]
    )
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as f:
                calibration = json.load(f)
            if "image_threshold" not in calibration or "pixel_threshold" not in calibration:
                raise ValueError(f"Invalid calibration file: {candidate}")
            print(f"Loaded calibration: {candidate}")
            return calibration
    raise FileNotFoundError(
        "No calibration.json found. Run train.py with the updated pipeline or pass --calibration_path."
    )


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
# Collected prediction
# -----------------------------
@dataclass
class Sample:
    path: Path
    gt_label: Optional[int]
    score_raw: float
    amap_hw: np.ndarray
    orig_rgb: np.ndarray
    gt_mask_orig: Optional[np.ndarray]
