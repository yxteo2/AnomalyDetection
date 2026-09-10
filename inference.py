# inference.py
# OOP FastFlow inference for MVTec:
# - exact Resize->CenterCrop preprocessing (torchvision v2)
# - image-level: Accuracy, Precision and AUROC with validation-calibrated thresholds
# - pixel-level: AUROC (sampled) + **SELF-IMPLEMENTED AUPRO** (region overlap) on globally-normalized maps
# - contour overlay drawn on ORIGINAL image, but thresholding done in crop-space then "uncropped" back

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from anomaly_detection.inference_utils import (
    Sample,
    SelfAUPRO,
    accuracy_from_counts,
    auroc_from_scores,
    confusion_counts,
    defect_name_from_path,
    infer_gt_label_from_path,
    iter_images_recursive,
    load_calibration,
    load_gt_mask_mvtec,
    precision_from_counts,
)
from anomaly_detection.modeling import FastFlowModel


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
        pretrained_backbone: bool,
        device: str,
        topk_ratio: float,
        crop_scale: float = 0.875,
    ):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model_cfg = ckpt.get("model_cfg", {}) if isinstance(ckpt, dict) else {}
        backbone = model_cfg.get("backbone_name", backbone)
        flow_steps = int(model_cfg.get("flow_steps", flow_steps))
        image_size = tuple(model_cfg.get("input_size", image_size))
        hidden_ratio = float(model_cfg.get("hidden_ratio", hidden_ratio))
        clamp = float(model_cfg.get("clamp", clamp))
        conv3x3_only = bool(model_cfg.get("conv3x3_only", conv3x3_only))
        pretrained_backbone = bool(model_cfg.get("pretrained_backbone", pretrained_backbone))
        crop_scale = float(model_cfg.get("crop_scale", crop_scale))
        reducer_channels = model_cfg.get("reducer_channels")

        self.image_size = tuple(image_size)
        self.backbone = backbone
        self.topk_ratio = float(topk_ratio)
        if not 0.0 < self.topk_ratio <= 1.0:
            raise ValueError(f"topk_ratio must be in (0, 1], got {self.topk_ratio}")

        if backbone == "wide_resnet50_2":
            self.model = FastFlowModel(
                backbone_name=backbone,
                flow_steps=flow_steps,
                input_size=self.image_size,
                hidden_ratio=hidden_ratio,
                reducer_channels=tuple(reducer_channels or (128, 192, 256)),
                clamp=clamp,
                conv3x3_only=conv3x3_only,
                # The checkpoint contains the frozen backbone weights; avoid a
                # redundant network download before strict state loading.
                pretrained_backbone=False,
            )
        else:
            self.model = FastFlowModel(
                backbone_name=backbone,
                flow_steps=flow_steps,
                input_size=self.image_size,
                hidden_ratio=hidden_ratio,
                clamp=clamp,
                conv3x3_only=conv3x3_only,
                pretrained_backbone=False,
            )

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
        pixel_threshold: float,
        min_area: int,
        contour_thickness: int,
    ) -> int:
        mask_hw = (anomaly_map_hw >= float(pixel_threshold)).astype(np.uint8) * 255

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
        backbone: str = "resnet18",
        calibration: Optional[dict] = None,
    ):
        self.engine = engine
        self.out_dir = out_dir / category / backbone
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.category = category
        self.min_area = int(min_area)
        self.thickness = int(thickness)
        self.fpr_limit = float(fpr_limit)
        self.aupro_downsample = max(1, int(aupro_downsample))
        self.save_mode = save_mode
        self.pixel_sample_per_image = int(pixel_sample_per_image)
        self.calibration = calibration or {}

        self.samples: List[Sample] = []

        self.img_thr: float = float(self.calibration.get("image_threshold", 0.0))
        self.global_min: float = 0.0
        self.global_max: float = 1.0
        self.pixel_threshold: float = float(self.calibration.get("pixel_threshold", 0.0))

        self.image_metrics = {}
        self.pixel_metrics = {}

    def collect(self, paths: List[Path]) -> None:
        self.samples.clear()
        for p in paths:
            gt = infer_gt_label_from_path(p)
            score, amap_hw, orig_rgb = self.engine.infer_one(str(p))
            gt_mask_orig = load_gt_mask_mvtec(p)
            self.samples.append(
                Sample(
                    path=p,
                    gt_label=int(gt) if gt is not None else None,
                    score_raw=float(score),
                    amap_hw=amap_hw,
                    orig_rgb=orig_rgb,
                    gt_mask_orig=gt_mask_orig,
                )
            )
        if not self.samples:
            raise RuntimeError("No readable images were collected.")

    def compute_image_metrics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y_true = np.array([s.gt_label if s.gt_label is not None else -1 for s in self.samples], dtype=np.int64)
        scores = np.array([s.score_raw for s in self.samples], dtype=np.float32)

        eff_scores = scores
        y_pred = (eff_scores >= self.img_thr).astype(np.int64)
        labeled = y_true >= 0
        tp, tn, fp, fn = confusion_counts(y_true[labeled], y_pred[labeled]) if labeled.any() else (0, 0, 0, 0)
        acc = accuracy_from_counts(tp, tn, fp, fn) if labeled.any() else float("nan")
        prec = precision_from_counts(tp, fp) if labeled.any() else float("nan")
        auc = auroc_from_scores(y_true[labeled], eff_scores[labeled]) if labeled.any() else float("nan")

        self.image_metrics = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc,
            "precision": prec,
            "auroc": auc,
            "threshold": self.img_thr,
        }

        print(f"[Image] fixed validation threshold={self.img_thr:.6f}")
        if labeled.any():
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
            # Keep raw model scores for fixed validation-threshold application.
            # AUROC is invariant to the monotonic min-max transform used by AUPRO.
            flat_pred = s.amap_hw.reshape(-1)
            flat_gt = gt01.reshape(-1).astype(np.int64)
            n = flat_pred.size
            m = min(n, self.pixel_sample_per_image)
            idx = np.arange(n) if n == m else rng.choice(n, size=m, replace=False)
            pix_scores_all.append(flat_pred[idx].astype(np.float32))
            pix_labels_all.append(flat_gt[idx].astype(np.int64))

        pix_scores_all = np.concatenate(pix_scores_all, axis=0)
        pix_labels_all = np.concatenate(pix_labels_all, axis=0)

        pix_auc = auroc_from_scores(pix_labels_all, pix_scores_all)
        pix_pred = (pix_scores_all >= self.pixel_threshold).astype(np.int64)
        tp, tn, fp, fn = confusion_counts(pix_labels_all, pix_pred)
        pix_prec = precision_from_counts(tp, fp)

        pixel_aupro = float(aupro_metric.compute().item())

        self.pixel_metrics = {
            "global_min": self.global_min,
            "global_max": self.global_max,
            "pixel_threshold": self.pixel_threshold,
            "pixel_auroc_sampled": pix_auc,
            "pixel_precision_sampled": pix_prec,
            "pixel_aupro": pixel_aupro,
            "aupro_downsample": ds,
            "fpr_limit": self.fpr_limit,
        }

        print(f"[Pixel] fixed validation threshold={self.pixel_threshold:.6f}")
        print(f"[Pixel] AUROC:     {pix_auc:.6f} (sampled)")
        print(f"[Pixel] AUPRO@FPR<={self.fpr_limit:.2f}: {pixel_aupro:.6f} (downsample={ds}x)")
        print(f"[Pixel] Precision: {pix_prec*100:.2f}%  (TP={tp}, FP={fp}) (sampled)")

    def save_overlays(self, eff_scores: np.ndarray, y_pred: np.ndarray) -> None:
        saved = 0
        tp_saved = fp_saved = 0

        for idx, s in enumerate(self.samples, 1):
            pred = int(y_pred[idx - 1])
            gt = int(s.gt_label) if s.gt_label is not None else -1
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
                pixel_threshold=self.pixel_threshold,
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
        _, _, eff_scores, y_pred = self.compute_image_metrics()
        if any(sample.gt_label is not None for sample in self.samples):
            self.compute_pixel_metrics()
        else:
            print("[Metrics] Unlabeled input: dataset metrics skipped.")
        self.save_overlays(eff_scores, y_pred)


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--calibration_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, required=True)   # folder (MVTec test) or single image
    parser.add_argument("--category", type=str, required=True)

    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--flow_steps", type=int, default=8)
    parser.add_argument("--image_size", type=int, nargs=2, default=[416, 416])  # H W
    parser.add_argument("--hidden_ratio", type=float, default=1.0)
    parser.add_argument("--clamp", type=float, default=2.0)
    parser.add_argument("--conv3x3_only", action="store_true")
    parser.add_argument("--pretrained_backbone", action=argparse.BooleanOptionalAction, default=True)
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
    calibration = load_calibration(args.checkpoint_path, args.calibration_path)

    engine = FastFlowInferenceEngine(
        checkpoint_path=args.checkpoint_path,
        backbone=args.backbone,
        flow_steps=args.flow_steps,
        image_size=tuple(args.image_size),
        hidden_ratio=args.hidden_ratio,
        clamp=args.clamp,
        conv3x3_only=args.conv3x3_only,
        pretrained_backbone=bool(args.pretrained_backbone),
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
        backbone=engine.backbone,
        calibration=calibration,
    )

    ip = Path(args.image_path)
    paths = list(iter_images_recursive(ip)) if ip.is_dir() else [ip]
    if not paths:
        print("Found 0 images.")
        return

    evaluator.run(paths)


if __name__ == "__main__":
    main()
