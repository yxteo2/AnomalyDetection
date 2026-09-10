# ssn_inference.py
# OOP SuperSimpleNet (SSN) inference for MVTec:
# - exact Resize->CenterCrop preprocessing (torchvision v2)
# - image-level: Accuracy, Precision and AUROC with validation-calibrated thresholds
# - pixel-level: AUROC (sampled) + SELF-IMPLEMENTED AUPRO (region overlap) on anomaly maps
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
from anomaly_detection.modeling import SuperSimpleNetModel


class SSNInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        backbone: str,
        image_size: Tuple[int, int],
        perlin_threshold: float,
        adapt_cls_features: bool,
        layers: List[str],
        pretrained_backbone: bool,
        device: str,
        crop_scale: float = 0.875,
    ):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model_cfg = ckpt.get("model_cfg", {}) if isinstance(ckpt, dict) else {}
        backbone = model_cfg.get("backbone_name", backbone)
        image_size = tuple(model_cfg.get("input_size", image_size))
        perlin_threshold = float(model_cfg.get("perlin_threshold", perlin_threshold))
        adapt_cls_features = bool(model_cfg.get("adapt_cls_features", adapt_cls_features))
        layers = list(model_cfg.get("layers", layers))
        pretrained_backbone = bool(model_cfg.get("pretrained_backbone", pretrained_backbone))
        crop_scale = float(model_cfg.get("crop_scale", crop_scale))
        self.image_size = tuple(image_size)
        self.backbone = backbone

        # Build SSN model (must match training config)
        self.model = SuperSimpleNetModel(
            perlin_threshold=float(perlin_threshold),
            backbone_name=backbone,
            layers=list(layers),
            stop_grad=True,  # anomalib default for unsupervised
            adapt_cls_features=bool(adapt_cls_features),
            input_size=self.image_size,
            # The checkpoint includes the frozen backbone weights.
            pretrained_backbone=False,
        )

        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)

        self.model.to(self.device)
        self.model.eval()

        # EXACT transform: Resize -> CenterCrop -> Normalize (torchvision v2)
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
        x = self.transform(pil).unsqueeze(0)  # [1,3,h,w]
        return x, orig_rgb

    @torch.inference_mode()
    def infer_one(self, image_path: str) -> Tuple[float, np.ndarray, np.ndarray]:
        x, orig_rgb = self.preprocess(image_path)
        x = x.to(self.device)

        # SSN returns (pred_map, pred_score) in eval
        pred_map, pred_score = self.model(x)  # pred_map [1,1,h,w], pred_score [1]
        if not isinstance(pred_map, torch.Tensor) or not isinstance(pred_score, torch.Tensor):
            raise RuntimeError("SSN model should return (pred_map, pred_score) tensors in eval mode.")

        score_val = float(pred_score.reshape(-1)[0].item())
        amap_np = pred_map[0, 0].detach().cpu().numpy().astype(np.float32)  # [0,1]
        return score_val, amap_np, orig_rgb

    def gt_mask_to_crop(self, gt_mask_orig: np.ndarray, orig_rgb: np.ndarray) -> np.ndarray:
        """Convert original-size GT mask -> crop-space (h,w) mask aligned with anomaly map."""
        H0, W0 = orig_rgb.shape[:2]
        m = gt_mask_orig
        if m.shape[:2] != (H0, W0):
            m = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST)
        m = cv2.resize(m, (self.pre_w, self.pre_h), interpolation=cv2.INTER_NEAREST)
        t, l = self.crop_top, self.crop_left
        m = m[t : t + self.h, l : l + self.w]
        return ((m > 0).astype(np.uint8) * 255)

    def uncrop_mask_to_original(self, orig_rgb: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
        """Put crop-space mask back into pre-resize canvas, then resize to original image size."""
        H0, W0 = orig_rgb.shape[:2]
        canvas = np.zeros((self.pre_h, self.pre_w), dtype=np.uint8)
        t, l = self.crop_top, self.crop_left
        canvas[t : t + self.h, l : l + self.w] = mask_hw
        return cv2.resize(canvas, (W0, H0), interpolation=cv2.INTER_NEAREST)

    def save_contour_overlay(
        self,
        orig_rgb: np.ndarray,
        anomaly_map_hw: np.ndarray,  # already [0,1]
        save_path: Path,
        pixel_thr: float,
        min_area: int,
        contour_thickness: int,
    ) -> int:
        vis = np.clip(anomaly_map_hw.astype(np.float32), 0.0, 1.0)
        mask_hw = (vis >= float(pixel_thr)).astype(np.uint8) * 255

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


class MVTecSSNEvaluator:
    def __init__(
        self,
        engine: SSNInferenceEngine,
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
        self.out_dir = (out_dir / category / backbone)
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
        self.pix_thr: float = float(self.calibration.get("pixel_threshold", 0.5))

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
        ds = self.aupro_downsample
        aupro_metric = SelfAUPRO(fpr_limit=self.fpr_limit)

        rng = np.random.default_rng(123)
        pix_scores_all = []
        pix_labels_all = []

        for s in self.samples:
            # SSN map already [0,1]
            pred01 = np.clip(s.amap_hw.astype(np.float32), 0.0, 1.0)

            if s.gt_mask_orig is None:
                gt_crop_255 = np.zeros_like(pred01, dtype=np.uint8)
            else:
                gt_crop_255 = self.engine.gt_mask_to_crop(s.gt_mask_orig, s.orig_rgb)
            gt01 = (gt_crop_255 > 0).astype(np.float32)

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
                torch.from_numpy(pred_ds).unsqueeze(0),
                torch.from_numpy(gt_ds).unsqueeze(0),
            )

            # sampled pixels for pixel-AUROC + pixel-threshold
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
        pix_pred = (pix_scores_all >= self.pix_thr).astype(np.int64)
        tp, tn, fp, fn = confusion_counts(pix_labels_all, pix_pred)
        pix_prec = precision_from_counts(tp, fp)

        pixel_aupro = float(aupro_metric.compute().item())

        self.pixel_metrics = {
            "pix_thr": self.pix_thr,
            "pixel_auroc_sampled": pix_auc,
            "pixel_precision_sampled": pix_prec,
            "pixel_aupro": pixel_aupro,
            "aupro_downsample": ds,
            "fpr_limit": self.fpr_limit,
        }

        print(f"[Pixel] fixed validation threshold={self.pix_thr:.6f}")
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

            if self.save_mode == "pred":
                do_save, subdir = (pred == 1), "Pred1"
            elif self.save_mode == "tp":
                do_save, subdir = (pred == 1 and gt == 1), "TP"
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
                pixel_thr=self.pix_thr,
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

    # SSN model args (must match training)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "wide_resnet50_2"])
    parser.add_argument("--layers", type=str, nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--perlin_threshold", type=float, default=0.2)
    parser.add_argument("--adapt_cls_features", action="store_true")
    parser.add_argument("--pretrained_backbone", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--image_size", type=int, nargs=2, default=[416, 416])  # H W

    parser.add_argument("--save_dir", type=str, default="./ssn_inference_results")
    parser.add_argument("--min_area", type=int, default=30)
    parser.add_argument("--thickness", type=int, default=2)

    # Pixel region metric
    parser.add_argument("--fpr_limit", type=float, default=0.3)
    parser.add_argument("--aupro_downsample", type=int, default=2, help="Downsample factor for AUPRO memory. 1=no downsample.")
    parser.add_argument("--pixel_sample_per_image", type=int, default=5000, help="Sample size per image for pixel AUROC/threshold (speed).")

    # Save behavior
    parser.add_argument("--save_mode", type=str, default="pred", choices=["pred", "tp", "all_pred"])

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    calibration = load_calibration(args.checkpoint_path, args.calibration_path)

    engine = SSNInferenceEngine(
        checkpoint_path=args.checkpoint_path,
        backbone=args.backbone,
        image_size=tuple(args.image_size),
        perlin_threshold=args.perlin_threshold,
        adapt_cls_features=args.adapt_cls_features,
        layers=list(args.layers),
        pretrained_backbone=bool(args.pretrained_backbone),
        device=args.device,
    )

    evaluator = MVTecSSNEvaluator(
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
