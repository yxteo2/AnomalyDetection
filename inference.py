import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2

from anomalib.metrics.threshold import F1AdaptiveThreshold
from model.fastflow import FastFlowModel


class FastFlowInference:
    def __init__(
        self,
        checkpoint_path: str,
        backbone: str = "resnet18",
        flow_steps: int = 8,
        image_size: tuple = (416, 416),  # (H, W)
        hidden_ratio: float = 1.0,
        clamp: float = 2.0,
        conv3x3_only: bool = False,
        device: str = "cuda",
        topk_ratio: float = 0.01,
    ):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.image_size = tuple(image_size)
        self.topk_ratio = float(topk_ratio)

        self.model = FastFlowModel(
            backbone_name=backbone,
            flow_steps=flow_steps,
            input_size=self.image_size,
            hidden_ratio=hidden_ratio,
            clamp=clamp,
            conv3x3_only=conv3x3_only,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state, strict=True)

        self.model.to(self.device)
        self.model.eval()

        from torchvision.transforms import v2 as T

        import math
        h, w = image_size
        crop_scale = 0.875
        pre_h = int(math.ceil(h / crop_scale))
        pre_w = int(math.ceil(w / crop_scale))  

        self.transform = T.Compose([
            T.ToImage(),                        # <-- important (PIL/np -> tensor image)
            T.Resize((pre_h, pre_w), antialias=True),
            T.CenterCrop((h, w)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_image(self, image_path: str):
        image_pil = Image.open(image_path).convert("RGB")
        orig = np.array(image_pil)                 # keep for visualization (HWC RGB uint8)

        x = self.transform(image_pil)              # <-- NO keyword args
        x = x.unsqueeze(0)                         # [1,3,H,W]
        return x, orig

    @torch.inference_mode()
    def get_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "get_anomaly_map"):
            return self.model.get_anomaly_map(x)
        return self.model(x)

    @torch.inference_mode()
    def score_only(self, image_path: str) -> float:
        x, _ = self.preprocess_image(image_path)
        x = x.to(self.device)

        amap = self.get_anomaly_map(x)             # [1,1,H,W]
        flat = amap.squeeze(1).flatten(1)          # [1,HW]
        k = max(1, int(self.topk_ratio * flat.shape[1]))
        score = flat.topk(k, dim=1).values.mean(dim=1)
        return float(score.item())

    @torch.inference_mode()
    def score_and_map(self, image_path: str):
        x, orig = self.preprocess_image(image_path)
        x = x.to(self.device)

        amap = self.get_anomaly_map(x)  # [1,1,H,W]
        flat = amap.squeeze(1).flatten(1)          # [1,HW]
        k = max(1, int(self.topk_ratio * flat.shape[1]))
        score = flat.topk(k, dim=1).values.mean(dim=1)
        score_val = float(score.item())

        amap_np = amap[0, 0].detach().cpu().numpy().astype(np.float32)  # HxW (model size)
        return score_val, amap_np, orig

    @staticmethod
    def _resize_map_to_orig(orig_rgb: np.ndarray, anomaly_map_hw: np.ndarray) -> np.ndarray:
        H, W = orig_rgb.shape[:2]
        if anomaly_map_hw.shape != (H, W):
            anomaly_map_hw = cv2.resize(anomaly_map_hw, (W, H), interpolation=cv2.INTER_LINEAR)
        return anomaly_map_hw

    @staticmethod
    def save_contour_only(
        orig_rgb: np.ndarray,
        anomaly_map_hw: np.ndarray,
        save_path: Path,
        pixel_thr: float = 0.5,
        min_area: int = 30,
        contour_thickness: int = 2,
    ):
        """
        Save only contour overlay (red contours) on the ORIGINAL image.
        """
        anomaly_map_hw = FastFlowInference._resize_map_to_orig(orig_rgb, anomaly_map_hw)

        mn, mx = float(anomaly_map_hw.min()), float(anomaly_map_hw.max())
        vis = (anomaly_map_hw - mn) / (mx - mn + 1e-12)

        mask = (vis >= pixel_thr).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # filter tiny blobs
        kept = [c for c in cnts if cv2.contourArea(c) >= float(min_area)]

        out_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        if kept:
            cv2.drawContours(out_bgr, kept, -1, (0, 0, 255), contour_thickness)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), out_bgr)


def iter_images_recursive(root: Path):
    """Recursively yields image files, skipping MVTec ground_truth masks."""
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if "ground_truth" in {x.lower() for x in p.parts}:
            continue
        yield p


def infer_gt_label_from_path(p: Path) -> int | None:
    """MVTec: .../test/good/... => 0; .../test/<defect>/... => 1"""
    parts = [x.lower() for x in p.parts]
    if "test" in parts:
        try:
            i = parts.index("test")
            if i + 1 < len(parts):
                return 0 if parts[i + 1] == "good" else 1
        except Exception:
            pass
    return None


def defect_name_from_path(p: Path) -> str:
    """
    Defect name = folder right after 'test' (MVTec style).
    If not found, fallback to parent folder name.
    """
    parts = list(p.parts)
    lower = [x.lower() for x in parts]
    if "test" in lower:
        i = lower.index("test")
        if i + 1 < len(parts):
            return parts[i + 1]
    return p.parent.name


def compute_f1adaptive_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    scores_t = torch.tensor(scores, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.int64)
    metric = F1AdaptiveThreshold()

    try:
        metric.update(scores_t, labels_t)
        thr = metric.compute()
        return float(thr.item() if hasattr(thr, "item") else thr)
    except Exception:
        thr = metric(scores_t, labels_t)
        return float(thr.item() if hasattr(thr, "item") else thr)


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

    # contour params
    parser.add_argument("--pixel_thr", type=float, default=0.5)
    parser.add_argument("--min_area", type=int, default=30)
    parser.add_argument("--thickness", type=int, default=2)

    args = parser.parse_args()

    infer = FastFlowInference(
        checkpoint_path=args.checkpoint_path,
        backbone=args.backbone,
        flow_steps=args.flow_steps,
        image_size=tuple(args.image_size),
        hidden_ratio=args.hidden_ratio,
        clamp=args.clamp,
        conv3x3_only=args.conv3x3_only,
        device="cuda",
        topk_ratio=args.topk_ratio,
    )

    ip = Path(args.image_path)
    out_dir = Path(args.save_dir) / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = list(iter_images_recursive(ip)) if ip.is_dir() else [ip]
    if len(paths) == 0:
        print("Found 0 images.")
        return

    # -------- Pass 1: collect scores + GT labels (needs MVTec test structure) --------
    valid_paths, gt_labels, raw_scores = [], [], []

    for p in paths:
        gt = infer_gt_label_from_path(p)
        if gt is None:
            continue
        s = infer.score_only(str(p))
        valid_paths.append(p)
        gt_labels.append(gt)
        raw_scores.append(s)

    if len(valid_paths) == 0:
        print("No images have inferable GT labels (expected MVTec: .../test/good or .../test/<defect>).")
        return

    raw_scores_np = np.asarray(raw_scores, dtype=np.float32)
    labels_np = np.asarray(gt_labels, dtype=np.int64)

    # -------- Auto-detect score direction --------
    n = raw_scores_np[labels_np == 0]
    a = raw_scores_np[labels_np == 1]
    direction = 1.0
    if len(n) > 0 and len(a) > 0:
        if float(a.mean()) < float(n.mean()):
            direction = -1.0
        print(f"Mean score normal: {float(n.mean()):.6f}")
        print(f"Mean score anomaly: {float(a.mean()):.6f}")
        print(f"Using direction = {direction:+.0f} (effective_score = direction * raw_score)")
    eff_scores_np = raw_scores_np * direction

    threshold = compute_f1adaptive_threshold(eff_scores_np, labels_np)
    print(f"F1AdaptiveThreshold (effective score) = {threshold:.6f} (N={len(valid_paths)})")

    # -------- Pass 2: print all results, save ONLY predicted defect images (contour only) --------
    TP = TN = FP = FN = 0
    saved = 0

    for idx, (p, gt, raw_s, eff_s) in enumerate(zip(valid_paths, gt_labels, raw_scores_np, eff_scores_np), 1):
        pred = 1 if (eff_s > threshold) else 0

        if pred == 1 and gt == 1:
            TP += 1
        elif pred == 0 and gt == 0:
            TN += 1
        elif pred == 1 and gt == 0:
            FP += 1
        else:
            FN += 1

        defect_name = defect_name_from_path(p)
        print(
            f"{idx:06d} | {p.name} | raw={float(raw_s):.6f} eff={float(eff_s):.6f} | "
            f"True={gt} Pred={pred} | folder={defect_name}"
        )

        # ✅ only save predicted defect
        if pred == 0:
            continue

        # compute map only when saving
        score_val, amap, orig = infer.score_and_map(str(p))

        save_name = f"TrueLabel_{gt}_Predicted_{pred}_{defect_name}_{p.name}"
        save_path = out_dir / save_name

        infer.save_contour_only(
            orig,
            amap,
            save_path,
            pixel_thr=args.pixel_thr,
            min_area=args.min_area,
            contour_thickness=args.thickness,
        )
        saved += 1

    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total > 0 else 0.0

    print("\n=== Summary ===")
    print(f"Total: {total}")
    print(f"TP: {TP}  TN: {TN}  FP: {FP}  FN: {FN}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Saved(pred=1 only): {saved}")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
