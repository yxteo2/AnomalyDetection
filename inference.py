import argparse
from pathlib import Path
import math

import numpy as np
import torch
from PIL import Image
import cv2

from anomalib.metrics.threshold import F1AdaptiveThreshold
from model import FastFlowModel


# -----------------------------
# MVTec GT mask loader
# -----------------------------
def load_gt_mask_mvtec(img_path: Path) -> np.ndarray | None:
    """
    For MVTec:
      test/<defect>/xxx.png
      ground_truth/<defect>/<stem>_mask.png
    Returns: uint8 mask in ORIGINAL image size (H0,W0) with values {0,255}, or None if good/missing.
    """
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

    m = ((m > 0).astype(np.uint8) * 255)
    return m


# -----------------------------
# Helpers
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


def infer_gt_label_from_path(p: Path) -> int | None:
    parts = [x.lower() for x in p.parts]
    if "test" in parts:
        i = parts.index("test")
        if i + 1 < len(parts):
            return 0 if parts[i + 1] == "good" else 1
    return None


def defect_name_from_path(p: Path) -> str:
    parts = list(p.parts)
    lower = [x.lower() for x in parts]
    if "test" in lower:
        i = lower.index("test")
        if i + 1 < len(parts):
            return parts[i + 1]
    return p.parent.name


def compute_f1_threshold_1d(scores_1d: np.ndarray, labels_1d: np.ndarray) -> float:
    scores_t = torch.tensor(scores_1d, dtype=torch.float32)
    labels_t = torch.tensor(labels_1d, dtype=torch.int64)
    metric = F1AdaptiveThreshold()
    metric.update(scores_t, labels_t)
    thr = metric.compute()
    return float(thr.item())


# -----------------------------
# Inference
# -----------------------------
class FastFlowInference:
    def __init__(
        self,
        checkpoint_path: str,
        backbone: str = "resnet18",
        flow_steps: int = 8,
        image_size: tuple[int, int] = (416, 416),  # (H,W)
        hidden_ratio: float = 1.0,
        clamp: float = 2.0,
        conv3x3_only: bool = False,
        device: str = "cuda",
        topk_ratio: float = 0.01,
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

        # EXACT transform: Resize -> CenterCrop -> Normalize
        from torchvision.transforms import v2 as T

        h, w = self.image_size
        crop_scale = 0.875
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
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_image(self, image_path: str):
        pil = Image.open(image_path).convert("RGB")
        orig_rgb = np.array(pil)  # original size (H0,W0,3)
        x = self.transform(pil).unsqueeze(0)  # [1,3,h,w]
        return x, orig_rgb

    @torch.inference_mode()
    def get_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "get_anomaly_map"):
            return self.model.get_anomaly_map(x)
        return self.model(x)  # your model returns tensor [B,1,h,w]

    def gt_mask_to_crop(self, gt_mask_orig: np.ndarray, orig_rgb: np.ndarray) -> np.ndarray:
        """
        Convert ORIGINAL-size GT mask -> crop-space mask (h,w) matching anomaly_map.
        """
        H0, W0 = orig_rgb.shape[:2]
        m = gt_mask_orig

        if m.shape[:2] != (H0, W0):
            m = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST)

        # resize to pre_h/pre_w
        m = cv2.resize(m, (self.pre_w, self.pre_h), interpolation=cv2.INTER_NEAREST)

        # center crop
        t, l = self.crop_top, self.crop_left
        m = m[t:t + self.h, l:l + self.w]

        return ((m > 0).astype(np.uint8) * 255)

    def uncrop_mask_to_original(self, orig_rgb: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
        """
        mask_hw: (h,w) uint8 {0,255} in crop-space.
        Return: (H0,W0) uint8 {0,255} aligned to original.
        """
        H0, W0 = orig_rgb.shape[:2]
        canvas = np.zeros((self.pre_h, self.pre_w), dtype=np.uint8)
        t, l = self.crop_top, self.crop_left
        canvas[t:t + self.h, l:l + self.w] = mask_hw
        return cv2.resize(canvas, (W0, H0), interpolation=cv2.INTER_NEAREST)

    @torch.inference_mode()
    def score_and_map(self, image_path: str):
        x, orig_rgb = self.preprocess_image(image_path)
        x = x.to(self.device)

        amap = self.get_anomaly_map(x)  # [1,1,h,w]
        flat = amap.squeeze(1).flatten(1)  # [1, h*w]

        k = max(1, int(self.topk_ratio * flat.shape[1]))
        score = flat.topk(k, dim=1).values.mean(dim=1)

        score_val = float(score.item())
        amap_np = amap[0, 0].detach().cpu().numpy().astype(np.float32)  # (h,w)
        return score_val, amap_np, orig_rgb

    def save_contour_overlay(
        self,
        orig_rgb: np.ndarray,
        anomaly_map_hw: np.ndarray,
        save_path: Path,
        pixel_thr_norm: float,
        global_min: float,
        global_max: float,
        min_area: int = 30,
        contour_thickness: int = 2,
    ):
        """
        Normalize in crop-space using global min/max, threshold in crop-space,
        uncrop to original, find contours, draw on original image.
        """
        mn, mx = float(global_min), float(global_max)
        vis = (anomaly_map_hw - mn) / (mx - mn + 1e-12)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)

    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--flow_steps", type=int, default=8)
    parser.add_argument("--image_size", type=int, nargs=2, default=[416, 416])
    parser.add_argument("--hidden_ratio", type=float, default=1.0)
    parser.add_argument("--clamp", type=float, default=2.0)
    parser.add_argument("--conv3x3_only", action="store_true")
    parser.add_argument("--topk_ratio", type=float, default=0.01)

    parser.add_argument("--save_dir", type=str, default="./inference_results")
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
    paths = list(iter_images_recursive(ip)) if ip.is_dir() else [ip]
    if not paths:
        print("Found 0 images.")
        return

    out_dir = Path(args.save_dir) / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Pass 1: collect image scores + maps + GT masks (for thresholds)
    # -----------------------------
    valid = []  # list of tuples: (path, gt_label, score, amap_hw, orig_rgb, gt_mask_orig)
    for p in paths:
        gt = infer_gt_label_from_path(p)
        if gt is None:
            continue

        score, amap_hw, orig_rgb = infer.score_and_map(str(p))
        gt_mask_orig = load_gt_mask_mvtec(p)  # None for good

        valid.append((p, gt, score, amap_hw, orig_rgb, gt_mask_orig))

    if not valid:
        print("No images have inferable GT labels.")
        return

    # image-level threshold
    scores = np.array([v[2] for v in valid], dtype=np.float32)
    labels = np.array([v[1] for v in valid], dtype=np.int64)

    n = scores[labels == 0]
    a = scores[labels == 1]
    direction = 1.0
    if len(n) and len(a) and float(a.mean()) < float(n.mean()):
        direction = -1.0
    eff_scores = scores * direction
    img_thr = compute_f1_threshold_1d(eff_scores, labels)
    print(f"[ImageThr] direction={direction:+.0f} img_thr={img_thr:.6f}")

    # global min/max across ALL anomaly maps (crop-space)
    global_min = min(float(v[3].min()) for v in valid)
    global_max = max(float(v[3].max()) for v in valid)

    # pixel-level threshold from ALL pixels, normalized globally, using GT pixel labels in CROP SPACE
    pixel_scores_all = []
    pixel_labels_all = []
    den = (global_max - global_min) + 1e-12

    for (p, gt, score, amap_hw, orig_rgb, gt_mask_orig) in valid:
        if gt_mask_orig is None:
            gt_crop = np.zeros_like(amap_hw, dtype=np.uint8)
        else:
            gt_crop = infer.gt_mask_to_crop(gt_mask_orig, orig_rgb)

        pix_scores_norm = (amap_hw - global_min) / den
        pix_scores_norm = np.clip(pix_scores_norm, 0.0, 1.0)

        pixel_scores_all.append(pix_scores_norm.reshape(-1).astype(np.float32))
        pixel_labels_all.append((gt_crop.reshape(-1) > 0).astype(np.int64))

    pixel_scores_all = np.concatenate(pixel_scores_all)
    pixel_labels_all = np.concatenate(pixel_labels_all)

    pix_thr_norm = compute_f1_threshold_1d(pixel_scores_all, pixel_labels_all)
    print(f"[PixelThr] global_min={global_min:.6f} global_max={global_max:.6f} pix_thr_norm={pix_thr_norm:.6f}")

    # -----------------------------
    # Pass 2: evaluate + save contour overlays for predicted anomalies
    # -----------------------------
    TP = TN = FP = FN = 0
    saved = 0

    for idx, (p, gt, score, amap_hw, orig_rgb, gt_mask_orig) in enumerate(valid, 1):
        eff_s = float(score) * direction
        pred = 1 if eff_s > img_thr else 0

        if pred == 1 and gt == 1:
            TP += 1
        elif pred == 0 and gt == 0:
            TN += 1
        elif pred == 1 and gt == 0:
            FP += 1
        else:
            FN += 1

        defect = defect_name_from_path(p)
        print(f"{idx:06d} | {p.name} | raw={score:.6f} eff={eff_s:.6f} | True={gt} Pred={pred} | folder={defect}")

        if pred == 0:
            continue

        save_name = f"True_{gt}_Pred_{pred}_{defect}_{p.name}"
        save_path = out_dir / save_name

        infer.save_contour_overlay(
            orig_rgb=orig_rgb,
            anomaly_map_hw=amap_hw,
            save_path=save_path,
            pixel_thr_norm=pix_thr_norm,
            global_min=global_min,
            global_max=global_max,
            min_area=args.min_area,
            contour_thickness=args.thickness,
        )
        saved += 1

    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total else 0.0
    print("\n=== Summary ===")
    print(f"Total: {total}")
    print(f"TP: {TP}  TN: {TN}  FP: {FP}  FN: {FN}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Saved(pred=1 only): {saved}")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
