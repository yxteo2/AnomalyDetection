import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2

from anomalib.metrics.threshold import F1AdaptiveThreshold

# -----------------------------
# Project models (your codebase)
# -----------------------------
# NOTE: Do NOT redefine SuperSimpleNetModel in this file.
from model import FastFlowModel, SuperSimpleNetModel


# =============================
# Helpers
# =============================

def iter_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            # skip mvtec ground-truth folders by default (if present)
            if "ground_truth" in {x.lower() for x in p.parts}:
                continue
            yield p


def infer_gt_label_from_path(p: Path) -> int | None:
    """Infer GT label from MVTec-style path.

    Supports:
      .../test/good/...
      .../test/<defect>/...
      .../test/images/good/...
      .../test/images/<defect>/...
    Returns: 0 normal, 1 anomaly, or None if can't infer.
    """
    parts = [x.lower() for x in p.parts]
    # try to locate "test"
    if "test" not in parts:
        return None
    idx = parts.index("test")
    # patterns:
    # test/good/xxx.png
    # test/<defect>/xxx.png
    # test/images/good/xxx.png
    # test/images/<defect>/xxx.png
    after = parts[idx + 1 :]
    if not after:
        return None
    if after[0] == "images" and len(after) >= 2:
        cls = after[1]
    else:
        cls = after[0]
    if cls == "good":
        return 0
    return 1


def normalize01(x: np.ndarray) -> np.ndarray:
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def save_heatmap_overlay(
    orig_bgr: np.ndarray,
    amap01: np.ndarray,
    out_path: Path,
    alpha: float = 0.5,
):
    """Save heatmap overlay image (BGR)."""
    h, w = orig_bgr.shape[:2]
    amap = cv2.resize(amap01, (w, h), interpolation=cv2.INTER_LINEAR)
    amap8 = np.clip(amap * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(amap8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_bgr, 1 - alpha, heat, alpha, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


# =============================
# Robust SSN checkpoint loading
# =============================

def _state_key_candidates(k: str):
    """Generate possible alternative key names for legacy checkpoints."""
    yield k

    # common wrappers
    for pref in ("model.", "module."):
        if k.startswith(pref):
            yield k[len(pref):]
        else:
            yield pref + k

    # adaptor naming variants
    if k.startswith("adaptor."):
        yield "adapter.projection." + k[len("adaptor."):]
    if k.startswith("adapter.projection."):
        yield "adaptor." + k[len("adapter.projection."):]

    # seg/cls block naming variants
    if k.startswith("segdec."):
        yield "disc." + k[len("segdec."):]
    if k.startswith("disc."):
        yield "segdec." + k[len("disc."):]

    # extractor naming variants
    if k.startswith("feature_extractor.feature_extractor."):
        yield "extractor.net." + k[len("feature_extractor.feature_extractor."):]
    if k.startswith("extractor.net."):
        yield "feature_extractor.feature_extractor." + k[len("extractor.net."):]
    if k.startswith("feature_extractor."):
        yield "extractor." + k[len("feature_extractor."):]
    if k.startswith("extractor."):
        yield "feature_extractor." + k[len("extractor."):]


def _remap_state_keys_to_model(state: dict, model_sd: dict) -> dict:
    """Try to remap checkpoint state keys to match model keys."""
    model_keys = set(model_sd.keys())
    remapped = {}

    for k, v in state.items():
        # direct match
        if k in model_keys:
            remapped[k] = v
            continue

        # try candidates
        cands = list(_state_key_candidates(k))

        # special-case exact keys adaptor.weight/bias
        if k == "adaptor.weight":
            cands.append("adaptor.projection.weight")
        if k == "adaptor.bias":
            cands.append("adaptor.projection.bias")

        # already in new style
        # feature_extractor.feature_extractor.* etc

        nk = None
        for cand in cands:
            if cand in model_keys:
                nk = cand
                break
        if nk is not None:
            remapped[nk] = v

    return remapped


def load_state_dict_forgiving(model: torch.nn.Module, state: dict, verbose: bool = True):
    """Load only keys that exist AND match shape.

    Returns (missing_keys, unexpected_keys, skipped_shape_keys)
    """
    model_sd = model.state_dict()

    # remap to match model naming first
    state = _remap_state_keys_to_model(state, model_sd)

    filtered = {}
    skipped = []
    for k, v in state.items():
        if k not in model_sd:
            continue
        if tuple(model_sd[k].shape) != tuple(v.shape):
            skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        filtered[k] = v

    ret = model.load_state_dict(filtered, strict=False)

    missing = getattr(ret, "missing_keys", [])
    unexpected = getattr(ret, "unexpected_keys", [])

    if verbose:
        print(f"[ForgivingLoad] loaded={len(filtered)} missing={len(missing)} unexpected={len(unexpected)} skipped_shape={len(skipped)}")
        if skipped:
            print("[ForgivingLoad] First 10 skipped (shape mismatch):")
            for k, s1, s2 in skipped[:10]:
                print(f"  {k} ckpt{list(s1)} != model{list(s2)}")

    return missing, unexpected, skipped


# =============================
# Inference wrappers
# =============================

class SuperSimpleNetInference:
    """Robust SSN inference that supports:
    - your new anomalib-aligned SSN checkpoints
    - legacy checkpoints (adapter.projection/segdec)
    - legacy/project checkpoints (extractor.net.* naming)
    """

    def __init__(
        self,
        checkpoint_path: str,
        backbone: str = "resnet18",
        image_size: tuple = (416, 416),
        device: str = "cuda",
        topk_ratio: float = 0.01,
        # new SSN args
        perlin_threshold: float = 0.2,
        adapt_cls_features: bool = False,
        layers: list[str] = None,
        stop_grad: bool = True,
        pretrained_backbone: bool = True,
    ):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.image_size = tuple(image_size)
        self.topk_ratio = float(topk_ratio)

        if layers is None:
            layers = ["layer2", "layer3"]

        # ---- load checkpoint first (for inspection) ----
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        if not isinstance(state, dict):
            raise RuntimeError("Checkpoint does not look like a state_dict or dict containing model_state_dict.")

        # ---- if checkpoint carries saved model config, prefer it (avoids backbone/layer mismatches) ----
        if isinstance(ckpt, dict) and isinstance(ckpt.get("model_cfg", None), dict):
            cfg = ckpt["model_cfg"]

            # Keep a copy of CLI args for debugging.
            cli_backbone = backbone
            cli_image_size = tuple(self.image_size)

            backbone = cfg.get("backbone_name", backbone)
            cfg_input_size = cfg.get("input_size", None)
            if cfg_input_size is not None:
                self.image_size = tuple(cfg_input_size)

            layers = cfg.get("layers", layers)
            perlin_threshold = cfg.get("perlin_threshold", perlin_threshold)
            adapt_cls_features = cfg.get("adapt_cls_features", adapt_cls_features)
            stop_grad = cfg.get("stop_grad", stop_grad)
            pretrained_backbone = cfg.get("pretrained_backbone", pretrained_backbone)

            if cli_backbone != backbone or cli_image_size != tuple(self.image_size):
                print(
                    f"[SSN] Using model_cfg from checkpoint: backbone={backbone}, input_size={tuple(self.image_size)}, layers={layers} "
                    f"(CLI was backbone={cli_backbone}, image_size={cli_image_size})"
                )

        # strip wrappers
        for pref in ("model.", "module."):
            if any(k.startswith(pref) for k in state.keys()):
                state = {k[len(pref):] if k.startswith(pref) else k: v for k, v in state.items()}

        # detect legacy ckpt naming
        legacy = any(k.startswith("extractor.net.") for k in state.keys())

        # detect resnet34-ish checkpoint (layer1.2 exists in resnet34/50 etc, not resnet18)
        if any(".layer1.2." in k for k in state.keys()):
            print("[Warn][SSN] Checkpoint contains layer1.2.* -> this is NOT resnet18. Likely resnet34.")
            print("[Warn][SSN] If you pass --backbone resnet18, strict load will fail or mapping will skip many keys.")

        # ---- build YOUR new anomalib-aligned SSN model ----
        # IMPORTANT: this must match your pasted SuperSimpleNetModel signature
        self.model = SuperSimpleNetModel(
            perlin_threshold=float(perlin_threshold),
            backbone_name=backbone,
            layers=list(layers),
            stop_grad=bool(stop_grad),
            adapt_cls_features=bool(adapt_cls_features),
            input_size=self.image_size,
            pretrained_backbone=bool(pretrained_backbone),
        )

        # ---- extend remapper to handle adaptor.weight -> adaptor.projection.weight ----
        def _remap_state_keys_to_model_extended(state: dict, model_sd: dict) -> dict:
            model_keys = set(model_sd.keys())
            remapped = {}

            for k, v in state.items():
                if k in model_keys:
                    remapped[k] = v
                    continue

                cands = list(_state_key_candidates(k))

                # extra: adaptor.weight/bias
                if k == "adaptor.weight":
                    cands.append("adaptor.projection.weight")
                if k == "adaptor.bias":
                    cands.append("adaptor.projection.bias")

                nk = None
                for cand in cands:
                    if cand in model_keys:
                        nk = cand
                        break
                if nk is not None:
                    remapped[nk] = v

            return remapped

        # ---- forgiving load (shape-safe) ----
        model_sd = self.model.state_dict()
        remapped = _remap_state_keys_to_model_extended(state, model_sd)

        filtered = {}
        skipped = []
        for k, v in remapped.items():
            if k not in model_sd:
                continue
            if tuple(model_sd[k].shape) != tuple(v.shape):
                skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
                continue
            filtered[k] = v

        ret = self.model.load_state_dict(filtered, strict=False)
        missing = getattr(ret, "missing_keys", [])
        unexpected = getattr(ret, "unexpected_keys", [])

        print(f"[SSN] legacy_ckpt={legacy} | loaded_keys={len(filtered)} | missing={len(missing)} | unexpected={len(unexpected)}")
        if skipped:
            print(f"[SSN] Skipped(shape mismatch) {len(skipped)} keys. First 10:")
            for k, s1, s2 in skipped[:10]:
                print(f"  skip: {k} ckpt{list(s1)} != model{list(s2)}")

        self.model.to(self.device)
        self.model.eval()

        # Preproc MUST match training
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.image_size, Image.BILINEAR)
        img_np = np.asarray(img).astype(np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
        # keep original for overlay
        orig_bgr = cv2.cvtColor(np.asarray(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        return x, orig_bgr

    @torch.inference_mode()
    def score_and_map(self, image_path: str):
        x, orig = self.preprocess_image(image_path)
        x = x.to(self.device)

        pred_map, pred_score = self.model(x)  # eval returns prob map + prob score
        if pred_map.ndim == 4 and pred_map.shape[1] == 1:
            amap = pred_map
        else:
            raise RuntimeError(f"Unexpected pred_map shape: {tuple(pred_map.shape)}")

        # Use the model's intended image-level anomaly score (matches training/eval pipeline).
        # SSN eval returns (prob_map, prob_score).
        score_val = float(pred_score.detach().reshape(-1)[0].item())

        amap_np = amap[0, 0].detach().cpu().numpy().astype(np.float32)
        return score_val, amap_np, orig

    @torch.inference_mode()
    def score_only(self, image_path: str) -> float:
        s, _, _ = self.score_and_map(image_path)
        return float(s)


class FastFlowInference:
    def __init__(
        self,
        checkpoint_path: str,
        backbone: str = "resnet18",
        image_size: tuple = (416, 416),
        device: str = "cuda",
    ):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.image_size = tuple(image_size)

        self.model = FastFlowModel(
            backbone_name=backbone,
            input_size=self.image_size,
            pretrained_backbone=True,
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        if not isinstance(state, dict):
            raise RuntimeError("Checkpoint does not look like a state_dict or dict containing model_state_dict.")

        for pref in ("model.", "module."):
            if any(k.startswith(pref) for k in state.keys()):
                state = {k[len(pref):] if k.startswith(pref) else k: v for k, v in state.items()}

        # forgiving load
        load_state_dict_forgiving(self.model, state, verbose=True)

        self.model.to(self.device)
        self.model.eval()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.image_size, Image.BILINEAR)
        img_np = np.asarray(img).astype(np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        orig_bgr = cv2.cvtColor(np.asarray(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        return x, orig_bgr

    @torch.inference_mode()
    def score_and_map(self, image_path: str):
        x, orig = self.preprocess_image(image_path)
        x = x.to(self.device)

        out = self.model(x)
        # Expect anomalib-like output dict or tuple
        if isinstance(out, dict):
            amap = out.get("anomaly_map", None)
            score = out.get("pred_score", None) or out.get("anomaly_score", None)
            if amap is None or score is None:
                raise RuntimeError(f"Unexpected output dict keys: {list(out.keys())}")
            if isinstance(score, torch.Tensor):
                score_val = float(score.detach().reshape(-1)[0].item())
            else:
                score_val = float(score)
            amap_np = amap.detach().cpu().numpy()
            if amap_np.ndim == 4:
                amap_np = amap_np[0, 0]
            elif amap_np.ndim == 3:
                amap_np = amap_np[0]
            return score_val, amap_np.astype(np.float32), orig

        # tuple fallback
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            pred_map, pred_score = out[0], out[1]
            score_val = float(pred_score.detach().reshape(-1)[0].item())
            amap_np = pred_map.detach().cpu().numpy()
            if amap_np.ndim == 4:
                amap_np = amap_np[0, 0]
            elif amap_np.ndim == 3:
                amap_np = amap_np[0]
            return score_val, amap_np.astype(np.float32), orig

        raise RuntimeError(f"Unsupported FastFlow output type: {type(out)}")

    @torch.inference_mode()
    def score_only(self, image_path: str) -> float:
        s, _, _ = self.score_and_map(image_path)
        return float(s)


# =============================
# Main
# =============================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="fastflow", choices=["fastflow", "ssn"])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--category", type=str, default=None, help="MVTec category name (optional, for GT inference)")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--image_size", type=int, nargs=2, default=[416, 416])
    parser.add_argument("--device", type=str, default="cuda")

    # thresholding / output
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--topk_ratio", type=float, default=0.01, help="SSN: (legacy) top-k ratio if you want to compare/debug")

    # SSN args (will be overridden by checkpoint model_cfg if present)
    parser.add_argument("--perlin_threshold", type=float, default=0.2)
    parser.add_argument("--adapt_cls_features", action="store_true")
    parser.add_argument("--layers", type=str, nargs="*", default=None)
    parser.add_argument("--stop_grad", action="store_true", default=True)
    parser.add_argument("--no_stop_grad", dest="stop_grad", action="store_false")
    parser.add_argument("--pretrained_backbone", action="store_true", default=True)
    parser.add_argument("--no_pretrained_backbone", dest="pretrained_backbone", action="store_false")

    args = parser.parse_args()

    image_size = tuple(args.image_size)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Select inference wrapper
    if args.model == "ssn":
        infer = SuperSimpleNetInference(
            checkpoint_path=args.checkpoint_path,
            backbone=args.backbone,
            image_size=image_size,
            device=args.device,
            topk_ratio=args.topk_ratio,
            perlin_threshold=args.perlin_threshold,
            adapt_cls_features=args.adapt_cls_features,
            layers=args.layers,
            stop_grad=args.stop_grad,
            pretrained_backbone=args.pretrained_backbone,
        )
    else:
        infer = FastFlowInference(
            checkpoint_path=args.checkpoint_path,
            backbone=args.backbone,
            image_size=image_size,
            device=args.device,
        )

    # Collect images
    img_root = Path(args.image_path)
    images = [p for p in iter_images(img_root)]
    if not images:
        raise RuntimeError(f"No images found under: {img_root}")

    # Run inference and gather scores for adaptive threshold
    scores = []
    paths = []
    maps = []
    origs = []
    gt_labels = []

    for p in images:
        s, amap, orig = infer.score_and_map(str(p))
        scores.append(float(s))
        paths.append(p)
        maps.append(amap)
        origs.append(orig)
        gt = infer_gt_label_from_path(p)
        gt_labels.append(gt if gt is not None else -1)

    scores_np = np.array(scores, dtype=np.float32)

    # Adaptive F1 threshold (if GT available)
    gt_available = all(x in (0, 1) for x in gt_labels)
    if gt_available:
        y_true = torch.tensor(gt_labels, dtype=torch.int)
        y_score = torch.tensor(scores_np, dtype=torch.float32)
        thr_estimator = F1AdaptiveThreshold()
        thr = float(thr_estimator(y_score, y_true).item())
        print(f"[Threshold] F1AdaptiveThreshold = {thr:.6f}")
    else:
        thr = float(np.median(scores_np))
        print(f"[Threshold] No GT inferred. Using median score as threshold = {thr:.6f}")

    # Save results
    out_dir = save_dir / f"{args.model}_{args.category or 'dataset'}"
    out_dir.mkdir(parents=True, exist_ok=True)

    TP = TN = FP = FN = 0
    saved = 0

    for p, s, amap, orig, gt in zip(paths, scores, maps, origs, gt_labels):
        pred = 1 if s >= thr else 0
        if gt_available:
            if pred == 1 and gt == 1:
                TP += 1
            elif pred == 0 and gt == 0:
                TN += 1
            elif pred == 1 and gt == 0:
                FP += 1
            elif pred == 0 and gt == 1:
                FN += 1

        # save overlay only for predicted anomalies (as you had)
        if args.save_overlays and pred == 1:
            amap01 = normalize01(amap)
            rel = p.relative_to(img_root)
            out_path = out_dir / rel
            out_path = out_path.with_suffix(".png")
            save_heatmap_overlay(orig, amap01, out_path, alpha=float(args.alpha))
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
