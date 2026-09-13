"""Reload any YAML-built detector and predict one image using saved transforms."""

import argparse
from pathlib import Path

import torch

from anomaly_detection.builder import build_model
from anomaly_detection.preprocessing import InferencePreprocessing


def load_model(checkpoint_path, device="cpu"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but is unavailable.")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not saved.get("experiment_cfg") or "model_state_dict" not in saved:
        raise ValueError("Shared inference requires a full YAML-training checkpoint with experiment_cfg and model_state_dict. "
                         "For legacy FastFlow/SSN checkpoints, use the model-specific inference script with explicit settings.")
    cfg = {**saved["experiment_cfg"], "pretrained_backbone": False}
    model = build_model(cfg)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return model.to(device).eval(), cfg


class AnomalyInferenceEngine(InferencePreprocessing):
    """Load once and reuse across images, for all five YAML-built detectors."""
    def __init__(self, checkpoint_path, device="cpu"):
        self.model, self.cfg = load_model(checkpoint_path, device)
        self.device = torch.device(device)
        self.setup_preprocessing(self.cfg["image_size"], self.cfg["crop_scale"])

    @torch.inference_mode()
    def predict(self, image_path, restore_original=False):
        tensor, original = self.preprocess(image_path)
        result = self.model(tensor.to(self.device))
        if self.cfg["model"] == "fastflow":
            # Match FastFlowEvaluator's default top-one-percent aggregation.
            maps = result
            flat = maps.flatten(1)
            scores = flat.topk(max(1, int(0.01 * flat.shape[1])), dim=1).values.mean(1)
        else:
            maps, scores = result
        if tuple(maps.shape) != (1, 1, self.h, self.w) or scores.numel() != 1:
            raise RuntimeError("Unexpected detector output dimensions.")
        if not torch.isfinite(maps).all() or not torch.isfinite(scores).all():
            raise RuntimeError("Detector produced non-finite predictions.")
        prediction = {"anomaly_map": maps.cpu(), "score": scores.cpu(),
                      "geometry": self.geometry(original.shape[:2])}
        if restore_original:
            restored, valid = self.restore_map(maps[0, 0], original.shape[:2])
            prediction.update(original_anomaly_map=restored, valid_region=valid)
        return prediction


def predict_image(checkpoint_path, image_path, device="cpu", restore_original=False):
    return AnomalyInferenceEngine(checkpoint_path, device).predict(image_path, restore_original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, help="Output .pt file containing map and raw anomaly score.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--restore-original", action="store_true", help="Also save source-size display map and valid region; cropped-away borders are NaN.")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    prediction = predict_image(args.checkpoint, args.image, args.device, args.restore_original)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prediction, output)
    print(f"Raw anomaly score: {prediction['score'].item():.6f}; saved {output}")


if __name__ == "__main__":
    main()
