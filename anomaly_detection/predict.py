"""Reload any YAML-built detector and predict one image using saved transforms."""

import argparse
import math
from pathlib import Path

from PIL import Image
import torch

from anomaly_detection.builder import build_model
from anomaly_detection.pipeline import make_det_tf


def load_model(checkpoint_path, device="cpu"):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = {**saved["experiment_cfg"], "pretrained_backbone": False}
    model = build_model(cfg)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return model.to(device).eval(), cfg


@torch.no_grad()
def predict_image(checkpoint_path, image_path, device="cpu"):
    model, cfg = load_model(checkpoint_path, device)
    h, w = cfg["image_size"]
    transform = make_det_tf(math.ceil(h / cfg["crop_scale"]), math.ceil(w / cfg["crop_scale"]), h, w)
    with Image.open(image_path) as source:
        tensor = transform(source.convert("RGB")).unsqueeze(0).to(device)
    result = model(tensor)
    if cfg["model"] == "fastflow":
        # Match FastFlowEvaluator's default top-one-percent aggregation.
        maps = result
        flat = maps.flatten(1)
        scores = flat.topk(max(1, int(0.01 * flat.shape[1])), dim=1).values.mean(1)
    else:
        maps, scores = result
    return {"anomaly_map": maps.cpu(), "score": scores.cpu()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, help="Output .pt file containing map and raw anomaly score.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    prediction = predict_image(args.checkpoint, args.image, args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prediction, output)
    print(f"Raw anomaly score: {prediction['score'].item():.6f}; saved {output}")


if __name__ == "__main__":
    main()
