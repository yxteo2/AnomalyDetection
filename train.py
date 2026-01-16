"""
Main training script for FastFlow / SSN anomaly detection (MVTec)

Usage:
  python train.py --data_path C:/.../MVTec --category candle --model fastflow --backbone resnet34
  python train.py --data_path C:/.../MVTec --category candle --model ssn --backbone resnet34 --layers layer2 layer3
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from torchvision.transforms import v2 as T

from model import FastFlowModel, SuperSimpleNetModel
from dataset import MVTecDataModule
from trainer import FastFlowTrainer, FastFlowEvaluator
from ssntrainer import SuperSimpleNetTrainer, SuperSimpleNetEvaluator


# -----------------------------
# Transforms (torchvision v2)
# -----------------------------
def make_train_tf(pre_h: int, pre_w: int, h: int, w: int):
    return T.Compose([
        T.ToImage(),
        T.Resize((pre_h, pre_w), antialias=True),
        T.CenterCrop((h, w)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def make_det_tf(pre_h: int, pre_w: int, h: int, w: int):
    return T.Compose([
        T.ToImage(),
        T.Resize((pre_h, pre_w), antialias=True),
        T.CenterCrop((h, w)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# -----------------------------
# Pipeline
# -----------------------------
class AnomalyPipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = None
        self.data_module = None
        self.trainer = None
        self.evaluator = None

        self.save_dir = Path(cfg["save_dir"]) / cfg["category"] / cfg["model"] / cfg["backbone"]
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _build_model(self):
        model_type = self.cfg["model"]
        backbone = self.cfg["backbone"]
        image_size = tuple(self.cfg["image_size"])

        if model_type == "ssn":
            layers: List[str] = self.cfg["layers"]
            print(f"\nInitializing SSN model (backbone={backbone}, layers={layers})")

            self.model = SuperSimpleNetModel(
                perlin_threshold=float(self.cfg["perlin_threshold"]),
                backbone_name=backbone,
                layers=layers,
                stop_grad=True,
                adapt_cls_features=bool(self.cfg["adapt_cls_features"]),
                input_size=image_size,
                pretrained_backbone=bool(self.cfg["pretrained_backbone"]),
            )
            return

        print(f"\nInitializing FastFlow model (backbone={backbone})")

        if backbone == "wide_resnet50_2":
            self.model = FastFlowModel(
                backbone_name=backbone,
                flow_steps=int(self.cfg["flow_steps"]),
                input_size=image_size,
                reducer_channels=(128, 192, 256),
                hidden_ratio=float(self.cfg["hidden_ratio"]),
                clamp=float(self.cfg["clamp"]),
                conv3x3_only=bool(self.cfg["conv3x3_only"]),
            )
        else:
            self.model = FastFlowModel(
                backbone_name=backbone,
                flow_steps=int(self.cfg["flow_steps"]),
                input_size=image_size,
                hidden_ratio=float(self.cfg["hidden_ratio"]),
                clamp=float(self.cfg["clamp"]),
                conv3x3_only=bool(self.cfg["conv3x3_only"]),
            )

    def _build_data(self):
        h, w = self.cfg["image_size"]
        crop_scale = float(self.cfg["crop_scale"])
        pre_h = int(math.ceil(h / crop_scale))
        pre_w = int(math.ceil(w / crop_scale))

        train_tf = make_train_tf(pre_h, pre_w, h, w)
        test_tf = make_det_tf(pre_h, pre_w, h, w)

        print(f"Loading MVTec dataset - Category: {self.cfg['category']}")
        self.data_module = MVTecDataModule(
            root_dir=self.cfg["data_path"],
            category=self.cfg["category"],
            batch_size=int(self.cfg["batch_size"]),
            num_workers=int(self.cfg["num_workers"]),
            image_size=(h, w),
            train_transform=train_tf,
            test_transform=test_tf,
        )
        self.data_module.setup()

    def _build_trainer(self):
        model_type = self.cfg["model"]

        if model_type == "ssn":
            self.trainer = SuperSimpleNetTrainer(
                model=self.model,
                device=str(self.device),
                save_dir=str(self.save_dir),
                monitor="image_auroc",
                maximize=True,
                model_cfg={
                    "perlin_threshold": float(self.cfg["perlin_threshold"]),
                    "backbone_name": self.cfg["backbone"],
                    "layers": list(self.cfg["layers"]),
                    "stop_grad": True,
                    "adapt_cls_features": bool(self.cfg["adapt_cls_features"]),
                    "input_size": tuple(self.cfg["image_size"]),
                    "pretrained_backbone": bool(self.cfg["pretrained_backbone"]),
                },
            )
        else:
            self.trainer = FastFlowTrainer(
                model=self.model,
                backbone_name=self.cfg["backbone"],
                device=str(self.device),
                learning_rate=float(self.cfg["learning_rate"]),
                weight_decay=float(self.cfg["weight_decay"]),
                save_dir=str(self.save_dir),
            )

    def setup(self):
        print("\n=== Setting up Pipeline ===")
        self._build_model()
        self._build_data()
        self._build_trainer()
        print("\n✓ Setup completed")

    def train(self):
        print("\n=== Starting Training ===")
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.test_dataloader()

        self.trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=int(self.cfg["num_epochs"]),
            patience=int(self.cfg["patience"]),
        )

    def evaluate(self) -> Dict[str, Any]:
        print("\n=== Evaluating ===")
        self.trainer.load_checkpoint("best_model.pth")

        if self.cfg["model"] == "ssn":
            self.evaluator = SuperSimpleNetEvaluator(model=self.model, device=str(self.device))
        else:
            self.evaluator = FastFlowEvaluator(model=self.model, device=str(self.device))

        test_loader = self.data_module.test_dataloader()
        preds = self.evaluator.predict(test_loader)
        metrics = self.evaluator.compute_metrics(preds)

        print("\n=== Results ===")
        print(f"Image AUROC: {metrics.get('image_auroc', float('nan')):.4f}")
        print(f"Pixel AUROC: {metrics.get('pixel_auroc', float('nan')):.4f}")

        # Save metrics (per model/backbone)
        metrics_path = self.save_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved: {metrics_path}")

        # (Optional) visualizations
        if hasattr(self.evaluator, "visualize_results"):
            vis_dir = self.save_dir / "visualizations"
            self.evaluator.visualize_results(
                test_loader,
                save_dir=str(vis_dir),
                num_samples=int(self.cfg["num_visualizations"]),
            )
        else:
            print("[Info] visualize_results not implemented, skipping.")

        return metrics

    def run(self) -> Dict[str, Any]:
        self.setup()
        self.train()
        return self.evaluate()


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--category", type=str, required=True)

    p.add_argument("--model", type=str, default="fastflow", choices=["fastflow", "ssn"])
    p.add_argument("--backbone", type=str, default="resnet18",
                   choices=["resnet18", "resnet34", "wide_resnet50_2"])

    # shared
    p.add_argument("--image_size", type=int, nargs=2, default=[416, 416])
    p.add_argument("--crop_scale", type=float, default=0.875)

    # fastflow
    p.add_argument("--flow_steps", type=int, default=8)
    p.add_argument("--hidden_ratio", type=float, default=1.0)
    p.add_argument("--clamp", type=float, default=2.0)
    p.add_argument("--conv3x3_only", action="store_true")

    # ssn
    p.add_argument("--perlin_threshold", type=float, default=0.2)
    p.add_argument("--adapt_cls_features", action="store_true")
    p.add_argument("--layers", type=str, nargs="+", default=["layer2", "layer3"])
    p.add_argument("--pretrained_backbone", action="store_true", default=True)

    # training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=30)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--num_visualizations", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = vars(args)

    print("=== Anomaly Training ===")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    pipeline = AnomalyPipeline(cfg)
    metrics = pipeline.run()

    print("\n=== Done ===")
    if "image_auroc" in metrics:
        print(f"Final Image AUROC: {metrics['image_auroc']:.4f}")
    if "pixel_auroc" in metrics:
        print(f"Final Pixel AUROC: {metrics['pixel_auroc']:.4f}")


if __name__ == "__main__":
    main()
