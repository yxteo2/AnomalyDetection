"""
Main training script for anomaly detection (FastFlow or SuperSimpleNet).

Examples
--------
FastFlow:
  python train.py --model fastflow --data_path /path/to/mvtec --category bottle

SuperSimpleNet (SSN):
  python train.py --model ssn --data_path /path/to/mvtec --category bottle
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torchvision.transforms import v2 as T

from dataset import MVTecDataModule
from model import FastFlowModel, SuperSimpleNetModel

from trainer import (
    FastFlowTrainer,
    FastFlowEvaluator,
    SuperSimpleNetTrainer,
    SuperSimpleNetEvaluator,
)


def make_train_tf(pre_h, pre_w, h, w):
    return T.Compose([
        T.Resize((pre_h, pre_w), antialias=True),
        T.CenterCrop((h, w)),

        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def make_det_tf(pre_h, pre_w, h, w):
    return T.Compose([
        T.Resize((pre_h, pre_w), antialias=True),
        T.CenterCrop((h, w)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class AnomalyPipeline:
    """End-to-end pipeline for training + evaluation."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = None
        self.data_module = None
        self.trainer = None
        self.evaluator = None

    def setup(self):
        print("\n=== Setting up Pipeline ===")

        # -------- model --------
        model_type = self.config["model"].lower()
        backbone = self.config["backbone"]
        image_size = tuple(self.config["image_size"])  # (H,W)

        if model_type == "fastflow":
            print(f"\nInitializing FastFlow model (backbone={backbone})")
            self.model = FastFlowModel(
                backbone_name=backbone,
                flow_steps=self.config["flow_steps"],
                input_size=image_size,
                hidden_ratio=self.config.get("hidden_ratio", 1.0),
                clamp=self.config.get("clamp", 2.0),
                conv3x3_only=self.config.get("conv3x3_only", False),
            )
        elif model_type == "ssn":
            print(f"\nInitializing SuperSimpleNet model (backbone={backbone})")
            self.model = SuperSimpleNetModel(
                backbone_name=backbone,
                input_size=image_size,
                adaptor_dim=self.config.get("adaptor_dim", 256),
                patch_k=self.config.get("patch_k", 3),
                noise_std=self.config.get("noise_std", 0.02),
                mask_thr=self.config.get("mask_thr", 0.5),
                base_res=self.config.get("base_res", 4),
                hidden=self.config.get("disc_hidden", 256),
            )
        else:
            raise ValueError(f"Unknown --model {model_type}")

        # -------- transforms + data --------
        h, w = image_size
        crop_scale = 0.875
        pre_h = int(math.ceil(h / crop_scale))
        pre_w = int(math.ceil(w / crop_scale))

        train_tf = make_train_tf(pre_h, pre_w, h, w)
        test_tf = make_det_tf(pre_h, pre_w, h, w)

        print(f"Loading MVTec dataset - Category: {self.config['category']}")
        self.data_module = MVTecDataModule(
            root_dir=self.config["data_path"],
            category=self.config["category"],
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            image_size=image_size,
            train_transform=train_tf,
            test_transform=test_tf,
        )
        self.data_module.setup()

        # -------- trainer --------
        save_dir = Path(self.config["save_dir"]) / self.config["category"] / model_type

        if model_type == "fastflow":
            self.trainer = FastFlowTrainer(
                model=self.model,
                device=str(self.device),
                learning_rate=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
                save_dir=str(save_dir),
                monitor=self.config.get("monitor", "image_auroc"),
            )
        else:
            self.trainer = SuperSimpleNetTrainer(
                model=self.model,
                device=str(self.device),
                learning_rate=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
                save_dir=str(save_dir),
                monitor=self.config.get("monitor", "image_auroc"),
                seg_weight=self.config.get("seg_weight", 1.0),
                cls_weight=self.config.get("cls_weight", 1.0),
            )

        print("\n✓ Setup completed successfully")

    def train(self):
        print("\n=== Starting Training ===")
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.test_dataloader()

        self.trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.config["num_epochs"],
            patience=self.config["patience"],
        )

    def evaluate(self):
        print("\n=== Evaluating Model ===")
        self.trainer.load_checkpoint("best_model.pth")

        model_type = self.config["model"].lower()
        if model_type == "fastflow":
            self.evaluator = FastFlowEvaluator(model=self.model, device=str(self.device))
        else:
            self.evaluator = SuperSimpleNetEvaluator(model=self.model, device=str(self.device))

        test_loader = self.data_module.test_dataloader()
        predictions = self.evaluator.predict(test_loader)
        metrics = self.evaluator.compute_metrics(predictions)

        print("\n=== Evaluation Results ===")
        print(f"Image-level AUROC: {metrics['image_auroc']:.4f}")
        print(f"Pixel-level AUROC: {metrics['pixel_auroc']:.4f}")

        save_dir = Path(self.config["save_dir"]) / self.config["category"] / model_type
        save_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = save_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"\nMetrics saved to {metrics_path}")

        results_dir = save_dir / "visualizations"
        self.evaluator.visualize_results(
            test_loader,
            save_dir=str(results_dir),
            num_samples=self.config["num_visualizations"],
        )

        return metrics

    def run(self):
        self.setup()
        self.train()
        return self.evaluate()


def parse_args():
    parser = argparse.ArgumentParser(description="Train anomaly models on MVTec AD")

    # data
    parser.add_argument("--data_path", type=str, required=True, help="Path to MVTec AD dataset")
    parser.add_argument("--category", type=str, required=True, help="Product category to train on")

    # model switch
    parser.add_argument("--model", type=str, default="fastflow", choices=["fastflow", "ssn"],
                        help="Which model to train")

    # shared backbone
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "wide_resnet50_2"],
        help="Backbone architecture",
    )

    # FastFlow-only
    parser.add_argument("--flow_steps", type=int, default=8, help="FastFlow: number of flow steps")
    parser.add_argument("--hidden_ratio", type=float, default=1.0, help="FastFlow: coupling subnet hidden ratio")
    parser.add_argument("--clamp", type=float, default=2.0, help="FastFlow: clamp for coupling scale")
    parser.add_argument("--conv3x3_only", action="store_true", help="FastFlow: use only 3x3 kernels in coupling")

    # SSN-only
    parser.add_argument("--adaptor_dim", type=int, default=256, help="SSN: adaptor (1x1) output channels")
    parser.add_argument("--patch_k", type=int, default=3, help="SSN: patch aggregation kernel size")
    parser.add_argument("--noise_std", type=float, default=0.02, help="SSN: synthetic feature noise std")
    parser.add_argument("--mask_thr", type=float, default=0.5, help="SSN: synthetic mask threshold")
    parser.add_argument("--base_res", type=int, default=4, help="SSN: synthetic mask base grid resolution")
    parser.add_argument("--disc_hidden", type=int, default=256, help="SSN: discriminator hidden channels")
    parser.add_argument("--seg_weight", type=float, default=1.0, help="SSN: weight for pixel loss")
    parser.add_argument("--cls_weight", type=float, default=1.0, help="SSN: weight for image loss")

    # training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--monitor", type=str, default="image_auroc", choices=["image_auroc", "pixel_auroc"])

    # misc
    parser.add_argument("--image_size", type=int, nargs=2, default=[416, 416], help="Input image size H W")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_visualizations", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()
    config = vars(args)

    print("=== Anomaly Detection Training ===")
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    pipeline = AnomalyPipeline(config)
    metrics = pipeline.run()

    print("\n=== Completed ===")
    print(f"Final Image AUROC: {metrics['image_auroc']:.4f}")
    print(f"Final Pixel AUROC: {metrics['pixel_auroc']:.4f}")


if __name__ == "__main__":
    main()
