"""Shared build, train, validation-calibration and final-test pipeline."""

import json
import math
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml
from torchvision.transforms import v2 as T

from anomaly_detection.builder import build_model, build_trainer
from anomaly_detection.config import as_yaml_config, resolve_config
from anomaly_detection.data import MVTecDataModule
from anomaly_detection.training import (
    FastFlowEvaluator,
    SuperSimpleNetEvaluator,
)


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
        self.cfg = resolve_config(cfg)
        for key in ("data_path", "save_dir"):
            self.cfg[key] = str(Path(self.cfg[key]).expanduser().resolve())
        requested_device = self.cfg["device"]
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but is unavailable. Set training.device to cpu or auto.")
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        print(f"Using device: {self.device}")

        self.model = None
        self.data_module = None
        self.trainer = None
        self.evaluator = None

        self.save_dir = Path(self.cfg["save_dir"]) / self.cfg["category"] / self.cfg["model"] / self.cfg["backbone"]
        if self.cfg["run_name"] is not None:
            self.save_dir /= self.cfg["run_name"]

    def _build_model(self):
        print(f"Building {self.cfg['model']} with {self.cfg['backbone']} and {self.cfg['loss_name']}")
        self.model = build_model(self.cfg)

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
            dataset_type=self.cfg["dataset_type"],
            val_ratio=float(self.cfg["val_ratio"]),
            seed=int(self.cfg["seed"]),
            pin_memory=self.device.type == "cuda",
        )
        self.data_module.setup()

    def _build_trainer(self):
        self.trainer = build_trainer(self.cfg, self.model, self.device, self.save_dir)

    def setup(self):
        print("\n=== Setting up Pipeline ===")
        if self.save_dir.exists() and any(self.save_dir.iterdir()) and not self.cfg["overwrite"]:
            raise FileExistsError(
                f"Output directory is not empty: {self.save_dir}. Choose output.run_name "
                "or explicitly set output.overwrite: true (--overwrite for the CLI)."
            )
        seed = self.cfg["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Validate the dataset before downloading backbone weights.
        self._build_data()
        self._build_model()
        self._build_trainer()
        with (self.save_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(as_yaml_config(self.cfg), stream, sort_keys=False)
        print(f"Outputs: {self.save_dir}")
        print("\n✓ Setup completed")

    def train(self):
        print("\n=== Starting Training ===")
        train_loader = self.data_module.train_dataloader()
        if self.cfg["model"] in ("padim", "patchcore"):
            train_loader = self.data_module.fit_dataloader()
        val_loader = self.data_module.val_dataloader()

        self.trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=int(self.cfg["num_epochs"]),
            patience=int(self.cfg["patience"]),
        )

    def evaluate(self) -> Dict[str, Any]:
        print("\n=== Evaluating ===")
        # Evaluation needs neither gradients nor Adam momentum buffers on GPU.
        if self.trainer.optimizer is not None:
            self.trainer.optimizer.zero_grad(set_to_none=True)
            self.trainer.optimizer.state.clear()
        self.trainer.load_checkpoint("best_model.pth", load_optimizer=False)

        if self.cfg["model"] != "fastflow":
            self.evaluator = SuperSimpleNetEvaluator(model=self.model, device=str(self.device))
        else:
            self.evaluator = FastFlowEvaluator(model=self.model, device=str(self.device))

        test_loader = self.data_module.test_dataloader()
        val_loader = self.data_module.val_dataloader()
        val_preds = self.evaluator.predict(val_loader)
        calibration = self.evaluator.calibrate(
            val_preds,
            image_quantile=float(self.cfg["image_threshold_quantile"]),
            pixel_quantile=float(self.cfg["pixel_threshold_quantile"]),
        )
        del val_preds  # Do not retain validation maps alongside all test maps in RAM.
        with open(self.save_dir / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=4)

        preds = self.evaluator.predict(test_loader)
        metrics = self.evaluator.compute_metrics(preds, calibration=calibration)

        print("\n=== Results ===")
        print(f"Image AUROC: {metrics.get('image_auroc', float('nan')):.4f}")
        print(f"Pixel AUROC: {metrics.get('pixel_auroc', float('nan')):.4f}")

        # Save metrics (per model/backbone)
        metrics_path = self.save_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved: {metrics_path}")

        # (Optional) visualizations
        if self.cfg["num_visualizations"] > 0 and hasattr(self.evaluator, "visualize_results"):
            vis_dir = self.save_dir / "visualizations"
            self.evaluator.visualize_results(
                test_loader,
                save_dir=str(vis_dir),
                num_samples=int(self.cfg["num_visualizations"]),
            )
        else:
            print("[Info] Visualizations disabled or not supported for this evaluator.")

        return metrics

    def run(self) -> Dict[str, Any]:
        self.setup()
        self.train()
        return self.evaluate()
