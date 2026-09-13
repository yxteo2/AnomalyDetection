"""Synthetic training/reload check with measured CUDA peak memory, not accuracy."""

import argparse
import gc
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from anomaly_detection.builder import build_model, build_trainer
from anomaly_detection.config import load_config


class SyntheticBatches:
    def __init__(self, cfg):
        self.batch_size = cfg["batch_size"]
        self.size = cfg["image_size"]
        self.count = max(2, cfg["accumulate_grad_batches"] + 1)
        self.seed = cfg["seed"]

    def __len__(self):
        return self.count

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed)
        for _ in range(self.count):
            # Generate one host batch at a time, not an entire synthetic dataset.
            yield {
                "image": torch.randn(self.batch_size, 3, *self.size, generator=rng),
                "mask": torch.zeros(self.batch_size, 1, *self.size),
                "label": torch.zeros(self.batch_size),
            }


def run_smoke(cfg, device="cuda", memory_budget_gb=14.0):
    if not math.isfinite(memory_budget_gb) or memory_budget_gb <= 0:
        raise ValueError("memory_budget_gb must be finite and positive.")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a Blackwell-compatible CUDA PyTorch build for RTX 5070 Ti.")
    info = {"device": device, "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "backbone": cfg["backbone"], "image_size": cfg["image_size"],
            "batch_size": cfg["batch_size"], "pretrained": cfg["pretrained_backbone"],
            "backbone_precision": cfg["backbone_precision"]}
    info["effective_backbone_precision"] = (
        cfg["backbone_precision"] if device == "cuda" and cfg["backbone"].startswith("dinov2") else "float32"
    )
    if device == "cuda":
        info.update(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                    total_gib=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30)
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(cfg["seed"])
    with TemporaryDirectory(prefix="anomaly-smoke-") as temporary:
        model = build_model(cfg)
        trainer = build_trainer(cfg, model, device, Path(temporary))
        loss = None
        if cfg["model"] in ("padim", "patchcore"):
            trainer.fit_statistics(SyntheticBatches(cfg))
        else:
            loss = trainer.train_epoch(SyntheticBatches(cfg))
        if loss is not None and not math.isfinite(loss):
            raise RuntimeError("Training produced a non-finite loss.")
        frozen = model.feature_extractor if cfg["model"] == "ssn" else model.backbone
        if any(p.requires_grad or p.grad is not None for p in frozen.parameters()):
            raise RuntimeError("Backbone was not fully frozen.")
        model.eval()
        images = torch.randn(1, 3, *cfg["image_size"], device=device)
        with torch.no_grad():
            prediction = model(images)
            expected = tuple(x.cpu() for x in prediction) if isinstance(prediction, tuple) else (prediction.cpu(),)
        if any(not torch.isfinite(x).all() for x in expected):
            raise RuntimeError("Inference produced non-finite outputs.")
        trainer.save_checkpoint("smoke.pth")
        if trainer.optimizer is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.optimizer.state.clear()
        # Do not keep two complete models or optimizer states on the GPU at reload.
        del prediction, trainer, model, frozen
        gc.collect()
        checkpoint = torch.load(Path(temporary) / "smoke.pth", map_location="cpu", weights_only=True)
        restored = build_model({**cfg, "pretrained_backbone": False})
        restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        restored.to(device).eval()
        with torch.no_grad():
            prediction = restored(images)
        actual = prediction if isinstance(prediction, tuple) else (prediction,)
        for before, after in zip(expected, actual):
            torch.testing.assert_close(before, after.cpu(), atol=1e-5, rtol=1e-4)
        if actual[0].shape[-2:] != tuple(cfg["image_size"]):
            raise RuntimeError("Reloaded anomaly map has incorrect image dimensions.")
        info.update(loss=loss, reload_matches=True, map_shape=list(actual[0].shape))
        if device == "cuda":
            torch.cuda.synchronize()
            info.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                        memory_budget_gib=memory_budget_gb)
            if info["peak_reserved_gib"] > memory_budget_gb:
                raise RuntimeError(f"CUDA memory budget exceeded: {json.dumps(info)}")
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--memory-budget-gb", type=float, default=14.0, help="GiB reserved by PyTorch; leaves headroom on 16 GB cards.")
    parser.add_argument("--no-pretrained", action="store_true", help="Offline architecture test, not a pretrained-feature validation.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.no_pretrained:
        cfg["pretrained_backbone"] = False
    print(json.dumps(run_smoke(cfg, args.device, args.memory_budget_gb), indent=2))


if __name__ == "__main__":
    main()
