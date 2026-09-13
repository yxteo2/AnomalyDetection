"""Offline builder checks and real YAML -> train -> reload smoke tests."""

import json

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F
import yaml

from anomaly_detection import builder
from anomaly_detection.cli import main
from anomaly_detection.config import load_config, resolve_config
from anomaly_detection.pipeline import AnomalyPipeline
from anomaly_detection.training.losses import FastFlowNLLLoss, SSNBCELoss, SSNLoss
from inference import FastFlowInferenceEngine
from ssn_inference import SSNInferenceEngine


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name", ["fastflow", "ssn"])
@pytest.mark.parametrize("backbone", ["resnet18", "resnet34", "wide_resnet50_2"])
def test_factory_forwards_backbone_and_model_options(monkeypatch, name, backbone):
    cfg = resolve_config({
        "data_path": "unused", "category": "bottle", "model": name, "backbone": backbone,
        "image_size": [32, 48], "pretrained_backbone": False, "flow_steps": 2,
        "layers": ["layer1", "layer2"], "adapt_cls_features": True,
    })
    monkeypatch.setitem(builder.MODEL_BUILDERS, name, lambda **kwargs: kwargs)
    built = builder.build_model(cfg)
    assert built["backbone_name"] == backbone
    assert built["input_size"] == (32, 48)
    assert built["pretrained_backbone"] is False
    if name == "fastflow":
        assert built["flow_steps"] == 2
        assert built["reducer_channels"] == ((128, 192, 256) if backbone == "wide_resnet50_2" else None)
    else:
        assert built["layers"] == ["layer1", "layer2"]
        assert built["adapt_cls_features"] is True


def test_flow_nll_matches_original_formula_and_backpropagates():
    latent = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]], requires_grad=True)
    jacobian = torch.tensor([0.5, 1.5], requires_grad=True)
    loss = FastFlowNLLLoss()([latent], [jacobian])
    assert loss.item() == pytest.approx(6.5)
    loss.backward()
    assert torch.isfinite(latent.grad).all()
    assert torch.isfinite(jacobian.grad).all()


def test_ssn_losses_use_selected_terms_and_backpropagate():
    maps = torch.tensor([[[[-1.0, 1.0]]]], requires_grad=True)
    scores = torch.tensor([0.2], requires_grad=True)
    masks, labels = torch.tensor([[[[0.0, 1.0]]]]), torch.tensor([1.0])
    bce = SSNBCELoss(seg_weight=2, cls_weight=3)
    expected = 2 * F.binary_cross_entropy_with_logits(maps, masks) + 3 * F.binary_cross_entropy_with_logits(scores, labels)
    torch.testing.assert_close(bce(maps, scores, masks, labels), expected)
    focal = SSNLoss(gamma=0, alpha=-1, seg_weight=2, cls_weight=3, truncation_weight=0)
    torch.testing.assert_close(focal(maps, scores, masks, labels), expected)
    focal(maps, scores, masks, labels).backward()
    assert torch.isfinite(maps.grad).all()
    assert torch.isfinite(scores.grad).all()


def make_dataset(root):
    """Tiny MVTec-shaped fixture; no external images, weights or GPU required."""
    rng = np.random.default_rng(123)
    for kind, count in (("train/good", 6), ("test/good", 1), ("test/crack", 1)):
        folder = root / "bottle" / kind
        folder.mkdir(parents=True)
        for i in range(count):
            image = rng.integers(20, 180, (32, 32, 3), dtype=np.uint8)
            if kind.endswith("crack"):
                image[10:20, 10:20] = 255
            Image.fromarray(image).save(folder / f"{i:03d}.png")
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    mask_dir = root / "bottle/ground_truth/crack"
    mask_dir.mkdir(parents=True)
    Image.fromarray(mask).save(mask_dir / "000_mask.png")


@pytest.mark.parametrize("model_name,loss_name,backbone", [
    ("fastflow", "fastflow_nll", "resnet18"), ("ssn", "ssn_focal", "resnet18"),
    ("ssn", "ssn_bce", "resnet18"), ("ssn", "ssn_bce_dice", "dinov2_vits14"),
    ("ssn", "ssn_focal_dice", "dinov2_vits14"),
    ("fastflow", "fastflow_nll", "dinov2_vits14"),
])
def test_yaml_runs_real_training_and_existing_inference(tmp_path, monkeypatch, model_name, loss_name, backbone):
    make_dataset(tmp_path / "data")
    loss_params = {} if model_name == "fastflow" else {"seg_weight": 2.0, "cls_weight": 0.5}
    doc = {
        "dataset": {"path": "data", "category": "bottle", "val_ratio": 0.25},
        "model": {
            "name": model_name, "backbone": backbone, "pretrained": False, "image_size": [32, 32],
            "params": {"flow_steps": 1} if model_name == "fastflow" else {"layers": ["layer2", "layer3"]},
        },
        "loss": {"name": loss_name, "params": loss_params},
        "training": {
            "epochs": 1, "batch_size": 2, "num_workers": 0, "seed": 123,
            "device": "cpu", "learning_rate": 0.0003, "weight_decay": 0.0002,
        },
        "evaluation": {"num_visualizations": 0},
        "output": {"save_dir": "runs", "run_name": "smoke"},
    }
    if model_name == "ssn":
        doc["training"].update(head_lr_multiplier=3.0, adaptor_weight_decay=0.005)
    if backbone.startswith("dinov2"):
        doc["model"]["params"].update(feature_channels=128, dino_layers=[11],
                                       backbone_precision="bfloat16")
        if model_name == "ssn":
            doc["model"]["params"]["adapt_cls_features"] = True
        doc["training"]["accumulate_grad_batches"] = 3  # Flush final two-batch partial window.
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")

    # Verify that the chosen loss is actually called by training AND validation.
    calls = []
    original = builder.build_loss

    def capture_loss(cfg):
        loss = original(cfg)
        assert isinstance(loss, builder.LOSS_BUILDERS[loss_name])
        loss.register_forward_pre_hook(lambda *args: calls.append(loss_name))
        return loss

    monkeypatch.setattr(builder, "build_loss", capture_loss)
    metrics = main(["--config", str(path)])
    assert len(calls) == 3  # two training batches + one held-out validation batch
    assert np.isfinite(metrics["image_auroc"])
    assert np.isfinite(metrics["pixel_auroc"])

    output = tmp_path / "runs/bottle" / model_name / backbone / "smoke"
    checkpoint_path = output / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["history"]["epoch"] == 1
    assert np.isfinite(checkpoint["history"]["train_loss"]).all()
    assert np.isfinite(checkpoint["history"]["val_loss"]).all()
    assert checkpoint["experiment_cfg"]["loss_name"] == loss_name
    assert checkpoint["experiment_cfg"]["loss_params"] == load_config(path)["loss_params"]
    assert checkpoint["model_cfg"]["input_size"] == (32, 32)
    groups = checkpoint["optimizer_state_dict"]["param_groups"]
    assert groups[0]["initial_lr"] == pytest.approx(0.0003)
    assert groups[0]["lr"] == pytest.approx(0.0003)
    if model_name == "ssn":
        assert groups[1]["initial_lr"] == pytest.approx(0.0009)
        assert groups[1]["lr"] == pytest.approx(0.0009)
        assert groups[0]["weight_decay"] == pytest.approx(0.005)
        assert groups[1]["weight_decay"] == pytest.approx(0.0002)
    else:
        assert groups[0]["weight_decay"] == pytest.approx(0.0002)
    assert load_config(output / "resolved_config.yaml") == load_config(path)
    assert json.loads((output / "metrics.json").read_text()) == metrics
    calibration = json.loads((output / "calibration.json").read_text())
    assert np.isfinite(calibration["image_threshold"])
    assert np.isfinite(calibration["pixel_threshold"])

    # Deliberately incorrect fallback sizes must be overridden by checkpoint metadata.
    common = dict(checkpoint_path=str(checkpoint_path), backbone="resnet34", image_size=(64, 64),
                  pretrained_backbone=False, device="cpu")
    if model_name == "fastflow":
        engine = FastFlowInferenceEngine(flow_steps=2, hidden_ratio=0.5, clamp=1, conv3x3_only=True,
                                         topk_ratio=0.01, **common)
    else:
        engine = SSNInferenceEngine(perlin_threshold=0.7, adapt_cls_features=True, layers=["layer1"], **common)
    assert engine.backbone == backbone
    assert engine.image_size == (32, 32)
    score, anomaly_map, _ = engine.infer_one(str(tmp_path / "data/bottle/test/good/000.png"))
    assert np.isfinite(score) and np.isfinite(anomaly_map).all()
    assert anomaly_map.shape == (32, 32)

    # A second invocation must not overwrite a completed experiment implicitly.
    with pytest.raises(FileExistsError, match="Output directory is not empty"):
        main(["--config", str(path)])


def test_explicit_unavailable_cuda_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        AnomalyPipeline({"data_path": "unused", "category": "bottle", "device": "cuda"})
