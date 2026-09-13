"""Real offline ViT tests: geometry, frozen features and trainable heads."""

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from anomaly_detection.builder import build_model, build_loss
from anomaly_detection.config import ConfigError, as_yaml_config, load_config, resolve_config
from anomaly_detection.modeling.dinov2 import pad_to_patch_grid
from anomaly_detection.training.accumulation import GradientAccumulator


def dino_config(**changes):
    return resolve_config({
        "data_path": "unused", "category": "bottle", "model": "ssn", "backbone": "dinov2_vits14",
        "pretrained_backbone": False, "adapt_cls_features": True, "feature_channels": 128,
        "image_size": [416, 416], "backbone_precision": "bfloat16", **changes,
    })


@pytest.mark.parametrize("height,width", [(416, 416), (512, 512), (704, 704), (416, 512)])
def test_real_dino_large_image_training_and_output_geometry(height, width):
    torch.set_num_threads(2)
    cfg = dino_config(image_size=[height, width], loss_name="ssn_bce_dice")
    model = build_model(cfg)
    model.train()
    backbone = model.feature_extractor
    assert not backbone.training
    assert not backbone.feature_extractor.training
    images = torch.randn(1, 3, height, width)
    logits, scores, masks, labels = model(images)
    assert logits.shape == (2, 1, (height + 13) // 14, (width + 13) // 14)
    loss = build_loss(cfg)(logits, scores, masks, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(not p.requires_grad and p.grad is None for p in backbone.parameters())
    assert model.adaptor.projection.weight.grad is not None
    assert torch.isfinite(model.adaptor.projection.weight.grad).all()
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        anomaly_map, score = model(images)
    assert anomaly_map.shape == (1, 1, height, width)
    assert score.shape == (1,)
    assert torch.isfinite(anomaly_map).all()


def test_padding_keeps_coordinates_and_only_extends_bottom_right():
    image = torch.arange(416 * 512).reshape(1, 1, 416, 512).float()
    padded = pad_to_patch_grid(image)
    assert padded.shape[-2:] == (420, 518)
    torch.testing.assert_close(padded[..., :416, :512], image)
    assert padded[..., 416:, :].count_nonzero() == 0
    assert padded[..., :, 512:].count_nonzero() == 0


def test_dino_rectangular_map_is_cropped_not_rescaled(monkeypatch):
    model = build_model(dino_config(image_size=[32, 48]))
    model.eval()
    observed = {}

    def output_grid(logits, final_size):
        observed["size"] = final_size
        h, w = final_size
        return torch.arange(h * w).reshape(1, 1, h, w).float() / (h * w)

    monkeypatch.setattr(model.anomaly_map_generator, "forward", output_grid)
    with torch.no_grad():
        result, _ = model(torch.randn(1, 3, 32, 48))
    assert observed["size"] == (42, 56)
    expected = torch.arange(42 * 56).reshape(1, 1, 42, 56).float() / (42 * 56)
    torch.testing.assert_close(result, expected.sigmoid()[..., :32, :48])


def test_base_backbone_and_multiple_blocks():
    model = build_model(dino_config(backbone="dinov2_vitb14", dino_layers=[8, 11], image_size=[32, 48]))
    assert model.adaptor.projection.in_channels == 768 * 2
    model.eval()
    with torch.no_grad():
        result, _ = model(torch.randn(1, 3, 32, 48))
    assert result.shape == (1, 1, 32, 48)


@pytest.mark.parametrize("changes", [
    {"model": "fastflow", "feature_channels": 127}, {"dino_layers": []}, {"dino_layers": [12]},
    {"dino_layers": [11, 8]}, {"dino_layers": [True]}, {"feature_channels": 0},
    {"adapt_cls_features": False}, {"backbone_precision": "float16"},
    {"accumulate_grad_batches": 0}, {"accumulate_grad_batches": True},
    {"layers": ["layer1"]}, {"backbone": "resnet18", "dino_layers": [8, 11]},
    {"loss_name": "ssn_bce", "loss_params": {"seg_pos_weight": 0}},
    {"loss_name": "ssn_bce_dice", "loss_params": {"dice_smooth": 0}},
])
def test_invalid_extensions_fail_before_model_construction(changes):
    with pytest.raises(ConfigError):
        dino_config(**changes)


@pytest.mark.parametrize("example", ["ssn_dinov2.yaml", "fastflow_dinov2.yaml", "padim_dinov2.yaml",
                                    "patchcore_dinov2.yaml", "dinomaly_dinov2.yaml"])
def test_example_roundtrip(tmp_path, example):
    import yaml

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / example)
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(as_yaml_config(cfg)))
    assert load_config(path) == cfg


@pytest.mark.parametrize("loss_name", ["ssn_bce", "ssn_bce_dice", "ssn_focal_dice"])
@pytest.mark.parametrize("target_value", [0., 1.])
def test_loss_variants_have_finite_gradients_on_empty_and_full_masks(loss_name, target_value):
    cfg = dino_config(loss_name=loss_name)
    maps = torch.zeros(2, 1, 3, 3, requires_grad=True)
    scores = torch.zeros(2, requires_grad=True)
    loss = build_loss(cfg)(maps, scores, torch.full_like(maps, target_value), torch.full_like(scores, target_value))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(maps.grad).all() and torch.isfinite(scores.grad).all()


def test_weighted_bce_matches_pytorch():
    cfg = dino_config(loss_name="ssn_bce", loss_params={"seg_pos_weight": 3., "cls_pos_weight": 2.})
    maps = torch.tensor([[[[-1., 2.]]]])
    scores = torch.tensor([0.4])
    mask, label = torch.tensor([[[[0., 1.]]]]), torch.ones(1)
    expected = (F.binary_cross_entropy_with_logits(maps, mask, pos_weight=torch.tensor(3.))
                + F.binary_cross_entropy_with_logits(scores, label, pos_weight=torch.tensor(2.)))
    torch.testing.assert_close(build_loss(cfg)(maps, scores, mask, label), expected)


def test_accumulation_matches_grouped_batches_and_flushes_short_window():
    small = torch.nn.Linear(2, 1, bias=False)
    large = torch.nn.Linear(2, 1, bias=False)
    large.load_state_dict(small.state_dict())
    small_opt, large_opt = torch.optim.SGD(small.parameters(), lr=.1), torch.optim.SGD(large.parameters(), lr=.1)
    x, y = torch.randn(7, 2), torch.randn(7, 1)
    accumulation = GradientAccumulator(small_opt, batches=3)
    for start, end in [(0, 2), (2, 4), (4, 5), (5, 7)]:
        accumulation.backward(F.mse_loss(small(x[start:end]), y[start:end]), end - start)
    accumulation.step()
    for start, end in [(0, 5), (5, 7)]:
        large_opt.zero_grad(set_to_none=True)
        F.mse_loss(large(x[start:end]), y[start:end]).backward()
        large_opt.step()
    torch.testing.assert_close(small.weight, large.weight)
    assert all(p.grad is None for p in small.parameters())


@pytest.mark.parametrize("loss_name", ["ssn_bce_dice", "ssn_focal_dice"])
def test_dice_composite_matches_analytic_value(loss_name):
    params = {"seg_weight": 2., "cls_weight": 3., "dice_weight": .25}
    if loss_name == "ssn_focal_dice":
        params.update(gamma=0., alpha=-1., truncation_weight=0.)
    cfg = dino_config(loss_name=loss_name, loss_params=params)
    loss = build_loss(cfg)(torch.zeros(1, 1, 1, 2), torch.zeros(1),
                           torch.tensor([[[[0., 1.]]]]), torch.ones(1))
    expected = 5 * torch.log(torch.tensor(2.)) + .25 * (1 - (1 + 1e-6) / (2 + 1e-6))
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("model_name", ["ssn", "fastflow"])
def test_memory_smoke_runs_training_and_reload_on_cpu(model_name):
    from anomaly_detection.smoke import run_smoke

    cfg = (dino_config(image_size=[32, 48], batch_size=1) if model_name == "ssn" else
           resolve_config({"data_path": "unused", "category": "bottle", "image_size": [32, 48],
                           "pretrained_backbone": False, "flow_steps": 1, "batch_size": 1}))
    result = run_smoke(cfg, device="cpu")
    assert result["reload_matches"]
    assert result["map_shape"] == [1, 1, 32, 48]
    assert result["effective_backbone_precision"] == "float32"
    assert "peak_reserved_gib" not in result


def test_cuda_smoke_fails_before_building_without_cuda(monkeypatch):
    from anomaly_detection import smoke

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(smoke, "build_model", lambda cfg: pytest.fail("Must not construct a model"))
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        smoke.run_smoke(dino_config())


def test_raw_dino_checkpoint_inference_with_explicit_architecture(tmp_path):
    from PIL import Image
    from ssn_inference import SSNInferenceEngine

    cfg = dino_config(image_size=[32, 48], dino_layers=[8, 11], feature_channels=64)
    model = build_model(cfg).eval()
    checkpoint = tmp_path / "weights.pth"
    torch.save(model.state_dict(), checkpoint)
    image_path = tmp_path / "image.png"
    Image.new("RGB", (48, 32), color=(30, 80, 120)).save(image_path)
    engine = SSNInferenceEngine(
        checkpoint_path=str(checkpoint), backbone="dinov2_vits14", image_size=(32, 48),
        perlin_threshold=.2, adapt_cls_features=True, layers=["layer2", "layer3"],
        pretrained_backbone=False, device="cpu", dino_layers=[8, 11], feature_channels=64,
        backbone_precision="float32",
    )
    score, anomaly_map, _ = engine.infer_one(str(image_path))
    assert anomaly_map.shape == (32, 48)
    assert torch.isfinite(torch.tensor(score))
    x, _ = engine.preprocess(str(image_path))
    with torch.no_grad():
        expected, expected_score = model(x)
    torch.testing.assert_close(torch.from_numpy(anomaly_map), expected[0, 0])
    assert score == pytest.approx(expected_score.item())
