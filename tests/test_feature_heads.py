"""Real backbones, analytic fitting checks, loss gradients and YAML round trips."""

import pytest
import torch
import yaml

from anomaly_detection.builder import build_model, build_loss
from anomaly_detection.config import resolve_config, ConfigError, as_yaml_config
from anomaly_detection.modeling.feature_heads import PadimModel, PatchcoreModel
from anomaly_detection.pipeline import AnomalyPipeline
from anomaly_detection.predict import predict_image
from anomaly_detection.smoke import run_smoke
from test_yaml_training import make_dataset


def config(model, **kwargs):
    return resolve_config(dict(data_path="unused", category="bottle", model=model, backbone="dinov2_vits14",
                               pretrained_backbone=False, image_size=[32, 48], batch_size=1, num_workers=0,
                               padim_channels=4, max_patches=32, memory_bank_size=8, distance_chunk_size=7,
                               decoder_depth=2, **kwargs))


@pytest.mark.parametrize("name", ["padim", "patchcore", "dinomaly"])
@pytest.mark.parametrize("backbone", ["dinov2_vits14", "dinov2_vitb14"])
def test_dino_heads_fit_train_reload(name, backbone):
    torch.set_num_threads(2)
    cfg = config(name)
    cfg.update(backbone=backbone, dino_layers=[8, 11])
    report = run_smoke(cfg, device="cpu")
    assert report["reload_matches"]
    assert report["map_shape"] == [1, 1, 32, 48]
    assert (report["loss"] is None) == (name != "dinomaly")


@pytest.mark.parametrize("head", ["padim", "patchcore"])
@pytest.mark.parametrize("backbone", ["resnet18", "resnet34", "wide_resnet50_2"])
def test_statistical_resnet_heads(head, backbone):
    cfg = config(head)
    cfg["backbone"] = backbone
    assert run_smoke(cfg, device="cpu")["reload_matches"]


@pytest.mark.parametrize("name", ["cosine", "mse", "smooth_l1", "cosine_mse"])
def test_reconstruction_loss_gradients_and_zero_at_match(name):
    loss = build_loss(config("dinomaly", loss_name=name))
    target = torch.randn(2, 5, 8, requires_grad=True)
    prediction = torch.randn_like(target, requires_grad=True)
    value = loss([target], [prediction])
    value.backward()
    assert target.grad is None
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
    assert abs(loss([target], [target]).item()) < 1e-6


@pytest.mark.parametrize("changes", [
    {"model": "padim", "loss_name": "mse"}, {"model": "patchcore", "loss_name": "ssn_bce"},
    {"model": "dinomaly", "backbone": "resnet18"}, {"model": "dinomaly", "decoder_depth": 1},
    {"model": "patchcore", "memory_bank_size": 20000}, {"model": "padim", "covariance_regularization": 0},
    {"model": "dinomaly", "loss_name": "cosine_mse", "loss_params": {"cosine_weight": 0, "mse_weight": 0}},
])
def test_invalid_combinations(changes):
    cfg = config("dinomaly")
    cfg.update(changes)
    with pytest.raises(ConfigError):
        resolve_config(cfg)


def test_padim_streaming_statistics_match_direct_covariance():
    model = PadimModel(backbone_name="resnet18", input_size=(32, 32), pretrained_backbone=False, padim_channels=3)
    x = torch.randn(5, model.backbone.channels, 2, 2)
    model.fit_features(iter((x[:2], x[2:4], x[4:])))
    selected = x[:, model.channel_indices].flatten(2).permute(2, 0, 1).double()
    expected_mean = selected.mean(1)
    delta = selected - expected_mean[:, None]
    covariance = delta.transpose(1, 2) @ delta / 4 + 0.01 * torch.eye(3)
    torch.testing.assert_close(model.mean, expected_mean.float())
    torch.testing.assert_close(model.precision, covariance.inverse().float())
    model.fit_memory_mb = 0
    with pytest.raises(ValueError, match="fit_memory_mb"):
        model.fit_features(iter((x,)))
    with pytest.raises(ValueError, match="at least two"):
        model.fit_features(iter(()))


def test_patchcore_bank_bound_and_chunked_nearest_neighbors(monkeypatch):
    model = PatchcoreModel(backbone_name="resnet18", input_size=(32, 32), pretrained_backbone=False,
                          max_patches=10, memory_bank_size=4, distance_chunk_size=3)
    features = torch.randn(2, 6, 2, 2)
    model.fit_features(iter((features, features + 2, features - 2)))
    assert model.memory_bank.shape == (4, 6)
    monkeypatch.setattr(model, "features", lambda images: features)
    result, _ = model(torch.zeros(2, 3, 32, 32))
    exact = torch.cdist(model.embed(features), model.memory_bank).min(1).values.reshape(2, 1, 2, 2)
    torch.testing.assert_close(result, model.output(exact)[0])


@pytest.mark.parametrize("name", ["padim", "patchcore", "dinomaly"])
def test_new_heads_full_yaml_pipeline_and_prediction(tmp_path, name):
    make_dataset(tmp_path / "data")
    cfg = config(name)
    cfg.update(data_path=str(tmp_path / "data"), save_dir=str(tmp_path / "runs"),
               num_epochs=1, device="cpu", num_visualizations=0)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(as_yaml_config(cfg)))
    from anomaly_detection.config import load_config
    pipeline = AnomalyPipeline(load_config(path))
    metrics = pipeline.run()
    assert 0 <= metrics["image_auroc"] <= 1
    assert set(pipeline.data_module.fit_dataset.indices).isdisjoint(pipeline.data_module.val_dataset.indices)
    result = predict_image(pipeline.save_dir / "best_model.pth", tmp_path / "data/bottle/test/good/000.png")
    assert result["anomaly_map"].shape == (1, 1, 32, 48)
    assert torch.isfinite(result["score"]).all()


@pytest.mark.parametrize("name", ["fastflow", "padim", "patchcore", "dinomaly"])
@pytest.mark.parametrize("size", [416, 512, 704])
def test_large_dino_head_geometry(name, size):
    cfg = config(name)
    cfg.update(image_size=[size, size], flow_steps=1)
    if name == "fastflow":
        cfg["feature_channels"] = 32
    model = build_model(cfg)
    images = torch.randn(1, 3, size, size)
    if name in ("padim", "patchcore"):
        model.fit_features(iter((model.features(images), model.features(images + 0.1))))
    elif name == "dinomaly":
        target, prediction, _ = model.reconstruct(images)
        build_loss(cfg)(target, prediction).backward()
        assert any(p.grad is not None for p in model.decoder.parameters())
    else:
        latent, jacobian = model(images)
        build_loss(cfg)(latent, jacobian).backward()
    assert all(p.grad is None and not p.requires_grad for p in model.backbone.parameters())
    model.eval()
    with torch.no_grad():
        output = model(images)
    maps = output[0] if isinstance(output, tuple) else output
    assert maps.shape == (1, 1, size, size)
    assert torch.isfinite(maps).all()
