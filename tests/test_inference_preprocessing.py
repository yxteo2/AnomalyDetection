"""Training/inference pixel parity, crop anchors and all-detector contracts."""

import math

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

from anomaly_detection.builder import build_model, model_kwargs
from anomaly_detection.config import resolve_config
from anomaly_detection.pipeline import make_det_tf
from anomaly_detection.predict import AnomalyInferenceEngine, load_model
from anomaly_detection.preprocessing import InferencePreprocessing
from inference import FastFlowInferenceEngine
from ssn_inference import SSNInferenceEngine


@pytest.mark.parametrize("engine_class", [InferencePreprocessing, FastFlowInferenceEngine, SSNInferenceEngine])
@pytest.mark.parametrize("size,scale", [((32, 48), 0.875), ((416, 512), 0.875), ((512, 704), 0.875),
                                      ((416, 416), 1.0), ((512, 512), 1.0)])
def test_pixel_and_mask_parity_with_dataset_transform(tmp_path, engine_class, size, scale):
    engine = object.__new__(engine_class)
    engine.setup_preprocessing(size, scale)
    rgb = np.random.default_rng(7).integers(0, 256, (83, 117, 3), dtype=np.uint8)
    mask = (rgb[..., 0] > 128).astype(np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(rgb).save(path)
    h, w = size
    transform = make_det_tf(math.ceil(h / scale), math.ceil(w / scale), h, w)
    expected_image, expected_mask = transform(tv_tensors.Image(Image.fromarray(rgb)),
                                             tv_tensors.Mask(torch.from_numpy(mask)))
    actual, original = engine.preprocess(path)
    torch.testing.assert_close(actual[0], expected_image, rtol=0, atol=0)
    np.testing.assert_array_equal(original, rgb)
    np.testing.assert_array_equal(engine.gt_mask_to_crop(mask, rgb), expected_mask.numpy() * 255)


def test_odd_crop_offsets_match_torchvision_and_restore_exact_canvas():
    engine = InferencePreprocessing()
    engine.setup_preprocessing((32, 48), 0.875)
    assert (engine.pre_h, engine.pre_w) == (37, 55)
    assert (engine.crop_top, engine.crop_left) == (2, 4)  # floor division incorrectly gave left=3
    coordinates = torch.arange(37 * 55).reshape(1, 37, 55)
    torch.testing.assert_close(T.CenterCrop((32, 48))(coordinates), coordinates[:, 2:34, 4:52])
    rgb = np.zeros((37, 55, 3), dtype=np.uint8)
    mask = np.ones((32, 48), dtype=np.uint8) * 255
    restored_mask = engine.uncrop_mask_to_original(rgb, mask)
    expected = np.zeros((37, 55), dtype=np.uint8)
    expected[2:34, 4:52] = 255
    np.testing.assert_array_equal(restored_mask, expected)
    restored_map, valid = engine.restore_map(torch.ones(32, 48), (37, 55))
    np.testing.assert_array_equal(valid.numpy(), expected > 0)
    assert torch.isnan(restored_map[~valid]).all()
    assert (restored_map[valid] == 1).all()


@pytest.mark.parametrize("name", ["fastflow", "ssn", "padim", "patchcore", "dinomaly"])
@pytest.mark.parametrize("scale", [0.875, 1.0])
def test_every_detector_uses_checkpoint_crop_and_matches_direct_prediction(tmp_path, name, scale):
    torch.set_num_threads(2)
    cfg = resolve_config({"model": name, "backbone": "dinov2_vits14", "pretrained_backbone": False,
                          "data_path": "unused", "category": "bottle", "image_size": [32, 48],
                          "crop_scale": scale, "flow_steps": 1, "decoder_depth": 2,
                          "feature_channels": 16 if name in ("fastflow", "ssn") else None,
                          "adapt_cls_features": True, "padim_channels": 4,
                          "max_patches": 16, "memory_bank_size": 4})
    model = build_model(cfg).eval()
    if name in ("padim", "patchcore"):
        model.fit_features(iter((model.features(torch.randn(2, 3, 32, 48)),)))
    path = tmp_path / "model.pth"
    torch.save({"model_state_dict": model.state_dict(), "experiment_cfg": cfg,
                "model_cfg": {**model_kwargs(cfg), "crop_scale": scale}}, path)
    del model
    image = np.random.default_rng(9).integers(0, 256, (79, 113, 3), dtype=np.uint8)
    image_path = tmp_path / "image.png"
    Image.fromarray(image).save(image_path)
    engine = AnomalyInferenceEngine(path)
    tensor = make_det_tf(math.ceil(32 / scale), math.ceil(48 / scale), 32, 48)(Image.fromarray(image)).unsqueeze(0)
    observed = []
    hook = engine.model.register_forward_pre_hook(lambda module, args: observed.append(args[0].clone()))
    prediction = engine.predict(image_path, restore_original=True)
    hook.remove()
    torch.testing.assert_close(observed[0], tensor, rtol=0, atol=0)
    with torch.inference_mode():
        output = engine.model(tensor)
    if name == "fastflow":
        maps = output
        flat = maps.flatten(1)
        score = flat.topk(max(1, int(0.01 * flat.shape[1])), dim=1).values.mean(1)
    else:
        maps, score = output
    torch.testing.assert_close(prediction["anomaly_map"], maps)
    torch.testing.assert_close(prediction["score"], score)
    assert prediction["geometry"]["crop_scale"] == scale
    assert prediction["original_anomaly_map"].shape == (79, 113)
    valid = prediction["valid_region"]
    assert torch.isfinite(prediction["original_anomaly_map"][valid]).all()
    assert bool(valid.all()) == (scale == 1.0)
    assert prediction["anomaly_map"].shape == (1, 1, 32, 48)  # DINO padding removed once by model
    if name in ("fastflow", "ssn"):
        # Deliberately wrong CLI/API fallback geometry must lose to saved metadata.
        common = dict(checkpoint_path=str(path), backbone="resnet18", image_size=(64, 64),
                      crop_scale=0.5, pretrained_backbone=False, device="cpu")
        if name == "fastflow":
            legacy = FastFlowInferenceEngine(flow_steps=1, hidden_ratio=1, clamp=2,
                                             conv3x3_only=False, topk_ratio=0.01, **common)
        else:
            legacy = SSNInferenceEngine(perlin_threshold=0.2, adapt_cls_features=False,
                                        layers=["layer2", "layer3"], **common)
        torch.testing.assert_close(legacy.preprocess(image_path)[0], tensor, rtol=0, atol=0)
        legacy_score, legacy_map, _ = legacy.infer_one(image_path)
        assert legacy_score == pytest.approx(score.item(), abs=1e-6)
        np.testing.assert_allclose(legacy_map, maps[0, 0].numpy(), rtol=1e-5, atol=1e-6)


def test_bad_geometry_and_legacy_metadata_fail_clearly(tmp_path, monkeypatch):
    engine = InferencePreprocessing()
    for scale in (0, -1, 2, float("nan")):
        with pytest.raises(ValueError, match="crop_scale"):
            engine.setup_preprocessing((32, 48), scale)
    engine.setup_preprocessing((32, 48), 0.875)
    with pytest.raises(ValueError, match="Ground-truth"):
        engine.gt_mask_to_crop(np.zeros((3, 4)), np.zeros((4, 5, 3)))
    path = tmp_path / "legacy.pth"
    torch.save({"weight": torch.ones(1)}, path)
    with pytest.raises(ValueError, match="full YAML-training checkpoint"):
        load_model(path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        load_model(path, "cuda")
