import json

import numpy as np
from PIL import Image
import pytest
import torch

from inference import auroc_from_scores, infer_gt_label_from_path
from anomaly_detection.inference_utils import (
    SelfAUPRO,
    defect_name_from_path,
    iter_images_recursive,
    load_calibration,
    load_gt_mask_mvtec,
)
from anomaly_detection.modeling import FastFlowModel, SuperSimpleNetModel
from anomaly_detection.training import FastFlowEvaluator, SuperSimpleNetEvaluator
from ssn_inference import infer_gt_label_from_path as infer_ssn_gt_label


def test_auroc_does_not_silently_reverse_scores():
    labels = np.array([0, 0, 1, 1])
    reversed_scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert auroc_from_scores(labels, reversed_scores) == pytest.approx(0.0)


@pytest.mark.parametrize("prediction,expected", [
    ("perfect", 1.0), ("reversed", 0.0), ("tied", 0.15), ("one_region", 0.575),
])
def test_aupro_ranking_ties_and_equal_region_weighting(prediction, expected):
    masks = torch.zeros(2, 1, 3, 3)
    masks[0, 0, 0, 0] = 1
    masks[1, 0, :2, :2] = 1  # A larger region in a separate image.
    if prediction == "perfect":
        scores = masks.clone()
    elif prediction == "reversed":
        scores = 1 - masks
    elif prediction == "tied":
        scores = torch.full_like(masks, 0.5)
    else:
        scores = torch.zeros_like(masks)
        scores[0] = masks[0]
    metric = SelfAUPRO(fpr_limit=0.3)
    # Separate updates and both supported binary-mask encodings.
    metric.update(scores[:1], masks[:1])
    metric.update(scores[1:], masks[1:] * 255)
    # With one region detected, PRO runs from 0.5 to 0.65 over FPR [0, 0.3].
    assert metric.compute().item() == pytest.approx(expected)


def test_calibration_lookup_priority_and_missing_or_invalid_files(tmp_path):
    checkpoint = tmp_path / "run/checkpoints/best_model.pth"
    checkpoint.parent.mkdir(parents=True)
    candidates = [
        tmp_path / "explicit.json",
        checkpoint.parent / "calibration.json",
        checkpoint.parent.parent / "calibration.json",
    ]
    for index, path in enumerate(candidates):
        path.write_text(json.dumps({"image_threshold": index, "pixel_threshold": index + 0.5}))
    assert load_calibration(str(checkpoint), str(candidates[0]))["image_threshold"] == 0
    assert load_calibration(str(checkpoint), None)["image_threshold"] == 1
    candidates[1].unlink()
    assert load_calibration(str(checkpoint), None)["image_threshold"] == 2
    candidates[2].unlink()
    with pytest.raises(FileNotFoundError, match="No calibration.json"):
        load_calibration(str(checkpoint), None)
    candidates[0].write_text('{"image_threshold": 0.5}')
    with pytest.raises(ValueError, match="Invalid calibration file"):
        load_calibration(str(checkpoint), str(candidates[0]))


@pytest.mark.parametrize("mask_name", ["001_mask.png", "001.png"])
def test_mvtec_file_helpers_skip_masks_and_find_both_mask_names(tmp_path, mask_name):
    good = tmp_path / "bottle/test/good/001.PNG"
    defect = tmp_path / "bottle/test/crack/001.png"
    mask = tmp_path / "bottle/ground_truth/crack" / mask_name
    for path in (good, defect, mask):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.array([[0, 1], [255, 0]], dtype=np.uint8)).save(path)
    (tmp_path / "notes.txt").write_text("not an image")
    assert set(iter_images_recursive(tmp_path)) == {good, defect}
    assert infer_gt_label_from_path(good) == 0
    assert infer_gt_label_from_path(defect) == 1
    assert defect_name_from_path(defect) == "crack"
    assert load_gt_mask_mvtec(good) is None
    np.testing.assert_array_equal(load_gt_mask_mvtec(defect), [[0, 255], [255, 0]])


def test_fastflow_calibration_uses_normal_validation_samples_only():
    evaluator = object.__new__(FastFlowEvaluator)
    predictions = {
        "labels": np.array([0, 0, 1]),
        "scores": np.array([0.1, 0.2, 100.0]),
        "anomaly_maps": np.array([[[[0.1]]], [[[0.3]]], [[[100.0]]]]),
    }
    calibration = evaluator.calibrate(
        predictions,
        image_quantile=1.0,
        pixel_quantile=1.0,
    )
    assert calibration["image_threshold"] == pytest.approx(0.2)
    assert calibration["pixel_threshold"] == pytest.approx(0.3)


def test_ssn_calibration_uses_normal_validation_samples_only():
    evaluator = object.__new__(SuperSimpleNetEvaluator)
    predictions = {
        "labels": np.array([0, 0, 1]),
        "scores": np.array([0.1, 0.2, 0.99]),
        "maps": np.array([[[[0.1]]], [[[0.3]]], [[[0.99]]]]),
    }
    calibration = evaluator.calibrate(
        predictions,
        image_quantile=1.0,
        pixel_quantile=1.0,
    )
    assert calibration["image_threshold"] == pytest.approx(0.2)
    assert calibration["pixel_threshold"] == pytest.approx(0.3)


def test_generic_single_images_do_not_require_mvtec_labels(tmp_path):
    image = tmp_path / "sample.png"
    assert infer_gt_label_from_path(image) is None
    assert infer_ssn_gt_label(image) is None


def test_fastflow_rejects_incompatible_layernorm_shape_before_model_download():
    with pytest.raises(ValueError, match="divisible by 16"):
        FastFlowModel(input_size=(417, 416))


def test_fastflow_train_and_eval_output_contracts():
    model = FastFlowModel(
        input_size=(32, 32),
        flow_steps=1,
        pretrained_backbone=False,
    )
    images = torch.rand(1, 3, 32, 32)
    model.train()
    hidden, jacobians = model(images)
    assert len(hidden) == len(jacobians) == 3

    model.eval()
    anomaly_map = model(images)
    assert anomaly_map.shape == (1, 1, 32, 32)


def test_ssn_keeps_frozen_feature_extractor_in_eval_mode():
    model = SuperSimpleNetModel(
        input_size=(32, 32),
        backbone_name="resnet18",
        pretrained_backbone=False,
    )
    model.train()
    assert model.training
    assert not model.feature_extractor.training
