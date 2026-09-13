import numpy as np
import pytest
import torch

from inference import auroc_from_scores, infer_gt_label_from_path
from anomaly_detection.modeling import FastFlowModel, SuperSimpleNetModel
from anomaly_detection.training import FastFlowEvaluator, SuperSimpleNetEvaluator
from ssn_inference import infer_gt_label_from_path as infer_ssn_gt_label


def test_auroc_does_not_silently_reverse_scores():
    labels = np.array([0, 0, 1, 1])
    reversed_scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert auroc_from_scores(labels, reversed_scores) == pytest.approx(0.0)


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
    assert not model.backbone.training
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
