"""YAML validation and command-line behavior without training a network."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from anomaly_detection.cli import parse_args
from anomaly_detection.config import ConfigError, as_yaml_config, load_config


ROOT = Path(__file__).resolve().parents[1]


def write_config(tmp_path, document):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.fixture
def document():
    return {"dataset": {"path": "images", "category": "bottle"}}


@pytest.mark.parametrize("filename,loss", [
    ("fastflow.yaml", "fastflow_nll"), ("ssn.yaml", "ssn_focal"), ("ssn_bce.yaml", "ssn_bce"),
])
def test_examples_and_resolved_config_round_trip(tmp_path, filename, loss):
    cfg = load_config(ROOT / "configs" / filename)
    assert cfg["loss_name"] == loss
    assert load_config(write_config(tmp_path, as_yaml_config(cfg))) == cfg


def test_defaults_relative_paths_and_scientific_notation(tmp_path, document, monkeypatch):
    document["training"] = {"learning_rate": "1e-4"}
    path = write_config(tmp_path, document)
    monkeypatch.chdir(ROOT)
    cfg = load_config(path)
    assert cfg["loss_name"] == "fastflow_nll"
    assert cfg["learning_rate"] == pytest.approx(0.0001)
    assert cfg["data_path"] == str(tmp_path / "images")
    assert cfg["save_dir"] == str(tmp_path / "checkpoints")
    cfg["image_size"][0] = 32
    assert load_config(path)["image_size"] == [416, 416]


@pytest.mark.parametrize("section,values,error", [
    ("model", {"name": "unknown"}, "model.name"),
    ("model", {"backbone": "unknown"}, "model.backbone"),
    ("model", {"pretrained": "false"}, "true or false"),
    ("model", {"image_size": [417, 416]}, "divisible by 16"),
    ("model", {"image_size": [32]}, "image_size"),
    ("model", {"image_size": [True, 32]}, "image_size"),
    ("model", {"crop_scale": 0}, "crop_scale"),
    ("model", {"params": {"flow_steps": 0}}, "flow_steps"),
    ("model", {"params": {"hidden_ratio": 0}}, "hidden_ratio"),
    ("model", {"name": "ssn", "params": {"flow_steps": 8}}, "model.params"),
    ("model", {"name": "ssn", "params": {"layers": ["layer3", "layer2"]}}, "layers"),
    ("model", {"name": "ssn", "params": {"layers": ["layer2", "layer2"]}}, "layers"),
    ("model", {"params": None}, "mapping"),
    ("loss", {"name": "ssn_focal"}, "loss.name for fastflow"),
    ("loss", {"params": {"gamma": 2}}, "loss.params"),
    ("training", {"epohs": 5}, "Unknown training"),
    ("training", {"epochs": 0}, "num_epochs"),
    ("training", {"batch_size": True}, "batch_size"),
    ("training", {"num_workers": -1}, "num_workers"),
    ("training", {"learning_rate": float("nan")}, "finite number"),
    ("training", {"weight_decay": -1}, "weight_decay"),
    ("training", {"seed": 2**32}, "seed"),
    ("training", {"device": "tpu"}, "training.device"),
    ("training", {"head_lr_multiplier": 2}, "SSN-only"),
    ("evaluation", {"image_threshold_quantile": 1.1}, "image_threshold_quantile"),
    ("output", {"run_name": "../outside"}, "directory name"),
    ("output", {"overwrite": "false"}, "true or false"),
])
def test_invalid_options_fail_early(tmp_path, document, section, values, error):
    document[section] = values
    with pytest.raises(ConfigError, match=error):
        load_config(write_config(tmp_path, document))


@pytest.mark.parametrize("name,params", [
    ("ssn_bce", {"gamma": 2}),
    ("ssn_focal", {"alpha": -0.5}),
    ("ssn_focal", {"truncation_term": 2}),
    ("ssn_focal", {"gamma": -1}),
    ("ssn_bce", {"seg_weight": 0, "cls_weight": 0}),
    ("fastflow_nll", {}),
])
def test_incompatible_ssn_losses(tmp_path, document, name, params):
    document["model"] = {"name": "ssn"}
    document["loss"] = {"name": name, "params": params}
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, document))


@pytest.mark.parametrize("contents", [
    "", "[]", "dataset: null", "unknown_section: {}", "dataset: [broken",
    "dataset: {path: images, category: bottle}\nmodel: {name: ssn, name: fastflow}",
    "dataset: {path: images, category: bottle}\ntraining: {epochs: 1, epochs: 2}",
    "!!python/object/apply:builtins.print ['must not execute']",
    "dataset: {path: images, category: bottle}\ntraining: {5: value}",
])
def test_malformed_or_unsafe_yaml(tmp_path, contents):
    path = tmp_path / "bad.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_file_and_missing_dataset(tmp_path):
    with pytest.raises(ConfigError, match="Cannot read"):
        load_config(tmp_path / "missing.yaml")
    with pytest.raises(ConfigError, match="data_path"):
        load_config(write_config(tmp_path, {"model": {"name": "ssn"}}))


def test_dataset_constraints(tmp_path, document):
    for change in ({"category": ".."}, {"val_ratio": 1}, {"path": ""}, {"type": "invalid"}):
        invalid = deepcopy(document)
        invalid["dataset"].update(change)
        with pytest.raises(ConfigError):
            load_config(write_config(tmp_path, invalid))


def test_legacy_flags_and_conflicting_overrides(tmp_path, document):
    args = parse_args(["--data_path", "images", "--category", "bottle", "--model", "ssn", "--no-pretrained_backbone"])
    assert args.config["loss_name"] == "ssn_focal"
    assert args.config["pretrained_backbone"] is False
    path = write_config(tmp_path, document)
    with pytest.raises(SystemExit) as exc:
        parse_args(["--config", str(path), "--num_epochs", "1"])
    assert exc.value.code == 2


@pytest.mark.parametrize("entrypoint", ["train.py", "anomaly_detection"])
def test_check_config_does_not_import_torch_or_create_outputs(tmp_path, document, entrypoint):
    path = write_config(tmp_path, document)
    code = (
        "import runpy, sys; "
        f"sys.argv = [{entrypoint!r}, '--config', {str(path)!r}, '--check-config']; "
        + ("runpy.run_path('train.py', run_name='__main__'); " if entrypoint == "train.py"
           else "runpy.run_module('anomaly_detection', run_name='__main__'); ")
        + "assert 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Configuration valid" in result.stdout
    assert not (tmp_path / "checkpoints").exists()
