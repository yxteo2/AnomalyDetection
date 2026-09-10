"""Strict, safe YAML configuration for the built-in training pipelines.

This module deliberately does not import torch: configuration checking must not
download weights, construct a model, or require an available GPU/dataset.
"""

from copy import deepcopy
import math
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """An invalid or unsupported experiment configuration."""


BACKBONES = ("resnet18", "resnet34", "wide_resnet50_2")
MODEL_PARAMS = {
    "fastflow": {"flow_steps", "hidden_ratio", "clamp", "conv3x3_only"},
    "ssn": {"perlin_threshold", "adapt_cls_features", "layers"},
}
LOSS_DEFAULTS = {
    "fastflow_nll": {},
    "ssn_focal": {
        "gamma": 4.0, "alpha": -1.0, "truncation_term": 0.5,
        "seg_weight": 1.0, "cls_weight": 1.0, "truncation_weight": 1.0,
    },
    "ssn_bce": {"seg_weight": 1.0, "cls_weight": 1.0},
}
MODEL_LOSSES = {"fastflow": ("fastflow_nll",), "ssn": ("ssn_focal", "ssn_bce")}

# Keep the existing command-line defaults, and share them with YAML experiments.
DEFAULTS = {
    "data_path": None, "category": None, "dataset_type": "auto",
    "model": "fastflow", "backbone": "resnet18", "pretrained_backbone": True,
    "image_size": [416, 416], "crop_scale": 0.875,
    "flow_steps": 8, "hidden_ratio": 1.0, "clamp": 2.0, "conv3x3_only": False,
    "perlin_threshold": 0.2, "adapt_cls_features": False, "layers": ["layer2", "layer3"],
    "loss_name": None, "loss_params": {},
    "batch_size": 32, "num_epochs": 100, "learning_rate": 1e-4,
    "weight_decay": 1e-5, "head_lr_multiplier": 2.0, "adaptor_weight_decay": 0.01,
    "patience": 30, "val_ratio": 0.2, "seed": 42, "device": "auto", "num_workers": 4,
    "image_threshold_quantile": 0.99, "pixel_threshold_quantile": 0.999,
    "save_dir": "./checkpoints", "run_name": None, "overwrite": False,
    "num_visualizations": 10,
}
SECTIONS = {
    "dataset": {"path": "data_path", "category": "category", "type": "dataset_type", "val_ratio": "val_ratio"},
    "model": {
        "name": "model", "backbone": "backbone", "pretrained": "pretrained_backbone",
        "image_size": "image_size", "crop_scale": "crop_scale", "params": None,
    },
    "loss": {"name": "loss_name", "params": "loss_params"},
    "training": {
        "epochs": "num_epochs", "batch_size": "batch_size", "learning_rate": "learning_rate",
        "weight_decay": "weight_decay", "patience": "patience", "seed": "seed",
        "device": "device", "num_workers": "num_workers",
        "head_lr_multiplier": "head_lr_multiplier", "adaptor_weight_decay": "adaptor_weight_decay",
    },
    "evaluation": {
        "image_threshold_quantile": "image_threshold_quantile",
        "pixel_threshold_quantile": "pixel_threshold_quantile",
        "num_visualizations": "num_visualizations",
    },
    "output": {"save_dir": "save_dir", "run_name": "run_name", "overwrite": "overwrite"},
}


def _mapping(value, name):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{name} must be a mapping with string keys.")
    return value


def _known_keys(mapping, allowed, name):
    unknown = set(mapping) - set(allowed)
    if unknown:
        raise ConfigError(f"Unknown {name} option(s): {', '.join(sorted(unknown))}")


def _choice(value, choices, name):
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"{name} must be one of {', '.join(choices)}; got {value!r}.")
    return value


def _number(value, name, lower=0.0, upper=None, inclusive=True):
    # PyYAML can parse scientific notation such as 1e-4 as a string.
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ConfigError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be a finite number.")
    if number < lower or (not inclusive and number == lower) or (upper is not None and number > upper):
        raise ConfigError(f"{name} is outside its supported range (got {value!r}).")
    return number


def _component(value, name):
    if not isinstance(value, str) or not value.strip() or value in (".", "..") or "/" in value or "\\" in value:
        raise ConfigError(f"{name} must be a non-empty single directory name.")
    return value


def resolve_config(options):
    """Validate flat pipeline options, resolve defaults and return a fresh dict."""
    options = _mapping(options, "configuration")
    _known_keys(options, DEFAULTS, "configuration")
    cfg = deepcopy(DEFAULTS)
    cfg.update(deepcopy(options))
    for name in ("data_path", "save_dir"):
        if not isinstance(cfg[name], str) or not cfg[name].strip():
            raise ConfigError(f"{name} must be a non-empty path.")
    _component(cfg["category"], "category")
    if cfg["run_name"] is not None:
        _component(cfg["run_name"], "output.run_name")
    _choice(cfg["model"], MODEL_PARAMS, "model.name")
    _choice(cfg["backbone"], BACKBONES, "model.backbone")
    _choice(cfg["dataset_type"], ("auto", "mvtec", "visa"), "dataset.type")
    _choice(cfg["device"], ("auto", "cpu", "cuda"), "training.device")
    for name in ("pretrained_backbone", "conv3x3_only", "adapt_cls_features", "overwrite"):
        if type(cfg[name]) is not bool:
            raise ConfigError(f"{name} must be true or false (not a quoted string).")
    for name, minimum in {
        "flow_steps": 1, "batch_size": 1, "num_epochs": 1, "patience": 1,
        "num_workers": 0, "seed": 0, "num_visualizations": 0,
    }.items():
        if type(cfg[name]) is not int or cfg[name] < minimum:
            raise ConfigError(f"{name} must be an integer >= {minimum}.")
    if cfg["seed"] >= 2**32:
        raise ConfigError("seed must be less than 2**32.")
    size = cfg["image_size"]
    if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(v) is not int or v < 32 for v in size):
        raise ConfigError("model.image_size must be [height, width], with each integer >= 32.")
    cfg["image_size"] = list(size)
    if cfg["model"] == "fastflow" and any(v % 16 for v in size):
        raise ConfigError("FastFlow image_size height and width must be divisible by 16.")
    layers = cfg["layers"]
    allowed_layers = ["layer1", "layer2", "layer3", "layer4"]
    if (not isinstance(layers, list) or not layers or
            any(not isinstance(v, str) or v not in allowed_layers for v in layers) or
            layers != sorted(set(layers))):
        raise ConfigError("layers must be unique ResNet stages in order, e.g. [layer2, layer3].")
    for name in ("learning_rate", "hidden_ratio", "clamp", "head_lr_multiplier"):
        cfg[name] = _number(cfg[name], name, inclusive=False)
    for name in ("weight_decay", "adaptor_weight_decay"):
        cfg[name] = _number(cfg[name], name)
    for name in ("crop_scale", "val_ratio", "image_threshold_quantile", "pixel_threshold_quantile"):
        cfg[name] = _number(cfg[name], name, upper=1.0, inclusive=False)
    if cfg["val_ratio"] == 1.0:
        raise ConfigError("val_ratio must be strictly between 0 and 1.")
    cfg["perlin_threshold"] = _number(cfg["perlin_threshold"], "perlin_threshold", upper=1.0)

    loss_name = cfg["loss_name"]
    if loss_name is None:
        loss_name = MODEL_LOSSES[cfg["model"]][0]
    _choice(loss_name, MODEL_LOSSES[cfg["model"]], f"loss.name for {cfg['model']}")
    params = _mapping(cfg["loss_params"], "loss.params")
    _known_keys(params, LOSS_DEFAULTS[loss_name], "loss.params")
    params = {**LOSS_DEFAULTS[loss_name], **params}
    for name in params:
        lower = -1.0 if name == "alpha" else 0.0
        upper = 1.0 if name in ("alpha", "truncation_term") else None
        params[name] = _number(params[name], f"loss.params.{name}", lower, upper)
    if "alpha" in params and params["alpha"] != -1.0 and params["alpha"] < 0:
        raise ConfigError("loss.params.alpha must be -1 or between 0 and 1.")
    weights = [v for k, v in params.items() if k.endswith("_weight")]
    if weights and not any(weights):
        raise ConfigError("At least one loss weight must be positive.")
    cfg["loss_name"], cfg["loss_params"] = loss_name, params
    return cfg


class _UniqueSafeLoader(yaml.SafeLoader):
    """SafeLoader with duplicate-key rejection instead of silent overwrites."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError("YAML mapping keys must be strings.")
            if key in result:
                raise ConfigError(f"Duplicate YAML key: {key!r} at line {key_node.start_mark.line + 1}.")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def load_config(path):
    """Read one YAML experiment. Relative paths are relative to the YAML file."""
    path = Path(path).expanduser().resolve()
    try:
        with path.open(encoding="utf-8") as stream:
            document = yaml.load(stream, Loader=_UniqueSafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read configuration {path}: {exc}") from exc
    document = _mapping(document, "YAML configuration")
    _known_keys(document, SECTIONS, "section")
    options = {}
    for section, keys in SECTIONS.items():
        values = _mapping(document.get(section, {}), section)
        _known_keys(values, keys, section)
        options.update({keys[key]: value for key, value in values.items() if keys[key] is not None})
    model_name = options.get("model", DEFAULTS["model"])
    _choice(model_name, MODEL_PARAMS, "model.name")
    params = _mapping(document.get("model", {}).get("params", {}), "model.params")
    _known_keys(params, MODEL_PARAMS[model_name], f"model.params for {model_name}")
    if model_name != "ssn" and {"head_lr_multiplier", "adaptor_weight_decay"} & set(document.get("training", {})):
        raise ConfigError("head_lr_multiplier and adaptor_weight_decay are SSN-only training options.")
    options.update(params)
    cfg = resolve_config(options)
    for key in ("data_path", "save_dir"):
        resolved = Path(cfg[key]).expanduser()
        if not resolved.is_absolute():
            resolved = path.parent / resolved
        cfg[key] = str(resolved.resolve())
    return cfg


def as_yaml_config(cfg):
    """Return a reloadable nested config, including the effective defaults."""
    cfg = resolve_config(cfg)
    result = {
        section: {key: cfg[flat] for key, flat in keys.items() if flat is not None}
        for section, keys in SECTIONS.items()
    }
    result["model"]["params"] = {key: cfg[key] for key in sorted(MODEL_PARAMS[cfg["model"]])}
    if cfg["model"] != "ssn":
        for key in ("head_lr_multiplier", "adaptor_weight_decay"):
            result["training"].pop(key)
    return result
