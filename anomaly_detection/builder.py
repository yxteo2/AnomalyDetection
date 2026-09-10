"""Build supported backbones, anomaly modules, losses and trainers from config.

Factories are explicit allowlists: YAML never imports or executes arbitrary code.
"""

from anomaly_detection.config import resolve_config
from anomaly_detection.modeling import FastFlowModel, SuperSimpleNetModel
from anomaly_detection.training import FastFlowTrainer, SuperSimpleNetTrainer
from anomaly_detection.training.losses import FastFlowNLLLoss, SSNBCELoss, SSNLoss, SSNBCEDiceLoss, SSNFocalDiceLoss


MODEL_BUILDERS = {"fastflow": FastFlowModel, "ssn": SuperSimpleNetModel}
LOSS_BUILDERS = {"fastflow_nll": FastFlowNLLLoss, "ssn_focal": SSNLoss, "ssn_bce": SSNBCELoss}
LOSS_BUILDERS.update(ssn_bce_dice=SSNBCEDiceLoss, ssn_focal_dice=SSNFocalDiceLoss)


def model_kwargs(cfg):
    """Constructor settings, also stored in checkpoints for existing inference."""
    cfg = resolve_config(cfg)
    kwargs = {
        "backbone_name": cfg["backbone"],
        "input_size": tuple(cfg["image_size"]),
        "pretrained_backbone": cfg["pretrained_backbone"],
    }
    if cfg["model"] == "fastflow":
        kwargs.update({key: cfg[key] for key in ("flow_steps", "hidden_ratio", "clamp", "conv3x3_only")})
        kwargs["reducer_channels"] = (128, 192, 256) if cfg["backbone"] == "wide_resnet50_2" else None
    else:
        kwargs.update({key: cfg[key] for key in ("perlin_threshold", "layers", "adapt_cls_features")})
        kwargs["stop_grad"] = True
        kwargs.update({key: cfg[key] for key in ("dino_layers", "feature_channels", "backbone_precision")})
    return kwargs


def build_model(cfg):
    cfg = resolve_config(cfg)
    return MODEL_BUILDERS[cfg["model"]](**model_kwargs(cfg))


def build_loss(cfg):
    cfg = resolve_config(cfg)
    return LOSS_BUILDERS[cfg["loss_name"]](**cfg["loss_params"])


def build_trainer(cfg, model, device, save_dir):
    cfg = resolve_config(cfg)
    kwargs = {
        "model": model, "device": str(device), "save_dir": str(save_dir),
        "learning_rate": cfg["learning_rate"], "weight_decay": cfg["weight_decay"],
        "model_cfg": {**model_kwargs(cfg), "crop_scale": cfg["crop_scale"]},
        "loss_fn": build_loss(cfg), "experiment_cfg": cfg,
        "monitor": "val_loss", "maximize": False,
        "accumulate_grad_batches": cfg["accumulate_grad_batches"],
    }
    if cfg["model"] == "fastflow":
        return FastFlowTrainer(backbone_name=cfg["backbone"], **kwargs)
    return SuperSimpleNetTrainer(
        head_lr_multiplier=cfg["head_lr_multiplier"],
        adaptor_weight_decay=cfg["adaptor_weight_decay"], **kwargs,
    )
