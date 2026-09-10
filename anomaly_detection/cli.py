"""YAML-first training CLI, with the original flag-based interface retained."""

import argparse

import yaml

from anomaly_detection.config import BACKBONES, LOSS_DEFAULTS, ConfigError, as_yaml_config, load_config, resolve_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build and train an anomaly detector from YAML or command-line options.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--config", help="YAML experiment path; cannot be combined with training flags.")
    parser.add_argument("--check-config", action="store_true", help="Validate and print resolved settings without training.")
    parser.add_argument("--data_path")
    parser.add_argument("--category")
    parser.add_argument("--model", choices=["fastflow", "ssn"])
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--image_size", type=int, nargs=2)
    parser.add_argument("--crop_scale", type=float)
    parser.add_argument("--flow_steps", type=int)
    parser.add_argument("--hidden_ratio", type=float)
    parser.add_argument("--clamp", type=float)
    parser.add_argument("--conv3x3_only", action="store_true")
    parser.add_argument("--perlin_threshold", type=float)
    parser.add_argument("--adapt_cls_features", action="store_true")
    parser.add_argument("--layers", nargs="+")
    parser.add_argument("--pretrained_backbone", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset_type", choices=["auto", "mvtec", "visa"])
    for name in ("batch_size", "num_epochs", "patience", "seed", "num_workers", "num_visualizations"):
        parser.add_argument(f"--{name}", type=int)
    for name in (
        "learning_rate", "weight_decay", "val_ratio", "head_lr_multiplier", "adaptor_weight_decay",
        "image_threshold_quantile", "pixel_threshold_quantile",
    ):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--save_dir")
    parser.add_argument("--run_name")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--loss", dest="loss_name", choices=LOSS_DEFAULTS)
    parser.add_argument("--accumulate_grad_batches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config", None)
    check_only = args.pop("check_config", False)
    try:
        if config_path is not None:
            if args:
                raise ConfigError("With --config, edit training options in YAML; CLI overrides are not supported.")
            cfg = load_config(config_path)
        else:
            cfg = resolve_config(args)
    except ConfigError as exc:
        parser.error(str(exc))
    return argparse.Namespace(config=cfg, check_config=check_only)


def main(argv=None):
    args = parse_args(argv)
    print(yaml.safe_dump(as_yaml_config(args.config), sort_keys=False))
    if args.check_config:
        print("Configuration valid. No model, dataset or output files were created.")
        return None
    # Lazy import lets --check-config run with PyYAML only, including on CPU hosts.
    from anomaly_detection.pipeline import AnomalyPipeline

    pipeline = AnomalyPipeline(args.config)
    metrics = pipeline.run()
    print(f"Training complete. Results saved to {pipeline.save_dir}")
    return metrics
