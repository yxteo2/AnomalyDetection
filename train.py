"""Train from a YAML experiment or the original command-line arguments."""

from anomaly_detection.cli import main, parse_args

__all__ = ["main", "parse_args"]


def __getattr__(name):
    # Preserve historical imports without loading torch for --check-config.
    if name in {"AnomalyPipeline", "make_det_tf", "make_train_tf"}:
        from anomaly_detection import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

if __name__ == "__main__":
    main()
