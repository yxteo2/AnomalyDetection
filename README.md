# AnomalyDetection

PyTorch implementations of FastFlow and SuperSimpleNet (SSN) for image-level
and pixel-level industrial anomaly detection. The data loader supports MVTec AD,
MVTec-style folders and the original VisA layout.

SSN also supports frozen **DINOv2 ViT-S/14 and ViT-B/14** backbones with automatic
patch-grid padding for 416, 512, 704 and rectangular image sizes. Start with
`configs/ssn_dinov2.yaml`; see the [GPU and memory guide](docs/gpu_memory.md) for
the conservative RTX 5070 Ti profile and the actual CUDA memory smoke test.
DINOv2 is not yet supported by this repository's FastFlow implementation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `train.py` | Shared training and final-evaluation command-line entry point |
| `inference.py` | FastFlow inference, metrics and visualizations |
| `ssn_inference.py` | SuperSimpleNet inference, metrics and visualizations |
| `anomaly_detection/data.py` | MVTec, MVTec-style and VisA dataset loading |
| `anomaly_detection/modeling/` | FastFlow and SuperSimpleNet implementations |
| `anomaly_detection/modeling/dinov2.py` | Frozen timm DINOv2 features and patch-grid padding |
| `anomaly_detection/training/` | Model-specific training and evaluation utilities |
| `anomaly_detection/builder.py` | Configuration-driven model, loss and trainer factories |
| `anomaly_detection/config.py` | YAML schema, defaults and validation |
| `anomaly_detection/pipeline.py` | Shared experiment lifecycle |
| `anomaly_detection/smoke.py` | Synthetic training/reload and CUDA peak-memory checks |
| `anomaly_detection/training/accumulation.py` | Sample-weighted gradient accumulation |
| `configs/` | Ready-to-edit FastFlow, SSN focal and SSN BCE YAML experiments |
| `docs/yaml_training.md` | YAML options, supported combinations and extension guide |
| `docs/gpu_memory.md` | DINOv2, larger inputs and GPU memory configuration |
| `tests/` | Regression tests for evaluation and model behavior |

## Installation

Use Python 3.10 or newer. Install a PyTorch build compatible with your CUDA
runtime first, then install the remaining dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset layout

```text
MVTec/
└── bottle/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   └── broken_large/
    └── ground_truth/
        └── broken_large/
```

Training normal images are split deterministically into training and validation
subsets. The validation subset selects the checkpoint and calibrates deployment
thresholds. The MVTec test split is evaluated only after training is complete.

## Training

### Build a model from YAML

Edit the dataset path/category in `configs/fastflow.yaml` or `configs/ssn.yaml`,
then run:

```bash
python train.py --config configs/fastflow.yaml --check-config
python train.py --config configs/fastflow.yaml
```

Choose the backbone with `model.backbone`, detector with `model.name`, and loss
with `loss.name`. Training settings, loss weights and model parameters are also
configurable. `python -m anomaly_detection --config configs/ssn.yaml` is an
equivalent entry point. See the [YAML training guide](docs/yaml_training.md) for
the supported combinations and full examples.

Relative paths resolve from the YAML file's directory. Resolved settings are
saved with each checkpoint. Use a distinct `output.run_name` for each experiment;
nonempty output folders require an explicit `output.overwrite: true` opt-in.

### Original command-line interface

FastFlow:

```bash
python train.py \
  --data_path /path/to/MVTec \
  --category bottle \
  --model fastflow \
  --backbone resnet18 \
  --image_size 416 416
```

SSN:

```bash
python train.py \
  --data_path /path/to/MVTec \
  --category bottle \
  --model ssn \
  --backbone resnet18 \
  --layers layer2 layer3 \
  --image_size 416 416
```

The output folder contains `best_model.pth`, `calibration.json`, `metrics.json`
and `resolved_config.yaml`, plus any model-supported plots. Checkpoints include
the architecture and preprocessing configuration required by inference.

FastFlow input height and width must be divisible by 16. Use
`--no-pretrained_backbone` to train SSN without ImageNet initialization.
Existing run folders require a new `--run_name` or explicit `--overwrite`.

## Inference

New checkpoints automatically provide their model configuration. Inference
also automatically loads `calibration.json` from the checkpoint directory:

```bash
python inference.py \
  --checkpoint_path checkpoints/bottle/fastflow/resnet18/best_model.pth \
  --image_path /path/to/image-or-MVTec-test-folder \
  --category bottle
```

```bash
python ssn_inference.py \
  --checkpoint_path checkpoints/bottle/ssn/resnet18/best_model.pth \
  --image_path /path/to/image-or-MVTec-test-folder \
  --category bottle
```

For older checkpoints, pass their original architecture arguments and provide a
calibration file explicitly with `--calibration_path`. Calibration JSON requires
`image_threshold` and `pixel_threshold` fields. Generic unlabeled images are
supported; dataset metrics are printed only when labels can be inferred from a
MVTec `test/good` or `test/<defect>` path.

## Evaluation policy

- Larger model outputs always mean “more anomalous”; score direction is never
  chosen using evaluation labels.
- AUROC is computed directly from raw scores.
- Accuracy, precision and contour masks use thresholds calibrated only on the
  held-out normal validation subset.
- Pixel AUROC includes pixels from both good and defective test images.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q .
python -m pyflakes .
```

Pytest settings are kept in `pyproject.toml`. Generated checkpoints, datasets,
inference outputs and local experiment files are intentionally excluded from
version control through `.gitignore`.
