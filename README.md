# AnomalyDetection

PyTorch implementations of FastFlow and SuperSimpleNet (SSN) for image-level
and pixel-level industrial anomaly detection. The data loader supports MVTec AD,
MVTec-style folders and the original VisA layout.

## Repository layout

| Path | Purpose |
| --- | --- |
| `train.py` | Shared training and final-evaluation command-line entry point |
| `inference.py` | FastFlow inference, metrics and visualizations |
| `ssn_inference.py` | SuperSimpleNet inference, metrics and visualizations |
| `dataset.py` | MVTec, MVTec-style and VisA dataset loading |
| `trainer.py` | FastFlow training and evaluation utilities |
| `ssntrainer.py` | SuperSimpleNet training and evaluation utilities |
| `model/` | Model implementations and public model exports |
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
and training plots. Checkpoints include the architecture and preprocessing
configuration required by inference.

FastFlow input height and width must be divisible by 16. Use
`--no-pretrained_backbone` to train SSN without ImageNet initialization.

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
