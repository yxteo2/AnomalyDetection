# Build and train from YAML

The YAML module combines the repository's supported backbone, anomaly detector
and loss implementations. No Python editing is needed to run an experiment.
It is not a general layer-by-layer network-description language.

## Quick start

From the repository root, install the dependencies and edit `dataset.path` and
`dataset.category` in one of the examples:

```bash
python -m pip install -r requirements.txt
python train.py --config configs/fastflow.yaml --check-config
python train.py --config configs/fastflow.yaml
```

Equivalent package entry point:

```bash
python -m anomaly_detection --config configs/ssn.yaml
```

`--check-config` validates and prints all effective settings, including defaults.
It does not open the dataset, build a model, download weights or create output
files. A valid schema does not establish that a dataset path exists; training
checks the dataset before downloading pretrained weights.

YAML is a configuration file, not an executable: pass it to either command above.
The original `python train.py --data_path ... --category ...` interface remains
available. Do not mix `--config` with training flags; edit the YAML instead.

## Minimal example

```yaml
dataset:
  path: ../data/MVTec
  category: bottle
model:
  name: fastflow
  backbone: resnet18
  image_size: [256, 256]
loss:
  name: fastflow_nll
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.0001
  device: auto
output:
  save_dir: ../checkpoints
  run_name: experiment_01
```

Paths in YAML are relative to the YAML file's directory, including the default
`output.save_dir`. Absolute paths are also supported. For Windows, prefer forward
slashes (`C:/datasets/MVTec`) or single-quoted backslash paths.

## Supported components

| Selection | Options |
| --- | --- |
| `model.backbone` | `resnet18`, `resnet34`, `wide_resnet50_2` |
| `model.name` | `fastflow`, `ssn` |
| FastFlow `loss.name` | `fastflow_nll` |
| SSN `loss.name` | `ssn_focal`, `ssn_bce` |
| `dataset.type` | `auto`, `mvtec`, `visa` |
| `training.device` | `auto`, `cpu`, `cuda` |

Use `fastflow_nll` for flow training and `ssn_focal` to retain the existing SSN
objective. `ssn_bce` is an alternative objective for controlled experiments, not
a claimed performance improvement. Arbitrary combinations are invalid: flow
latents/Jacobians and SSN segmentation/classification logits have different loss
contracts. An incompatible combination is rejected before building a model.

Both detectors support all three listed backbones. FastFlow uses ResNet stages
1–3; its `wide_resnet50_2` variant retains the existing channel reducers. SSN's
feature stages are selectable. Pretrained initialization defaults to true;
set `model.pretrained: false` for offline smoke tests. This is generally not a
replacement for pretrained features in a real anomaly-detection experiment.

## Configuration sections

Unspecified options use the defaults printed by `--check-config`. Unknown keys,
duplicate keys, invalid types and unsupported combinations produce errors.

| Section | Fields |
| --- | --- |
| `dataset` | `path` and `category` (required); `type`, `val_ratio` |
| `model` | `name`, `backbone`, `pretrained`, `image_size`, `crop_scale`, `params` |
| `loss` | `name`, `params` |
| `training` | `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `patience`, `num_workers`, `seed`, `device`; SSN-only `head_lr_multiplier`, `adaptor_weight_decay` |
| `evaluation` | `image_threshold_quantile`, `pixel_threshold_quantile`, `num_visualizations` |
| `output` | `save_dir`, `run_name`, `overwrite` |

`image_size` is `[height, width]`, with both integers at least 32. FastFlow also
requires both to be divisible by 16. `crop_scale` must be in `(0, 1]` and
`dataset.val_ratio` strictly between 0 and 1. The dataset needs at least two
normal training images to make separate training and validation subsets.

### Model parameters

| Model | `model.params` fields and defaults |
| --- | --- |
| FastFlow | `flow_steps: 8`, `hidden_ratio: 1.0`, `clamp: 2.0`, `conv3x3_only: false` |
| SSN | `layers: [layer2, layer3]`, `perlin_threshold: 0.2`, `adapt_cls_features: false` |

SSN layers must be unique members of `layer1` through `layer4`, in shallow-to-deep
order. Parameters belonging to the other detector are rejected.

### Loss parameters

| Loss | `loss.params` fields and defaults |
| --- | --- |
| `fastflow_nll` | None; sums Gaussian NLL minus log-Jacobian across feature levels |
| `ssn_focal` | `gamma: 4.0`, `alpha: -1.0`, `truncation_term: 0.5`, `seg_weight: 1.0`, `cls_weight: 1.0`, `truncation_weight: 1.0` |
| `ssn_bce` | `seg_weight: 1.0`, `cls_weight: 1.0` |

SSN focal loss is the weighted sum of map focal loss, image focal loss and
truncation loss on map probabilities. SSN BCE uses binary cross entropy with
logits for both heads. Weights must be nonnegative, with at least one positive.
Focal `alpha` must be `-1` (disable alpha balancing) or in `[0, 1]`; `gamma` must
be nonnegative. See the [Torchvision focal-loss reference](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.sigmoid_focal_loss.html).

### Training and repeat runs

The existing optimizers/schedulers are retained: Adam/cosine scheduling for
FastFlow and AdamW/multistep scheduling for SSN. SSN uses `learning_rate` for its
adaptor and `learning_rate * head_lr_multiplier` for its heads. `weight_decay`
controls the heads; `adaptor_weight_decay` controls the adaptor. The defaults
retain the original SSN optimizer settings.

`seed` seeds Python, NumPy and PyTorch before model construction and controls the
normal-data split. It does not guarantee bitwise-identical results across devices
or library versions; see [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Runs save to:

```text
<save_dir>/<category>/<model>/<backbone>/<run_name>/
```

The final `run_name` directory is omitted when no name is supplied. Outputs include
`best_model.pth`, `resolved_config.yaml`, `calibration.json`, `metrics.json`, and
any model-supported plots. The checkpoint includes `model_cfg` for the existing
inference scripts and `experiment_cfg` for the training/loss settings.

By default a nonempty run directory is rejected. Prefer a new `run_name` when
comparing models or losses. `output.overwrite: true` (or legacy `--overwrite`)
explicitly permits overwriting matching output files; it does not resume training
or clear unrelated old files. A saved resolved config is reloadable, but requires
a new run name/output folder or that overwrite opt-in.

The original held-out validation and final-test policy is unchanged: training
uses normal training images, validation selects the checkpoint and calibrates
thresholds, and test labels are not used for either decision.

## Implementation and extension points

- `anomaly_detection/config.py`: schema, safe loader, validation and defaults.
- `anomaly_detection/builder.py`: explicit model/loss factories and trainer construction.
- `anomaly_detection/training/losses.py`: loss implementations.
- `anomaly_detection/pipeline.py`: shared training/calibration/evaluation lifecycle.
- `anomaly_detection/cli.py`: both supported entry points.

YAML uses a restricted `yaml.SafeLoader` subclass, with duplicate-key rejection,
and cannot name/import arbitrary Python classes. See the [PyYAML loading guidance](https://pyyaml.org/wiki/PyYAMLDocumentation).
To add a new component in Python, extend the schema/compatibility tables and the
matching factory, implement the required trainer/evaluator contract, and add
validation and training tests. Merely adding an arbitrary name to YAML is not
enough to support a new architecture.
