# Backbone, detector and loss selection

Select a complete detector with `model.name`, its frozen feature extractor with
`model.backbone`, and a compatible objective with `loss.name`. These detectors
have different fitting contracts; they are not interchangeable logit heads.

| Detector | Backbones | Loss choices | Procedure |
| --- | --- | --- | --- |
| `fastflow` | All three ResNets; DINOv2 small/base | `fastflow_nll` | Train flow likelihood |
| `ssn` | All three ResNets; DINOv2 small/base | `ssn_focal`, `ssn_bce`, `ssn_bce_dice`, `ssn_focal_dice` | Train synthetic-anomaly segmentation/classification |
| `padim` | All three ResNets; DINOv2 small/base | `none` | Fit regularized per-location Gaussian statistics once |
| `patchcore` | All three ResNets; DINOv2 small/base | `none` | Fit patch reservoir and greedy coreset once |
| `dinomaly` | DINOv2 small/base | `cosine`, `mse`, `smooth_l1`, `cosine_mse` | Train feature reconstruction |

Backbone names: `resnet18`, `resnet34`, `wide_resnet50_2`, `dinov2_vits14`,
`dinov2_vitb14`. DINO selectors are zero-based `dino_layers` indices 0–11.
The name DINO here means these two **DINOv2** architectures; DINOv1, DINOv3,
register-token and large/giant models are not implemented by this selector.
Incompatible losses fail validation before model construction. PaDiM and
PatchCore have no optimizer; `epochs` does not repeat their one fitting pass and
gradient accumulation must be 1. Their fitting loader uses deterministic
preprocessing and only the training partition; held-out validation calibrates
thresholds, and the test partition is used only for final evaluation.

## Run an example

Edit the dataset path/category and choose an example:

```bash
python train.py --config configs/padim_dinov2.yaml --check-config
python train.py --config configs/padim_dinov2.yaml
```

Other examples: `patchcore_dinov2.yaml`, `dinomaly_dinov2.yaml`,
`fastflow_dinov2.yaml`, `ssn_dinov2.yaml`. Set `model.image_size` to
`[416, 416]`, `[512, 512]`, `[704, 704]` or another supported size. DINO pads to
the patch grid and crops the final map back to the requested dimensions.
For `cosine_mse`, `loss.params` accepts nonnegative `cosine_weight` and
`mse_weight` (both default 1; at least one must be positive). MSE and Smooth L1
act on reconstructed features, not binary defect masks. Changing the loss
requires an accuracy experiment; no loss is promised to improve every dataset.

All YAML-built detectors can reload without downloading weights:

```bash
python -m anomaly_detection.predict --checkpoint path/to/best_model.pth \
  --image path/to/image.png --output prediction.pt --device cuda
```

The output contains the raw anomaly score and map at the configured input size.
It does not apply calibration or restore the source image's original size.

## Implementation scope and memory

These new combinations are experimental local variants, not wrappers around
anomalib and not claims to reproduce published benchmark scores:

- **FastFlow + DINO:** selected blocks are concatenated on their native patch
  grid and processed as one feature level. Optional even `feature_channels`
  projects into a smaller space before the flow. The flow models that projected
  distribution; it is not the original ResNet pyramid.
- **PaDiM:** randomly selects up to `padim_channels` feature channels (default
  32), fits streaming FP64 statistics on CPU, and stores FP32 means/inverse
  covariances. `covariance_regularization` defaults to 0.01. The covariance
  allocation estimate is checked against `fit_memory_mb` (default 512 MiB).
  This is an allocation guard, not a total-process RAM limit. At least two
  normal training images are required. Fixed-position statistics may be
  unsuitable for poorly aligned objects.
- **PatchCore:** locally pools features, retains a CPU random-priority reservoir
  capped by `max_patches` (default 10,000), then selects up to
  `memory_bank_size` (1,000) patches by greedy k-center. Query and bank chunks
  are bounded by `distance_chunk_size` (256); no full pairwise distance matrix
  is constructed. Scores use the maximum nearest-neighbor distance, without
  the original paper's image-score reweighting. A bounded reservoir trades
  coverage for memory. Large banks can still take substantial fitting time.
- **Dinomaly-style:** frozen normalized DINO patch features feed a dropout MLP
  and linear-attention reconstruction decoder with two layer-fusion groups.
  `decoder_depth` defaults to 4 and must be at least 2; `bottleneck_dropout`
  defaults to 0.2. `gradient_checkpointing: true` recomputes decoder activations
  during backward. This compact variant omits original register/class-token
  processing, original hard-mining hooks, StableAdamW and its learning-rate
  schedule. It uses AdamW and the chosen reconstruction loss. Anomaly maps use
  cosine reconstruction distance for every loss. It is **not the exact
  official Dinomaly implementation**.

All backbones remain frozen/eval. CUDA BF16 is confined to DINO extraction;
trainable heads, covariance evaluation, nearest-neighbor distances and losses
use FP32. Start with batch size 1 on a 16 GB GPU. Accumulation is available for
FastFlow, SSN and Dinomaly. Larger images, ViT-B and extra selected blocks still
increase memory; no arbitrary-size or 16 GB fit guarantee is made.

Measure each configuration on the actual GPU:

```bash
python -m anomaly_detection.smoke --config configs/dinomaly_dinov2.yaml --memory-budget-gb 14
```

The smoke command also supports PaDiM/PatchCore fitting and checkpoint reload
(reported `loss: null` means no training objective). It uses a small synthetic
dataset, so it does not saturate a large PatchCore reservoir or measure accuracy.
Monitor host RAM and GPU memory during full fitting too. See
[GPU setup and geometry](gpu_memory.md). CPU software tests are not RTX 5070 Ti
hardware certification.

## Algorithm references

- [PaDiM paper](https://arxiv.org/abs/2011.08785)
- [PatchCore reference implementation](https://github.com/amazon-science/patchcore-inspection)
- [Official Dinomaly implementation](https://github.com/guojiajeremy/Dinomaly)
- [Anomalib Dinomaly architecture reference](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/dinomaly.html)
