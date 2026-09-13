# Inference and preprocessing parity

Use the shared command for **FastFlow, SSN, PaDiM, PatchCore and the Dinomaly-style
variant**, with either supported DINOv2 backbone or a compatible ResNet:

```bash
python -m anomaly_detection.predict --checkpoint path/to/best_model.pth \
  --image path/to/image.png --output prediction.pt --device cuda --restore-original
```

The checkpoint must be a full YAML-training checkpoint, not a bare state dict.
The detector, backbone, selected layers, input dimensions and `crop_scale` come
from its saved configuration. Backbone weights load from the checkpoint without
another pretrained download. An unavailable requested CUDA device raises an
error instead of silently switching to CPU. The output file is never overwritten.

For repeated images, load the model once:

```python
from anomaly_detection.predict import AnomalyInferenceEngine

engine = AnomalyInferenceEngine("path/to/best_model.pth", device="cuda")
prediction = engine.predict("path/to/image.png", restore_original=True)
```

## Exact preprocessing sequence

Training's deterministic validation/test transform and every inference entry
point share `anomaly_detection/preprocessing.py`:

1. Convert the source image to RGB.
2. Resize to `(ceil(height / crop_scale), ceil(width / crop_scale))` using
   torchvision's antialiased bilinear tensor resize. This is a fixed rectangular
   resize, not letterboxing or aspect-ratio-preserving resizing.
3. Center crop to the checkpoint's `(height, width)`. Offsets use torchvision's
   `round((resize_dimension - crop_dimension) / 2)`, including round-to-even ties.
4. Convert uint8 values to FP32 `[0, 1]` and normalize with ImageNet mean
   `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
5. The detector handles DINO patch padding internally and crops its output back
   to the configured input dimensions. Do not pad or crop a second time outside
   the model. ResNet models do not receive this DINO padding.

Training alone additionally applies its existing random horizontal/vertical
flips. These are intentionally absent during evaluation/inference.

`crop_scale: 1.0` preserves the entire resized field of view. Smaller values
discard borders. Do not change cropping only at inference: retrain/refit and
recalibrate if you change the preprocessing used in an experiment.

## Outputs and coordinates

| Output key | Meaning |
| --- | --- |
| `anomaly_map` | Raw `[1, 1, height, width]` anomaly map in crop coordinates |
| `score` | Raw image score; higher means more anomalous |
| `geometry` | Source dimensions, resize dimensions, crop dimensions, offsets and scale |
| `original_anomaly_map` | Optional source-size display map; NaN outside inspected crop |
| `valid_region` | Optional source-size boolean mask marking inspected pixels |

Source-size restoration inserts the crop into its resized canvas and uses
nearest-neighbor projection back to the source size. It is a display mapping,
not an exact inverse of resizing, and cannot recover discarded borders. Use the
valid mask when visualizing or thresholding. Do not interpret excluded regions
as normal or compute pixel metrics over those regions.

FastFlow image scores use the same top-1%-pixel mean as the training evaluator.
Other detectors use their model-provided score. Maps are not arbitrarily clamped
or min-max normalized: PaDiM/PatchCore distances are not probabilities, and
SSN's image score is not replaced by a map maximum. Apply the matching run's
`calibration.json` thresholds to raw scores/maps when classification is needed;
the shared command itself does not make binary decisions.

## Ground-truth masks and legacy scripts

`engine.gt_mask_to_crop(mask, original_rgb)` uses the same torchvision Mask
resize and center-crop operations as dataset evaluation. Input masks must match
the source image dimensions; mismatches fail explicitly. Masks use nearest
interpolation, not bilinear smoothing.

The existing `inference.py` and `ssn_inference.py` remain FastFlow- and SSN-specific.
They now share the same preprocessing and corrected crop offsets, including
ground-truth alignment and contour restoration. For legacy checkpoints without
saved preprocessing, supply the original training size and crop scale explicitly:
use `--image_size HEIGHT WIDTH --crop_scale SCALE` (the legacy scale fallback
is 0.875, while saved metadata always takes precedence).
Missing metadata cannot be inferred safely. Use the shared command above for the
three new detector types.

Tests compare preprocessing pixels and masks against the dataset transform,
exercise odd crop margins, and verify checkpoint-based predictions for all five
DINO-backed detectors with and without center cropping. Additional geometry
checks cover 416, 512, 704 and rectangular sizes. These are CPU checks, not GPU
performance or anomaly-accuracy benchmarks.
