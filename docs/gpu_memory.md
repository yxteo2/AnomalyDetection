# DINOv2, large images and the RTX 5070 Ti

## Scope

SSN supports `dinov2_vits14` and `dinov2_vitb14` through timm, in addition to the
three existing ResNet backbones. FastFlow still supports ResNet only: its flow
stages assume the ResNet spatial hierarchy. Unsupported combinations fail early.

All backbones are frozen and kept in evaluation mode. Adapters, detection heads
and flow modules remain trainable. DINOv2 uses timm's native PyTorch attention;
there is no xFormers dependency or runtime import of Python from a GitHub URL.
Pretrained weights download from the official timm Hugging Face model repository
on first use. Checkpoints include the backbone weights, so inference rebuilds
without a download. `pretrained: false` is for offline software tests, not a
substitute for pretrained features in an anomaly-detection experiment.

Prefer the full `best_model.pth` checkpoint with metadata. For a state-dict-only
file, SSN inference also accepts `--dino_layers`, `--feature_channels` and the
existing architecture flags explicitly. `--backbone_precision float32` can
override a full checkpoint's BF16 execution setting on another GPU; this changes
execution precision, not the stored model weights.

## Conservative starting profile

Edit the dataset in `configs/ssn_dinov2.yaml`, then run:

```bash
python train.py --config configs/ssn_dinov2.yaml --check-config
python train.py --config configs/ssn_dinov2.yaml
```

The profile uses ViT-S/14, one transformer block (`dino_layers: [11]`), a trainable
128-channel projection, batch size 1, and accumulation over four microbatches.
Change `model.image_size` to `[416, 416]`, `[512, 512]`, `[704, 704]` or a
rectangular size. Use a different `output.run_name` for each experiment.
There is no unlimited-resolution or 16 GB fit guarantee. ViT-B, more feature
blocks, larger batches and larger resolutions need separate memory measurements.

DINO block indices are zero-based, ordered and unique (0 through 11). The existing
`layers` option refers only to ResNet stages. DINO retains the native patch grid
instead of upscaling and concatenating large high-channel feature maps. Channel
projection requires `adapt_cls_features: true`, so both SSN heads consume the
projected features. This is a configurable SSN variant, not a claim of improved
accuracy over the original ResNet implementation.

## Padding and coordinate alignment

| Requested image | DINO padded input | Returned anomaly map |
| --- | --- | --- |
| 416 x 416 | 420 x 420 | 416 x 416 |
| 512 x 512 | 518 x 518 | 512 x 512 |
| 704 x 704 | 714 x 714 | 704 x 704 |
| 416 x 512 | 420 x 518 | 416 x 512 |

After the existing resize/crop/ImageNet normalization, the model zero-pads only
the bottom/right edges to multiples of 14. Input masks receive the same padding.
The output map is interpolated to the padded dimensions and then cropped, not
squeezed into the original dimensions. Padding still participates in attention;
it is not an attention-masked region. The profile uses `crop_scale: 1.0` to keep
the full resized field of view. Larger input sizes should be evaluated for both
small-defect sensitivity and runtime; more pixels are not automatically better.

## Memory behavior

- No backbone gradients or retained backbone backward graph.
- `backbone_precision: bfloat16` autocasts only DINO's frozen feature extraction
  on a CUDA device that supports BF16. CPU falls back to FP32. Heads and losses
  remain FP32; FastFlow likelihood/log-determinant calculations are not put in
  reduced precision.
- `feature_channels: 128` projects channels before synthetic-feature duplication
  and the heads. `null` preserves the original channel count.
- `training.accumulate_grad_batches` works for both detectors. Each microbatch
  is backpropagated immediately; gradients are sample-weighted, and a final
  partial window is flushed. No list of computation graphs is retained.
- Accumulation is not mathematically identical to a larger physical batch for
  models with BatchNorm, stochastic augmentation or synthetic anomaly generation.
- ResNet and DINO channel dimensions are obtained without full-image probe passes.
- Frozen parameters receive no gradient or Adam momentum buffers; gradients are
  cleared with `set_to_none`. FastFlow retains its historical parameter-group
  layout for optimizer-checkpoint compatibility.
- Checkpoints deserialize on CPU. Final evaluation releases gradients and
  optimizer states and does not reload optimizer states to GPU.
- Validation predictions are released before test predictions are collected.
  Exact AUROC still retains prediction arrays in host RAM; RAM use scales with
  dataset size. CPU DataLoaders do not unnecessarily pin memory.
- The example uses zero loader workers to limit host prefetch memory. Increase
  workers only after checking host RAM and throughput.

## CUDA installation and verification

The desktop RTX 5070 Ti has 16 GB memory and compute capability 12.0 (Blackwell).
Use a current NVIDIA driver and a matching CUDA-enabled PyTorch/torchvision pair
with Blackwell support. PyTorch 2.7 introduced support with CUDA 12.8 wheels;
the general project dependency floor alone does not guarantee GPU compatibility.
In a clean environment, a CUDA 12.8 wheel installation can be selected explicitly:

```bash
python -m pip install "torch>=2.7,<3" "torchvision>=0.22,<1" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Use the official PyTorch installer for another supported CUDA build or platform.
Do not assume that an older CUDA 11.8/12.1 build supports this card.

Measure the actual configuration before full training:

```bash
python -m anomaly_detection.smoke --config configs/ssn_dinov2.yaml --memory-budget-gb 14
```

This command ignores the dataset and output paths. It generates synthetic images,
runs enough batches for optimizer initialization and a partial accumulation
window, checks frozen gradients, and saves/reloads a temporary checkpoint.
It checks map dimensions, finite outputs and matching reloaded predictions.
It reports GPU identity, CUDA build and peak PyTorch allocated/reserved GiB, and
fails on CUDA absence, a memory-budget excess, or a training/reload error. The
temporary checkpoint is removed automatically. The 14 GiB default leaves some
headroom; other applications and non-PyTorch allocations are not fully included
in PyTorch's allocator statistics. Close GPU-heavy applications when measuring.

Repeat after changing the backbone, loss, feature blocks, image size or batch size.
If memory is insufficient, reduce physical batch size, selected blocks or feature
channels; accumulation cannot reduce the memory of an individual image.

Offline CPU smoke test (no pretrained download, no GPU-memory measurement):

```bash
python -m anomaly_detection.smoke --config configs/ssn_dinov2.yaml --device cpu --no-pretrained
```

Software tests exercise real randomly initialized DINO models at 416, 512 and 704
pixels and rectangular inputs. They do not establish pretrained anomaly accuracy
or certify a run on a physical RTX 5070 Ti. Run the CUDA command on that card.

## References

- [timm DINOv2 ViT-S/14 model and weights](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)
- [DINOv2 reference implementation](https://github.com/facebookresearch/dinov2)
- [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus)
- [RTX 5070 family specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/)
- [PyTorch Blackwell / CUDA 12.8 introduction](https://pytorch.org/blog/pytorch-2-7/)
- [PyTorch installation selector](https://pytorch.org/get-started/locally/)
