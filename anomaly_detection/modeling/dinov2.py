"""Frozen DINOv2 patch features; no remote Python execution or xFormers required."""

import torch
from torch import nn
from torch.nn import functional as F


DINO_MODELS = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
}


def pad_to_patch_grid(images, patch_size=14):
    """Pad bottom/right only, retaining the original coordinate origin."""
    h, w = images.shape[-2:]
    return F.pad(images, (0, (-w) % patch_size, 0, (-h) % patch_size))


class DinoFeatureExtractor(nn.Module):
    def __init__(self, backbone_name, layers=(11,), pretrained=True, precision="float32"):
        super().__init__()
        import timm

        self.layers = tuple(layers)
        if not self.layers or any(type(i) is not int or not 0 <= i < 12 for i in self.layers):
            raise ValueError("DINO layers must be block indices in [0, 11].")
        if list(self.layers) != sorted(set(self.layers)):
            raise ValueError("DINO layers must be unique and ordered.")
        if precision not in ("float32", "bfloat16"):
            raise ValueError("backbone_precision must be float32 or bfloat16.")
        self.precision = precision
        self.feature_extractor = timm.create_model(
            DINO_MODELS[backbone_name], pretrained=pretrained,
            num_classes=0, dynamic_img_size=True,
        )
        self.feature_extractor.requires_grad_(False)
        self.feature_extractor.eval()
        self.channels = self.feature_extractor.num_features * len(self.layers)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def get_channels_dim(self, probe_hw=None):
        # Metadata avoids an expensive dummy transformer forward at construction.
        return self.channels

    @torch.no_grad()
    def forward(self, images):
        if any(size % 14 for size in images.shape[-2:]):
            raise ValueError("Pad images to the DINO patch grid before feature extraction.")
        self.feature_extractor.eval()
        use_bf16 = images.is_cuda and self.precision == "bfloat16"
        if use_bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 backbone inference is unavailable; use backbone_precision: float32.")
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=use_bf16):
            features = self.feature_extractor.get_intermediate_layers(
                images, n=self.layers, reshape=True, norm=True,
            )
            # Retain the native patch grid. Upscaling large-channel features before
            # the adapter wastes memory; only the final anomaly map is upscaled.
            fused = features[0] if len(features) == 1 else torch.cat(features, dim=1)
        return fused.float()
