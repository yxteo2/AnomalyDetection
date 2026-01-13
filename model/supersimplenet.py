import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torchvision.models as models
import math
from typing import List, Tuple, Optional

from torchvision.models.feature_extraction import create_feature_extractor

def _init_weights(module: nn.Module) -> None:
    """Xavier init for Linear/Conv and constant=1 for BN weights."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if getattr(module, "weight", None) is not None:
            nn.init.constant_(module.weight, 1)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


class GaussianBlur2d(nn.Module):
    """Small, dependency-free Gaussian blur (depthwise conv)."""

    def __init__(self, kernel_size: int, sigma: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

        # Create 2D Gaussian kernel
        ax = torch.arange(self.kernel_size) - self.kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * (self.sigma**2)))
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None, :, :])  # (1,1,k,k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W) typically
        b, c, h, w = x.shape
        k = self.kernel.to(dtype=x.dtype, device=x.device)
        # depthwise: apply same kernel per channel
        k = k.expand(c, 1, self.kernel_size, self.kernel_size)
        return F.conv2d(x, k, padding=self.kernel_size // 2, groups=c)


class UpscalingFeatureExtractor(nn.Module):
    """Frozen backbone feature extractor with upscaling + patch aggregation.

    Matches the anomalib SSN idea:
    - extract layers (typically layer2, layer3)
    - upsample ALL selected layers to (2x the largest selected layer)
    - channel-wise concat
    - avgpool with stride=1 for local patch aggregation
    """

    def __init__(self, backbone_name: str, layers: List[str], patch_size: int = 3, pretrained: bool = True):
        super().__init__()
        self.layers = list(layers)

        if backbone_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone_name == "resnet34":
            backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone_name == "wide_resnet50_2":
            backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        return_nodes = {l: l for l in self.layers}
        self.feature_extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)

        self.pooler = nn.AvgPool2d(kernel_size=patch_size, stride=1, padding=patch_size // 2)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        feat_list = [feats[k] for k in self.layers]

        # "first" (largest) is typically layer2 for ResNets
        _, _, h, w = feat_list[0].shape
        target_hw = (h * 2, w * 2)

        up = [F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False) for f in feat_list]
        fused = torch.cat(up, dim=1)
        return self.pooler(fused)

    def get_channels_dim(self, probe_hw: Tuple[int, int] = (256, 256)) -> int:
        self.feature_extractor.eval()
        with torch.no_grad():
            feats = self.feature_extractor(torch.rand(1, 3, probe_hw[0], probe_hw[1]))
        return sum(v.shape[1] for v in feats.values())


class FeatureAdapter(nn.Module):
    """1x1 conv projection (linear per spatial location)."""

    def __init__(self, channel_dim: int):
        super().__init__()
        self.projection = nn.Conv2d(channel_dim, channel_dim, kernel_size=1, stride=1)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class _PerlinLikeMask(nn.Module):
    """Lightweight Perlin-like smooth noise mask via low-res noise upsample.

    The anomalib repo uses real Perlin noise; this is a close, fast approximation.
    """

    def __init__(self, base_res: int = 4):
        super().__init__()
        self.base_res = int(base_res)

    def forward(self, b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n = torch.rand((b, 1, self.base_res, self.base_res), device=device, dtype=dtype)
        n = F.interpolate(n, size=(h, w), mode="bicubic", align_corners=False)
        # normalize per-sample
        nmin = n.amin(dim=(2, 3), keepdim=True)
        nmax = n.amax(dim=(2, 3), keepdim=True)
        return (n - nmin) / (nmax - nmin + 1e-6)


class AnomalyGenerator(nn.Module):
    """Feature-level synthetic anomaly generator (train-time only).

    Creates a batch of size 2B:
      - first B are untouched (label 0)
      - second B are noised inside a smooth mask (label 1)

    If both `input_features` and `adapted_features` are given, applies the SAME noise mask to both.
    """

    def __init__(self, noise_mean: float = 0.0, noise_std: float = 0.015, threshold: float = 0.2, base_res: int = 4):
        super().__init__()
        self.noise_mean = float(noise_mean)
        self.noise_std = float(noise_std)
        self.threshold = float(threshold)
        self.mask_gen = _PerlinLikeMask(base_res=base_res)

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        adapted_features: torch.Tensor,
        masks: torch.Tensor,
        labels: Optional[torch.Tensor],
    ):
        # expected masks are already at feature resolution (B,1,H,W)
        b, c_ad, h, w = adapted_features.shape
        device, dtype = adapted_features.device, adapted_features.dtype

        # duplicate
        adapted2 = adapted_features.repeat(2, 1, 1, 1)
        input2 = input_features.repeat(2, 1, 1, 1) if input_features is not None else None

        # build synthetic mask for second half
        smooth = self.mask_gen(b, h, w, device=device, dtype=dtype)
        m = (smooth > self.threshold).float()

        noise_ad = torch.randn((b, c_ad, h, w), device=device, dtype=dtype) * self.noise_std + self.noise_mean
        adapted2[b:] = adapted2[b:] + noise_ad * m

        if input2 is not None:
            c_in = input2.shape[1]
            noise_in = torch.randn((b, c_in, h, w), device=device, dtype=dtype) * self.noise_std + self.noise_mean
            input2[b:] = input2[b:] + noise_in * m

        masks2 = masks.repeat(2, 1, 1, 1)
        labels2 = torch.zeros((2 * b,), device=device, dtype=torch.float32)

        # in unsupervised setting, original masks are all zeros; synthetic mask becomes target for second half
        masks2[b:] = m
        labels2[b:] = 1.0

        # if user provided labels (supervised/mixed), keep them for first half
        if labels is not None:
            labels2[:b] = labels.view(-1).to(dtype=torch.float32)

        return input2, adapted2, masks2, labels2


class SegmentationDetectionModule(nn.Module):
    """Anomalib SSN seg+cls module."""

    def __init__(self, channel_dim: int, stop_grad: bool = True):
        super().__init__()
        self.stop_grad = bool(stop_grad)

        # segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(channel_dim, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1, stride=1, bias=False),
        )

        # classification head
        self.cls_conv = nn.Sequential(
            nn.Conv2d(channel_dim + 1, 128, kernel_size=5, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # pooling
        self.map_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.map_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dec_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dec_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.cls_fc = nn.Linear(128 * 2 + 2, 1)
        self.apply(_init_weights)

    def forward(self, seg_features: torch.Tensor, cls_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ano_map = self.seg_head(seg_features)  # (B,1,H,W) logits

        map_dec_copy = ano_map.detach() if self.stop_grad else ano_map
        dec_in = torch.cat([cls_features, map_dec_copy], dim=1)
        dec_out = self.cls_conv(dec_in)

        dec_max = self.dec_max_pool(dec_out)
        dec_avg = self.dec_avg_pool(dec_out)

        map_max = self.map_max_pool(ano_map)
        map_avg = self.map_avg_pool(ano_map)
        if self.stop_grad:
            map_max = map_max.detach()
            map_avg = map_avg.detach()

        dec_cat = torch.cat([dec_max, dec_avg, map_max, map_avg], dim=1).squeeze()
        ano_score = self.cls_fc(dec_cat).reshape(-1)  # (B,)

        return ano_map, ano_score


class SSNAnomalyMapGenerator(nn.Module):
    """Upscale + smooth anomaly map (matches anomalib)."""

    def __init__(self, sigma: float = 4.0):
        super().__init__()
        kernel_size = 2 * math.ceil(3 * sigma) + 1
        self.smoothing = GaussianBlur2d(kernel_size=int(kernel_size), sigma=float(sigma))

    def forward(self, out_map: torch.Tensor, final_size: Tuple[int, int]) -> torch.Tensor:
        amap = F.interpolate(out_map, size=final_size, mode="bilinear", align_corners=False)
        return self.smoothing(amap)


class SuperSimpleNetModel(nn.Module):
    """SuperSimpleNet model aligned to anomalib's implementation.

    Key behavioral points (same as anomalib):
    - backbone frozen
    - anomaly generator duplicates batch (normal + synthetic)
    - stop_grad=True in unsupervised mode
    - during training returns logits + targets
    - during eval returns sigmoid(map/score)
    """

    def __init__(
        self,
        perlin_threshold: float = 0.2,
        backbone_name: str = "wide_resnet50_2",
        layers: List[str] = ["layer2", "layer3"],
        stop_grad: bool = True,
        adapt_cls_features: bool = False,
        input_size: Tuple[int, int] = (256, 256),
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.input_size = tuple(input_size)
        self.adapt_cls_features = bool(adapt_cls_features)

        self.feature_extractor = UpscalingFeatureExtractor(
            backbone_name=backbone_name,
            layers=layers,
            patch_size=3,
            pretrained=pretrained_backbone,
        )
        channels = self.feature_extractor.get_channels_dim()

        self.adaptor = FeatureAdapter(channels)
        self.segdec = SegmentationDetectionModule(channel_dim=channels, stop_grad=stop_grad)
        self.anomaly_generator = AnomalyGenerator(noise_mean=0.0, noise_std=0.015, threshold=perlin_threshold)
        self.anomaly_map_generator = SSNAnomalyMapGenerator(sigma=4.0)

    @staticmethod
    def downsample_mask(masks: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
        masks = masks.to(dtype=torch.float32)
        masks = F.interpolate(masks, size=(feat_h, feat_w), mode="bilinear", align_corners=False)
        return torch.where(masks < 0.5, torch.zeros_like(masks), torch.ones_like(masks))

    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        out_hw = images.shape[-2:]

        features = self.feature_extractor(images)
        adapted = self.adaptor(features)

        if self.training:
            # unsupervised: masks/labels can be None -> create zeros at feature resolution
            if masks is None:
                b, _, h, w = features.shape
                masks = torch.zeros((b, 1, h, w), dtype=torch.float32, device=features.device)
            else:
                masks = self.downsample_mask(masks, *features.shape[-2:])

            if labels is not None:
                labels = labels.to(dtype=torch.float32)

            if self.adapt_cls_features:
                # ICPR style: both heads use adapted
                _, noised_adapt, masks2, labels2 = self.anomaly_generator(
                    input_features=None,
                    adapted_features=adapted,
                    masks=masks,
                    labels=labels,
                )
                seg_feats = noised_adapt
                cls_feats = noised_adapt
            else:
                # JIMS extension: apply same noise to raw + adapted; cls uses raw features
                noised_feat, noised_adapt, masks2, labels2 = self.anomaly_generator(
                    input_features=features,
                    adapted_features=adapted,
                    masks=masks,
                    labels=labels,
                )
                seg_feats = noised_adapt
                cls_feats = noised_feat

            pred_map, pred_score = self.segdec(seg_features=seg_feats, cls_features=cls_feats)
            return pred_map, pred_score, masks2, labels2

        # inference
        seg_feats = adapted
        cls_feats = adapted if self.adapt_cls_features else features
        pred_map, pred_score = self.segdec(seg_features=seg_feats, cls_features=cls_feats)

        pred_map = self.anomaly_map_generator(pred_map, final_size=out_hw).sigmoid()
        pred_score = pred_score.sigmoid()

        return pred_map, pred_score
