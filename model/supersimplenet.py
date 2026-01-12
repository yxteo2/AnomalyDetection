import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torchvision.models as models

class _SSNFeatureExtractor(nn.Module):
    """Frozen CNN backbone + multi-layer feature fusion (SSN-style).

    Extracts layer2 and layer3, upsamples layer3 to layer2 resolution, concatenates,
    then applies stride-1 avg pooling (patch aggregation) to mix local context.
    """

    def __init__(self, backbone_name: str = "wide_resnet50_2", patch_k: int = 3):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.c2, self.c3 = 128, 256
        elif backbone_name == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.c2, self.c3 = 128, 256
        elif backbone_name == "wide_resnet50_2":
            net = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            self.c2, self.c3 = 512, 1024
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        for p in net.parameters():
            p.requires_grad = False
        net.eval()
        self.net = net

        self.patch_k = int(patch_k)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.net
        x = n.conv1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)

        x = n.layer1(x)       # /4
        f2 = n.layer2(x)      # /8
        f3 = n.layer3(f2)     # /16

        f3u = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([f2, f3u], dim=1)  # (B, C2+C3, H/8, W/8)

        if self.patch_k > 1:
            fused = F.avg_pool2d(fused, kernel_size=self.patch_k, stride=1, padding=self.patch_k // 2)

        return fused

    @property
    def out_channels(self) -> int:
        return int(self.c2 + self.c3)


def _ssn_smooth_noise_mask(
    b: int,
    h: int,
    w: int,
    device: torch.device,
    base_res: int = 4,
    thr: float = 0.5,
) -> torch.Tensor:
    """Perlin-ish smooth binary masks without extra dependencies.

    1) random grid at (base_res, base_res)
    2) bicubic upsample to (h,w) => smooth blobs
    3) normalize then threshold => binary mask

    returns: (B,1,H,W) float in {0,1}
    """
    base_res = max(2, int(base_res))
    grid = torch.rand((b, 1, base_res, base_res), device=device)
    noise = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)

    mn = noise.amin(dim=(2, 3), keepdim=True)
    mx = noise.amax(dim=(2, 3), keepdim=True)
    noise = (noise - mn) / (mx - mn + 1e-12)

    mask = (noise > float(thr)).float()
    return mask


class _SSNAnomalyGenerator(nn.Module):
    """Inject synthetic anomalies into features during training.

    Duplicates batch:
      - first half: original (normal)
      - second half: features + (mask * gaussian_noise)

    Produces synthetic masks and image labels.
    """

    def __init__(self, noise_std: float = 0.02, mask_thr: float = 0.5, base_res: int = 4):
        super().__init__()
        self.noise_std = float(noise_std)
        self.mask_thr = float(mask_thr)
        self.base_res = int(base_res)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = feat.shape
        feat2 = feat.repeat(2, 1, 1, 1)

        mask = torch.zeros((2 * b, 1, h, w), device=feat.device, dtype=feat.dtype)
        y = torch.zeros((2 * b,), device=feat.device, dtype=torch.long)

        m = _ssn_smooth_noise_mask(b, h, w, device=feat.device, base_res=self.base_res, thr=self.mask_thr)
        n = torch.randn((b, c, h, w), device=feat.device, dtype=feat.dtype) * self.noise_std

        feat2[b:] = feat2[b:] + n * m
        mask[b:] = m
        y[b:] = 1

        return feat2, mask, y


class _SSNDiscriminator(nn.Module):
    """Dual head: pixel map logits + image score logits."""

    def __init__(self, in_ch: int, hidden: int = 256):
        super().__init__()

        self.seg_head = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),  # anomaly logits map
        )

        self.dec = nn.Sequential(
            nn.Conv2d(in_ch + 1, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        amap_logits = self.seg_head(feat)  # (B,1,h,w)

        x = torch.cat([feat, amap_logits], dim=1)
        x = self.dec(x)

        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        score_logits = self.fc(torch.cat([avg, mx], dim=1)).squeeze(1)  # (B,)

        return amap_logits, score_logits


class SuperSimpleNetModel(nn.Module):
    """SuperSimpleNet model.

    Training forward returns:
      amap_logits: (2B,1,h,w) feature resolution logits
      score_logits: (2B,) image-level logits
      syn_mask: (2B,1,h,w) synthetic pixel mask (0 for normal half)
      syn_label: (2B,) image label (0 normal, 1 synthetic)

    Eval forward returns:
      anomaly_map: (B,1,H,W) in [0,1] at input resolution.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        input_size: Tuple[int, int] = (256, 256),
        adaptor_dim: int = 256,
        patch_k: int = 3,
        noise_std: float = 0.02,
        mask_thr: float = 0.5,
        base_res: int = 4,
        hidden: int = 256,
    ):
        super().__init__()
        self.input_size = tuple(input_size)

        self.extractor = _SSNFeatureExtractor(backbone_name=backbone_name, patch_k=patch_k)
        self.adaptor = nn.Conv2d(self.extractor.out_channels, int(adaptor_dim), kernel_size=1)

        self.anom_gen = _SSNAnomalyGenerator(noise_std=noise_std, mask_thr=mask_thr, base_res=base_res)
        self.disc = _SSNDiscriminator(in_ch=int(adaptor_dim), hidden=int(hidden))

    def get_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return anomaly map at input resolution in [0,1]."""
        self.eval()
        with torch.no_grad():
            feat = self.extractor(x)
            feat = self.adaptor(feat)
            amap_logits, _ = self.disc(feat)
            amap = torch.sigmoid(amap_logits)
            amap = F.interpolate(amap, size=self.input_size, mode="bilinear", align_corners=False)
            return amap

    def forward(self, x: torch.Tensor):
        feat = self.extractor(x)
        feat = self.adaptor(feat)

        if self.training:
            feat2, syn_mask, syn_label = self.anom_gen(feat)
            amap_logits, score_logits = self.disc(feat2)
            return amap_logits, score_logits, syn_mask, syn_label

        amap_logits, _ = self.disc(feat)
        amap = torch.sigmoid(amap_logits)
        amap = F.interpolate(amap, size=self.input_size, mode="bilinear", align_corners=False)
        return amap
