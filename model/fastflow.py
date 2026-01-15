import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import torchvision.models as models


# -----------------------------
# Glow-style invertible parts
# (we keep ActNorm + Inv1x1Conv, but expose anomalib-style outputs)
# -----------------------------
class LocalConvContext(nn.Module):
    """Learnable local neighborhood mixing: x + PW(DW(x))."""
    def __init__(self, c: int, k: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=k, padding=k//2, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=False)

        # Start as exact identity (pw=0 => output = x), but gradients can still flow
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity="linear")
        nn.init.zeros_(self.pw.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw(self.dw(x))
    
class ActNorm2d(nn.Module):
    """ActNorm (Glow): per-channel affine transform with data-dependent init."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("_initialized", torch.tensor(0, dtype=torch.uint8))

    @torch.no_grad()
    def _data_init(self, x: torch.Tensor):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std = x.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
        self.bias.data = -mean
        self.log_scale.data = torch.log(1.0 / (std + self.eps))
        self._initialized.fill_(1)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if (self._initialized.item() == 0) and (not reverse):
            self._data_init(x)

        b, c, h, w = x.shape
        if not reverse:
            y = (x + self.bias) * torch.exp(self.log_scale)
            # log|det| = H*W*sum(log_scale)
            ld = (h * w) * self.log_scale.view(1, c).sum(dim=1)  # [1]
            return y, ld.expand(b)  # [B]
        else:
            y = x * torch.exp(-self.log_scale) - self.bias
            ld = (h * w) * self.log_scale.view(1, c).sum(dim=1)
            return y, (-ld).expand(b)


class InvConv1x1(nn.Module):
    """Invertible 1x1 conv (Glow): mixes channels; logdet = H*W*slogdet(W)."""
    def __init__(self, num_channels: int):
        super().__init__()
        w = torch.randn(num_channels, num_channels)
        q, _ = torch.linalg.qr(w)
        self.weight = nn.Parameter(q)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        sign, logabsdet = torch.linalg.slogdet(self.weight)
        # (if sign flips it is still invertible; but keep init orthogonal for stability)

        if not reverse:
            y = F.conv2d(x, self.weight.view(c, c, 1, 1))
            log_det = (h * w) * logabsdet
            return y, log_det.expand(b)
        else:
            w_inv = torch.linalg.inv(self.weight).view(c, c, 1, 1)
            y = F.conv2d(x, w_inv)
            log_det = -(h * w) * logabsdet
            return y, log_det.expand(b)


class Subnet(nn.Module):
    """Subnet used inside coupling (anomalib uses 2 convs with ReLU)."""
    def __init__(self, in_ch: int, out_ch: int, hidden_ratio: float, kernel_size: int):
        super().__init__()
        hidden_ch = max(8, int(in_ch * hidden_ratio))
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, kernel_size, padding=padding),
        )
        # stability: start near identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AffineCoupling(nn.Module):
    """
    RealNVP affine coupling with clamping (like anomalib's AllInOneBlock):
      a *= 0.1
      s = clamp * tanh(a_s)
      y2 = x2 * exp(s) + t
      logdet = sum(s)
    """
    def __init__(self, channels: int, hidden_ratio: float, kernel_size: int, clamp: float = 2.0):
        super().__init__()
        assert channels % 2 == 0, "need even channels for split"
        self.channels = channels
        self.clamp = clamp

        self.c1 = channels // 2
        self.c2 = channels - self.c1

        # predict [s,t] for transformed half (size 2*c2)
        self.subnet = Subnet(self.c1, 2 * self.c2, hidden_ratio=hidden_ratio, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = torch.split(x, [self.c1, self.c2], dim=1)

        a = self.subnet(x1) * 0.1
        s, t = torch.split(a, [self.c2, self.c2], dim=1)
        s = self.clamp * torch.tanh(s)

        if not reverse:
            y2 = x2 * torch.exp(s) + t
            y = torch.cat([x1, y2], dim=1)
            log_det = s.sum(dim=(1, 2, 3))
            return y, log_det
        else:
            y2 = (x2 - t) * torch.exp(-s)
            y = torch.cat([x1, y2], dim=1)
            log_det = (-s).sum(dim=(1, 2, 3))
            return y, log_det


class FastFlowStep(nn.Module):
    """
    One step (keeps your Glow ingredients):
      Coupling -> Inv1x1Conv -> ActNorm
    This “follows anomalib” in outputs (z, logJ), not necessarily exact op order.
    """
    def __init__(self, channels: int, hidden_ratio: float, kernel_size: int, clamp: float = 2.0):
        super().__init__()
        self.coupling = AffineCoupling(channels, hidden_ratio=hidden_ratio, kernel_size=kernel_size, clamp=clamp)
        self.invconv = InvConv1x1(channels)
        self.actnorm = ActNorm2d(channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, ld1 = self.coupling(x, reverse=False)
        x, ld2 = self.invconv(x, reverse=False)
        x, ld3 = self.actnorm(x, reverse=False)
        return x, (ld1 + ld2 + ld3)


class AnomalyMapGenerator(nn.Module):
    """
    Anomalib FastFlow anomaly map:
      log_prob = -0.5 * mean(z^2 over channels)
      prob = exp(log_prob)
      upsample probs to input size, average, then negate
    :contentReference[oaicite:2]{index=2}
    """
    def __init__(self, input_size: Tuple[int, int]):
        super().__init__()
        self.input_size = input_size

    def forward(self, hidden_variables: List[torch.Tensor]) -> torch.Tensor:
        maps = []
        for z in hidden_variables:
            log_prob = -0.5 * torch.mean(z ** 2, dim=1, keepdim=True)   # [B,1,h,w]
            prob = torch.exp(log_prob)                                  # [B,1,h,w]
            prob = F.interpolate(prob, size=self.input_size, mode="bilinear", align_corners=False)
            maps.append(prob)
        flow_map = torch.stack(maps, dim=0).mean(dim=0)                 # [B,1,H,W]
        return -flow_map


class FastFlowModel(nn.Module):
    """
    Anomalib-style interface:
      - train(): returns (hidden_variables, jacobians)
      - eval():  returns anomaly_map
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        flow_steps: int = 8,
        input_size: Tuple[int, int] = (256, 256),
        reducer_channels: Tuple[int,int,int] | None = None,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
        clamp: float = 2.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.flow_steps = flow_steps
        self.conv3x3_only = conv3x3_only
        self.hidden_ratio = hidden_ratio
        self.clamp = clamp

        self.feat_drop = nn.ModuleList([
            nn.Dropout2d(p=0.2),  # try 0.05~0.2
            nn.Dropout2d(p=0.2),
            nn.Dropout2d(p=0.2),
        ])
        self.backbone, backbone_channels, self.scales = self._build_backbone(backbone_name, input_size)

        # reducers (optional)
        self.reducers = None
        if reducer_channels is not None:
            self.reducers = nn.ModuleList([
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
                for in_ch, out_ch in zip(backbone_channels, reducer_channels)
            ])
            self.feature_channels = list(reducer_channels)
        else:
            self.feature_channels = list(backbone_channels)

        # NOW build modules that depend on feature_channels
        self.context = nn.ModuleList([LocalConvContext(ch, k=3) for ch in self.feature_channels])

        self.norms = nn.ModuleList()
        for ch, sc in zip(self.feature_channels, self.scales):
            h = int(input_size[0] / sc)
            w = int(input_size[1] / sc)
            self.norms.append(nn.LayerNorm([ch, h, w], elementwise_affine=True))

        # flows per feature level
        self.blocks = nn.ModuleList()
        for ch in self.feature_channels:
            steps = nn.ModuleList()
            for i in range(flow_steps):
                if (i % 2 == 1) and (not conv3x3_only):
                    k = 1
                else:
                    k = 3
                steps.append(FastFlowStep(ch, hidden_ratio=hidden_ratio, kernel_size=k, clamp=clamp))
            self.blocks.append(steps)

        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

    def _build_backbone(self, name: str, input_size: Tuple[int, int]):
        if name == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            channels = [64, 128, 256]   # layer1, layer2, layer3
        elif name == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            channels = [64, 128, 256]
        elif name == "wide_resnet50_2":
            net = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            channels = [256, 512, 1024]
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        for p in net.parameters():
            p.requires_grad = False

        # scales for resnet with input H,W: after maxpool => /4, layer2 => /8, layer3 => /16
        scales = [4, 8, 16]

        return net, channels, scales

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        # ResNet forward manually for layer1/2/3 features
        net = self.backbone
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)

        f1 = net.layer1(x)  # scale /4
        f2 = net.layer2(f1) # /8
        f3 = net.layer3(f2) # /16

        feats = [f1, f2, f3]

        if self.reducers is not None:
            feats = [self.reducers[i](feats[i]) for i in range(3)]

        feats = [self.norms[i](feat) for i, feat in enumerate(feats)]

        feats = [self.context[i](feat) for i, feat in enumerate(feats)]
        if self.training:
            feats = [self.feat_drop[i](feat) for i, feat in enumerate(feats)]
        return feats

    def forward(self, x: torch.Tensor):
        # extract frozen features
        self.backbone.eval()
        features = self._extract_features(x)

        hidden_variables: List[torch.Tensor] = []
        jacobians: List[torch.Tensor] = []

        for feat, steps in zip(features, self.blocks):
            z = feat
            log_j = torch.zeros(z.size(0), device=z.device)
            for step in steps:
                z, ld = step(z)
                log_j = log_j + ld  # [B]
            hidden_variables.append(z)
            jacobians.append(log_j)

        # anomalib behavior:
        if self.training:
            return hidden_variables, jacobians

        return self.anomaly_map_generator(hidden_variables)  