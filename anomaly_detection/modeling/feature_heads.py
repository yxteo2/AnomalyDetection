"""Experimental frozen-feature detectors, with bounded fitting allocations.

PaDiM uses per-location Gaussians; PatchCore uses a reservoir plus greedy
coreset. Dinomaly is a compact, patch-token reconstruction variant of the
published architecture, not an exact reproduction of its training recipe.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from anomaly_detection.modeling.dinov2 import DINO_MODELS, DinoFeatureExtractor, pad_to_patch_grid


class FrozenFeatures(nn.Module):
    def __init__(self, backbone_name, layers, dino_layers, pretrained_backbone, backbone_precision):
        super().__init__()
        self.is_dino = backbone_name in DINO_MODELS
        if self.is_dino:
            self.net = DinoFeatureExtractor(backbone_name, dino_layers, pretrained_backbone, backbone_precision)
            self.channels = self.net.channels
        else:
            import timm
            self.net = timm.create_model(backbone_name, pretrained=pretrained_backbone,
                                         features_only=True, out_indices=tuple(int(s[-1]) for s in layers))
            self.channels = sum(self.net.feature_info.channels())
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        if self.is_dino:
            return self.net(pad_to_patch_grid(images))
        maps = self.net(images)
        return torch.cat([F.interpolate(m, maps[0].shape[-2:], mode="bilinear", align_corners=False)
                          for m in maps], dim=1)


class FeatureDetector(nn.Module):
    def __init__(self, backbone_name, input_size, pretrained_backbone=True, layers=("layer2", "layer3"),
                 dino_layers=(11,), backbone_precision="float32", **kwargs):
        super().__init__()
        self.input_size = tuple(input_size)
        self.backbone = FrozenFeatures(backbone_name, layers, dino_layers, pretrained_backbone, backbone_precision)

    def features(self, images):
        if tuple(images.shape[-2:]) != self.input_size:
            raise ValueError(f"Expected image size {self.input_size}.")
        return self.backbone(images)

    def output(self, patch_map):
        h, w = self.input_size
        size = ((h + 13) // 14 * 14, (w + 13) // 14 * 14) if self.backbone.is_dino else (h, w)
        maps = F.interpolate(patch_map, size, mode="bilinear", align_corners=False)[..., :h, :w]
        return maps, maps.flatten(1).amax(1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Fitting determines bank/grid dimensions. Preserve destination device.
        for name in ("mean", "precision", "memory_bank"):
            if name in self._buffers and prefix + name in state_dict:
                self._buffers[name] = torch.empty_like(state_dict[prefix + name], device=self._buffers[name].device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class PadimModel(FeatureDetector):
    def __init__(self, padim_channels=32, covariance_regularization=0.01, fit_memory_mb=512, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("channel_indices", torch.randperm(self.backbone.channels)[:padim_channels])
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("precision", torch.empty(0))
        self.regularization = covariance_regularization
        self.fit_memory_mb = fit_memory_mb

    @torch.no_grad()
    def fit_features(self, batches):
        count, mean, scatter = 0, None, None
        for features in batches:
            x = features[:, self.channel_indices].flatten(2).permute(2, 0, 1).cpu().double()
            patches, n, channels = x.shape
            # Budget includes scatter, covariance, inverse and factorization workspace.
            if patches * channels * channels * 8 * 5 > self.fit_memory_mb * 2**20:
                raise ValueError("PaDiM covariance exceeds fit_memory_mb; reduce padim_channels or image size.")
            batch_mean = x.mean(1)
            centered = x - batch_mean[:, None]
            batch_scatter = centered.transpose(1, 2) @ centered
            if mean is None:
                mean, scatter = batch_mean, batch_scatter
            else:
                delta = batch_mean - mean
                scatter.add_(batch_scatter).add_(delta.unsqueeze(2) * delta.unsqueeze(1), alpha=count * n / (count + n))
                mean.add_(delta, alpha=n / (count + n))
            count += n
        if count < 2:
            raise ValueError("PaDiM needs at least two normal training images.")
        covariance = scatter / (count - 1)
        covariance.diagonal(dim1=-2, dim2=-1).add_(self.regularization)
        self.mean = mean.float().to(self.channel_indices.device)
        self.precision = torch.linalg.inv(covariance).float().to(self.channel_indices.device)

    def forward(self, images):
        if not self.mean.numel():
            raise RuntimeError("Fit PaDiM before inference.")
        features = self.features(images)
        x = features[:, self.channel_indices].flatten(2).permute(0, 2, 1) - self.mean
        distances = torch.einsum("bpc,pcd,bpd->bp", x, self.precision, x).clamp_min(0).sqrt()
        return self.output(distances.reshape(images.shape[0], 1, *features.shape[-2:]))


class PatchcoreModel(FeatureDetector):
    def __init__(self, max_patches=10000, memory_bank_size=1000, distance_chunk_size=256, **kwargs):
        super().__init__(**kwargs)
        self.max_patches, self.memory_bank_size = max_patches, memory_bank_size
        self.distance_chunk_size = distance_chunk_size
        self.register_buffer("memory_bank", torch.empty(0))

    @staticmethod
    def embed(features):
        return F.avg_pool2d(features, 3, stride=1, padding=1).permute(0, 2, 3, 1).reshape(-1, features.shape[1])

    @torch.no_grad()
    def fit_features(self, batches):
        bank, keys = None, None
        for features in batches:
            # CPU reservoir with random priorities, processed in bounded chunks.
            for x in self.embed(features).split(self.distance_chunk_size):
                x = x.cpu()
                k = torch.rand(len(x))
                bank = x if bank is None else torch.cat((bank, x))
                keys = k if keys is None else torch.cat((keys, k))
                keep = keys.topk(min(self.max_patches, len(keys))).indices
                bank, keys = bank[keep], keys[keep]
        if bank is None:
            raise ValueError("PatchCore needs normal training images.")
        # Greedy k-center on the bounded reservoir. No N x N distance matrix.
        selected, nearest, index = [], torch.full((len(bank),), float("inf")), 0
        for _ in range(min(self.memory_bank_size, len(bank))):
            selected.append(index)
            for start in range(0, len(bank), self.distance_chunk_size):
                stop = start + self.distance_chunk_size
                distance = (bank[start:stop] - bank[index]).square().sum(1)
                nearest[start:stop] = torch.minimum(nearest[start:stop], distance)
            nearest[selected] = -1
            index = int(nearest.argmax())
        self.memory_bank = bank[selected].to(self.memory_bank.device)

    def forward(self, images):
        if not self.memory_bank.numel():
            raise RuntimeError("Fit PatchCore before inference.")
        features = self.features(images)
        distances = []
        for query in self.embed(features).split(self.distance_chunk_size):
            best = torch.full((len(query),), float("inf"), device=query.device)
            for bank in self.memory_bank.split(self.distance_chunk_size):
                best = torch.minimum(best, torch.cdist(query, bank).amin(1))
            distances.append(best)
        maps = torch.cat(distances).reshape(images.shape[0], 1, *features.shape[-2:])
        return self.output(maps)


class LinearDecoderBlock(nn.Module):
    """Positive-kernel linear attention avoids a quadratic token attention matrix."""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim, bias=False), nn.GELU(), nn.Linear(4 * dim, dim, bias=False))

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        kv = k.transpose(-2, -1) @ v / n
        denominator = (q * k.mean(-2, keepdim=True)).sum(-1, keepdim=True).clamp_min(1e-6)
        attention = ((q @ kv) / denominator).transpose(1, 2).reshape(b, n, c)
        x = x + self.proj(attention)
        return x + self.mlp(self.norm2(x))


class DinomalyModel(FeatureDetector):
    """Dinomaly-style variant: frozen patch features, dropout MLP, linear decoder.

    Uses selected normalized patch layers, two fusion groups, AdamW and optional
    reconstruction losses. Does not reproduce original register tokens, optimizer
    schedule or hard-mining hooks; benchmark separately before production use.
    """
    def __init__(self, decoder_depth=4, bottleneck_dropout=0.2, gradient_checkpointing=True, **kwargs):
        if kwargs["backbone_name"] not in DINO_MODELS:
            raise ValueError("Dinomaly requires a DINOv2 backbone.")
        super().__init__(**kwargs)
        self.layer_count = len(kwargs.get("dino_layers", (11,)))
        dim = self.backbone.channels // self.layer_count
        self.checkpointing = gradient_checkpointing
        self.bottleneck = nn.Sequential(nn.Dropout(bottleneck_dropout), nn.Linear(dim, dim * 4, bias=False),
                                        nn.GELU(), nn.Dropout(bottleneck_dropout), nn.Linear(dim * 4, dim, bias=False))
        self.decoder = nn.ModuleList([LinearDecoderBlock(dim, 6 if dim == 384 else 12) for _ in range(decoder_depth)])

    def reconstruct(self, images):
        features = self.features(images).chunk(self.layer_count, dim=1)
        tokens = [f.flatten(2).transpose(1, 2) for f in features]
        x = self.bottleneck(sum(tokens) / len(tokens))
        decoded = []
        for block in self.decoder:
            x = checkpoint(block, x, use_reentrant=False) if self.training and self.checkpointing else block(x)
            decoded.append(x)
        decoded.reverse()
        def groups(values):
            split = max(1, len(values) // 2)
            return [values[:split], values[split:] or values[:split]]
        targets = [sum(group) / len(group) for group in groups(tokens)]
        predictions = [sum(group) / len(group) for group in groups(decoded)]
        return targets, predictions, features[0].shape[-2:]

    def forward(self, images):
        targets, predictions, size = self.reconstruct(images)
        maps = sum(1 - F.cosine_similarity(a, b, dim=-1) for a, b in zip(targets, predictions)) / len(targets)
        return self.output(maps.reshape(images.shape[0], 1, *size))
