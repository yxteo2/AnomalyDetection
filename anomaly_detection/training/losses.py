"""Loss modules with explicit contracts for flow latents and SSN logits."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss


class FastFlowNLLLoss(nn.Module):
    """Sum over feature levels of mean Gaussian NLL minus log-Jacobian."""

    def forward(self, hidden_vars, jacobians):
        if not hidden_vars or len(hidden_vars) != len(jacobians):
            raise ValueError("FastFlow NLL requires matching non-empty latent/Jacobian lists.")
        loss = hidden_vars[0].new_zeros(())
        for z, log_j in zip(hidden_vars, jacobians):
            loss = loss + (0.5 * z.square().sum(dim=(1, 2, 3)) - log_j).mean()
        return loss


class SSNLoss(nn.Module):
    """SSN focal + probability-truncation objective; defaults match the trainer."""

    def __init__(
        self, truncation_term=0.5, gamma=4.0, alpha=-1.0,
        seg_weight=1.0, cls_weight=1.0, truncation_weight=1.0,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.th = float(truncation_term)
        self.seg_weight = float(seg_weight)
        self.cls_weight = float(cls_weight)
        self.truncation_weight = float(truncation_weight)

    def focal(self, logits, target):
        return sigmoid_focal_loss(logits, target, alpha=self.alpha, gamma=self.gamma, reduction="mean")

    def trunc_l1_loss(self, pred_map_logits, target_mask):
        pred = pred_map_logits.sigmoid()
        normal_scores = pred[target_mask == 0]
        anomalous_scores = pred[target_mask > 0]
        true_loss = torch.clamp(normal_scores - (1.0 - self.th), min=0.0)
        fake_loss = torch.clamp(self.th - anomalous_scores, min=0.0)
        true_loss = true_loss.mean() if true_loss.numel() else pred.new_tensor(0.0)
        fake_loss = fake_loss.mean() if fake_loss.numel() else pred.new_tensor(0.0)
        return true_loss + fake_loss

    def forward(self, pred_map_logits, pred_score_logits, target_mask, target_label):
        map_focal = self.focal(pred_map_logits, target_mask)
        map_trunc = self.trunc_l1_loss(pred_map_logits, target_mask)
        score_focal = self.focal(pred_score_logits, target_label)
        return self.seg_weight * map_focal + self.truncation_weight * map_trunc + self.cls_weight * score_focal


class SSNBCELoss(nn.Module):
    """Weighted segmentation/classification BCE on logits, for SSN ablations."""

    def __init__(self, seg_weight=1.0, cls_weight=1.0):
        super().__init__()
        self.seg_weight = float(seg_weight)
        self.cls_weight = float(cls_weight)

    def forward(self, pred_map_logits, pred_score_logits, target_mask, target_label):
        segmentation = F.binary_cross_entropy_with_logits(pred_map_logits, target_mask)
        classification = F.binary_cross_entropy_with_logits(pred_score_logits, target_label)
        return self.seg_weight * segmentation + self.cls_weight * classification
