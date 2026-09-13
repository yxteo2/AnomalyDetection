"""Normal-only fitting and reconstruction training for frozen-feature heads."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from anomaly_detection.training.accumulation import GradientAccumulator


class ReconstructionLoss(nn.Module):
    def __init__(self, name="cosine", cosine_weight=1.0, mse_weight=1.0):
        super().__init__()
        self.name, self.cosine_weight, self.mse_weight = name, cosine_weight, mse_weight

    def forward(self, targets, predictions):
        values = []
        for target, prediction in zip(targets, predictions):
            target, prediction = target.detach().float(), prediction.float()
            cosine = (1 - F.cosine_similarity(target, prediction, dim=-1)).mean()
            if self.name == "cosine":
                value = cosine
            elif self.name == "mse":
                value = F.mse_loss(prediction, target)
            elif self.name == "smooth_l1":
                value = F.smooth_l1_loss(prediction, target)
            else:
                value = self.cosine_weight * cosine + self.mse_weight * F.mse_loss(prediction, target)
            values.append(value)
        return sum(values) / len(values)


class FeatureHeadTrainer:
    def __init__(self, model, device, save_dir, loss_fn, experiment_cfg, model_cfg, learning_rate,
                 weight_decay, accumulate_grad_batches, **kwargs):
        self.model, self.device = model.to(device), device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn, self.experiment_cfg, self.model_cfg = loss_fn, experiment_cfg, model_cfg
        self.accumulate_grad_batches = accumulate_grad_batches
        self.optimizer = None if loss_fn is None else torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=weight_decay)
        self.history = {}

    def normal_images(self, loader):
        for batch in loader:
            images = batch["image"]
            if "label" in batch:
                images = images[batch["label"].view(-1) == 0]
            if len(images):
                yield images.to(self.device)

    def fit_statistics(self, loader):
        self.model.eval()
        with torch.no_grad():
            self.model.fit_features(self.model.features(images) for images in self.normal_images(loader))

    def train_epoch(self, dataloader):
        if self.optimizer is None:
            raise RuntimeError("PaDiM/PatchCore fit statistics; call fit_statistics, not train_epoch.")
        self.model.train()
        accumulator = GradientAccumulator(self.optimizer, self.accumulate_grad_batches, max_norm=1.0)
        total, count = 0.0, 0
        for images in self.normal_images(dataloader):
            targets, predictions, _ = self.model.reconstruct(images)
            loss = self.loss_fn(targets, predictions)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite reconstruction loss.")
            accumulator.backward(loss, len(images))
            total += loss.item() * len(images)
            count += len(images)
        accumulator.step()
        if not count:
            raise ValueError("No normal training images.")
        return total / count

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total, count = 0.0, 0
        for images in self.normal_images(loader):
            targets, predictions, _ = self.model.reconstruct(images)
            loss = self.loss_fn(targets, predictions)
            total += loss.item() * len(images)
            count += len(images)
        if not count:
            raise ValueError("No normal validation images.")
        return total / count

    def fit(self, train_loader, val_loader, num_epochs=100, patience=10):
        if self.optimizer is None:
            self.fit_statistics(train_loader)
            self.history = {"fit_passes": 1}
            self.save_checkpoint("best_model.pth")
            return
        best, stale = float("inf"), 0
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            if not torch.isfinite(torch.tensor(val_loss)):
                raise RuntimeError("Non-finite validation loss.")
            self.history = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            print(f"Epoch {epoch + 1}: train={train_loss:.6f}, val={val_loss:.6f}")
            if val_loss < best:
                best, stale = val_loss, 0
                self.save_checkpoint("best_model.pth")
            else:
                stale += 1
            if stale >= patience:
                break

    def save_checkpoint(self, filename):
        torch.save({"model_state_dict": self.model.state_dict(), "experiment_cfg": self.experiment_cfg,
                    "model_cfg": self.model_cfg, "history": self.history,
                    "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None},
                   self.save_dir / filename)

    def load_checkpoint(self, filename, strict=True, load_optimizer=True):
        checkpoint_data = torch.load(self.save_dir / filename, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint_data["model_state_dict"], strict=strict)
        if load_optimizer and self.optimizer and checkpoint_data.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        self.history = checkpoint_data.get("history", {})
