# trainer.py
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class FastFlowTrainer:
    """Trainer for FastFlow anomaly detection model (anomalib-style monitoring)."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        save_dir: str = "./checkpoints",
        monitor: str = "image_auroc",   # or "pixel_auroc"
        maximize: bool = True,          # AUROC higher is better
    ):
        self.model = model.to(device)
        self.device = device

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.maximize = maximize

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.history = {
            "train_loss": [],
            "val_loss": [],          # normal-only loss if labels exist
            "image_auroc": [],
            "pixel_auroc": [],
            "best_metric": -float("inf") if maximize else float("inf"),
            "best_epoch": 0,
            "epoch": 0,
        }

    def _fastflow_loss(self, hidden_vars, jacobians) -> torch.Tensor:
        # sum_l mean( 0.5*sum(z^2) - log_detJ )
        loss = torch.zeros((), device=self.device)
        for z, log_j in zip(hidden_vars, jacobians):
            loss = loss + (0.5 * (z ** 2).sum(dim=(1, 2, 3)) - log_j).mean()
        return loss

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            images = batch["image"].to(self.device)

            hidden_vars, jacobians = self.model(images)  # anomalib-style output in train
            loss = self._fastflow_loss(hidden_vars, jacobians)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def validate_nll_on_normals(self, dataloader) -> float:
        """
        Compute FastFlow loss but ONLY on normal samples if labels exist.
        This makes val_loss meaningful even if you pass test_dataloader (mixed).
        """
        self.model.train()  # keep train=True so forward returns (hidden,jac)
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Validation(NormalOnly)"):
            images = batch["image"].to(self.device)

            # filter normals if label exists (MVTec: 0=good, 1=anomaly)
            if "label" in batch:
                labels = batch["label"].to(self.device).view(-1)
                normal_mask = (labels == 0)
                if not normal_mask.any():
                    continue
                images = images[normal_mask]

            hidden_vars, jacobians = self.model(images)
            loss = self._fastflow_loss(hidden_vars, jacobians)

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(1, num_batches)

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        patience: int = 10,
        eval_every: int = 1,
    ):
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        best_metric = self.history["best_metric"]
        patience_counter = 0

        evaluator = FastFlowEvaluator(self.model, device=self.device)

        for epoch in range(num_epochs):
            self.history["epoch"] = epoch + 1
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # normal-only val loss (safe even if val_loader is test_loader)
            val_loss = self.validate_nll_on_normals(val_loader)
            self.history["val_loss"].append(val_loss)

            if (epoch + 1) % eval_every == 0:
                preds = evaluator.predict(val_loader)
                metrics = evaluator.compute_metrics(preds)

                img_auc = metrics["image_auroc"]
                px_auc = metrics["pixel_auroc"]

                self.history["image_auroc"].append(img_auc)
                self.history["pixel_auroc"].append(px_auc)

                print(
                    f"Train Loss: {train_loss:.4f} | Val Loss(normal): {val_loss:.4f} | "
                    f"Val Image-AUROC: {img_auc:.4f} | Val Pixel-AUROC: {px_auc:.4f}"
                )

                current = metrics[self.monitor]
                improved = (current > best_metric) if self.maximize else (current < best_metric)

                if improved:
                    best_metric = current
                    self.history["best_metric"] = best_metric
                    self.history["best_epoch"] = epoch + 1
                    self.save_checkpoint("best_model.pth")
                    patience_counter = 0
                    print(f"✓ New best model saved ({self.monitor}={best_metric:.4f})")
                else:
                    patience_counter += 1
            else:
                print(f"Train Loss: {train_loss:.4f} | Val Loss(normal): {val_loss:.4f}")
                patience_counter += 1

            scheduler.step()

            if patience_counter >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch + 1} "
                    f"(best {self.monitor}={best_metric:.4f} @ epoch {self.history['best_epoch']})"
                )
                break

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

        self.plot_history()
        print("\nTraining completed!")
        print(f"Best {self.monitor}: {best_metric:.4f} (epoch {self.history['best_epoch']})")

    def save_checkpoint(self, filename: str):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        torch.save(checkpoint, self.save_dir / filename)

    def load_checkpoint(self, filename: str):
        path = self.save_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        print(f"Checkpoint loaded from {path}")

    def plot_history(self):
        if len(self.history["train_loss"]) == 0:
            return

        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, self.history["train_loss"], label="Train Loss", marker="o")
        ax.plot(epochs, self.history["val_loss"], label="Val Loss (normal-only)", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(self.save_dir / "training_history_loss.png", dpi=150, bbox_inches="tight")
        plt.close()

        if len(self.history["image_auroc"]) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            e2 = range(1, len(self.history["image_auroc"]) + 1)
            ax.plot(e2, self.history["image_auroc"], label="Val Image AUROC", marker="o")
            ax.plot(e2, self.history["pixel_auroc"], label="Val Pixel AUROC", marker="s")
            ax.set_xlabel("Eval step")
            ax.set_ylabel("AUROC")
            ax.set_title("Validation AUROC")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.savefig(self.save_dir / "training_history_auroc.png", dpi=150, bbox_inches="tight")
            plt.close()


class FastFlowEvaluator:
    """Evaluator that supports both: model.get_anomaly_map(x) OR model(x) in eval returns map."""

    def __init__(self, model: nn.Module, device: str = "cuda", topk_ratio: float = 0.01):
        self.model = model.to(device)
        self.device = device
        self.topk_ratio = topk_ratio

    def _anomaly_map(self, images: torch.Tensor) -> torch.Tensor:
        # Prefer explicit method if exists
        if hasattr(self.model, "get_anomaly_map"):
            return self.model.get_anomaly_map(images)
        # Otherwise assume eval forward returns map
        self.model.eval()
        return self.model(images)

    @torch.no_grad()
    def predict(self, dataloader) -> Dict[str, np.ndarray]:
        self.model.eval()
        all_scores, all_labels, all_masks, all_maps = [], [], [], []

        for batch in tqdm(dataloader, desc="Predicting"):
            images = batch["image"].to(self.device)

            labels = batch.get("label", torch.zeros(images.size(0)))
            labels_np = labels.detach().cpu().numpy().reshape(-1)

            masks = batch.get(
                "mask",
                torch.zeros(images.size(0), 1, images.size(2), images.size(3), device=images.device),
            )
            masks_np = masks.detach().cpu().numpy()

            anomaly_map = self._anomaly_map(images)       # [B,1,H,W]
            amap = anomaly_map.squeeze(1)                 # [B,H,W]
            flat = amap.flatten(1)                        # [B,HW]

            k = max(1, int(self.topk_ratio * flat.shape[1]))
            scores = flat.topk(k, dim=1).values.mean(dim=1)   # [B]
            scores_np = scores.detach().cpu().numpy()

            all_scores.append(scores_np)
            all_labels.append(labels_np)
            all_masks.append(masks_np)
            all_maps.append(anomaly_map.detach().cpu().numpy())

        return {
            "scores": np.concatenate(all_scores, axis=0),
            "labels": np.concatenate(all_labels, axis=0),
            "masks": np.concatenate(all_masks, axis=0),
            "anomaly_maps": np.concatenate(all_maps, axis=0),
        }

    def compute_metrics(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        scores = predictions["scores"]
        labels = predictions["labels"]
        masks = predictions["masks"]
        maps = predictions["anomaly_maps"]

        image_auroc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0

        has_mask = np.array([m.max() > 0 for m in masks])
        if has_mask.sum() > 0:
            pixel_labels = masks[has_mask].reshape(-1)
            pixel_scores = maps[has_mask].reshape(-1)
            pixel_auroc = roc_auc_score(pixel_labels, pixel_scores) if len(np.unique(pixel_labels)) > 1 else 0.0
        else:
            pixel_auroc = 0.0

        return {"image_auroc": float(image_auroc), "pixel_auroc": float(pixel_auroc)}

    @torch.no_grad()
    def visualize_results(self, dataloader, save_dir: str = "./results", num_samples: int = 10):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        shown = 0

        for batch in dataloader:
            if shown >= num_samples:
                break

            images = batch["image"].to(self.device)
            labels = batch.get("label", torch.zeros(images.size(0))).detach().cpu().numpy().reshape(-1)
            masks = batch.get(
                "mask",
                torch.zeros(images.size(0), 1, images.size(2), images.size(3), device=images.device),
            ).detach().cpu().numpy()

            anomaly_maps = self._anomaly_map(images).detach().cpu().numpy()

            images_np = images.detach().cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            images_np = np.clip(images_np * std + mean, 0, 1)

            for i in range(images.shape[0]):
                if shown >= num_samples:
                    break

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                img = images_np[i].transpose(1, 2, 0)
                axes[0].imshow(img)
                axes[0].set_title(f"Input (Label: {'Anomaly' if labels[i] else 'Normal'})")
                axes[0].axis("off")

                axes[1].imshow(masks[i, 0], cmap="gray")
                axes[1].set_title("Ground Truth Mask")
                axes[1].axis("off")

                im = axes[2].imshow(anomaly_maps[i, 0], cmap="jet")
                axes[2].set_title("Predicted Anomaly Map")
                axes[2].axis("off")
                plt.colorbar(im, ax=axes[2])

                plt.tight_layout()
                plt.savefig(save_dir / f"sample_{shown + 1}.png", dpi=150, bbox_inches="tight")
                plt.close()

                shown += 1

        print(f"Visualizations saved to {save_dir}")


# ======================================================================
# SuperSimpleNet (SSN) trainer + evaluator
# ======================================================================


class SuperSimpleNetTrainer:
    """Trainer for SuperSimpleNet (SSN).

    SSN is discriminative: it learns to segment synthetic anomalies (pixel loss)
    and classify images as normal/anomalous (image loss) using a dual-head
    discriminator.

    Training is *unsupervised* by default: it only needs normal images because
    it injects synthetic anomalies in feature space.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        save_dir: str = "./checkpoints",
        monitor: str = "image_auroc",
        maximize: bool = True,
        seg_weight: float = 1.0,
        cls_weight: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.maximize = maximize

        self.seg_weight = float(seg_weight)
        self.cls_weight = float(cls_weight)

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = None

        self.bce_map = nn.BCEWithLogitsLoss()
        self.bce_cls = nn.BCEWithLogitsLoss()

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "image_auroc": [],
            "pixel_auroc": [],
            "best_metric": -float("inf") if maximize else float("inf"),
            "best_epoch": 0,
            "epoch": 0,
        }

    def _loss(self, amap_logits, score_logits, syn_mask, syn_label) -> torch.Tensor:
        # amap_logits: (2B,1,h,w)  syn_mask: (2B,1,h,w)
        seg_loss = self.bce_map(amap_logits, syn_mask)
        # score_logits: (2B,)  syn_label: (2B,)
        cls_loss = self.bce_cls(score_logits, syn_label.float())
        return self.seg_weight * seg_loss + self.cls_weight * cls_loss

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training(SSN)")
        for batch in pbar:
            images = batch["image"].to(self.device)

            amap_logits, score_logits, syn_mask, syn_label = self.model(images)
            loss = self._loss(amap_logits, score_logits, syn_mask, syn_label)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def validate_loss_on_normals(self, dataloader) -> float:
        """Compute SSN training loss on *normal-only* samples (if labels exist).

        This keeps val_loss aligned with the SSN training distribution (good-only).
        """
        self.model.train()  # need train=True to get logits + synthetic targets
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="ValidationLoss(SSN,NormalOnly)"):
            images = batch["image"].to(self.device)

            if "label" in batch:
                labels = batch["label"].to(self.device).view(-1)
                normal_mask = labels == 0
                if not normal_mask.any():
                    continue
                images = images[normal_mask]

            amap_logits, score_logits, syn_mask, syn_label = self.model(images)
            loss = self._loss(amap_logits, score_logits, syn_mask, syn_label)

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(1, num_batches)

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        patience: int = 10,
        eval_every: int = 1,
    ):
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        best_metric = self.history["best_metric"]
        patience_counter = 0

        evaluator = SuperSimpleNetEvaluator(self.model, device=self.device)

        for epoch in range(num_epochs):
            self.history["epoch"] = epoch + 1
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            val_loss = self.validate_loss_on_normals(val_loader)
            self.history["val_loss"].append(val_loss)

            if (epoch + 1) % eval_every == 0:
                preds = evaluator.predict(val_loader)
                metrics = evaluator.compute_metrics(preds)

                img_auc = metrics["image_auroc"]
                px_auc = metrics["pixel_auroc"]

                self.history["image_auroc"].append(img_auc)
                self.history["pixel_auroc"].append(px_auc)

                print(
                    f"Train Loss: {train_loss:.4f} | Val Loss(normal): {val_loss:.4f} | "
                    f"Val Image-AUROC: {img_auc:.4f} | Val Pixel-AUROC: {px_auc:.4f}"
                )

                current = metrics[self.monitor]
                improved = (current > best_metric) if self.maximize else (current < best_metric)

                if improved:
                    best_metric = current
                    self.history["best_metric"] = best_metric
                    self.history["best_epoch"] = epoch + 1
                    self.save_checkpoint("best_model.pth")
                    patience_counter = 0
                    print(f"✓ New best model saved ({self.monitor}={best_metric:.4f})")
                else:
                    patience_counter += 1
            else:
                print(f"Train Loss: {train_loss:.4f} | Val Loss(normal): {val_loss:.4f}")
                patience_counter += 1

            self.scheduler.step()

            if patience_counter >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch + 1} "
                    f"(best {self.monitor}={best_metric:.4f} @ epoch {self.history['best_epoch']})"
                )
                break

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

        self.plot_history()
        print("\nTraining completed!")
        print(f"Best {self.monitor}: {best_metric:.4f} (epoch {self.history['best_epoch']})")

    def save_checkpoint(self, filename: str):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        torch.save(checkpoint, self.save_dir / filename)

    def load_checkpoint(self, filename: str):
        path = self.save_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        print(f"Checkpoint loaded from {path}")

    def plot_history(self):
        if len(self.history["train_loss"]) == 0:
            return

        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, self.history["train_loss"], label="Train Loss", marker="o")
        ax.plot(epochs, self.history["val_loss"], label="Val Loss (normal-only)", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("SSN Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(self.save_dir / "training_history_loss.png", dpi=150, bbox_inches="tight")
        plt.close()

        if len(self.history["image_auroc"]) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            e2 = range(1, len(self.history["image_auroc"]) + 1)
            ax.plot(e2, self.history["image_auroc"], label="Val Image AUROC", marker="o")
            ax.plot(e2, self.history["pixel_auroc"], label="Val Pixel AUROC", marker="s")
            ax.set_xlabel("Eval step")
            ax.set_ylabel("AUROC")
            ax.set_title("SSN Validation AUROC")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.savefig(self.save_dir / "training_history_auroc.png", dpi=150, bbox_inches="tight")
            plt.close()


class SuperSimpleNetEvaluator:
    """Evaluator for SSN.

    Uses the model's anomaly map and derives an image score via top-k mean.
    """

    def __init__(self, model: nn.Module, device: str = "cuda", topk_ratio: float = 0.01):
        self.model = model.to(device)
        self.device = device
        self.topk_ratio = float(topk_ratio)

    def _anomaly_map(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        if hasattr(self.model, "get_anomaly_map"):
            return self.model.get_anomaly_map(images)
        return self.model(images)

    @torch.no_grad()
    def predict(self, dataloader) -> Dict[str, np.ndarray]:
        self.model.eval()
        all_scores, all_labels, all_masks, all_maps = [], [], [], []

        for batch in tqdm(dataloader, desc="Predicting(SSN)"):
            images = batch["image"].to(self.device)

            labels = batch.get("label", torch.zeros(images.size(0)))
            labels_np = labels.detach().cpu().numpy().reshape(-1)

            masks = batch.get(
                "mask",
                torch.zeros(images.size(0), 1, images.size(2), images.size(3), device=images.device),
            )
            masks_np = masks.detach().cpu().numpy()

            anomaly_map = self._anomaly_map(images)        # [B,1,H,W] in [0,1]
            amap = anomaly_map.squeeze(1)                  # [B,H,W]
            flat = amap.flatten(1)                         # [B,HW]

            k = max(1, int(self.topk_ratio * flat.shape[1]))
            scores = flat.topk(k, dim=1).values.mean(dim=1)
            scores_np = scores.detach().cpu().numpy()

            all_scores.append(scores_np)
            all_labels.append(labels_np)
            all_masks.append(masks_np)
            all_maps.append(anomaly_map.detach().cpu().numpy())

        return {
            "scores": np.concatenate(all_scores, axis=0),
            "labels": np.concatenate(all_labels, axis=0),
            "masks": np.concatenate(all_masks, axis=0),
            "anomaly_maps": np.concatenate(all_maps, axis=0),
        }

    def compute_metrics(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute image- and pixel-level AUROC (same computation as FastFlowEvaluator)."""
        scores = predictions["scores"]
        labels = predictions["labels"]
        masks = predictions["masks"]
        maps = predictions["anomaly_maps"]

        image_auroc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0

        has_mask = np.array([m.max() > 0 for m in masks])
        if has_mask.sum() > 0:
            pixel_labels = masks[has_mask].reshape(-1)
            pixel_scores = maps[has_mask].reshape(-1)
            pixel_auroc = roc_auc_score(pixel_labels, pixel_scores) if len(np.unique(pixel_labels)) > 1 else 0.0
        else:
            pixel_auroc = 0.0

        return {"image_auroc": float(image_auroc), "pixel_auroc": float(pixel_auroc)}

    @torch.no_grad()
    def visualize_results(self, dataloader, save_dir: str = "./results", num_samples: int = 10):
        """Save a few qualitative examples (input / GT mask / predicted map)."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        shown = 0

        for batch in dataloader:
            if shown >= num_samples:
                break

            images = batch["image"].to(self.device)
            labels = batch.get("label", torch.zeros(images.size(0))).detach().cpu().numpy().reshape(-1)
            masks = batch.get(
                "mask",
                torch.zeros(images.size(0), 1, images.size(2), images.size(3), device=images.device),
            ).detach().cpu().numpy()

            anomaly_maps = self._anomaly_map(images).detach().cpu().numpy()

            # de-normalize for visualization
            images_np = images.detach().cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            images_np = np.clip(images_np * std + mean, 0, 1)

            for i in range(images.shape[0]):
                if shown >= num_samples:
                    break

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                img = images_np[i].transpose(1, 2, 0)
                axes[0].imshow(img)
                axes[0].set_title(f"Input (Label: {'Anomaly' if labels[i] else 'Normal'})")
                axes[0].axis("off")

                axes[1].imshow(masks[i, 0], cmap="gray")
                axes[1].set_title("Ground Truth Mask")
                axes[1].axis("off")

                im = axes[2].imshow(anomaly_maps[i, 0], cmap="jet")
                axes[2].set_title("Predicted Anomaly Map")
                axes[2].axis("off")
                plt.colorbar(im, ax=axes[2])

                plt.tight_layout()
                plt.savefig(save_dir / f"sample_{shown + 1}.png", dpi=150, bbox_inches="tight")
                plt.close()

                shown += 1

        print(f"Visualizations saved to {save_dir}")
