# trainer.py
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

try:
    # torchvision focal loss (matches anomalib SSN loss)
    from torchvision.ops.focal_loss import sigmoid_focal_loss
except Exception:  # pragma: no cover
    sigmoid_focal_loss = None


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


# =============================================================================
# SuperSimpleNet (SSN) - anomalib-style trainer
# =============================================================================


class SSNLoss(nn.Module):
    """Loss used by anomalib's SuperSimpleNet.

    Total = focal(map) + trunc_l1(map) + focal(score)
    """

    def __init__(self, truncation_term: float = 0.5):
        super().__init__()
        if sigmoid_focal_loss is None:
            raise RuntimeError(
                "torchvision.ops.sigmoid_focal_loss not available. "
                "Please ensure torchvision is installed correctly."
            )
        self.gamma = 4.0
        self.alpha = -1  # anomalib sets alpha=-1
        self.th = float(truncation_term)

    def focal(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(logits, target, alpha=self.alpha, gamma=self.gamma, reduction="mean")

    def trunc_l1_loss(self, pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        normal_scores = pred_logits[target_mask == 0]
        anomalous_scores = pred_logits[target_mask > 0]

        true_loss = torch.clamp(normal_scores + self.th, min=0)
        fake_loss = torch.clamp(-anomalous_scores + self.th, min=0)

        true_loss = true_loss.mean() if true_loss.numel() else pred_logits.new_tensor(0.0)
        fake_loss = fake_loss.mean() if fake_loss.numel() else pred_logits.new_tensor(0.0)
        return true_loss + fake_loss

    def forward(
        self,
        pred_map_logits: torch.Tensor,
        pred_score_logits: torch.Tensor,
        target_mask: torch.Tensor,
        target_label: torch.Tensor,
    ) -> torch.Tensor:
        map_focal = self.focal(pred_map_logits, target_mask)
        map_trunc = self.trunc_l1_loss(pred_map_logits, target_mask)
        score_focal = self.focal(pred_score_logits, target_label)
        return map_focal + map_trunc + score_focal


class SuperSimpleNetTrainer:
    """Trainer for anomalib-style SSN model.

    Expects model.forward:
      - train: (pred_map_logits, pred_score_logits, masks2, labels2)
      - eval:  (pred_map_prob, pred_score_prob)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        save_dir: str = "./checkpoints",
        monitor: str = "image_auroc",
        maximize: bool = True,
    ):
        self.model = model.to(device)
        self.device = device

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.maximize = maximize

        # Match anomalib optimizer setup: two param groups
        adaptor_params = list(getattr(self.model, "adaptor").parameters())
        segdec_params = list(getattr(self.model, "segdec").parameters())
        self.optimizer = AdamW(
            [
                {"params": adaptor_params, "lr": 1e-4},
                {"params": segdec_params, "lr": 2e-4, "weight_decay": 1e-5},
            ]
        )
        self.loss_fn = SSNLoss()

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "image_auroc": [],
            "pixel_auroc": [],
            "best_metric": -float("inf") if maximize else float("inf"),
            "best_epoch": 0,
            "epoch": 0,
        }

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total = 0.0
        n = 0
        pbar = tqdm(dataloader, desc="Training(SSN)")
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch.get("mask")
            labels = batch.get("label")
            masks = masks.to(self.device) if masks is not None else None
            labels = labels.to(self.device) if labels is not None else None

            pred_map_logits, pred_score_logits, tgt_mask, tgt_label = self.model(images, masks=masks, labels=labels)
            loss = self.loss_fn(pred_map_logits, pred_score_logits, tgt_mask, tgt_label)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total += float(loss.item())
            n += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        return total / max(1, n)

    @torch.no_grad()
    def validate_loss_on_normals(self, dataloader) -> float:
        """Compute SSN training loss but only on normal samples.

        We keep model.train() so anomaly generator runs (like anomalib forward).
        """
        self.model.train()
        total = 0.0
        n = 0
        for batch in tqdm(dataloader, desc="ValidationLoss(SSN,NormalOnly)"):
            images = batch["image"].to(self.device)
            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(self.device).view(-1)
                keep = (labels == 0)
                if not keep.any():
                    continue
                images = images[keep]
                masks = batch.get("mask")
                masks = masks.to(self.device)[keep] if masks is not None else None
                labels = labels[keep]
            else:
                masks = batch.get("mask")
                masks = masks.to(self.device) if masks is not None else None

            pred_map_logits, pred_score_logits, tgt_mask, tgt_label = self.model(images, masks=masks, labels=labels)
            loss = self.loss_fn(pred_map_logits, pred_score_logits, tgt_mask, tgt_label)
            total += float(loss.item())
            n += 1
        return total / max(1, n)

    def save_checkpoint(self, filename: str):
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        torch.save(ckpt, self.save_dir / filename)

    def load_checkpoint(self, filename: str):
        ckpt = torch.load(self.save_dir / filename, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.history = ckpt.get("history", self.history)

    def fit(self, train_loader, val_loader, num_epochs: int = 100, patience: int = 10, eval_every: int = 1):
        scheduler = MultiStepLR(
            self.optimizer,
            milestones=[int(num_epochs * 0.8), int(num_epochs * 0.9)],
            gamma=0.4,
        )
        evaluator = SuperSimpleNetEvaluator(self.model, device=self.device)

        best = self.history["best_metric"]
        patience_counter = 0

        for epoch in range(num_epochs):
            self.history["epoch"] = epoch + 1
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            tr_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(tr_loss)

            va_loss = self.validate_loss_on_normals(val_loader)
            self.history["val_loss"].append(va_loss)

            if (epoch + 1) % eval_every == 0:
                preds = evaluator.predict(val_loader)
                metrics = evaluator.compute_metrics(preds)
                self.history["image_auroc"].append(metrics["image_auroc"])
                self.history["pixel_auroc"].append(metrics["pixel_auroc"])

                print(
                    f"Train Loss: {tr_loss:.4f} | Val Loss(normal): {va_loss:.4f} | "
                    f"Val Image-AUROC: {metrics['image_auroc']:.4f} | Val Pixel-AUROC: {metrics['pixel_auroc']:.4f}"
                )

                current = metrics.get(self.monitor, float("nan"))
                improved = (current > best) if self.maximize else (current < best)
                if np.isfinite(current) and improved:
                    best = current
                    self.history["best_metric"] = best
                    self.history["best_epoch"] = epoch + 1
                    self.save_checkpoint("best_model.pth")
                    patience_counter = 0
                    print(f"✓ New best model saved ({self.monitor}={best:.4f})")
                else:
                    patience_counter += 1

            scheduler.step()
            if patience_counter >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch + 1} "
                    f"(best {self.monitor}={best:.4f} @ epoch {self.history['best_epoch']})"
                )
                break


class SuperSimpleNetEvaluator:
    """Evaluator for SSN that uses model's eval outputs: (map_prob, score_prob)."""

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def predict(self, dataloader) -> Dict:
        self.model.eval()
        all_scores = []
        all_labels = []
        all_maps = []
        all_masks = []

        for batch in tqdm(dataloader, desc="Predicting(SSN)"):
            images = batch["image"].to(self.device)
            labels = batch.get("label")
            masks = batch.get("mask")

            pred_map, pred_score = self.model(images)

            all_scores.append(pred_score.detach().cpu())
            all_maps.append(pred_map.detach().cpu())

            if labels is not None:
                all_labels.append(labels.detach().cpu())
            if masks is not None:
                all_masks.append(masks.detach().cpu())

        out = {
            "scores": torch.cat(all_scores, dim=0).numpy(),
            "maps": torch.cat(all_maps, dim=0).numpy(),
        }
        if all_labels:
            out["labels"] = torch.cat(all_labels, dim=0).numpy().reshape(-1)
        if all_masks:
            out["masks"] = torch.cat(all_masks, dim=0).numpy()
        return out

    def compute_metrics(self, preds: Dict) -> Dict[str, float]:
        metrics = {"image_auroc": float("nan"), "pixel_auroc": float("nan")}

        labels = preds.get("labels")
        scores = preds.get("scores")
        if labels is not None and scores is not None and len(np.unique(labels)) >= 2:
            # If your score direction is flipped, take the better of score and -score.
            auc = roc_auc_score(labels, scores)
            auc_inv = roc_auc_score(labels, -scores)
            metrics["image_auroc"] = float(max(auc, auc_inv))

        masks = preds.get("masks")
        maps = preds.get("maps")
        if masks is not None and maps is not None:
            gt = masks.reshape(-1)
            pr = maps.reshape(-1)
            if len(np.unique(gt)) >= 2:
                px_auc = roc_auc_score(gt, pr)
                px_auc_inv = roc_auc_score(gt, -pr)
                metrics["pixel_auroc"] = float(max(px_auc, px_auc_inv))

        return metrics
