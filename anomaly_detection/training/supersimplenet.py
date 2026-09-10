"""SuperSimpleNet training, validation calibration and final evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from anomaly_detection.training.losses import SSNLoss


# =============================================================================
# Helpers
# =============================================================================
def _strip_prefix_if_present(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip common wrappers (model./module.) so inference can load strict=True."""
    for pref in ("model.", "module."):
        if any(k.startswith(pref) for k in state.keys()):
            state = {k[len(pref):] if k.startswith(pref) else k: v for k, v in state.items()}
    return state


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def _accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    return float((tp + tn) / (tp + tn + fp + fn + 1e-12))


def _precision(tp: int, fp: int) -> float:
    return float(tp / (tp + fp + 1e-12))


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC with the fixed contract that larger scores are more anomalous."""
    y_true = y_true.astype(np.int64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score.astype(np.float64)))
    except Exception:
        return float("nan")


def _binarize_mask_np(mask: np.ndarray) -> np.ndarray:
    """Accept {0,1} or {0,255} or float; return {0,1} int64."""
    if mask.dtype != np.float32 and mask.dtype != np.float64:
        m = mask.astype(np.float32)
    else:
        m = mask
    if m.max() > 1.0:
        m = (m > 0.0).astype(np.int64)
    else:
        m = (m > 0.5).astype(np.int64)
    return m


# =============================================================================
# Trainer
# =============================================================================
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
        monitor: str = "val_loss",
        maximize: bool = False,
        model_cfg: Optional[Dict[str, Any]] = None,  # store SSN init args for strict inference
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        head_lr_multiplier: float = 2.0,
        adaptor_weight_decay: float = 0.01,
        loss_fn: Optional[nn.Module] = None,
        experiment_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.model = model.to(device)
        self.device = device

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.maximize = maximize

        self.model_cfg = model_cfg or {}
        self.experiment_cfg = experiment_cfg or {}

        # anomalib-style optimizer: two param groups
        adaptor_params = list(getattr(self.model, "adaptor").parameters())
        segdec_params = list(getattr(self.model, "segdec").parameters())
        self.optimizer = AdamW(
            [
                {"params": adaptor_params, "lr": learning_rate, "weight_decay": adaptor_weight_decay},
                {"params": segdec_params, "lr": learning_rate * head_lr_multiplier, "weight_decay": weight_decay},
            ]
        )

        self.loss_fn = (loss_fn if loss_fn is not None else SSNLoss()).to(device)

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

            pred_map_logits, pred_score_logits, tgt_mask, tgt_label = self.model(
                images,
                masks=masks,
                labels=labels,
                generate_synthetic=True,
            )
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
        Synthetic generation is requested explicitly while the model remains in
        evaluation mode, preventing BatchNorm state updates from validation data.
        """
        self.model.eval()
        total = 0.0
        n = 0

        for batch in tqdm(dataloader, desc="ValidationLoss(SSN,NormalOnly)"):
            images = batch["image"].to(self.device)
            labels = batch.get("label")
            masks = batch.get("mask")

            if labels is not None:
                labels = labels.to(self.device).view(-1)
                keep = (labels == 0)
                if not keep.any():
                    continue
                images = images[keep]
                masks = masks.to(self.device)[keep] if masks is not None else None
                labels = labels[keep]
            else:
                masks = masks.to(self.device) if masks is not None else None

            pred_map_logits, pred_score_logits, tgt_mask, tgt_label = self.model(
                images,
                masks=masks,
                labels=labels,
                generate_synthetic=True,
            )
            loss = self.loss_fn(pred_map_logits, pred_score_logits, tgt_mask, tgt_label)

            total += float(loss.item())
            n += 1

        return total / max(1, n)

    def save_checkpoint(self, filename: str, model_only: bool = False):
        """Save checkpoint. model_only=True saves just state_dict."""
        if model_only:
            torch.save(_strip_prefix_if_present(self.model.state_dict()), self.save_dir / filename)
            return

        ckpt = {
            "model_state_dict": _strip_prefix_if_present(self.model.state_dict()),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "model_cfg": self.model_cfg,
            "experiment_cfg": self.experiment_cfg,
        }
        torch.save(ckpt, self.save_dir / filename)

    def load_checkpoint(self, filename: str, strict: bool = True):
        ckpt = torch.load(self.save_dir / filename, map_location=self.device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = _strip_prefix_if_present(ckpt["model_state_dict"])
            self.model.load_state_dict(state, strict=strict)

            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "history" in ckpt:
                self.history = ckpt["history"]
            if "model_cfg" in ckpt:
                self.model_cfg = ckpt["model_cfg"]
        else:
            state = _strip_prefix_if_present(ckpt)
            self.model.load_state_dict(state, strict=strict)

    def fit(self, train_loader, val_loader, num_epochs: int = 100, patience: int = 10, eval_every: int = 1):
        scheduler = MultiStepLR(
            self.optimizer,
            # Short YAML smoke runs must start at the requested learning rate;
            # milestone zero would decay it before the first optimizer step.
            milestones=sorted({int(num_epochs * fraction) for fraction in (0.8, 0.9)
                               if 0 < int(num_epochs * fraction) < num_epochs}),
            gamma=0.4,
        )
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
                print(f"Train Loss: {tr_loss:.4f} | Val Loss(normal): {va_loss:.4f}")
                current = va_loss
                improved = (current > best) if self.maximize else (current < best)

                if np.isfinite(current) and improved:
                    best = current
                    self.history["best_metric"] = best
                    self.history["best_epoch"] = epoch + 1

                    self.save_checkpoint("best_model.pth", model_only=False)
                    self.save_checkpoint("best_model_state_dict.pth", model_only=True)

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


# =============================================================================
# Evaluator
# =============================================================================
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

            pred_map, pred_score = self.model(images)  # (B,1,H,W), (B,) or (B,1)

            # score -> (B,)
            score_1d = pred_score.detach().reshape(-1)
            all_scores.append(score_1d.cpu())

            # map -> (B,1,H,W)
            pred_map = pred_map.detach()
            all_maps.append(pred_map.cpu())

            if labels is not None:
                all_labels.append(labels.detach().cpu().reshape(-1))

            if masks is not None:
                # Ensure mask matches pred_map spatial size
                m = masks.detach()
                if m.dim() == 3:
                    m = m.unsqueeze(1)
                if m.shape[-2:] != pred_map.shape[-2:]:
                    m = F.interpolate(m.float(), size=pred_map.shape[-2:], mode="nearest")
                all_masks.append(m.cpu())

        out = {
            "scores": torch.cat(all_scores, dim=0).numpy(),
            "maps": torch.cat(all_maps, dim=0).numpy(),  # (N,1,H,W)
        }
        if all_labels:
            out["labels"] = torch.cat(all_labels, dim=0).numpy().reshape(-1)
        if all_masks:
            out["masks"] = torch.cat(all_masks, dim=0).numpy()  # (N,1,H,W)
        return out

    def calibrate(
        self,
        preds: Dict,
        image_quantile: float = 0.99,
        pixel_quantile: float = 0.999,
    ) -> Dict[str, float]:
        """Fit fixed deployment thresholds on normal validation data."""
        labels = preds.get("labels")
        if labels is None:
            raise ValueError("Calibration requires validation labels.")
        normal = labels.reshape(-1) == 0
        if not normal.any():
            raise ValueError("Calibration requires at least one normal validation image.")
        return {
            "image_threshold": float(np.quantile(preds["scores"][normal], image_quantile)),
            "pixel_threshold": float(np.quantile(preds["maps"][normal], pixel_quantile)),
            "image_quantile": float(image_quantile),
            "pixel_quantile": float(pixel_quantile),
        }

    def compute_metrics(
        self,
        preds: Dict,
        calibration: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        labels = preds.get("labels")
        scores = preds.get("scores")

        if labels is not None and scores is not None and len(np.unique(labels)) >= 2:
            labels = labels.astype(np.int64)
            scores = scores.astype(np.float64)
            metrics["image_auroc"] = _auroc(labels, scores)
            if calibration is not None:
                img_thr = float(calibration["image_threshold"])
                y_pred = (scores >= img_thr).astype(np.int64)
                tp, tn, fp, fn = _confusion_counts(labels, y_pred)
                metrics.update(
                    {
                        "image_threshold": img_thr,
                        "image_accuracy": _accuracy(tp, tn, fp, fn),
                        "image_precision": _precision(tp, fp),
                        "image_tp": float(tp),
                        "image_fp": float(fp),
                        "image_tn": float(tn),
                        "image_fn": float(fn),
                    }
                )
        else:
            metrics["image_auroc"] = float("nan")

        masks = preds.get("masks")
        maps = preds.get("maps")
        if masks is not None and maps is not None:
            gt = _binarize_mask_np(masks).reshape(-1)
            pr = maps.astype(np.float32).reshape(-1)
            metrics["pixel_auroc"] = _auroc(gt, pr) if len(np.unique(gt)) >= 2 else float("nan")
            if calibration is not None:
                px_thr = float(calibration["pixel_threshold"])
                px_pred = (pr >= px_thr).astype(np.int64)
                tp, _, fp, _ = _confusion_counts(gt, px_pred)
                metrics["pixel_threshold"] = px_thr
                metrics["pixel_precision"] = _precision(tp, fp)
        else:
            metrics["pixel_auroc"] = float("nan")

        return metrics
