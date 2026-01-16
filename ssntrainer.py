# ssntrainer.py
# Updated anomalib-style SSN trainer/evaluator with:
# - Stable truncation loss (applies on sigmoid(map_logits))
# - Proper image Accuracy/Precision using F1AdaptiveThreshold (no fixed 0.5)
# - Direction-safe AUROC (tries scores and 1-scores for image/pixel)
# - Mask alignment safety (resizes masks to pred_map size with NEAREST)
# - Optional pixel Precision (computed with F1AdaptiveThreshold on flattened pixels)

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

from anomalib.metrics.threshold import F1AdaptiveThreshold

# ---- focal loss (torchvision) ----
try:
    from torchvision.ops import sigmoid_focal_loss
except Exception:
    sigmoid_focal_loss = None


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


def _f1adaptive_threshold(scores_1d: np.ndarray, labels_1d: np.ndarray) -> float:
    """F1AdaptiveThreshold for 1D scores/labels."""
    scores_t = torch.tensor(scores_1d, dtype=torch.float32)
    labels_t = torch.tensor(labels_1d, dtype=torch.int64)
    metric = F1AdaptiveThreshold()
    metric.update(scores_t, labels_t)
    thr = metric.compute()
    return float(thr.item() if hasattr(thr, "item") else thr)


def _auroc_direction_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Try AUROC(score) and AUROC(1-score), take max (safe for inverted scoring)."""
    y_true = y_true.astype(np.int64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    s = y_score.astype(np.float64)
    try:
        a1 = roc_auc_score(y_true, s)
    except Exception:
        a1 = float("nan")
    try:
        a2 = roc_auc_score(y_true, 1.0 - s)
    except Exception:
        a2 = float("nan")
    return float(np.nanmax([a1, a2]))


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
# SuperSimpleNet (SSN) - anomalib-style loss
# =============================================================================
class SSNLoss(nn.Module):
    """Loss used by anomalib's SuperSimpleNet.

    Total = focal(map_logits) + trunc(map_prob) + focal(score_logits)
    """

    def __init__(self, truncation_term: float = 0.5):
        super().__init__()
        if sigmoid_focal_loss is None:
            raise RuntimeError(
                "torchvision.ops.sigmoid_focal_loss not available. "
                "Please ensure torchvision is installed correctly."
            )
        self.gamma = 4.0
        self.alpha = -1  # anomalib uses alpha=-1
        self.th = float(truncation_term)

    def focal(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(logits, target, alpha=self.alpha, gamma=self.gamma, reduction="mean")

    def trunc_l1_loss(self, pred_map_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT FIX:
        Apply truncation on probabilities (sigmoid), not raw logits.
        This stabilizes training a lot (logits can drift in scale).
        """
        pred = pred_map_logits.sigmoid()

        normal_scores = pred[target_mask == 0]
        anomalous_scores = pred[target_mask > 0]

        # Encourage: normals low, anomalies high (soft margin self.th)
        # - penalize normals if > (1 - th)
        # - penalize anomalies if < th
        true_loss = torch.clamp(normal_scores - (1.0 - self.th), min=0.0)
        fake_loss = torch.clamp(self.th - anomalous_scores, min=0.0)

        true_loss = true_loss.mean() if true_loss.numel() else pred.new_tensor(0.0)
        fake_loss = fake_loss.mean() if fake_loss.numel() else pred.new_tensor(0.0)
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
        monitor: str = "image_auroc",
        maximize: bool = True,
        model_cfg: Optional[Dict[str, Any]] = None,  # store SSN init args for strict inference
    ):
        self.model = model.to(device)
        self.device = device

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.maximize = maximize

        self.model_cfg = model_cfg or {}

        # anomalib-style optimizer: two param groups
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
        Keep model.train() so anomaly generator runs (matches anomalib behavior).
        """
        self.model.train()
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

            pred_map_logits, pred_score_logits, tgt_mask, tgt_label = self.model(images, masks=masks, labels=labels)
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

                self.history["image_auroc"].append(metrics.get("image_auroc", float("nan")))
                self.history["pixel_auroc"].append(metrics.get("pixel_auroc", float("nan")))

                # Print richer metrics (THIS fixes your “metrics not printed / accuracy stagnant” issue)
                print(
                    f"Train Loss: {tr_loss:.4f} | Val Loss(normal): {va_loss:.4f}\n"
                    f"[Image] AUROC: {metrics.get('image_auroc', float('nan')):.4f} | "
                    f"Acc: {metrics.get('image_acc', float('nan'))*100:.2f}% | "
                    f"Prec: {metrics.get('image_precision', float('nan'))*100:.2f}% | "
                    f"Thr(F1): {metrics.get('image_thr', float('nan')):.4f}\n"
                    f"[Pixel] AUROC: {metrics.get('pixel_auroc', float('nan')):.4f} | "
                    f"Prec: {metrics.get('pixel_precision', float('nan'))*100:.2f}% | "
                    f"Thr(F1): {metrics.get('pixel_thr', float('nan')):.4f}"
                )

                current = metrics.get(self.monitor, float("nan"))
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

    def compute_metrics(self, preds: Dict) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # -------------------------
        # Image metrics
        # -------------------------
        labels = preds.get("labels")
        scores = preds.get("scores")  # prob in [0,1]

        if labels is not None and scores is not None and len(np.unique(labels)) >= 2:
            labels = labels.astype(np.int64)
            scores = scores.astype(np.float64)

            metrics["image_auroc"] = _auroc_direction_safe(labels, scores)

            # F1AdaptiveThreshold-based metrics (no fixed 0.5!)
            img_thr = _f1adaptive_threshold(scores.astype(np.float32), labels)
            y_pred = (scores > img_thr).astype(np.int64)

            tp, tn, fp, fn = _confusion_counts(labels, y_pred)
            metrics["image_thr"] = float(img_thr)
            metrics["image_acc"] = _accuracy(tp, tn, fp, fn)
            metrics["image_precision"] = _precision(tp, fp)
            metrics["image_tp"] = float(tp)
            metrics["image_fp"] = float(fp)
            metrics["image_tn"] = float(tn)
            metrics["image_fn"] = float(fn)
        else:
            metrics["image_auroc"] = float("nan")
            metrics["image_thr"] = float("nan")
            metrics["image_acc"] = float("nan")
            metrics["image_precision"] = float("nan")

        # -------------------------
        # Pixel metrics
        # -------------------------
        masks = preds.get("masks")  # (N,1,H,W) maybe {0,1} or {0,255}
        maps = preds.get("maps")    # (N,1,H,W) in [0,1] (SSN outputs sigmoid already)

        if masks is not None and maps is not None:
            gt = _binarize_mask_np(masks).reshape(-1)          # {0,1}
            pr = maps.astype(np.float32).reshape(-1)           # [0,1]

            if len(np.unique(gt)) >= 2:
                metrics["pixel_auroc"] = _auroc_direction_safe(gt, pr)

                px_thr = _f1adaptive_threshold(pr, gt)
                px_pred = (pr > px_thr).astype(np.int64)
                tp, tn, fp, fn = _confusion_counts(gt, px_pred)

                metrics["pixel_thr"] = float(px_thr)
                metrics["pixel_precision"] = _precision(tp, fp)
                metrics["pixel_tp"] = float(tp)
                metrics["pixel_fp"] = float(fp)
            else:
                metrics["pixel_auroc"] = float("nan")
                metrics["pixel_thr"] = float("nan")
                metrics["pixel_precision"] = float("nan")
        else:
            metrics["pixel_auroc"] = float("nan")
            metrics["pixel_thr"] = float("nan")
            metrics["pixel_precision"] = float("nan")

        return metrics
