# trainer.py
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

# ---- FIX: import focal loss properly ----
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
        self.alpha = -1  # anomalib uses alpha=-1
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
        model_cfg: Optional[Dict[str, Any]] = None,   # ✅ store SSN init args for strict inference
    ):
        self.model = model.to(device)
        self.device = device

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.maximize = maximize

        # store model init args used in training (so inference can rebuild exactly)
        self.model_cfg = model_cfg or {}

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
        Keep model.train() so anomaly generator runs (same behavior as anomalib).
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
        """Save checkpoint.

        model_only=True saves a smaller file with just the state_dict (useful for deployment).
        """
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
        """Load checkpoint (strict by default)."""
        ckpt = torch.load(self.save_dir / filename, map_location=self.device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            state = _strip_prefix_if_present(state)
            self.model.load_state_dict(state, strict=strict)

            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "history" in ckpt:
                self.history = ckpt["history"]
            if "model_cfg" in ckpt:
                self.model_cfg = ckpt["model_cfg"]
        else:
            # allow loading from pure state_dict files
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

                    # ✅ save full checkpoint + also model-only export
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

            all_scores.append(pred_score.detach().cpu().reshape(-1))
            all_maps.append(pred_map.detach().cpu())

            if labels is not None:
                all_labels.append(labels.detach().cpu().reshape(-1))
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
