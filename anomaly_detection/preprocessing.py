"""Shared train/evaluation transforms and inference crop geometry.

Images are RGB, resized with antialiased bilinear interpolation, center-cropped,
scaled to [0, 1], then ImageNet-normalized. Masks use torchvision's mask kernel.
DINO patch padding belongs to the model, never this preprocessing stage.
"""

import math

import numpy as np
from PIL import Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF


def make_det_tf(pre_h, pre_w, h, w):
    return _transform(pre_h, pre_w, h, w, augment=False)


def make_train_tf(pre_h, pre_w, h, w):
    return _transform(pre_h, pre_w, h, w, augment=True)


def _transform(pre_h, pre_w, h, w, augment):
    operations = [T.ToImage(), T.Resize((pre_h, pre_w), antialias=True), T.CenterCrop((h, w))]
    if augment:
        operations.extend([T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)])
    return T.Compose(operations + [T.ToDtype(torch.float32, scale=True),
                                  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class InferencePreprocessing:
    """Reusable geometry mixin for the shared and legacy inference engines."""

    def setup_preprocessing(self, image_size, crop_scale):
        if len(image_size) != 2 or any(type(v) is not int or v <= 0 for v in image_size):
            raise ValueError("image_size must contain two positive integers.")
        if not math.isfinite(crop_scale) or not 0 < crop_scale <= 1:
            raise ValueError("crop_scale must be in (0, 1].")
        self.image_size = tuple(image_size)
        self.crop_scale = float(crop_scale)
        self.h, self.w = self.image_size
        self.pre_h, self.pre_w = (math.ceil(v / crop_scale) for v in self.image_size)
        # Match torchvision CenterCrop exactly, including round-to-even ties.
        self.crop_top = int(round((self.pre_h - self.h) / 2.0))
        self.crop_left = int(round((self.pre_w - self.w) / 2.0))
        self.transform = make_det_tf(self.pre_h, self.pre_w, self.h, self.w)

    def preprocess(self, image_path):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            original = np.array(rgb)
            tensor = self.transform(rgb).unsqueeze(0)
        return tensor, original

    def gt_mask_to_crop(self, gt_mask_orig, orig_rgb):
        """Use the same mask resize/crop kernels as MVTecDataset evaluation."""
        if tuple(gt_mask_orig.shape) != tuple(orig_rgb.shape[:2]):
            raise ValueError("Ground-truth mask must match the original image dimensions.")
        mask = tv_tensors.Mask(torch.from_numpy((gt_mask_orig > 0).astype(np.uint8)))
        mask = T.Resize((self.pre_h, self.pre_w), antialias=True)(mask)
        mask = T.CenterCrop(self.image_size)(mask)
        return mask.as_subclass(torch.Tensor).numpy().astype(np.uint8) * 255

    def uncrop_mask_to_original(self, orig_rgb, mask_hw):
        """Restore a binary crop mask for display; excluded borders stay zero."""
        if tuple(mask_hw.shape) != self.image_size:
            raise ValueError("Mask dimensions must match the configured crop.")
        canvas = torch.zeros((self.pre_h, self.pre_w), dtype=torch.uint8)
        canvas[self.crop_top:self.crop_top + self.h, self.crop_left:self.crop_left + self.w] = torch.as_tensor(mask_hw)
        restored = TF.resize(tv_tensors.Mask(canvas), list(orig_rgb.shape[:2]))
        return restored.as_subclass(torch.Tensor).numpy()

    def restore_map(self, anomaly_map, original_size):
        """Nearest-neighbor display projection; NaN marks unobserved borders.

        This is not an inverse of resizing. Compute metrics in crop space.
        Nearest projection avoids blending predictions with unknown borders.
        """
        if tuple(anomaly_map.shape) != self.image_size:
            raise ValueError("Anomaly map dimensions must match the configured crop.")
        canvas = torch.full((1, 1, self.pre_h, self.pre_w), float("nan"))
        canvas[..., self.crop_top:self.crop_top + self.h, self.crop_left:self.crop_left + self.w] = anomaly_map.detach().cpu()
        restored = torch.nn.functional.interpolate(canvas, size=tuple(original_size), mode="nearest")[0, 0]
        return restored, torch.isfinite(restored)

    def geometry(self, original_size):
        return {"original_size": list(original_size), "resize_size": [self.pre_h, self.pre_w],
                "crop_size": list(self.image_size), "crop_top": self.crop_top,
                "crop_left": self.crop_left, "crop_scale": self.crop_scale}
