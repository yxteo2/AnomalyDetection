import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List

from torchvision.transforms import v2 as T
from torchvision import tv_tensors


def build_torchvision_transform(image_size: Tuple[int, int], split: str):
    return T.Compose([
        T.Resize(image_size, antialias=True),
        T.ToDtype(torch.float32, scale=True),  # Image -> float32 in [0,1]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class MVTecDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.image_size = image_size
        self.transform = transform if transform is not None else build_torchvision_transform(image_size, split)

        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.mask_paths: List[Optional[Path]] = []

        self._load_dataset()

    def _load_dataset(self):
        category_dir = self.root_dir / self.category

        if self.split == "train":
            train_dir = category_dir / "train" / "good"
            if train_dir.exists():
                self.image_paths = sorted([p for ext in ("*.png", "*.jpg", "*.jpeg") for p in train_dir.glob(ext)])
                self.labels = [0] * len(self.image_paths)
                self.mask_paths = [None] * len(self.image_paths)

        elif self.split == "test":
            test_dir = category_dir / "test"

            # good
            good_dir = test_dir / "good"
            if good_dir.exists():
                good_images = sorted([p for ext in ("*.png", "*.jpg", "*.jpeg") for p in good_dir.glob(ext)])
                self.image_paths.extend(good_images)
                self.labels.extend([0] * len(good_images))
                self.mask_paths.extend([None] * len(good_images))

            # defects
            mask_dir = category_dir / "ground_truth"
            IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]

            for defect_dir in test_dir.iterdir():
                if defect_dir.name == "good" or not defect_dir.is_dir():
                    continue

                defect_images = sorted([p for ext in ("*.png", "*.jpg", "*.jpeg") for p in defect_dir.glob(ext)])

                for img_path in defect_images:
                    self.image_paths.append(img_path)
                    self.labels.append(1)

                    # find mask (always append something to keep alignment!)
                    mask_path = None

                    for ext in IMG_EXTS:
                        p = mask_dir / defect_dir.name / f"{img_path.stem}_mask{ext}"
                        if p.exists():
                            mask_path = p
                            break

                    if mask_path is None:
                        for ext in IMG_EXTS:
                            p = mask_dir / defect_dir.name / f"{img_path.stem}{ext}"
                            if p.exists():
                                mask_path = p
                                break

                    self.mask_paths.append(mask_path)

        print(f"Loaded {len(self.image_paths)} images for {self.category} ({self.split})")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        image_pil = Image.open(img_path).convert("RGB")

        # wrap as tv_tensors.Image so v2 transforms behave nicely
        image = tv_tensors.Image(image_pil)

        mask_path = self.mask_paths[idx]
        if mask_path is not None and mask_path.exists():
            mask_pil = Image.open(mask_path).convert("L")
            mask_np = (np.array(mask_pil) > 0).astype(np.uint8)  # 0/1
            mask = tv_tensors.Mask(torch.from_numpy(mask_np))     # [H,W]
        else:
            # create zero mask at original size (will be resized by transform)
            w, h = image_pil.size
            mask = tv_tensors.Mask(torch.zeros((h, w), dtype=torch.uint8))

        # IMPORTANT: torchvision v2 takes (image, mask) and returns (image, mask)
        image, mask = self.transform(image, mask)

        # ensure mask shape [1,H,W] float
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = mask.float()

        label = self.labels[idx]

        return {
            "image": image,  # [3,H,W] float normalized
            "label": torch.tensor(label, dtype=torch.long),
            "mask": mask,    # [1,H,W] float 0/1
            "path": str(img_path),
        }

class MVTecDataModule:
    def __init__(self, root_dir, category, batch_size=32, num_workers=4,
                 image_size=(256,256), train_transform=None, test_transform=None):
        self.root_dir = root_dir
        self.category = category
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.train_dataset = None
        self.test_dataset = None

    def setup(self):
        self.train_dataset = MVTecDataset(
            root_dir=self.root_dir,
            category=self.category,
            split="train",
            image_size=self.image_size,
            transform=self.train_transform,
        )
        self.test_dataset = MVTecDataset(
            root_dir=self.root_dir,
            category=self.category,
            split="test",
            image_size=self.image_size,
            transform=self.test_transform,
        )
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    @staticmethod
    def get_available_categories(root_dir: str) -> List[str]:
        """Get list of available categories in MVTec dataset"""
        root = Path(root_dir)
        categories = [d.name for d in root.iterdir() if d.is_dir()]
        return sorted(categories)
