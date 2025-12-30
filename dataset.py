import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MVTecDataset(Dataset):
    """MVTec AD Dataset for anomaly detection"""
    
    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[A.Compose] = None
    ):
        """
        Args:
            root_dir: Root directory of MVTec dataset
            category: Product category (e.g., 'bottle', 'cable')
            split: 'train' or 'test'
            image_size: Target image size
            transform: Albumentations transform
        """
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.image_size = image_size
        
        # Setup transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        # Load image paths
        self.image_paths = []
        self.labels = []
        self.mask_paths = []
        
        self._load_dataset()
    
    def _get_default_transform(self) -> A.Compose:
        """Get default transforms"""
        if self.split == 'train':
            return A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _load_dataset(self):
        """Load dataset paths and labels"""
        category_dir = self.root_dir / self.category
        
        if self.split == 'train':
            # Training only contains normal images
            train_dir = category_dir / 'train' / 'good'
            if train_dir.exists():
                self.image_paths = sorted(list(train_dir.glob('*.png')))
                self.labels = [0] * len(self.image_paths)  # 0 = normal
                self.mask_paths = [None] * len(self.image_paths)
        
        elif self.split == 'test':
            test_dir = category_dir / 'test'
            
            # Load normal test images
            good_dir = test_dir / 'good'
            if good_dir.exists():
                good_images = sorted(list(good_dir.glob('*.png')))
                self.image_paths.extend(good_images)
                self.labels.extend([0] * len(good_images))
                self.mask_paths.extend([None] * len(good_images))
            
            # Load anomalous images
            mask_dir = category_dir / 'ground_truth'
            for defect_dir in test_dir.iterdir():
                if defect_dir.name == 'good' or not defect_dir.is_dir():
                    continue
                
                defect_images = sorted(list(defect_dir.glob('*.png')))
                self.image_paths.extend(defect_images)
                self.labels.extend([1] * len(defect_images))  # 1 = anomaly
                
                # Load corresponding masks
                for img_path in defect_images:
                    mask_path = mask_dir / defect_dir.name / (img_path.stem + '_mask.png')
                    self.mask_paths.append(mask_path if mask_path.exists() else None)
        
        print(f"Loaded {len(self.image_paths)} images for {self.category} ({self.split})")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """Get dataset item"""
        # Load image
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load mask if available
        mask = None
        if self.mask_paths[idx] is not None:
            mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))
            mask = (mask > 0).astype(np.float32)
        
        # Apply transforms
        if mask is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]

            mask_t = transformed["mask"]  # could be Tensor (ToTensorV2) or ndarray (older pipeline)
            if isinstance(mask_t, torch.Tensor):
                mask = mask_t
            else:
                mask = torch.from_numpy(mask_t)

            # Ensure shape [1, H, W] and float
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()
        else:
            transformed = self.transform(image=image)
            image = transformed['image']
            mask = torch.zeros(1, *self.image_size)
        
        label = self.labels[idx]
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'mask': mask,
            'path': str(img_path)
        }


class MVTecDataModule:
    """Data module for MVTec dataset"""
    
    def __init__(
        self,
        root_dir: str,
        category: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (256, 256)
    ):
        self.root_dir = root_dir
        self.category = category
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        self.train_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets"""
        self.train_dataset = MVTecDataset(
            root_dir=self.root_dir,
            category=self.category,
            split='train',
            image_size=self.image_size
        )
        
        self.test_dataset = MVTecDataset(
            root_dir=self.root_dir,
            category=self.category,
            split='test',
            image_size=self.image_size
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
