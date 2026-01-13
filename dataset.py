import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

IMG_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")


# ---------------------------------
# Transforms
# ---------------------------------
def build_torchvision_transform(image_size: Tuple[int, int], split: str):
    """Default transform for both MVTec and VisA.

    Uses torchvision v2 so it can accept (image, mask) and resize both together.
    - Image is converted to float [0,1] then normalized (ImageNet stats).
    - Mask stays as a Mask tensor (Normalize will not touch it).
    """
    return T.Compose(
        [
            T.Resize(image_size, antialias=True),
            T.ToDtype(torch.float32, scale=True),  # image -> float32 [0,1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ---------------------------------
# Helpers
# ---------------------------------
def _list_images_recursive(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    for pat in IMG_GLOBS:
        out.extend(root.rglob(pat))
    return sorted(out)


def _is_mvtec_category_dir(category_dir: Path) -> bool:
    return (category_dir / "train").is_dir() and (category_dir / "test").is_dir()


def _is_visa_original_root(root_dir: Path) -> bool:
    if not (root_dir / "split_csv").is_dir():
        return False
    for d in root_dir.iterdir():
        if d.is_dir() and d.name != "split_csv":
            if (d / "Data" / "Images").is_dir():
                return True
    return False


def _parse_visa_split_csv(split_csv_path: Path) -> List[Dict[str, str]]:
    if not split_csv_path.exists():
        return []

    rows: List[Dict[str, str]] = []
    with split_csv_path.open("r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = any(h in sample.lower() for h in ["object", "split", "label", "image"])
        if has_header:
            reader = csv.DictReader(f)
            for r in reader:
                if not r:
                    continue
                rows.append({(k or "").strip(): (v.strip() if isinstance(v, str) else "") for k, v in r.items()})
        else:
            reader = csv.reader(f)
            for r in reader:
                if not r:
                    continue
                rows.append(
                    {
                        "object": (r[0].strip() if len(r) > 0 else ""),
                        "split": (r[1].strip() if len(r) > 1 else ""),
                        "label": (r[2].strip() if len(r) > 2 else ""),
                        "image": (r[3].strip() if len(r) > 3 else ""),
                        "mask": (r[4].strip() if len(r) > 4 else ""),
                    }
                )
    return rows


def _visa_label_is_anomaly(label: str) -> bool:
    lab = (label or "").strip().lower()
    return lab in {"anomaly", "anomalous", "defect", "bad", "1", "true", "yes"}


def _guess_visa_mask_from_image(img_path: Path) -> Optional[Path]:
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "Images":
            parts[i] = "Masks"
            break
    else:
        return None

    cand = Path(*parts).with_suffix(".png")
    if cand.exists():
        return cand
    cand2 = Path(*parts)
    if cand2.exists():
        return cand2
    return None


def _load_visa_original(root_dir: Path, category: str, split: str) -> Tuple[List[Path], List[int], List[Optional[Path]]]:
    split = split.lower()
    cat_dir = root_dir / category

    image_paths: List[Path] = []
    labels: List[int] = []
    mask_paths: List[Optional[Path]] = []

    split_csv = root_dir / "split_csv" / "1cls.csv"
    rows = _parse_visa_split_csv(split_csv)

    # Strategy A: use split_csv
    if rows:
        for r in rows:
            obj = (r.get("object") or r.get("obj") or "").strip()
            if obj.lower() != category.lower():
                continue

            sp = (r.get("split") or "").strip().lower()
            if split not in sp:
                continue

            is_anom = _visa_label_is_anomaly(r.get("label", ""))

            if split == "train" and is_anom:
                continue

            img_rel = (r.get("image") or r.get("img") or r.get("image_path") or "").strip()
            if not img_rel:
                continue

            img_path = (root_dir / img_rel)
            if not img_path.exists():
                img_path = (cat_dir / img_rel)
            if not img_path.exists():
                img_path = (cat_dir / "Data" / img_rel)
            if not img_path.exists():
                continue

            image_paths.append(img_path)
            labels.append(1 if is_anom else 0)

            mpath: Optional[Path] = None
            if is_anom:
                mask_rel = (r.get("mask") or r.get("mask_path") or "").strip()
                if mask_rel:
                    cand = (root_dir / mask_rel)
                    if not cand.exists():
                        cand = (cat_dir / mask_rel)
                    if cand.exists():
                        mpath = cand
                if mpath is None:
                    mpath = _guess_visa_mask_from_image(img_path)
            mask_paths.append(mpath)

        return image_paths, labels, mask_paths

    # Strategy B: directory-based fallback
    base = cat_dir / "Data"
    normal_dir = base / "Images" / "Normal"
    anomaly_dir = base / "Images" / "Anomaly"

    if split == "train":
        imgs = _list_images_recursive(normal_dir)
        return imgs, [0] * len(imgs), [None] * len(imgs)

    good_imgs = _list_images_recursive(normal_dir)
    image_paths.extend(good_imgs)
    labels.extend([0] * len(good_imgs))
    mask_paths.extend([None] * len(good_imgs))

    anom_imgs = _list_images_recursive(anomaly_dir)
    for p in anom_imgs:
        image_paths.append(p)
        labels.append(1)
        mask_paths.append(_guess_visa_mask_from_image(p))

    return image_paths, labels, mask_paths


def _find_mvtec_mask(mask_dir: Path, defect_name: str, img_stem: str) -> Optional[Path]:
    cand = mask_dir / defect_name / f"{img_stem}.png"
    if cand.exists():
        return cand

    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
    for ext in exts:
        cand2 = mask_dir / defect_name / f"{img_stem}_mask{ext}"
        if cand2.exists():
            return cand2
    for ext in exts:
        cand3 = mask_dir / defect_name / f"{img_stem}{ext}"
        if cand3.exists():
            return cand3

    return None


class MVTecDataset(Dataset):
    """Unified dataset class that supports BOTH MVTec AD and VisA (original) formats.

    Name kept for compatibility with your existing train.py.
    """

    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        transform=None,
        dataset_type: str = "auto",  # auto|mvtec|visa
    ):
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split.lower()
        self.image_size = image_size
        self.transform = transform if transform is not None else build_torchvision_transform(image_size, split)
        self.dataset_type = dataset_type.lower()

        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.mask_paths: List[Optional[Path]] = []

        self._load_dataset()

    def _detect_type(self) -> str:
        if self.dataset_type in {"mvtec", "visa"}:
            return self.dataset_type

        cat_dir = self.root_dir / self.category
        if _is_mvtec_category_dir(cat_dir):
            return "mvtec"
        if _is_visa_original_root(self.root_dir) and (cat_dir / "Data" / "Images").is_dir():
            return "visa"
        # also accept VisA already converted to MVTec
        if (cat_dir / "train").is_dir() and (cat_dir / "test").is_dir():
            return "mvtec"

        raise ValueError(
            "Could not detect dataset format. Expected either:\n"
            "  MVTec: <root>/<category>/{train,test,ground_truth}\n"
            "  VisA (original): <root>/split_csv/1cls.csv and <root>/<category>/Data/Images\n"
            f"Got root={self.root_dir}, category={self.category}"
        )

    def _load_dataset(self):
        dtype = self._detect_type()

        if dtype == "visa":
            imgs, labs, masks = _load_visa_original(self.root_dir, self.category, self.split)
            self.image_paths = imgs
            self.labels = labs
            self.mask_paths = masks
            print(f"[Dataset] VisA(original) loaded {len(self.image_paths)} images for {self.category} ({self.split})")
            return

        category_dir = self.root_dir / self.category

        if self.split == "train":
            train_dir = category_dir / "train" / "good"
            self.image_paths = sorted([p for pat in IMG_GLOBS for p in train_dir.glob(pat)])
            self.labels = [0] * len(self.image_paths)
            self.mask_paths = [None] * len(self.image_paths)

        elif self.split == "test":
            test_dir = category_dir / "test"
            mask_dir = category_dir / "ground_truth"

            good_dir = test_dir / "good"
            good_images = sorted([p for pat in IMG_GLOBS for p in good_dir.glob(pat)])
            self.image_paths.extend(good_images)
            self.labels.extend([0] * len(good_images))
            self.mask_paths.extend([None] * len(good_images))

            if test_dir.exists():
                for defect_dir in sorted([d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]):
                    defect_images = sorted([p for pat in IMG_GLOBS for p in defect_dir.glob(pat)])
                    for img_path in defect_images:
                        self.image_paths.append(img_path)
                        self.labels.append(1)
                        self.mask_paths.append(_find_mvtec_mask(mask_dir, defect_dir.name, img_path.stem))
        else:
            raise ValueError(f"split must be 'train' or 'test', got {self.split}")

        print(f"[Dataset] MVTec-style loaded {len(self.image_paths)} images for {self.category} ({self.split})")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        image_pil = Image.open(img_path).convert("RGB")
        image = tv_tensors.Image(image_pil)

        mask_path = self.mask_paths[idx]
        if mask_path is not None and mask_path.exists():
            mask_pil = Image.open(mask_path).convert("L")
            mask_np = (np.array(mask_pil) > 0).astype(np.uint8)
            mask = tv_tensors.Mask(torch.from_numpy(mask_np))
        else:
            w, h = image_pil.size
            mask = tv_tensors.Mask(torch.zeros((h, w), dtype=torch.uint8))

        image, mask = self.transform(image, mask)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = mask.float()

        label = int(self.labels[idx])

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "mask": mask,
            "path": str(img_path),
        }


class MVTecDataModule:
    """DataModule wrapper. Name kept for compatibility with your train.py."""

    def __init__(
        self,
        root_dir: str,
        category: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (256, 256),
        train_transform=None,
        test_transform=None,
        dataset_type: str = "auto",
    ):
        self.root_dir = root_dir
        self.category = category
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.dataset_type = dataset_type

        self.train_dataset: Optional[MVTecDataset] = None
        self.test_dataset: Optional[MVTecDataset] = None

    def setup(self):
        self.train_dataset = MVTecDataset(
            root_dir=self.root_dir,
            category=self.category,
            split="train",
            image_size=self.image_size,
            transform=self.train_transform,
            dataset_type=self.dataset_type,
        )
        self.test_dataset = MVTecDataset(
            root_dir=self.root_dir,
            category=self.category,
            split="test",
            image_size=self.image_size,
            transform=self.test_transform,
            dataset_type=self.dataset_type,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @staticmethod
    def get_available_categories(root_dir: str) -> List[str]:
        root = Path(root_dir)

        # VisA original: categories are dirs with Data/Images, excluding split_csv
        if (root / "split_csv").is_dir():
            cats: List[str] = []
            for d in root.iterdir():
                if d.is_dir() and d.name != "split_csv" and (d / "Data" / "Images").is_dir():
                    cats.append(d.name)
            if cats:
                return sorted(cats)

        # MVTec-style
        cats = [d.name for d in root.iterdir() if d.is_dir() and d.name != "split_csv"]
        return sorted(cats)
