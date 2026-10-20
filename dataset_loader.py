import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def get_dataset_root() -> Path:
    """
    Resolve the dataset root directory.
    Checks in order:
    1. LUNAR_DATASET_ROOT environment variable (if set)
    2. Local project-relative './dataset' directory
    3. User Downloads directory fallback for existing local development
    """
    env_path = os.environ.get("LUNAR_DATASET_ROOT")
    if env_path:
        return Path(env_path).resolve()

    # Project-relative dataset location
    local_rel = Path(__file__).resolve().parent / "dataset"
    if local_rel.is_dir():
        return local_rel

    # Fallback to local user Downloads directory if present
    downloads_fallback = (
        Path.home()
        / "Downloads"
        / "moon_ohrc_detection.v2i.coco-segmentation"
    )
    if downloads_fallback.is_dir():
        return downloads_fallback

    return local_rel


DATASET_ROOT = get_dataset_root()
MASK_ROOT = Path(__file__).resolve().parent / "dataset_masks"


class LunarLandslideDataset(Dataset):
    def __init__(self, split, image_size=256, dataset_root=None, mask_root=None):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid dataset split: {split}")

        self.dataset_root = Path(dataset_root) if dataset_root else get_dataset_root()
        self.image_dir = self.dataset_root / split
        self.mask_dir = (Path(mask_root) if mask_root else MASK_ROOT) / split
        self.image_size = image_size

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset split directory not found: '{self.image_dir}'.\n"
                "Please set the LUNAR_DATASET_ROOT environment variable to the path of your dataset,\n"
                "e.g. on Windows (PowerShell):\n"
                "  $env:LUNAR_DATASET_ROOT = 'C:\\path\\to\\moon_ohrc_detection.v2i.coco-segmentation'\n"
                "or place the dataset files in a './dataset' folder in the project root."
            )

        self.image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        self.pairs = []

        for image_path in self.image_paths:
            mask_name = f"{image_path.stem}_mask.png"
            mask_path = self.mask_dir / mask_name

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing mask for {image_path.name}: {mask_path}"
                )

            self.pairs.append((image_path, mask_path))

        if not self.pairs:
            raise RuntimeError(f"No image-mask pairs found for {split}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image-mask size mismatch: {image_path.name}, "
                f"{image.shape[:2]} vs {mask.shape[:2]}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask[None, :, :])
        )

        return image_tensor, mask_tensor