"""Separate RGB-only Mars adapter. No downloads, writes, training or lunar access."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1] / 'datasets/mmlsv2'

def read_pair(image_path, mask_path):
    image = tifffile.imread(image_path)
    with Image.open(mask_path) as source:
        mask = np.array(source)
    if image.shape != (128, 128, 7) or image.dtype != np.float32:
        raise ValueError(f'Unexpected image shape/type: {image_path}')
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError(f'Invalid normalized image: {image_path}')
    if mask.shape != image.shape[:2] or mask.dtype != np.uint8:
        raise ValueError(f'Unexpected mask shape/type: {mask_path}')
    if not set(np.unique(mask)) <= {0, 1}:
        raise ValueError(f'Unexpected mask classes: {mask_path}')
    return image, mask

def preprocess(image, mask, size=256):
    # Source bands are RGB already and normalized: no BGR swap or /255.
    rgb = cv2.resize(image[:, :, :3], (size, size), interpolation=cv2.INTER_AREA)
    target = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(target[None].astype(np.float32))))

class MMLSv2RGB(Dataset):
    def __init__(self, split, root=ROOT, image_size=256):
        if split not in {'train', 'val'}:
            raise ValueError('Only Mars train and val are enabled')
        if image_size != 256:
            raise ValueError('This adapter is fixed at 256x256 for the existing U-Net')
        self.image_size = image_size
        root = Path(root)
        images = {p.name: p for p in (root/split/'images').glob('*.tif')}
        masks = {p.name: p for p in (root/split/'masks').glob('*.tif')}
        if images.keys() != masks.keys() or len(images) != {'train':465,'val':66}[split]:
            raise ValueError('Missing/unmatched files or unexpected pair count')
        self.pairs = [(images[name], masks[name]) for name in sorted(images)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return preprocess(*read_pair(*self.pairs[index]), self.image_size)
