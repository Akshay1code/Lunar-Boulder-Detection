import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

DATASET_PATH = (
    Path.home()
    / "Downloads"
    / "moon_ohrc_detection.v2i.coco-segmentation"
)

MASK_PATH = Path("dataset_masks")

# Check the training split
split = "train"

annotation_file = (
    DATASET_PATH / split / "_annotations.coco.json"
)

with open(annotation_file, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = {
    category["id"]: category["name"]
    for category in data["categories"]
}

landslide_ids = {
    category_id
    for category_id, name in categories.items()
    if name.lower() == "landslide"
}

# Find images containing landslide annotations
landslide_image_ids = {
    annotation["image_id"]
    for annotation in data["annotations"]
    if annotation["category_id"] in landslide_ids
}

image_info = next(
    image
    for image in data["images"]
    if image["id"] in landslide_image_ids
)

image_path = DATASET_PATH / split / image_info["file_name"]

mask_path = (
    MASK_PATH / split /
    f"{Path(image_info['file_name']).stem}_mask.png"
)

image = cv2.imread(str(image_path))
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if image is None or mask is None:
    raise FileNotFoundError(
        f"Could not load image or mask:\n{image_path}\n{mask_path}"
    )

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

overlay = image.copy()

# Highlight the mask in red
overlay[mask > 0] = (
    0.6 * overlay[mask > 0] +
    0.4 * np.array([255, 0, 0])
).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title("Original Image")

axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Landslide Mask")

axes[2].imshow(overlay)
axes[2].set_title("Landslide Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

print("Image:", image_path)
print("Mask:", mask_path)
print("Mask dimensions:", mask.shape)
print("Unique mask values:", np.unique(mask))
print("Landslide pixels:", np.count_nonzero(mask))