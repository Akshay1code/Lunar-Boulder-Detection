import os
import json
import cv2
import numpy as np

from pathlib import Path
from collections import defaultdict


def get_dataset_path() -> Path:
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

    local_rel = Path(__file__).resolve().parent / "dataset"
    if local_rel.is_dir():
        return local_rel

    downloads_fallback = (
        Path.home()
        / "Downloads"
        / "moon_ohrc_detection.v2i.coco-segmentation"
    )
    if downloads_fallback.is_dir():
        return downloads_fallback

    return local_rel


DATASET_PATH = get_dataset_path()
OUTPUT_PATH = Path(__file__).resolve().parent / "dataset_masks"


def create_all_masks(dataset_path=None, output_path=None):
    base_dataset = Path(dataset_path) if dataset_path else DATASET_PATH
    base_output = Path(output_path) if output_path else OUTPUT_PATH

    for split in ["train", "valid", "test"]:

        annotation_file = (
            base_dataset / split / "_annotations.coco.json"
        )

        if not annotation_file.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: '{annotation_file}'.\n"
                "Please configure the dataset location by setting the LUNAR_DATASET_ROOT "
                "environment variable, e.g.:\n"
                "  set LUNAR_DATASET_ROOT=C:\\path\\to\\moon_ohrc_detection.v2i.coco-segmentation"
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

        image_info = {
            image["id"]: image
            for image in data["images"]
        }

        annotations_by_image = defaultdict(list)

        for annotation in data["annotations"]:
            if annotation["category_id"] in landslide_ids:
                annotations_by_image[
                    annotation["image_id"]
                ].append(annotation)

        output_folder = base_output / split
        output_folder.mkdir(parents=True, exist_ok=True)

        generated = 0

        for image_id, image in image_info.items():

            width = image["width"]
            height = image["height"]

            mask = np.zeros((height, width), dtype=np.uint8)

            for annotation in annotations_by_image[image_id]:

                segmentation = annotation.get("segmentation", [])

                if not isinstance(segmentation, list):
                    continue

                for polygon in segmentation:

                    if not isinstance(polygon, list):
                        continue

                    if len(polygon) < 6 or len(polygon) % 2 != 0:
                        continue

                    points = np.array(
                        polygon,
                        dtype=np.float32
                    ).reshape(-1, 2)

                    points = np.round(points).astype(np.int32)

                    cv2.fillPoly(mask, [points], 255)

            image_name = Path(image["file_name"]).stem

            mask_name = f"{image_name}_mask.png"

            cv2.imwrite(
                str(output_folder / mask_name),
                mask
            )

            generated += 1

        print(f"{split}: generated {generated} masks")

    print("\nMask generation completed!")


if __name__ == "__main__":
    create_all_masks()
