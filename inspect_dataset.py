import json
from pathlib import Path
from collections import Counter

DATASET_PATH = (
    Path.home()
    / "Downloads"
    / "moon_ohrc_detection.v2i.coco-segmentation"
)

for split in ["train", "valid", "test"]:

    annotation_file = (
        DATASET_PATH / split / "_annotations.coco.json"
    )

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    landslide_id = next(
        category_id
        for category_id, name in categories.items()
        if name.lower() == "landslide"
    )

    landslide_annotations = [
        annotation
        for annotation in data["annotations"]
        if annotation["category_id"] == landslide_id
    ]

    rectangular = 0

    for annotation in landslide_annotations:
        segmentation = annotation.get("segmentation", [])

        if not isinstance(segmentation, list) or not segmentation:
            continue

        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue

            points = list(zip(polygon[0::2], polygon[1::2]))

            xs = {point[0] for point in points}
            ys = {point[1] for point in points}

            if len(xs) == 2 and len(ys) == 2:
                rectangular += 1
                break

    print(f"\n{split.upper()}")
    print("Landslide annotations:", len(landslide_annotations))
    print("Rectangular polygons:", rectangular)