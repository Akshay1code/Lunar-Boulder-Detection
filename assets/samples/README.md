# Lunar Sample Images

This directory contains 17 representative lunar surface images featuring craters and rilles, provided for quick inference testing and validation without needing to download the entire ~100MB dataset.

## Quick Inference Usage

### Desktop GUI App
Launch the Lunar-CV desktop detector and select an image from this folder:
```powershell
python scripts/app.py
```
Click **Browse Image** or **Browse Folder** and navigate to `assets/samples/`.

### CLI / Python Script
Run model prediction on a sample image using Ultralytics:
```python
from ultralytics import YOLO

model = YOLO("weights/yolo11s.pt")  # or your trained weights
results = model.predict("assets/samples/lunar_sample_01.jpg", save=True, conf=0.25)
```
