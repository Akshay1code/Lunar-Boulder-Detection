"""Extract top 5 largest high-priority crater bounding boxes for review."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
csv_in = ROOT / "reports/isro/large_bbox_review.csv"
csv_out = ROOT / "reports/isro/top5_crater_review.csv"

if not csv_in.exists():
    raise FileNotFoundError(f"Missing input at {csv_in}")

df = pd.read_csv(csv_in)
top5 = df[
    (df["class_name"] == "crater") &
    (df["review_priority"] == "HIGH")
].nlargest(5, "relative_area")

top5.to_csv(csv_out, index=False)

print("Top 5 crater boxes selected:")
print(top5[["split", "image", "relative_area"]].to_string(index=False))
print(f"\nCSV saved to {csv_out}")
