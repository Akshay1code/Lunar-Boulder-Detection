"""
Evaluate a trained YOLO model on the lunar crater/rille dataset.

Usage (from project root):
    python scripts/evaluate.py
    python scripts/evaluate.py --weights runs/lunar_detection/v2/weights/best.pt --splits val test
    python scripts/evaluate.py --tta
"""
import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def get_default_weights() -> Path:
    candidates = [
        ROOT / "runs" / "lunar_detection" / "v3" / "weights" / "best.pt",
        ROOT / "runs" / "lunar_detection" / "v2" / "weights" / "best.pt",
        ROOT / "runs" / "lunar_detection" / "baseline" / "weights" / "best.pt",
        ROOT / "weights" / "best.pt",
        ROOT / "weights" / "yolo11s.pt",
        ROOT / "weights" / "yolo11n.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return ROOT / "weights" / "yolo11s.pt"

DEFAULT_WEIGHTS = get_default_weights()


def best_f1_threshold(box):
    """Confidence that maximises the class-averaged F1 curve, with P/R/F1 there."""
    try:
        f1 = np.asarray(box.f1_curve).mean(0)
        idx = int(f1.argmax())
        conf = float(np.asarray(box.px)[idx])
        p = float(np.asarray(box.p_curve).mean(0)[idx])
        r = float(np.asarray(box.r_curve).mean(0)[idx])
        return conf, p, r, float(f1[idx])
    except Exception:
        return None


def evaluate(model, data, split, imgsz, tta):
    metrics = model.val(data=data, split=split, imgsz=imgsz, augment=tta, plots=True,
                        project=str(ROOT / "runs" / "lunar_detection" / "eval"),
                        name=f"{split}{'_tta' if tta else ''}", exist_ok=True)
    box = metrics.box

    print(f"\n{'=' * 60}\nSplit: {split}   TTA: {tta}\n{'=' * 60}")
    print(f"Precision: {box.mp:.4f}")
    print(f"Recall:    {box.mr:.4f}")
    print(f"mAP50:     {box.map50:.4f}")
    print(f"mAP50-95:  {box.map:.4f}")

    print("\nPer-class results:")
    print(f"  {'class':<10}{'P':>8}{'R':>8}{'AP50':>8}{'AP50-95':>10}")
    for i, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        print(f"  {metrics.names[int(c)]:<10}{p:>8.3f}{r:>8.3f}{ap50:>8.3f}{ap:>10.3f}")

    best = best_f1_threshold(box)
    if best:
        conf, p, r, f1 = best
        print(f"\nBest-F1 confidence threshold: {conf:.3f}  (P={p:.3f}, R={r:.3f}, F1={f1:.3f})")
        print("  -> use this as the confidence value in the app / model.predict().")
    print(f"\nPlots (PR/F1 curves, confusion matrix) saved in: {metrics.save_dir}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Evaluate YOLO model on Lunar-CV data")
    ap.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--data", type=str, default=str(ROOT / "data.yaml"))
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"],
                    help="Dataset split(s) to evaluate (default: test)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    args = ap.parse_args()

    # Resolve weights path
    weights_path = Path(args.weights)
    if not weights_path.is_file() and (ROOT / "weights" / args.weights).is_file():
        weights_path = ROOT / "weights" / args.weights
    elif not weights_path.is_file() and (ROOT / args.weights).is_file():
        weights_path = ROOT / args.weights

    if not weights_path.is_file():
        raise SystemExit(f"Weights not found: {args.weights}\nTip: Place weights in 'weights/' or specify valid path.")

    # Resolve dataset yaml path
    data_path = Path(args.data)
    if not data_path.is_file() and (ROOT / "configs" / args.data).is_file():
        data_path = ROOT / "configs" / args.data
    elif not data_path.is_file() and (ROOT / args.data).is_file():
        data_path = ROOT / args.data

    model = YOLO(str(weights_path))
    for split in args.splits:
        evaluate(model, str(data_path), split, args.imgsz, args.tta)


if __name__ == "__main__":
    main()