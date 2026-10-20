"""Preview Experiment E on the six previously compared training images.

CPU inference only. Reference annotations are unverified rectangular masks;
agreement does not establish geological accuracy. Use --check-only to run
inference and render PNGs in memory without writing preview files.
"""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import preview_unet_predictions as preview

CHECKPOINT = preview.ROOT / (
    "experiments/experiment_b_exposure/runs/"
    "20261013T133640Z_402e62786bab407598214d53f521fae9/best.pth"
)
# Exact filenames from the completed comparison, in its deterministic order.
EXPECTED_FILENAMES = (
    "ohr_000_patch_0154_png.rf.6d9141dc701bdcaa729c6fc4eef9edac.jpg",
    "ohr_000_patch_0355_png.rf.17323d58a923a316796ca6f76ce1f625.jpg",
    "ohr_001_patch_1715_png.rf.9723a06f2ed11acf2ec84f2df2532410.jpg",
    "ohr_002_patch_0474_png.rf.0a6a1e77f7ae7be5785f25c940994006.jpg",
    "ohr_002_patch_0582_png.rf.21aebaec41078a6d596242ae13aaffc7.jpg",
    "ohr_002_patch_3698_png.rf.5c34df1a945dee5ad5cf32233bc17a88.jpg",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Check all six inferences and PNG rendering without saving files")
    args = parser.parse_args(argv)
    before = {}
    try:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"Experiment E checkpoint missing: {CHECKPOINT}")
        if preview.THRESHOLD != 0.5:
            raise ValueError("Existing preview threshold changed; expected 0.5")
        dataset = preview.training_dataset()
        indices = preview.select_images(dataset, 6)
        actual = tuple(dataset.pairs[index][0].name for index in indices)
        if actual != EXPECTED_FILENAMES:
            raise ValueError(f"Training selection differs from the completed comparison: {actual}")
        protected = [CHECKPOINT] + [preview.ROOT / name for name in (
            "unet_model.py", "dataset_loader.py", "predict_image.py",
            "tools/preview_unet_predictions.py", "tools/compare_unet_checkpoints.py",
        )] + [path for index in indices for path in dataset.pairs[index]]
        before = {path: preview.digest(path) for path in protected}
        print("Experiment E: verified the same six training filenames and order.")
        print("Reference annotations are unverified; these previews do not establish geological accuracy.")
        options = ["--checkpoint", str(CHECKPOINT), "--count", "6"]
        if args.check_only:
            options.append("--check-only")
        # Reuse unchanged loader, UNet, eval/no_grad CPU inference, rendering,
        # unique output directories, and exclusive PNG writes.
        return preview.main(options)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1
    finally:
        changed = [str(path) for path, expected in before.items()
                   if not path.is_file() or preview.digest(path) != expected]
        if changed:
            raise RuntimeError(f"Protected file hash discrepancy; no repairs attempted: {changed}")


if __name__ == "__main__":
    raise SystemExit(main())
