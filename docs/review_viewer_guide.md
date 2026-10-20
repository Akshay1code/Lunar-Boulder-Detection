# Viewing the annotation-review shortlist

The viewer reads the approved 20-record shortlist and existing images and masks.
It does not load a model, generate masks, classify geological features, save review
decisions, or update any files or review statuses. All current records remain
unreviewed. No test images are included.

## Start in VS Code on Windows

1. Open `C:\Dbit project\lunar-landslide-segmentation` in VS Code.
2. Select **Terminal > New Terminal** (PowerShell).
3. Run:

```powershell
.\.venv311\Scripts\python.exe -B review_shortlist.py
```

The viewer uses the installed OpenCV, NumPy, and Matplotlib packages. It does not
need PyTorch or a checkpoint. `-B` prevents Python bytecode-cache writes.
The script finds its manifest relative to its own location, not the terminal's
working directory. The manifest retains the original asset paths.

## Inspect each pair

- Left: the original exported lunar image, without model predictions or overlays.
- Right: the existing rectangular reference mask, explicitly marked UNVERIFIED.
  Black represents stored zero values; white represents stored foreground values.
  These values are not verified geological boundaries or presence/absence labels.
- The heading shows the review ID, position, existing split, landslide annotation
  count, and unchanged review status. The exported filename appears below it.
- Selection reasons and uncertainty notes appear below the images.
- Images without landslide labels have an explicit warning. An empty mask is
  never proof that the image contains no landslide.

Use **Previous/Next** buttons or click inside the viewer and press the **left/right
arrow keys**. Navigation wraps from R20 to R01 and back. Use the standard plot
toolbar's zoom/pan/home controls to examine pixels; zoom history resets when you
change image. Maximize the window if text is difficult to read. Close the window
to finish. Browsing an image never marks it as reviewed.

Do not use the toolbar's optional save-figure action if you want a strictly
viewing-only session; the viewer itself does not save screenshots or other files.

## Missing files and integrity warnings

The viewer recalculates SHA-256 from the exact bytes it decodes and compares it
with the shortlist. A missing file, absent hash, hash mismatch, decode failure,
or dimension mismatch displays an error in that panel instead of substituted
image/mask data. The other panel and navigation remain available. A missing mask
is not rendered as an empty mask. Errors do not change records or repair assets.

If an error appears, note the review ID and message and check the expected path
in `annotation/review_shortlist.proposed.json`. Do not update hashes or substitute
files merely to clear a warning; investigate the discrepancy first.

To check all 20 pairs without opening a window:

```powershell
.\.venv311\Scripts\python.exe -B review_shortlist.py --check
```

This prints per-record results and returns a nonzero exit code if any pair fails.

## Review boundaries

This viewer supports visual inspection only. Existing labels, selection reasons,
brightness, and texture do not confirm a landslide. Missing labels are not confirmed
negatives. Geological interpretation and any future review decisions require the
agreed protocol and appropriate human expertise. No annotation-editing or status-
updating controls are provided. Consult `annotation_review_plan.md`,
`annotation_limitations.md`, and `manual_annotation_protocol.md` for context.
