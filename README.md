# Lunar Landslide Segmentation Prototype

College project: **Lunar Terrain Hazard Detection for Safer Exploration**.
This component demonstrates binary segmentation of candidate regions in lunar
surface imagery using a lightweight U-Net. It does not certify terrain safety.

**Reference annotations are unverified rectangular polygons. Scores measure
agreement with those annotations, not confirmed geological accuracy.**

## Launch the completed demo

Use the existing Python 3.11 environment; no installation is needed on this laptop:

```powershell
Set-Location "C:\Dbit project\lunar-landslide-segmentation"
.\.venv311\Scripts\python.exe -B demo.py
```

Open **http://127.0.0.1:8000**. Choose a validation or external lunar image,
click **Generate prediction**, and inspect the original, binary mask and red
overlay. The threshold slider defaults to 0.50 and supports 0.20–0.80.
It reuses the current probability map without rerunning the model.
Download controls save the native 256×256 mask or the displayed overlay.
Selecting another image clears the previous result. Stop with Ctrl+C.

If port 8000 is occupied, use `demo.py --port 8001` and open port 8001 instead.
The application is a local, single-user Python HTTP application, not Streamlit.
It binds to localhost, accepts up to 15 MB / 16 million pixels, and keeps
uploads in memory. Predictions are written only through browser downloads.
Console diagnostics include image hashes, dimensions and model-score statistics.
Do not upload test-split images; browser uploads do not retain source paths,
so the application cannot establish an arbitrary uploaded file's provenance.

The demo loads only:

```text
experiments/experiment_i_rotation/runs/20261018T164118Z_c8c8656b2bf14d69b77b8b02585ef040/best.pth
```

This is Experiment I's best validation checkpoint, update **3,080**, not its
final checkpoint. `unet_model.py`, the checkpoint, and the inference helper
modules under `tools/` must remain in their existing locations.
Alternatively, the `LUNAR_CHECKPOINT_PATH` environment variable can be set to point to `best.pth` if stored at a custom location.

## Dependencies and verification

The tested environment is Python 3.11, CPU PyTorch 2.14.1+cpu,
NumPy 2.4.6, OpenCV 5.0.0 and Pillow 12.3.0. The demo does not require the
datasets, a GPU, Streamlit or an internet connection.
For a separate Python 3.11 environment, install the CPU runtime dependencies:

```powershell
python -m venv .venv311
.\.venv311\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv311\Scripts\pip.exe install -r requirements.txt
```

Do not recreate the already working environment before submission. These
commands are setup guidance, not a guarantee that future package versions
will reproduce the tested environment. Preserve the existing environment.

Run the synthetic checks:

```powershell
.\.venv311\Scripts\python.exe -B demo.py --self-test
```

They check HTTP upload, exact checkpoint weights, preprocessing equality,
mask/overlay encoding, threshold behavior, cache reuse, invalid inputs,
native-resolution download data and protected-file hashes. They do not
replace an interactive browser test.

## Dataset and preprocessing

The existing Moon OHRC Detection v2 export is recorded at:
`https://universe.roboflow.com/moonohrcdetection/moon_ohrc_detection-zqh88`.

Dataset location is configured via the `LUNAR_DATASET_ROOT` environment variable, or by placing the dataset in a `./dataset` folder in the project root:
```powershell
$env:LUNAR_DATASET_ROOT = "path\to\moon_ohrc_detection.v2i.coco-segmentation"
```

Reference masks are in `dataset_masks/<split>/`, paired by
`<image-stem>_mask.png`. `create_masks.py` rasterizes only category 3,
named `landslide`; the targets do not combine other object categories.
The original polygons are axis-aligned rectangles. Their geological meaning
and annotation completeness are not independently established.

Training contains 279 images, of which 73 have nonempty masks. Validation
contains 80 images: 19 nonempty and 61 empty references. The completed
experiments use the same selected 73 training examples. An empty reference
does not establish absence of a landslide. No test evaluation is performed
by this demo or the project-completion checks.

Preprocessing is unchanged: OpenCV decoding, BGR→RGB, direct
`INTER_AREA` resize to 256×256, float32 division by 255, then CHW layout.
There is no additional mean/std normalization or padding. Nonsquare inputs
are stretched. Masks use nearest-neighbor resizing and binary foreground.
Inference uses evaluation mode, no gradients, sigmoid, and `p >= threshold`.
The display restores predictions to the preview dimensions with nearest
neighbor. Predicted pixel coverage is calculated on the 256×256 output,
not measured physical area or a calibrated geological probability.

## Architecture and completed training

`unet_model.py` defines a 482,033-parameter U-Net: three encoder stages
(16/32/64 channels), a 128-channel bottleneck, skip connections, transposed
convolution upsampling, and a one-channel logit output. Double-convolution
blocks use ReLU; the output sigmoid is applied during inference.

Experiment I used seed 42, CPU, Adam 0.001, batch size 2, and 5,180 optimizer
updates (140 complete passes through 73 images). The loss is mean
BCEWithLogitsLoss plus mean per-image soft Dice loss, with equal coefficients:
`1 - (2*sum(sigmoid(logits)*target)+1)/(sum(sigmoid(logits))+sum(target)+1)`.
Joint random quarter-turn rotations use a separate generator. Validation
is unaugmented and excluded from gradients. Evaluation occurs at update 0
and every 140 updates; strictly improving mean Dice on nonempty validation
references selects the checkpoint, keeping the earlier checkpoint on ties.

The transfer experiment used MMLSv2: 465 Martian training pairs and 66
validation pairs. RGB bands 1–3 were already normalized to 0–1. It pretrained
for 2,330 updates (10 passes), then reset the optimizer and random generators
for the same 5,180-update lunar stage. Martian validation data was excluded
from gradient updates. Missing image georeferencing limits independent
verification of image-mask alignment despite technical and visual checks.
See `experiments/mmlsv2_transfer/README.md` and its run reports for provenance.

## Evaluation and results

Binary Dice is `2TP/(2TP+FP+FN)` and IoU is `TP/(TP+FP+FN)`.
Precision is `TP/(TP+FP)`; recall is `TP/(TP+FN)`.
Per-image scores are averaged over defined values. Both-empty Dice/IoU,
precision without predictions, and recall without reference foreground are
undefined, not automatically perfect. Empty-reference behavior is reported
separately and is not interpreted as verified geological false positives.

Same threshold 0.5, same 19 nonempty validation references:

| Approach | Best Dice | Best IoU |
|---|---:|---:|
| Lunar-only Experiment I | **0.556983** | **0.418620** |
| Mars pretraining → lunar fine-tuning | 0.515292 | 0.382620 |

The lunar-only checkpoint remains the demo choice. Transfer did not improve
overlap in this run, although mean coverage on the 61 empty references
fell from 63.964% to 3.922% (59 versus 27 images with predictions).
The comparison uses a historical lunar baseline and one seed; transfer
receives extra compute. It is not a compute-matched statistical study.

Completed transfer results:
`experiments/mmlsv2_transfer/runs/20261020T033814Z_continued_86dec9b655f8490e97d0645068464127/`.
Read `final_report.md`, `comparison.json`, `per_image.csv`, and comparison PNGs.
Presentation figure:
`experiments/mmlsv2_transfer/presentation_figures/20261020T091120Z_af4e137143734550aa12b9bdda6c84be/lunar_baseline_vs_mars_transfer.png`.

## Limitations and submission scope

### Known horizon/background failure

Experiment I can predict foreground over black space in horizon images.
The reproduced D1 example has 65.002441% coverage at threshold 0.5; its
black top quarter has 99.945% foreground. The model probabilities cause this;
the existing preprocessing and mask/overlay rendering were checked and match.
The separate D2 surface example produces 2.899170% coverage.

The demo displays a prominent warning at **40% or greater predicted coverage**.
This is a chosen UI caution threshold, not a confidence score or a detector
of incorrect predictions. Results below it are not certified correct.
A persistent notice explains that horizon/space scenes are not reliably supported.
Predictions and downloads remain raw and are not silently filtered.
Darkness, even when connected to an image edge, cannot reliably distinguish
space from shadowed terrain. No automatic horizon detector or dark-pixel
removal is enabled. A reliable exclusion mask would require a verified horizon
or explicit user-reviewed surface boundary, including tests on shadows,
sloping/curved horizons and terrain-only scenes. Until then, demonstrate
surface-only examples and disclose this limitation. Low coverage never
establishes absence of a landslide.

### Optional terrain appearance and connected regions

After predicting an image, expand **Optional visual terrain appearance (not
geology)**. This is a transparent image-texture heuristic, independent of
the U-Net: grayscale resized to 256×256, 3×3 Gaussian blur, then Canny edges
with thresholds 50/100. Edge density below 4% is called relatively
smooth-looking, 4–<12% textured-looking, and ≥12% highly textured-looking.
The result is inconclusive if grayscale standard deviation is below 10
or more than 40% of pixels are at intensities ≤5 or ≥250.
These chosen thresholds are illustrative, not calibrated or validated.
Lighting, shadows, image scale, compression and resizing affect the result.
It does not measure slope/elevation, identify rocks, or assign geological classes.

Connected regions use 8-connectivity on the thresholded 256×256 prediction.
All regions, including tiny ones, are counted without filtering; the UI shows
the ten largest sizes in output pixels. These are not physical surface areas
or verified individual landslides. Changing the threshold changes these counts,
but not the image-appearance assessment. Small coverage does not establish
that a landslide is absent. Geological confirmation is unavailable.

### Optional offline video demonstration

To keep the working image-upload server responsive, video processing is a
separate command, not a browser upload mode. Use a permitted local video;
do not supply test-split content or renamed copies of it.

```powershell
Set-Location "C:\Dbit project\lunar-landslide-segmentation"
.\.venv311\Scripts\python.exe -B tools\demo_video.py "C:\path\to\your\clip.mp4" --fps 0.5 --max-frames 20 --threshold 0.5
```

Replace the example path with an actual file. MP4/AVI/MOV/MKV containers are
accepted, but codec availability depends on installed OpenCV decoders.
Limits: 100 MiB, 60 seconds, at most 1920×1080 pixels per frame,
0.1–2 samples/second, and at most 30 processed frames. Defaults sample one
frame every two seconds, up to 20 frames. A 120-second budget is checked
between frames; it cannot interrupt a blocking native decoder call.
Missing timing metadata or an unreadable selected frame causes an explicit
failure rather than silently skipping it. Seek accuracy depends on the codec;
requested indices/times and reported decoder positions are recorded.

The script uses the same Experiment I checkpoint, preprocessing, thresholding
and overlay logic. Frames are processed individually, not loaded as a whole
video into RAM. Each run creates a unique `video_demonstrations/<timestamp_uuid>/`
with original-frame PNGs, 256×256 binary masks, overlays and `report.json`.
The console shows processed count, threshold, coverage and output location.
On failure after output creation, partial results remain with an incomplete
status. Video resources are released in a `finally` block; source/checkpoint
hashes are checked. Open the PNGs in VS Code or an image viewer.
There is no claim of real-time operation, motion estimation or geological
activity detection. The local test used synthetic AVI footage solely to
verify the code path; real lunar video performance has not been established.

### Optional MP4 from saved video frames

After `demo_video.py` finishes, copy its printed output directory into:

```powershell
.\.venv311\Scripts\python.exe -B tools\export_demo_video.py "video_demonstrations/<completed-run-directory>"
```

Replace `<completed-run-directory>` with the actual directory name. This
separate utility performs **no inference** and leaves the original PNG workflow
unchanged. It creates `video_presentations/<timestamp_uuid>/sampled_predictions.mp4`
and an integrity/export report. Original frames and overlays appear side by side,
with source timestamps, threshold, coverage and experimental warnings.
The MP4 is a **slideshow: one second per sampled frame, encoded at 10 fps**.
It does not reproduce source video timing, estimate motion or demonstrate
real-time detection. Synthetic inputs are labelled explicitly.

The installed OpenCV `mp4v` encoder passed a synthetic export/decode test.
Playback compatibility varies by player; use the PNGs if MP4 playback fails.
If encoding is unavailable, the utility reports failure without modifying
the source frames. Partial export output is marked incomplete.

### Repository structure and Git exclusions

Due to GitHub file size limits and repository hygiene, binary model weights, raw datasets, virtual environments, and generated runtime artifacts are excluded from Git tracking via `.gitignore`:
- **Model Checkpoints (`*.pth`, `*.pt`, `checkpoints/`)**: Binary trained weights (such as the Experiment I baseline `best.pth`) are kept locally. To run the demo, keep `best.pth` at `experiments/experiment_i_rotation/runs/20261018T164118Z_c8c8656b2bf14d69b77b8b02585ef040/best.pth` or set the `LUNAR_CHECKPOINT_PATH` environment variable to its location.
- **Virtual Environments (`.venv/`, `.venv311/`)**: Create an isolated environment and install runtime requirements via `pip install -r requirements.txt`.
- **Raw Image Datasets & Generated Masks (`dataset_masks/`, `datasets/`)**: Stored locally and configured via `LUNAR_DATASET_ROOT`.
- **Generated Demonstrations and Predictions (`video_demonstrations/`, `predictions/`, `runs/`)**: Created dynamically during runtime inference.

### Final deliverables

- Application: `demo.py`; offline video: `tools/demo_video.py`.
- Optional MP4 exporter: `tools/export_demo_video.py`.
- Selected checkpoint: the Experiment I `best.pth` path listed above.
- Methodology and usage: this `README.md`.
- Final verification and deliverable index:
  `project_completion/20261020T144921Z_submission_b24467c73ca64146aad243658f26be77/final_report.md`.
- Presentation PNG and PDF:
  `experiments/mmlsv2_transfer/presentation_figures/20261020T091120Z_af4e137143734550aa12b9bdda6c84be/`.

Unverified rectangles, incomplete labels, only 73 annotated training images,
domain differences and repeated validation use limit scientific conclusions.
Matching references or visible terrain features does not confirm landslides.
Predictions can miss references or cover large unrelated-looking regions.
Performance on arbitrary uploads is unknown; threshold changes trade coverage
against missed regions without establishing correctness.

This submission delivers a working experimental visualization application,
completed training comparisons, evaluation artifacts and documented limitations.
No further training is needed to demonstrate it. Preserve all experiment runs
and checkpoints; do not use the application to certify exploration safety.
