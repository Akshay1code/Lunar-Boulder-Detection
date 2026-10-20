# Existing annotation limitations

## Verified local observations

The inspected export contains 399 images: 279 train, 80 valid, and 40 test.
All 276 landslide annotations (191 train, 42 valid, 43 test) are axis-aligned
rectangular polygons. They do not provide verified pixel-level landslide boundaries.
The audit manifest records actual file SHA-256 hashes, COCO annotation IDs, and
JPEG-header dimensions compared with COCO metadata.

Existing rasterized masks are not independently verified ground truth. The
existing create_masks.py fills supplied polygons; do not run it to manufacture
ground truth from the current rectangles. The manifest's mask status describes
the known polygon-filling workflow, not independent confirmation of each mask's
creation history or pixel accuracy.

There are 295 images without landslide labels (206 train, 61 valid, 28 test).
They are not confirmed negatives. Label completeness is unknown. Visual review
candidates remain uncertain and have not received geological confirmation.
Dark or uninterpretable image regions must not be assumed to be background.

One exact file-duplicate group spans training and testing. Its three members
are documented in duplicate_groups.proposed.json using freshly calculated hashes.
No member has been removed, excluded, or reassigned. Exact-hash matching does
not establish the absence of cropped, shifted, or otherwise near-duplicate images.

## Missing information and uncertainty

Original-resolution imagery, annotation methodology, exhaustive-labeling policy,
and human review history are unavailable or unverified in this export.
The export README reports automatic orientation and stretching to 640 x 640.
Resized images do not recover the original spatial detail.

The public project's description did not explain annotation methodology during
the earlier investigation:
https://universe.roboflow.com/moonohrcdetection/moon_ohrc_detection-zqh88
Version-specific pages could not be retrieved in that investigation. Better
annotations may exist privately or in another version; this audit does not claim
that they do not exist. No owner-provided review provenance is invented.

Source-prefix overlap raises possible scene leakage, but filename prefixes are
unverified hints, not established source-scene or geographic identifiers.
Patch coordinates and verified source-scene IDs remain null. The numeric patch
suffix must not be interpreted as coordinates. Source-scene independence and
near-duplicate relationships remain unresolved.

## Permitted interpretation

Segmentation performance claims require independently reviewed boundary labels
and a defensible evaluation split. A nonrectangular polygon alone is not evidence
of an accurate boundary. An absent label is not evidence of geological absence.

All manifest entries are observations or proposals. They do not change dataset
splits or automatically exclude images. Existing test images remain reserved
from training. Any future split proposal requires source and overlap checks.

The manual annotation JSON is a non-executable specification, not a tool-specific
import file. Its disabled flags document policy; they do not enforce controls
in existing scripts. Annotation, mask export, and training are not enabled.
