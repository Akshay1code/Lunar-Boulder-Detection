# Proposed manual annotation protocol

Status: draft pending domain review. This document does not authorize annotation,
mask generation, or training. manual_annotation.proposed.json is a non-executable
project specification; its flags do not alter any existing scripts.

1. Obtain the original-resolution imagery, original annotation files if available,
   source-scene metadata, patch coordinates, and annotation guidance from the
   owner. Record actual provenance only. Keep unknown information null.
2. Have a qualified lunar-domain reviewer approve the operational target
   definition, included/excluded regions, and treatment of uncertainty. Do not
   derive a geological definition from existing rectangles or suspected examples.
3. Calibrate annotators on a small, varied sample before scaling up. Include
   suspected candidates, images without landslide labels, and difficult viewing
   conditions. Document disagreements and version the approved guidelines.
4. Review each image and record whether review is complete, partial, or uncertain.
   Use the image states in the specification. Missing labels never establish
   negative status. A reviewed-no-identifiable-target decision applies only
   under the approved protocol and available evidence, not absolute absence.
5. Once separately authorized, trace only regions supported by the approved
   definition on the best available imagery, consulting larger source context.
   Existing boxes are reference aids only. Do not force tracing to their edges.
   Preserve holes and multipart regions where supported. Do not infer boundaries
   through shadow, blur, missing detail, or other obscuration. Mark ambiguous
   regions unresolved rather than foreground or background.
6. Require a second reviewer for all proposed landslide boundaries and all
   proposed negative images. Record actual reviewer identities, timestamps,
   decisions, and uncertainty notes. Adjudicate disagreements with domain
   expertise. Keep unresolved examples outside supervised evaluation.
7. Validate image identity, dimensions, coordinate frame, geometric validity,
   class consistency, completeness, and overlays before accepting annotations.
   Preserve original files and retain a versioned history of future annotations.
   Do not automatically convert rectangles to masks. Any future mask export
   requires separate authorization and approved annotations, including a defined
   mechanism for ignoring unresolved pixels.
8. Keep test-label review independent of model predictions. Existing test images
   must not enter training. Verify source-scene and geographic overlap groups
   before proposing revised splits; keep related patches and duplicate groups
   together. Do not apply the duplicate manifest's recommendation automatically.
9. Annotation, mask export, and training remain disabled in this infrastructure
   task. A future workflow must explicitly implement and validate these controls;
   the specification alone provides no runtime enforcement.

## Manifest conventions

- image_key is the split-relative filename, resolved from dataset_root; COCO
  image and annotation IDs are scoped to their original split, not globally.
- Existing mask paths resolve from project_root. File hashes identify observed
  bytes, not annotation reliability.
- source_scene_hint is filename-derived and explicitly unverified; verified
  source-scene ID and patch coordinates remain null until evidence is supplied.
- The audit's not_domain_reviewed status does not deny prior thumbnail screening;
  it records the lack of qualified geological review.
- Proposed required provenance fields are requirements for future work, not
  assertions that annotation or review has already occurred.
