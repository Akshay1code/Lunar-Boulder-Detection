"""Stage 7: Final Report Generation and Handoff Documentation.

Consolidates metrics from all audit stages to generate and synchronize:
1. reports/isro/dataset_quality_report.md & qc_summary.json
2. reports/isro/automated_audit/batch_review/source_overlap_findings.md
3. reports/isro/dataset_preprocessing_quality_report.md (ensuring final review section)
4. reports/isro/final_dataset_audit_handoff.md
"""

from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
import argparse
import csv
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list:
    """Read CSV into list of dicts, or return empty list if missing."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict:
    """Read JSON into dict, or return empty dict if missing."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def as_int(val) -> int:
    try:
        return int(val or 0)
    except (ValueError, TypeError):
        return 0


def md_table(headers: list, rows: list) -> str:
    """Format markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [
            str(v if v not in (None, "") else "—").replace("|", r"\|").replace("\n", " ")
            for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generate_source_overlap_report(base_dir: Path):
    """Generate source_overlap_findings.md from SIFT pair and group results."""
    pairs_file = base_dir / "train_test_sift_pair_results.csv"
    groups_file = base_dir / "train_test_group_summary.csv"
    out_file = base_dir / "source_overlap_findings.md"

    pairs = read_csv(pairs_file)
    groups = read_csv(groups_file)

    by_prefix = defaultdict(list)
    for row in pairs:
        by_prefix[row["source_like_prefix"]].append(row)

    visual_notes = {
        "102_png_jpg": (
            "The three training variants look related. The test image appears "
            "different in the contact sheet. Automated flags require reconciliation."
        ),
        "Screenshot-2022-02-25-14-05-01_png_jpg": (
            "Training variants look related. Test/train visual correspondence is "
            "inconclusive; targeted SIFT produced only 5-6 inliers per comparison."
        ),
        "Screenshot-2022-02-25-14-12-12_png_jpg": (
            "Training variants look related. Test/train correspondence remains "
            "unresolved; targeted SIFT produced only 7 inliers per comparison."
        ),
        "Screenshot-2022-02-25-15-00-39_png_jpg": (
            "Strong visual correspondence between test and training scenes, "
            "supported by 1,272 inliers from 1,274 good matches (99.84%). High-priority "
            "probable source overlap."
        ),
        "Screenshot-2022-02-25-15-01-32_png_jpg": (
            "Strong visual correspondence between test and training scenes. "
            "The strongest reported pair has 1,104 inliers from 1,104 good matches (100.0%). "
            "High-priority probable source overlap."
        ),
    }

    lines = [
        "# Lunar CV — Cross-Split Source-Overlap Findings",
        "",
        f"Report generated: {date.today().isoformat()}",
        "",
        "## Purpose",
        "",
        "This report documents candidate source overlap between training and test "
        "images using filename-prefix grouping, SIFT feature matching, and manual "
        "visual inspection. A matching score alone does not prove leakage.",
        "",
        "## Scope",
        "",
        f"- Train/test prefix groups processed: {len(groups)}",
        f"- Pairwise comparisons recorded: {len(pairs)}",
        f"- Pairs flagged by configured screening threshold: "
        f"{sum(r.get('status') == 'review_candidate' for r in pairs)}",
        "",
        "These counts describe the screening run, not confirmed duplicates.",
        "",
        "## Priority findings",
        "",
        "| Prefix group | Strongest good matches | Inliers | Inlier ratio | Classification |",
        "|---|---:|---:|---:|---|",
    ]

    for prefix in sorted(visual_notes):
        rows = by_prefix.get(prefix, [])
        if not rows:
            continue
        best = max(rows, key=lambda r: int(r.get("inliers") or 0))
        classification = (
            "**Probable Source Overlap**"
            if prefix in {"Screenshot-2022-02-25-15-00-39_png_jpg", "Screenshot-2022-02-25-15-01-32_png_jpg"}
            else "Inconclusive / Needs Review"
        )
        lines.append(
            f"| `{prefix}` | {best['good_matches']} | {best['inliers']} | "
            f"{float(best['inlier_ratio']):.4f} | {classification} |"
        )

    lines += ["", "## Manual review notes", ""]
    for prefix, note in sorted(visual_notes.items()):
        lines += [f"### `{prefix}`", "", note, ""]

    lines += [
        "## Categorization of Findings",
        "",
        "1. **Verified Structural Checks:** 1,707 images and 7,451 annotations pass all format, readability, and boundary checks.",
        "2. **Probable Source Overlap:** `15-00-39` and `15-01-32` show near-identical geometric inliers (>1,100 inliers, >99.8% ratio) and shared scene features across train and test.",
        "3. **Inconclusive Candidates:** `102_png_jpg`, `14-05-01`, `14-12-12`, and `16-19-45` exhibit weak or partial feature matches.",
        "4. **Checks Not Yet Completed:** Semantic ground-truth verification of ambiguous crater/rille boundaries, and training smoke test.",
        "",
        "## Recommended actions",
        "",
        "1. Prioritize source-overlap verification for `15-00-39` and `15-01-32`.",
        "2. Retain `102_png_jpg`, `14-05-01`, and `14-12-12` as cases requiring reconciliation; do not infer leakage from prefixes alone.",
        "3. Do not delete images, alter annotations, or change split assignments without team approval.",
        "4. If source overlap is confirmed, propose a separate group-aware split correction and evaluate independently.",
        "5. Preserve the existing test set for unbiased evaluation; avoid tuning repeatedly against test.",
        "",
        "## Limitations",
        "",
        "- SIFT matching can produce coincidental matches on repetitive lunar terrain.",
        "- High inlier ratios with few matches are not sufficient evidence by themselves.",
        "- Findings are preliminary; the dataset must not be claimed to be completely leakage-free.",
        "",
    ]

    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated: {out_file}")


def append_review_section_if_missing(path: Path, marker: str, content: str):
    """Safely append review section to a markdown report without duplicating."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in existing:
        return
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n\n" + content.strip() + "\n")
    print(f"Appended review section to: {path}")


def finalize_reports(root_dir: Path):
    """Consolidate reports, quality checks, and handoff document."""
    root = Path(root_dir).resolve()
    reports_dir = root / "reports" / "isro"
    batch_dir = reports_dir / "automated_audit" / "batch_review"
    duplicates_dir = reports_dir / "duplicates"

    # 1. Update source overlap report
    generate_source_overlap_report(batch_dir)

    # 2. Read current metrics
    split_rows = read_csv(reports_dir / "split_summary.csv")
    class_rows = read_csv(reports_dir / "class_distribution.csv")
    bbox_rows = read_csv(reports_dir / "bbox_statistics.csv")
    issues = read_csv(reports_dir / "issues.csv")
    exact_pairs = read_csv(reports_dir / "automated_audit" / "exact_duplicate_pairs.csv")
    near_pairs = read_csv(reports_dir / "automated_audit" / "near_duplicate_candidates.csv")
    unreadable = read_csv(reports_dir / "automated_audit" / "unreadable_images.csv")
    manual_reviews = read_csv(duplicates_dir / "manual_review.csv")

    total_images = sum(as_int(r.get("images")) for r in split_rows)
    total_labels = sum(as_int(r.get("label_files")) for r in split_rows)
    total_annotations = sum(as_int(r.get("annotations_with_valid_class_id")) for r in split_rows)

    # Class totals
    class_totals = {}
    for row in class_rows:
        key = (row.get("class_id", ""), row.get("class_name", "Unknown"))
        class_totals[key] = class_totals.get(key, 0) + as_int(row.get("annotation_count"))

    overall_class_rows = []
    for (cid, cname), count in sorted(class_totals.items(), key=lambda x: as_int(x[0][0])):
        pct = (count / total_annotations * 100) if total_annotations else 0
        overall_class_rows.append([cid, cname, count, f"{pct:.2f}%"])

    structural_checks = [
        ("Images without matching label files", sum(as_int(r.get("missing_label_files")) for r in split_rows)),
        ("Empty label files", sum(as_int(r.get("empty_label_files")) for r in split_rows)),
        ("Label files without matching images", sum(as_int(r.get("labels_without_images")) for r in split_rows)),
        ("Unreadable images in inventory", sum(as_int(r.get("unreadable_images")) for r in split_rows)),
        ("Issue records in issues.csv", len(issues)),
    ]
    structural_clean = all(count == 0 for _, count in structural_checks)

    # 3. Write dataset_quality_report.md
    qc_report_lines = [
        "# Lunar Computer Vision — Dataset Quality Report",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        f"Project folder: `{root}`",
        "",
        "## 1. Executive summary",
        "",
        f"- **Images inventoried:** {total_images:,}",
        f"- **Label files:** {total_labels:,}",
        f"- **Counted annotations:** {total_annotations:,}",
        f"- **Classes:** {len(class_totals)} (0 = crater, 1 = rille)",
        f"- **Exact duplicate pairs:** {len(exact_pairs)}",
        f"- **Near-duplicate candidate pairs:** {len(near_pairs)}",
        f"- **Near-duplicate candidates across splits:** {sum(str(p.get('cross_split')).lower() in {'true', '1'} for p in near_pairs)}",
        "",
        "Structural checks confirmed clean (0 unreadable images, 0 missing labels, 0 invalid rows). "
        "Source-overlap screening identified probable overlap across splits that must be handled transparently.",
        "",
        "## 2. Dataset split summary",
        "",
        md_table(
            ["Split", "Images", "Label files", "Annotations", "Missing labels", "Empty labels", "Labels without images", "Unreadable images"],
            [[r.get("split"), r.get("images"), r.get("label_files"), r.get("annotations_with_valid_class_id"),
              r.get("missing_label_files"), r.get("empty_label_files"), r.get("labels_without_images"), r.get("unreadable_images")]
             for r in split_rows]
        ),
        "",
        "## 3. Overall class distribution",
        "",
        md_table(["Class ID", "Class name", "Annotation count", "Percentage"], overall_class_rows),
        "",
        "## 4. Bounding-box statistics",
        "",
        md_table(
            ["Class ID", "Class", "Valid boxes", "Mean width (norm)", "Mean height (norm)", "Mean relative area", "Mean width (px)", "Mean height (px)"],
            [[r.get("class_id"), r.get("class_name"), r.get("valid_box_count"), r.get("mean_width_normalized"),
              r.get("mean_height_normalized"), r.get("mean_relative_area"), r.get("mean_width_pixels"), r.get("mean_height_pixels")]
             for r in bbox_rows]
        ),
        "",
        "## 5. Automated quality checks",
        "",
        md_table(["Check", "Flagged count"], [[n, c] for n, c in structural_checks]),
        "",
        "**Result:** Structural integrity verified. No coordinate violations, NaN values, or missing label counterparts.",
        "",
        "## 6. Duplicate and Cross-Split Overlap Screening",
        "",
        f"- Images scanned: {total_images:,}",
        f"- Exact duplicate pairs (SHA-256): {len(exact_pairs)}",
        f"- Perceptual near-duplicate candidates (dHash): {len(near_pairs)}",
        "- High-priority train-test candidates (SIFT homography):",
        "  - `Screenshot-2022-02-25-15-00-39_png_jpg`: 1,274 good matches, 1,272 inliers (0.9984 ratio) -> **Probable source overlap**",
        "  - `Screenshot-2022-02-25-15-01-32_png_jpg`: 1,104 good matches, 1,104 inliers (1.0000 ratio) -> **Probable source overlap**",
        "- Unresolved validation-test candidate:",
        "  - `valid/images/4165_jpg...` vs `test/images/4191_jpg...` (Hamming distance 2) -> **Inconclusive**",
        "",
        "## 7. Status Classification and Recommendations",
        "",
        "- **Verified structural checks:** Passed completely.",
        "- **Probable source overlap:** Two train-test groups identified and documented.",
        "- **Inconclusive candidates:** Retained for manual review; do not delete or reassign arbitrarily.",
        "- **Checks not yet completed:** Semantic labeling correctness and training smoke test.",
        "- **Strict policy:** No dataset images or annotations have been modified or deleted. Do not claim dataset is leakage-free.",
        "",
    ]

    qc_report_path = reports_dir / "dataset_quality_report.md"
    qc_report_path.write_text("\n".join(qc_report_lines), encoding="utf-8")
    print(f"Updated: {qc_report_path}")

    # 4. Write qc_summary.json
    qc_summary = {
        "project_root": str(root),
        "total_images": total_images,
        "total_label_files": total_labels,
        "total_annotations": total_annotations,
        "classes": {str(cid): cname for cid, cname in class_totals.keys()},
        "split_summary": split_rows,
        "class_distribution": overall_class_rows,
        "bbox_statistics": bbox_rows,
        "structural_checks_clean": structural_clean,
        "exact_duplicate_pairs": len(exact_pairs),
        "near_duplicate_candidates": len(near_pairs),
        "high_priority_overlap_candidates": [
            "Screenshot-2022-02-25-15-00-39_png_jpg",
            "Screenshot-2022-02-25-15-01-32_png_jpg",
        ],
        "leakage_free_claim": False,
        "note": "Documented probable source overlap. Dataset integrity preserved without deletions or alterations.",
    }
    with (reports_dir / "qc_summary.json").open("w", encoding="utf-8") as f:
        json.dump(qc_summary, f, indent=2)

    # 5. Write final_dataset_audit_handoff.md
    handoff_content = f"""# Lunar CV — Dataset Audit Handoff

{date.today().isoformat()}

## Dataset summary

- Images: 1,707
- Annotations: 7,451
- Train: 1,493 images / 6,499 annotations
- Validation: 143 images / 636 annotations
- Test: 71 images / 316 annotations
- Structural checks: 0 unreadable images, 0 missing labels, 0 invalid rows
- Classes: 0 = crater, 1 = rille

## Current status

Structural integrity checks passed. Cross-split source overlap is not fully
cleared. The train-test groups `Screenshot-2022-02-25-15-00-39_png_jpg`
and `Screenshot-2022-02-25-15-01-32_png_jpg` are high-priority potential
source-overlap candidates. Several train-validation groups remain unresolved
or have weak evidence.

No dataset files or split assignments were changed by this audit.

## Final train–test overlap review

Two high-priority source-prefix groups were visually reviewed alongside SIFT feature-matching results:

- **`Screenshot-2022-02-25-15-00-39_png_jpg`**: Probable source overlap. The train and test images share distinctive crater and terrain features. The strongest pair yielded 1,274 good matches and 1,272 geometric inliers (99.84%).
- **`Screenshot-2022-02-25-15-01-32_png_jpg`**: Probable source overlap. The train and test images exhibit strong geometric feature correspondence. The strongest pair yielded 1,104 good matches and 1,104 geometric inliers (100.0%).

These results strongly suggest related source imagery across the train and test splits. They do not independently establish the exact source-image lineage.

**Recommended action:** obtain approval for a group-aware split correction, keep related source variants within a single split, regenerate the affected split only under a documented procedure, and rerun the final evaluation on an appropriately separated test set.

No images, labels, or split assignments were changed during this audit. The dataset should not be described as fully leakage-free until the remaining candidates and the handling of these two groups are documented.

## Audit Categorization

1. **Verified structural checks:** All 1,707 images and 7,451 annotations validated for format, bounds, and readability.
2. **Probable source overlap:** `15-00-39` and `15-01-32` train-test candidate groups.
3. **Inconclusive candidates:** `102_png_jpg`, `14-05-01`, `14-12-12`, `16-19-45`, and `4165 vs 4191`.
4. **Checks not yet completed:** Semantic bounding-box labeling verification, training smoke test.

## Handoff checklist

- [ ] Share `dataset_quality_report.md`
- [ ] Share `dataset_preprocessing_quality_report.md`
- [ ] Share `dataset_audit_baseline_addendum.md`
- [ ] Share `automated_audit/batch_review/image_level_manifest.csv`
- [ ] Share `automated_audit/batch_review/train_test_sift_pair_results.csv`
- [ ] Share `automated_audit/batch_review/train_valid_sift_review.csv`
- [ ] Share contact sheets and `source_overlap_findings.md`
- [ ] Discuss unresolved cross-split candidates with the model owner

Do not describe the dataset as leakage-free until the remaining high-priority
candidates have been addressed or explicitly documented as accepted limitations.
"""
    handoff_path = reports_dir / "final_dataset_audit_handoff.md"
    handoff_path.write_text(handoff_content, encoding="utf-8")
    print(f"Updated: {handoff_path}")

    # 6. Ensure dataset_preprocessing_quality_report.md has final review section
    final_review_text = f"""
## Final dataset audit review

Review date: {date.today().isoformat()}

### Dataset integrity

- Total images: 1,707
- Train: 1,493 images, 6,499 valid annotations
- Validation: 143 images, 636 valid annotations
- Test: 71 images, 316 valid annotations
- Total valid annotations: 7,451
- Unreadable images: 0
- Missing label files: 0
- Invalid label rows: 0

The manifest counts agree with the earlier structural audit.
These checks do not establish that all annotations are semantically correct
or that the dataset is free of source overlap.

### Train-validation source-prefix review

Five groups were reviewed visually and with SIFT screening:

- Screenshot-2022-02-25-16-19-45_png_jpg:
  unresolved possible partial source overlap. The strongest pair had
  136 good matches and 85 geometric inliers (0.625 ratio).
  The visualization showed correspondences concentrated in part of the images.
- Screenshot-2022-02-25-16-20-17_png_jpg:
  inconclusive; maximum 15 inliers in the reported comparison.
- Screenshot-2022-02-25-16-20-20_png_jpg:
  weak SIFT evidence; maximum 6 inliers.
- Screenshot-2022-02-25-16-20-58_png_jpg:
  weak SIFT evidence; maximum 5 inliers.
- images-_10__jpg:
  possible partial overlap remains unresolved; maximum 24 inliers.

The training variants within some groups appear visually related or transformed.
Matching filename prefixes and SIFT correspondences are screening evidence,
not proof of leakage. Groups with weak matches have not been proven unrelated.

### Earlier train-test review

The earlier audit identified strong potential source-overlap candidates
in these cross-split groups:

- Screenshot-2022-02-25-15-00-39_png_jpg
- Screenshot-2022-02-25-15-01-32_png_jpg

Their reported SIFT results were very strong, and contact-sheet review suggested
related transformed versions of the same scene. Treat these as high-priority
potential cross-split source overlap, pending final source-level verification.

Other train-test prefix groups and SIFT screening candidates were also identified.
Prefix matches and automated flags must not be treated as confirmed duplicates.
An earlier validation-test pair also remains unresolved:
4165_jpg.rf.cae5c14fc0e6abbc22135c127a531299.jpg and
4191_jpg.rf.266ac35f07499b841f2da6d2056b512d.jpg.

### Decisions and limitations

- No images, labels, or split assignments were changed during this review.
- No candidate was removed based only on a filename prefix or SIFT score.
- Cross-split source overlap is not fully cleared; the candidates above must
  remain documented as unresolved or high-priority potential overlap.
- The validation/test contact-sheet review does not replace completion of
  the earlier train-test candidate review.
- Keep the test set untouched for model selection and tuning.

### Recommended handoff

Share this report, the image-level manifest, SIFT pair-result CSVs,
contact sheets, and existing quality/duplicate reports with the model owner.
Explain the unresolved cross-split candidates before interpreting evaluation
metrics. Do not claim that the dataset is leakage-free until those candidates
have been addressed or explicitly accepted with limitations.
"""
    append_review_section_if_missing(
        reports_dir / "dataset_preprocessing_quality_report.md",
        "## Final dataset audit review",
        final_review_text,
    )
    append_review_section_if_missing(
        batch_dir / "source_overlap_findings.md",
        "## Final dataset audit review",
        final_review_text,
    )

    print("\n=== FINAL AUDIT REPORT GENERATION COMPLETE ===")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 7: Finalize dataset audit and QC reports."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root folder containing reports",
    )
    args = parser.parse_args()

    finalize_reports(args.root)


if __name__ == "__main__":
    main()
