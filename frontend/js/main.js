/**
 * main.js — Lunar Mission Dashboard (4-Step Auto-Run)
 *
 * Flow: Upload → Preview → Validate → Results
 *
 * After validation passes the system automatically dispatches BOTH
 * YOLO11s detection AND U-Net segmentation in parallel and renders
 * results for whichever models are available.
 */

import { store } from "./state.js";
import { getHealth, runDetection, runSegmentation } from "./api.js";
import { initUpload } from "./upload.js";
import { runValidationStep } from "./validation.js";
import { getSceneController } from "./scene.js";
import { $, formatBytes, truncateFilename, clearChildren, createEl } from "./utils.js";

/* ══════════════════════════════════════════════
   HEALTH CHECK
══════════════════════════════════════════════ */
const beacon      = $("statusBeacon");
const statusLabel = $("statusLabel");

async function checkHealth() {
  try {
    const data = await getHealth();
    const yoloOk = !!(data.detection?.available);
    const unetOk = !!(data.segmentation?.available);

    store.state.modelAvailability = { yolo11n: yoloOk, unet: unetOk };

    if (yoloOk && unetOk) {
      beacon.className = "status-beacon is-online";
      statusLabel.textContent = "All systems ready";
    } else if (yoloOk || unetOk) {
      beacon.className = "status-beacon is-partial";
      statusLabel.textContent = yoloOk
        ? "Detection ready · Segmentation unavailable"
        : "Segmentation ready · Detection unavailable";
    } else {
      beacon.className = "status-beacon is-offline";
      statusLabel.textContent = "Models unavailable";
    }
  } catch {
    beacon.className = "status-beacon is-offline";
    statusLabel.textContent = "Server unreachable";
    store.state.modelAvailability = { yolo11n: false, unet: false };
  }
}

/* ══════════════════════════════════════════════
   STEPPER  (4 steps)
══════════════════════════════════════════════ */
const STEP_COUNT = 4;
const panels   = Array.from({ length: STEP_COUNT }, (_, i) => $(`panel${i + 1}`));
const stepBtns = Array.from({ length: STEP_COUNT }, (_, i) => $(`stepBtn${i + 1}`));

let currentStep = 1;

function goToStep(step) {
  if (step < 1 || step > STEP_COUNT) return;
  currentStep = step;

  panels.forEach((p, i) => {
    if (!p) return;
    p.classList.toggle("is-hidden", i + 1 !== step);
  });

  stepBtns.forEach((btn, i) => {
    if (!btn) return;
    const s = i + 1;
    if (s === step) btn.setAttribute("aria-current", "step");
    else            btn.removeAttribute("aria-current");
    // Unlock all steps up to current
    if (s <= step) {
      btn.disabled = false;
      btn.classList.add("is-clickable");
    }
  });

  window.scrollTo({ top: 0, behavior: "smooth" });
}

// Allow clicking earlier steps to jump back
stepBtns.forEach((btn, i) => {
  btn?.addEventListener("click", () => {
    const s = i + 1;
    if (btn.disabled || s > currentStep) return;
    if (s === 1) { store.reset(); }
    goToStep(s);
  });
});

/* ══════════════════════════════════════════════
   STEP 1 — UPLOAD
══════════════════════════════════════════════ */
initUpload({
  allowed_extensions: [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"],
  max_upload_bytes: 16 * 1024 * 1024,
});

/* ══════════════════════════════════════════════
   STEP 2 — PREVIEW
══════════════════════════════════════════════ */
const previewImg      = $("previewImg");
const previewFileMeta = $("previewFileMeta");

function renderPreview(state) {
  if (!state.previewURL || !state.file) return;
  previewImg.src = state.previewURL;
  previewImg.alt = `Preview of ${state.fileMeta.name}`;
  previewFileMeta.textContent =
    `${truncateFilename(state.fileMeta.name, 40)} — ${formatBytes(state.fileMeta.size)} — ${state.fileMeta.width}×${state.fileMeta.height}px`;
}

$("btnConfirmImage")?.addEventListener("click", () => {
  store.transition("image_confirmed");
  goToStep(3);
  runValidationStep();
});

$("btnChangeImage")?.addEventListener("click", () => {
  store.reset();
  goToStep(1);
});

/* ══════════════════════════════════════════════
   STEP 3 — VALIDATION (handled by validation.js)
   validation.js transitions to "accepted" / "rejected" / "ambiguous"
   and for "accepted" it calls store.transition("model_selection")
   which we intercept here to go to step 4 and auto-run.
══════════════════════════════════════════════ */

/* ══════════════════════════════════════════════
   THRESHOLD SLIDERS
══════════════════════════════════════════════ */
const confSlider    = $("confSlider");
const confVal       = $("confVal");
const iouSlider     = $("iouSlider");
const iouVal        = $("iouVal");
const segThreshSl   = $("segThreshSlider");
const segThreshVal  = $("segThreshVal");

confSlider?.addEventListener("input", () => {
  confVal.textContent = parseFloat(confSlider.value).toFixed(2);
});
iouSlider?.addEventListener("input", () => {
  iouVal.textContent = parseFloat(iouSlider.value).toFixed(2);
});
segThreshSl?.addEventListener("input", () => {
  segThreshVal.textContent = parseFloat(segThreshSl.value).toFixed(2);
});

$("btnRerun")?.addEventListener("click", () => {
  const state = store.getState();
  if (state.file) runBothModels(state);
});

/* ══════════════════════════════════════════════
   STEP 4 — AUTO-RUN BOTH MODELS IN PARALLEL
══════════════════════════════════════════════ */

// ── Detection helpers ──────────────────────
const detectSpinner    = $("detectSpinner");
const detectStatusText = $("detectStatusText");
const detectMetrics    = $("detectMetricsGrid");
const detectBody       = $("detectResultBody");
const detectErrorEl    = $("detectError");
const detectUnavail    = $("detectUnavailable");
const detectDownload   = $("btnDownloadDetect");

let lastDetectData = null;

function detectSetState(state) {
  // state: "loading" | "done" | "error" | "unavailable"
  detectSpinner.style.display  = state === "loading"  ? "block" : "none";
  detectMetrics.style.display  = state === "done"     ? "flex"  : "none";
  detectBody.style.display     = state === "done"     ? "block" : "none";
  detectErrorEl.style.display  = state === "error"    ? "block" : "none";
  detectUnavail.style.display  = state === "unavailable" ? "flex" : "none";
  detectDownload.hidden        = state !== "done";

  const labels = {
    loading: "Running…",
    done: "Complete",
    error: "Failed",
    unavailable: "Unavailable",
  };
  detectStatusText.textContent = labels[state] || "";
  detectStatusText.style.color = {
    loading: "var(--text-muted)",
    done: "var(--success)",
    error: "var(--danger)",
    unavailable: "var(--text-muted)",
  }[state] || "var(--text-muted)";
}

function chipSet(chipEl, val, lbl) {
  if (!chipEl) return;
  const v = chipEl.querySelector(".chip-val");
  const l = chipEl.querySelector(".chip-lbl");
  if (v) v.textContent = val;
  if (l) l.textContent = lbl;
}

function clsClass(name) {
  const n = (name || "").toLowerCase();
  if (n === "crater")  return "cls-crater";
  if (n === "rille")   return "cls-rille";
  if (n === "boulder") return "cls-boulder";
  return "cls-default";
}

function renderDetectionResult(data, file) {
  chipSet($("metricCount"),   String(data.count ?? 0),                        "Detections");
  chipSet($("metricTime"),    `${data.inference_time_ms ?? "—"} ms`,           "Inference");
  chipSet($("metricClasses"), String(Object.keys(data.class_summary || {}).length), "Classes found");
  chipSet($("metricSize"),    `${data.original_size?.[0]}×${data.original_size?.[1]}`, "Image size");

  $("origImgDetect").src = URL.createObjectURL(file);
  $("annotatedImg").src  = data.annotated_image || "";

  const tbody = $("detectTableBody");
  clearChildren(tbody);
  if (!data.detections?.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="4" style="text-align:center;color:var(--text-muted);padding:20px;">No detections above confidence threshold.</td>`;
    tbody.appendChild(tr);
  } else {
    data.detections.forEach((det, i) => {
      const [x1, y1, x2, y2] = det.bbox_xyxy || [];
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="color:var(--text-muted);">${i + 1}</td>
        <td><span class="cls-badge ${clsClass(det.class_name)}">${det.class_name}</span></td>
        <td style="font-family:var(--font-mono);">${(det.confidence * 100).toFixed(1)}%</td>
        <td style="font-family:var(--font-mono);font-size:var(--text-xs);">${x1}, ${y1}, ${x2}, ${y2}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  const summary = $("classSummary");
  clearChildren(summary);
  Object.entries(data.class_summary || {}).forEach(([cls, cnt]) => {
    const pill = document.createElement("div");
    pill.className = "summary-pill";
    pill.appendChild(createEl("span", { className: `cls-badge ${clsClass(cls)}`, text: cls }));
    pill.appendChild(createEl("span", { text: String(cnt) }));
    summary.appendChild(pill);
  });

  lastDetectData = data;
  detectSetState("done");
}

// ── Segmentation helpers ───────────────────
const segSpinner    = $("segSpinner");
const segStatusText = $("segStatusText");
const segMetrics    = $("segMetricsGrid");
const segBody       = $("segResultBody");
const segErrorEl    = $("segError");
const segUnavail    = $("segUnavailable");
const segDownload   = $("btnDownloadSeg");

let lastSegData = null;

function segSetState(state) {
  segSpinner.style.display  = state === "loading"     ? "block" : "none";
  segMetrics.style.display  = state === "done"        ? "flex"  : "none";
  segBody.style.display     = state === "done"        ? "block" : "none";
  segErrorEl.style.display  = state === "error"       ? "block" : "none";
  segUnavail.style.display  = state === "unavailable" ? "flex"  : "none";
  segDownload.hidden         = state !== "done";

  const labels = { loading: "Running…", done: "Complete", error: "Failed", unavailable: "Unavailable" };
  segStatusText.textContent = labels[state] || "";
  segStatusText.style.color = {
    loading: "var(--text-muted)",
    done: "var(--success)",
    error: "var(--danger)",
    unavailable: "var(--text-muted)",
  }[state] || "var(--text-muted)";
}

function renderSegmentationResult(data) {
  chipSet($("segCoverage"),  `${data.coverage_percent?.toFixed(2) ?? "—"}%`, "Coverage");
  chipSet($("segTime"),      `${data.inference_time_ms ?? "—"} ms`,          "Inference");
  chipSet($("segThreshChip"), String(data.threshold?.toFixed(2) ?? "—"),     "Threshold");
  chipSet($("segSize"),      `${data.original_size?.[0]}×${data.original_size?.[1]}`, "Image size");

  $("origImgSeg").src  = data.original_image || "";
  $("overlayImg").src  = data.overlay_image  || "";
  $("maskImg").src     = data.mask_image     || "";

  const warn = $("segWarning");
  if (data.warning) {
    warn.textContent = `⚠️  ${data.warning}`;
    warn.style.display = "block";
  } else {
    warn.style.display = "none";
  }

  lastSegData = data;
  segSetState("done");
}

// ── Parallel dispatch ──────────────────────
async function runBothModels(state) {
  const avail = store.state.modelAvailability;
  const conf      = parseFloat(confSlider?.value  ?? 0.25);
  const iou       = parseFloat(iouSlider?.value   ?? 0.45);
  const threshold = parseFloat(segThreshSl?.value ?? 0.5);
  const subtitle  = $("resultsSubtitle");

  lastDetectData = null;
  lastSegData    = null;

  const modelsList = [];
  if (avail.yolo11n) modelsList.push("YOLO11s Detection");
  if (avail.unet)    modelsList.push("U-Net Segmentation");
  if (subtitle) {
    subtitle.textContent = modelsList.length
      ? `Running: ${modelsList.join(" + ")}…`
      : "No models are currently available.";
  }

  // ─── Detection ───────────────────────────
  if (avail.yolo11n) {
    detectSetState("loading");
    runDetection(state.file, { conf, iou })
      .then(data => {
        if (!data.success) throw new Error(data.error || "Detection failed");
        renderDetectionResult(data, state.file);
      })
      .catch(err => {
        detectErrorEl.textContent = `⚠️  Detection error: ${err.message}`;
        detectSetState("error");
      });
  } else {
    detectSetState("unavailable");
  }

  // ─── Segmentation ─────────────────────────
  if (avail.unet) {
    segSetState("loading");
    runSegmentation(state.file, { threshold })
      .then(data => {
        if (!data.success) throw new Error(data.error || "Segmentation failed");
        renderSegmentationResult(data);
        if (subtitle && !subtitle.textContent.includes("Running")) {
          subtitle.textContent = "Analysis complete.";
        }
      })
      .catch(err => {
        segErrorEl.textContent = `⚠️  Segmentation error: ${err.message}`;
        segSetState("error");
      });
  } else {
    segSetState("unavailable");
  }

  // Update subtitle when both done
  Promise.allSettled([
    avail.yolo11n ? runDetection(state.file, { conf, iou }).catch(() => null) : Promise.resolve(null),
    avail.unet    ? runSegmentation(state.file, { threshold }).catch(() => null) : Promise.resolve(null),
  ]).then(() => {
    if (subtitle && subtitle.textContent.startsWith("Running")) {
      subtitle.textContent = "Analysis complete.";
    }
  });
}

// ── Download buttons ───────────────────────
detectDownload?.addEventListener("click", () => {
  if (!lastDetectData?.annotated_image) return;
  const a = document.createElement("a");
  a.href = lastDetectData.annotated_image;
  a.download = `detection_${store.getState().fileMeta?.name?.replace(/\.[^.]+$/, "") || "result"}.png`;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
});

segDownload?.addEventListener("click", () => {
  if (!lastSegData?.overlay_image) return;
  const a = document.createElement("a");
  a.href = lastSegData.overlay_image;
  a.download = `segmentation_overlay_${store.getState().fileMeta?.name?.replace(/\.[^.]+$/, "") || "result"}.png`;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
});

$("btnNewMission")?.addEventListener("click", () => {
  store.reset();
  lastDetectData = null;
  lastSegData    = null;
  goToStep(1);
});

/* ══════════════════════════════════════════════
   STATE SUBSCRIBER — drives step navigation
══════════════════════════════════════════════ */
store.subscribe((state) => {
  const phase = state.phase;

  if (phase === "image_selected") {
    renderPreview(state);
    stepBtns[1]?.removeAttribute("disabled");
    stepBtns[1]?.classList.add("is-clickable");
    goToStep(2);
    return;
  }

  // "model_selection" is what validation.js transitions to when accepted
  if (phase === "model_selection") {
    stepBtns[3]?.removeAttribute("disabled");
    stepBtns[3]?.classList.add("is-clickable");
    goToStep(4);
    // Reset UI state before running
    detectSetState("loading");
    segSetState("loading");
    runBothModels(state);
    return;
  }

  if (phase === "empty") {
    for (let i = 1; i < STEP_COUNT; i++) {
      stepBtns[i]?.setAttribute("disabled", "");
      stepBtns[i]?.classList.remove("is-clickable");
      stepBtns[i]?.removeAttribute("aria-current");
    }
    goToStep(1);
  }
});

/* ══════════════════════════════════════════════
   INIT
══════════════════════════════════════════════ */
goToStep(1);
checkHealth();
setInterval(checkHealth, 30_000);
