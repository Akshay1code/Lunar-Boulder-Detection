"use strict";

/* ============================================================
   Lunar Boulder & Landslide Detection — Frontend Logic
   Pure vanilla JS — no frameworks
   ============================================================ */

/* ─── DOM references ─── */
const $ = id => document.getElementById(id);

const uploadZone       = $("uploadZone");
const fileInput        = $("fileInput");
const uploadFilename   = $("uploadFilename");
const uploadContent    = $("uploadContent");
const runBtn           = $("runBtn");
const downloadBtn      = $("downloadBtn");

const statusDot        = $("statusDot");
const statusLabel      = $("statusLabel");

const tabDetect        = $("tab-detect");
const tabSegment       = $("tab-segment");
const modelInfoDetect  = $("modelInfoDetect");
const modelInfoSegment = $("modelInfoSegment");
const detectOptions    = $("detectOptions");
const segmentOptions   = $("segmentOptions");

const confSlider       = $("confSlider");
const confVal          = $("confVal");
const iouSlider        = $("iouSlider");
const iouVal           = $("iouVal");
const segThreshSlider  = $("segThreshSlider");
const segThreshVal     = $("segThreshVal");
const segStatusInline  = $("segStatusInline");

const loadingOverlay   = $("loadingOverlay");
const loadingText      = $("loadingText");
const emptyState       = $("emptyState");
const errorState       = $("errorState");
const errorMessage     = $("errorMessage");

const resultDetect     = $("resultDetect");
const resultSegment    = $("resultSegment");

// Detection result elements
const origImgDetect    = $("origImgDetect");
const annotatedImg     = $("annotatedImg");
const detectTableBody  = $("detectTableBody");
const classSummary     = $("classSummary");
const metricCount      = $("metricCount");
const metricTime       = $("metricTime");
const metricClasses    = $("metricClasses");
const metricSize       = $("metricSize");
const detectClasses    = $("detectClasses");

// Segmentation result elements
const origImgSeg       = $("origImgSeg");
const overlayImg       = $("overlayImg");
const maskImg          = $("maskImg");
const segWarning       = $("segWarning");
const segCoverage      = $("segCoverage");
const segTime          = $("segTime");
const segThreshChip    = $("segThreshChip");
const segSize          = $("segSize");

const modelSelectSection = $("modelSelectSection");
const validationGateCard = $("validationGateCard");
const gateIcon           = $("gateIcon");
const gateTitle          = $("gateTitle");
const gateBadge          = $("gateBadge");
const gateExplanation    = $("gateExplanation");
const gateConfidence     = $("gateConfidence");
const gateDimensions     = $("gateDimensions");

/* ─── State ─── */
let selectedFile          = null;
let activeModel           = "detect";  // "detect" | "segment"
let lastResult            = null;      // last successful response data
let detAvailable          = false;
let segAvailable          = false;
let isValidationPermitted = false;

const API_BASE = window.API_BASE || (
  window.location.origin && window.location.origin.includes(":5000")
    ? ""
    : "http://127.0.0.1:5000"
);

/* ============================================================
   HEALTH CHECK
   ============================================================ */
async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();

    detAvailable = !!(data.detection && data.detection.available);
    segAvailable = !!(data.segmentation && data.segmentation.available);

    if (detAvailable && segAvailable) {
      setStatus("online", "All systems ready");
    } else if (detAvailable) {
      setStatus("partial", "Detection ready · Segmentation unavailable");
    } else if (segAvailable) {
      setStatus("partial", "Segmentation ready · Detection unavailable");
    } else {
      setStatus("offline", "Models unavailable");
    }

    // Update segmentation status in the model info panel
    segStatusInline.textContent = segAvailable ? "Available" : "Unavailable — checkpoint missing";
    segStatusInline.style.color = segAvailable ? "var(--accent-green)" : "var(--accent-red)";

    // Update class names if detection model is loaded
    if (data.detection && data.detection.classes) {
      const names = Object.values(data.detection.classes);
      if (names.length > 0) {
        detectClasses.textContent = names.join(" · ");
      }
    }

    updateRunButton();
  } catch {
    setStatus("offline", "Server unreachable");
    updateRunButton();
  }
}

function setStatus(cls, label) {
  statusDot.className   = `status-dot ${cls}`;
  statusLabel.textContent = label;
}

/* ============================================================
   FILE UPLOAD
   ============================================================ */
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

uploadZone.addEventListener("dragover", e => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", e => {
  if (!uploadZone.contains(e.relatedTarget)) uploadZone.classList.remove("drag-over");
});
uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

async function handleFile(file) {
  const ALLOWED = ["image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"];
  if (!ALLOWED.includes(file.type) && !file.name.match(/\.(jpe?g|png|bmp|tiff?|webp)$/i)) {
    showError(`Unsupported file type: ${file.type || file.name.split(".").pop()}.\nPlease upload a JPEG, PNG, BMP, or TIFF.`);
    resetGateUI();
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showError(`File too large (${(file.size / 1048576).toFixed(1)} MB). Maximum is 16 MB.`);
    resetGateUI();
    return;
  }
  selectedFile = file;
  uploadZone.classList.add("has-file");
  uploadFilename.textContent = `📎 ${file.name}  (${(file.size / 1024).toFixed(0)} KB)`;
  uploadFilename.hidden = false;

  // Reset results and lock models while validating
  hideAll();
  emptyState.classList.remove("hidden");
  lastResult = null;
  downloadBtn.hidden = true;
  isValidationPermitted = false;
  lockModelSelection(true);
  updateRunButton();

  // Trigger Validation Gate check
  await runValidationGate(file);
}

function resetGateUI() {
  validationGateCard.className = "validation-gate-card hidden";
  isValidationPermitted = false;
  lockModelSelection(true);
  updateRunButton();
}

function lockModelSelection(locked) {
  if (modelSelectSection) {
    modelSelectSection.classList.toggle("disabled-section", locked);
  }
}

async function runValidationGate(file) {
  validationGateCard.className = "validation-gate-card gate-checking";
  validationGateCard.classList.remove("hidden");
  gateIcon.textContent = "🔍";
  gateTitle.textContent = "Validating Lunar Imagery";
  gateBadge.textContent = "Screening…";
  gateExplanation.textContent = "Executing file integrity and lunar domain classification gate...";
  gateConfidence.textContent = "—";
  gateDimensions.textContent = "—";

  const fd = new FormData();
  fd.append("image", file);

  try {
    const res = await fetch(`${API_BASE}/api/validate`, { method: "POST", body: fd });
    const data = await res.json();

    if (!res.ok && res.status !== 400 && res.status !== 503) {
      showGateResult("rejected", "Error", "Validation endpoint returned an unexpected server error.", 0, "—");
      return;
    }

    const status = data.status || "rejected";
    const confidence = typeof data.confidence === "number" ? `${(data.confidence * 100).toFixed(1)}%` : "—";
    const dims = data.file_metadata && data.file_metadata.dimensions ? data.file_metadata.dimensions : "—";
    const explanation = data.explanation || data.error || "Validation check complete.";

    if (data.is_permitted && status === "accepted") {
      isValidationPermitted = true;
      showGateResult("accepted", "Accepted", explanation, confidence, dims);
      lockModelSelection(false);
    } else if (status === "ambiguous") {
      isValidationPermitted = false;
      showGateResult("ambiguous", "Ambiguous", explanation, confidence, dims);
      lockModelSelection(true);
    } else if (status === "unavailable") {
      isValidationPermitted = false;
      showGateResult("ambiguous", "Classifier Unavailable", explanation, "0%", dims);
      lockModelSelection(true);
    } else {
      isValidationPermitted = false;
      showGateResult("rejected", "Rejected", explanation, confidence, dims);
      lockModelSelection(true);
    }
  } catch (err) {
    isValidationPermitted = false;
    showGateResult("rejected", "Connection Error", "Failed to reach validation service: " + err.message, "—", "—");
    lockModelSelection(true);
  } finally {
    updateRunButton();
  }
}

function showGateResult(stateCls, badgeText, explanation, confText, dimText) {
  validationGateCard.className = `validation-gate-card gate-${stateCls}`;
  validationGateCard.classList.remove("hidden");
  if (stateCls === "accepted") {
    gateIcon.textContent = "✅";
  } else if (stateCls === "rejected") {
    gateIcon.textContent = "❌";
  } else {
    gateIcon.textContent = "⚠️";
  }
  gateTitle.textContent = "Lunar Validation Gate";
  gateBadge.textContent = badgeText;
  gateExplanation.textContent = explanation;
  gateConfidence.textContent = confText;
  gateDimensions.textContent = dimText;
}

/* ============================================================
   MODEL TABS
   ============================================================ */
tabDetect.addEventListener("click", () => {
  if (!isValidationPermitted) return;
  setModel("detect");
});
tabSegment.addEventListener("click", () => {
  if (!isValidationPermitted) return;
  setModel("segment");
});

function setModel(model) {
  activeModel = model;

  tabDetect.classList.toggle("active", model === "detect");
  tabDetect.setAttribute("aria-selected", model === "detect");
  tabSegment.classList.toggle("active", model === "segment");
  tabSegment.setAttribute("aria-selected", model === "segment");

  modelInfoDetect.classList.toggle("hidden", model !== "detect");
  modelInfoSegment.classList.toggle("hidden", model !== "segment");
  detectOptions.classList.toggle("hidden", model !== "detect");
  segmentOptions.classList.toggle("hidden", model !== "segment");

  // Hide results from the other model
  hideAll();
  emptyState.classList.remove("hidden");
  lastResult = null;
  downloadBtn.hidden = true;
  updateRunButton();
}

/* ============================================================
   SLIDERS
   ============================================================ */
confSlider.addEventListener("input", () => {
  confVal.textContent = parseFloat(confSlider.value).toFixed(2);
});
iouSlider.addEventListener("input", () => {
  iouVal.textContent = parseFloat(iouSlider.value).toFixed(2);
});
segThreshSlider.addEventListener("input", () => {
  segThreshVal.textContent = parseFloat(segThreshSlider.value).toFixed(2);
});

/* ============================================================
   RUN ANALYSIS
   ============================================================ */
function updateRunButton() {
  const modelOk = activeModel === "detect" ? detAvailable : segAvailable;
  runBtn.disabled = !selectedFile || !isValidationPermitted || !modelOk;
  if (!selectedFile) {
    runBtn.title = "Upload an image to begin.";
  } else if (!isValidationPermitted) {
    runBtn.title = "Image must pass Lunar Validation Gate before running analysis.";
  } else if (!modelOk) {
    runBtn.title = `${activeModel === "detect" ? "Detection" : "Segmentation"} model is not available.`;
  } else {
    runBtn.title = "";
  }
}

runBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  if (!selectedFile) return;
  if (runBtn.disabled) return;

  const model = activeModel;
  const endpoint = `${API_BASE}${model === "detect" ? "/api/detect" : "/api/segment"}`;

  const fd = new FormData();
  fd.append("image", selectedFile);

  if (model === "detect") {
    fd.append("conf", confSlider.value);
    fd.append("iou",  iouSlider.value);
  } else {
    fd.append("threshold", segThreshSlider.value);
  }

  // UI into loading state
  hideAll();
  loadingOverlay.classList.remove("hidden");
  loadingText.textContent = model === "detect"
    ? "Running YOLO11s object detection…"
    : "Running U-Net landslide segmentation…";
  runBtn.disabled = true;
  runBtn.classList.add("loading");

  try {
    const res = await fetch(endpoint, { method: "POST", body: fd });
    const data = await res.json();

    if (!res.ok || !data.success) {
      showError(data.error || `Server error (${res.status})`);
      return;
    }

    lastResult = { model, data };

    if (model === "detect") {
      showDetectionResult(data);
    } else {
      showSegmentationResult(data);
    }

    downloadBtn.hidden = false;
  } catch (err) {
    showError(`Network error: ${err.message}.\nCheck that the Flask server is running.`);
  } finally {
    loadingOverlay.classList.add("hidden");
    runBtn.disabled = false;
    runBtn.classList.remove("loading");
    updateRunButton();
  }
}

/* ============================================================
   DISPLAY RESULTS — DETECTION
   ============================================================ */
function showDetectionResult(data) {
  hideAll();
  resultDetect.classList.remove("hidden");

  // Images
  // The annotated image from YOLO is the only image we have after upload
  // We display the annotated image on the right
  // For the "original" we display the uploaded file directly
  const origUrl = URL.createObjectURL(selectedFile);
  origImgDetect.src = origUrl;
  annotatedImg.src  = data.annotated_image;

  // Metrics
  chipSet(metricCount,   data.count.toString(), "Detections");
  chipSet(metricTime,    `${data.inference_time_ms} ms`, "Inference time");
  chipSet(metricClasses, Object.keys(data.class_summary).length.toString(), "Classes detected");
  chipSet(metricSize,    `${data.original_size[0]}×${data.original_size[1]}`, "Image size");

  // Detection table
  detectTableBody.innerHTML = "";
  if (data.detections.length === 0) {
    detectTableBody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--text-muted);padding:20px;">No detections above confidence threshold.</td></tr>`;
  } else {
    data.detections.forEach((det, i) => {
      const [x1,y1,x2,y2] = det.bbox_xyxy;
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${i + 1}</td>
        <td><span class="cls-badge ${clsClass(det.class_name)}">${det.class_name}</span></td>
        <td>${(det.confidence * 100).toFixed(1)}%</td>
        <td>${x1}, ${y1}, ${x2}, ${y2}</td>
      `;
      detectTableBody.appendChild(row);
    });
  }

  // Class summary pills
  classSummary.innerHTML = "";
  Object.entries(data.class_summary).forEach(([cls, cnt]) => {
    const pill = document.createElement("div");
    pill.className = "summary-pill";
    pill.innerHTML = `<span class="cls-badge ${clsClass(cls)}">${cls}</span><span class="count">${cnt}</span>`;
    classSummary.appendChild(pill);
  });
}

function clsClass(name) {
  const n = name.toLowerCase();
  if (n === "crater")  return "cls-crater";
  if (n === "rille")   return "cls-rille";
  if (n === "boulder") return "cls-boulder";
  return "cls-default";
}

/* ============================================================
   DISPLAY RESULTS — SEGMENTATION
   ============================================================ */
function showSegmentationResult(data) {
  hideAll();
  resultSegment.classList.remove("hidden");

  origImgSeg.src  = data.original_image;
  overlayImg.src  = data.overlay_image;
  maskImg.src     = data.mask_image;

  chipSet(segCoverage,    `${data.coverage_percent.toFixed(2)}%`, "Pixel coverage");
  chipSet(segTime,        `${data.inference_time_ms} ms`,         "Inference time");
  chipSet(segThreshChip,  data.threshold.toFixed(2),              "Threshold used");
  chipSet(segSize,        `${data.original_size[0]}×${data.original_size[1]}`, "Image size");

  if (data.warning) {
    segWarning.textContent = `⚠️  ${data.warning}`;
    segWarning.classList.remove("hidden");
  } else {
    segWarning.classList.add("hidden");
  }
}

/* ============================================================
   DOWNLOAD
   ============================================================ */
downloadBtn.addEventListener("click", () => {
  if (!lastResult) return;
  const { model, data } = lastResult;

  let src, filename;
  if (model === "detect") {
    src      = data.annotated_image;
    filename = `detection_${selectedFile ? selectedFile.name.replace(/\.[^.]+$/, "") : "result"}.png`;
  } else {
    src      = data.overlay_image;
    filename = `segmentation_overlay_${selectedFile ? selectedFile.name.replace(/\.[^.]+$/, "") : "result"}.png`;
  }

  const a = document.createElement("a");
  a.href     = src;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

/* ============================================================
   UTILITIES
   ============================================================ */
function hideAll() {
  [emptyState, errorState, resultDetect, resultSegment].forEach(el => el.classList.add("hidden"));
}

function showError(msg) {
  hideAll();
  errorState.classList.remove("hidden");
  errorMessage.textContent = msg;
  downloadBtn.hidden = true;
  lastResult = null;
}

function chipSet(chipEl, val, lbl) {
  const valEl = chipEl.querySelector(".chip-val");
  const lblEl = chipEl.querySelector(".chip-lbl");
  if (valEl) valEl.textContent = val;
  if (lblEl) lblEl.textContent = lbl;
}

/* ============================================================
   INIT
   ============================================================ */
lockModelSelection(true);
checkHealth();
// Re-check health periodically in case models finish loading
setInterval(checkHealth, 30_000);
