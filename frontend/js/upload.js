/**
 * Step 1: Upload Logic
 * Handles drag/drop, local format & integrity checks, and continues to Confirmation.
 */

import { $, formatBytes, announce } from "./utils.js";
import { store } from "./state.js";

export function initUpload(config) {
  const dropzone = $("dropzoneContainer");
  const fileInput = $("fileInput");
  const uploadError = $("uploadError");

  if (!dropzone || !fileInput) return;

  const allowedExts = config?.allowed_extensions || [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"];
  const maxBytes = config?.max_upload_bytes || 16 * 1024 * 1024;

  // Dropzone click & keyboard trigger
  // NOTE: fileInput must NOT be a child of dropzone — its click bubbles up and
  // would re-trigger this listener creating an infinite loop.
  dropzone.addEventListener("click", (e) => {
    // Guard: never re-trigger when the click comes FROM the file input itself
    if (e.target === fileInput) return;
    fileInput.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  // Drag and drop handlers
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("is-dragover");
  });

  dropzone.addEventListener("dragleave", (e) => {
    if (!dropzone.contains(e.relatedTarget)) {
      dropzone.classList.remove("is-dragover");
    }
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processSelectedFile(e.dataTransfer.files[0], allowedExts, maxBytes);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      processSelectedFile(fileInput.files[0], allowedExts, maxBytes);
    }
  });
}

async function processSelectedFile(file, allowedExts, maxBytes) {
  const uploadError = $("uploadError");
  if (uploadError) uploadError.textContent = "";

  if (!file) return;

  // 1. File size check
  if (file.size === 0) {
    showError("The selected file is empty (0 bytes). Please upload a valid image.");
    return;
  }
  if (file.size > maxBytes) {
    showError(`File size exceeds limit (${formatBytes(file.size)} > ${formatBytes(maxBytes)}).`);
    return;
  }

  // 2. Extension check
  const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
  if (!allowedExts.includes(ext)) {
    showError(`Unsupported file format '${ext}'. Please select a JPEG, PNG, BMP, TIFF, or WEBP image.`);
    return;
  }

  // 3. Local image decodability check via Image object / createImageBitmap
  try {
    const previewURL = URL.createObjectURL(file);
    const img = new Image();

    await new Promise((resolve, reject) => {
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Image decoding failed. File may be corrupt."));
      img.src = previewURL;
    });

    const meta = {
      name: file.name,
      size: file.size,
      width: img.naturalWidth,
      height: img.naturalHeight,
      type: file.type || ext,
    };

    // Calculate lightweight content fingerprint for client-state tracking
    const fingerprint = `${file.name}_${file.size}_${file.lastModified}`;

    store.setFile(file, meta, fingerprint, previewURL);
    announce(`Selected file ${file.name}, resolution ${meta.width} by ${meta.height} pixels.`);
  } catch (err) {
    showError(err.message || "Failed to decode image file. Please verify it is a valid image.");
  }
}

function showError(msg) {
  const uploadError = $("uploadError");
  if (uploadError) {
    uploadError.textContent = msg;
    uploadError.classList.remove("visually-hidden");
  }
  announce(`Upload Error: ${msg}`);
}
