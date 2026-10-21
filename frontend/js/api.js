/**
 * REST API Client Module
 * Provides fetch wrappers, timeout AbortController handling, and error normalization.
 */

const API_BASE = window.API_BASE || (
  window.location.origin && window.location.origin.includes(":5000")
    ? ""
    : "http://127.0.0.1:5000"
);

export async function fetchWithTimeout(endpoint, options = {}, timeoutMs = 60000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const signal = options.signal || controller.signal;

  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      signal,
    });
    clearTimeout(timeoutId);

    const isJson = (res.headers.get("content-type") || "").includes("application/json");
    const data = isJson ? await res.json() : null;

    if (!res.ok) {
      const errMessage = (data && (data.error?.message || data.error || data.explanation)) || `HTTP ${res.status}: ${res.statusText}`;
      const errorObj = new Error(errMessage);
      errorObj.status = res.status;
      errorObj.data = data;
      throw errorObj;
    }

    return data;
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      const abortErr = new Error("Request timed out or was cancelled by user.");
      abortErr.isTimeout = true;
      throw abortErr;
    }
    throw err;
  }
}

export async function getConfig(signal) {
  return fetchWithTimeout("/api/config", { method: "GET", signal }, 10000);
}

export async function getModels(signal) {
  return fetchWithTimeout("/api/models", { method: "GET", signal }, 10000);
}

export async function getHealth(signal) {
  return fetchWithTimeout("/api/health", { method: "GET", signal }, 10000);
}

export async function validateImage(file, thresholds = {}, signal) {
  const fd = new FormData();
  fd.append("image", file);
  if (thresholds.acceptance) fd.append("acceptance_threshold", thresholds.acceptance);
  if (thresholds.ambiguity) fd.append("ambiguity_threshold", thresholds.ambiguity);

  return fetchWithTimeout("/api/validate", {
    method: "POST",
    body: fd,
    signal,
  }, 60000);
}

export async function runDetection(file, options = {}, signal) {
  const fd = new FormData();
  fd.append("image", file);
  if (options.conf !== undefined) fd.append("conf", options.conf);
  if (options.iou !== undefined) fd.append("iou", options.iou);

  return fetchWithTimeout("/api/detect", {
    method: "POST",
    body: fd,
    signal,
  }, 90000);
}

export async function runSegmentation(file, options = {}, signal) {
  const fd = new FormData();
  fd.append("image", file);
  if (options.threshold !== undefined) fd.append("threshold", options.threshold);

  return fetchWithTimeout("/api/segment", {
    method: "POST",
    body: fd,
    signal,
  }, 90000);
}
