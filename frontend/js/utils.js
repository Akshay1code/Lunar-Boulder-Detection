/**
 * DOM and Formatting Utilities
 * Secure DOM builders preventing innerHTML injection.
 */

export const $ = (id) => document.getElementById(id);

export function formatBytes(bytes, decimals = 1) {
  if (!bytes || bytes === 0) return "0 B";
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

export function truncateFilename(filename, maxLen = 32) {
  if (!filename || filename.length <= maxLen) return filename;
  const extIndex = filename.lastIndexOf(".");
  if (extIndex === -1) return `${filename.slice(0, maxLen - 3)}...`;
  const ext = filename.slice(extIndex);
  const base = filename.slice(0, extIndex);
  const partLen = Math.floor((maxLen - ext.length - 3) / 2);
  return `${base.slice(0, partLen)}...${base.slice(-partLen)}${ext}`;
}

export function announce(message) {
  const liveRegion = $("liveAnnouncer");
  if (liveRegion) {
    liveRegion.textContent = message;
  }
}

export function clearChildren(element) {
  while (element && element.firstChild) {
    element.removeChild(element.firstChild);
  }
}

export function createEl(tag, options = {}) {
  const el = document.createElement(tag);
  if (options.className) el.className = options.className;
  if (options.text) el.textContent = options.text;
  if (options.attrs) {
    for (const [key, val] of Object.entries(options.attrs)) {
      if (val !== null && val !== undefined) {
        el.setAttribute(key, val);
      }
    }
  }
  return el;
}
