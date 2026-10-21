/**
 * Step 3: Validation View & Gate Execution
 * Manages validation UI stages, decision cards, and retry/replace actions.
 */

import { $, clearChildren, createEl, announce } from "./utils.js";
import { store } from "./state.js";
import { validateImage } from "./api.js";
import { getSceneController } from "./scene.js";

let currentAbortController = null;

export async function runValidationStep() {
  const state = store.getState();
  if (!state.file) {
    store.transition("empty");
    return;
  }

  // Cancel prior in-flight request if present
  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();
  const requestToken = store.incrementToken();

  store.transition("validating");

  const scene = getSceneController();
  scene.start("validation", "Validating lunar surface imagery authenticity...");

  renderValidatingState();

  try {
    const result = await validateImage(state.file, {}, currentAbortController.signal);

    // Stale response guard
    if (store.getState().requestToken !== requestToken) {
      console.log("[Validation] Ignored stale validation response.");
      return;
    }

    const decision = result.decision || (result.is_permitted ? "accepted" : "rejected");
    const label = result.label || "uncertain";
    const confidence = result.confidence || 0.0;

    store.transition(decision, {
      validation: result,
      imageId: result.image_id,
    });

    renderValidationOutcome(decision, result);
  } catch (err) {
    if (store.getState().requestToken !== requestToken) return;

    if (err.isTimeout) {
      renderErrorState("Validation request timed out. Please check backend connection and retry.");
    } else {
      renderErrorState(err.message || "Failed to reach backend validation gate.");
    }
    store.transition("validation_error", { error: err.message });
  }
}

function renderValidatingState() {
  const container = $("validationStatusContainer");
  if (!container) return;
  clearChildren(container);

  const card = createEl("div", { className: "status-card is-elevated" });
  
  const title = createEl("h3", { text: "Orbital Image Screening Gate" });
  const desc = createEl("p", {
    text: "Analyzing image structural integrity and running convolutional lunar terrain classifier..."
  });

  const spinnerRow = createEl("div", {
    className: "stepper-list",
    attrs: { style: "display: flex; align-items: center; gap: 12px; margin-top: 16px;" }
  });
  const spinner = createEl("div", { className: "spinner" });
  const spinnerText = createEl("span", {
    text: "Executing server-side validation checks...",
    className: "brand-sub"
  });
  spinnerRow.appendChild(spinner);
  spinnerRow.appendChild(spinnerText);

  card.appendChild(title);
  card.appendChild(desc);
  card.appendChild(spinnerRow);

  container.appendChild(card);
}

function renderValidationOutcome(decision, result) {
  const container = $("validationStatusContainer");
  if (!container) return;
  clearChildren(container);

  const confPercent = `${(result.confidence * 100).toFixed(1)}%`;
  const explanation = result.explanation || result.reason || "";

  const card = createEl("div", {
    className: `status-card state-${decision}`
  });

  if (decision === "accepted") {
    const heading = createEl("h3", { text: "✓ Lunar Imagery Confirmed" });
    const copy = createEl("p", {
      text: `Surface structure matches lunar terrain (Classifier Confidence: ${confPercent}). Both available models will now run automatically.`
    });
    card.appendChild(heading);
    card.appendChild(copy);

    // Checks breakdown
    if (result.checks && result.checks.length > 0) {
      const table = createChecksTable(result.checks);
      card.appendChild(table);
    }

    const actionRow = createEl("div", {
      attrs: { style: "display: flex; gap: 16px; margin-top: 16px;" }
    });
    const continueBtn = createEl("button", {
      className: "btn btn-primary",
      text: "Run Analysis on Both Models →",
      attrs: { id: "btnProceedToModels" }
    });
    continueBtn.addEventListener("click", () => {
      store.transition("model_selection");
    });
    actionRow.appendChild(continueBtn);
    card.appendChild(actionRow);

    announce(`Image Accepted. Confidence ${confPercent}. Continue to model selection.`);
  } else if (decision === "ambiguous") {
    const heading = createEl("h3", { text: "⚠️ Ambiguous Imagery — Review Required" });
    const copy = createEl("p", {
      text: `Classification confidence (${confPercent}) is inconclusive. The image could not be verified as authentic lunar terrain. Deep learning inference is blocked.`
    });
    card.appendChild(heading);
    card.appendChild(copy);

    const actionRow = createEl("div", {
      attrs: { style: "display: flex; gap: 16px; margin-top: 16px;" }
    });
    const replaceBtn = createEl("button", {
      className: "btn btn-secondary",
      text: "Try Another Image",
      attrs: { id: "btnTryAnotherAmbiguous" }
    });
    replaceBtn.addEventListener("click", () => store.reset());
    actionRow.appendChild(replaceBtn);
    card.appendChild(actionRow);

    announce(`Image Ambiguous. Confidence ${confPercent}. Model inference blocked.`);
  } else {
    // Rejected
    const heading = createEl("h3", { text: "✕ Non-Lunar Image Detected" });
    const copy = createEl("p", {
      text: explanation || "Input identified as non-lunar imagery. Analysis blocked to preserve pipeline integrity."
    });
    card.appendChild(heading);
    card.appendChild(copy);

    const actionRow = createEl("div", {
      attrs: { style: "display: flex; gap: 16px; margin-top: 16px;" }
    });
    const replaceBtn = createEl("button", {
      className: "btn btn-danger-outline",
      text: "Upload New Image",
      attrs: { id: "btnTryAnotherRejected" }
    });
    replaceBtn.addEventListener("click", () => store.reset());
    actionRow.appendChild(replaceBtn);
    card.appendChild(actionRow);

    announce(`Image Rejected. ${explanation}`);
  }

  container.appendChild(card);
}

function renderErrorState(errorMessage) {
  const container = $("validationStatusContainer");
  if (!container) return;
  clearChildren(container);

  const card = createEl("div", { className: "status-card state-rejected" });
  const heading = createEl("h3", { text: "Validation Service Error" });
  const copy = createEl("p", { text: errorMessage });
  const retryBtn = createEl("button", {
    className: "btn btn-secondary",
    text: "Retry Validation ⟳",
    attrs: { style: "margin-top: 12px;" }
  });
  retryBtn.addEventListener("click", () => runValidationStep());

  card.appendChild(heading);
  card.appendChild(copy);
  card.appendChild(retryBtn);
  container.appendChild(card);
}

function createChecksTable(checks) {
  const table = createEl("table", { className: "checks-table", attrs: { style: "margin-top: 12px;" } });
  for (const check of checks) {
    const tr = createEl("tr");
    const tdStatus = createEl("td", {
      text: check.status === "passed" ? "✓ PASSED" : (check.status === "needs_review" ? "⚠ REVIEW" : "✕ FAILED"),
      attrs: {
        style: `font-weight: 600; width: 110px; color: ${check.status === "passed" ? "var(--success)" : (check.status === "needs_review" ? "var(--warning)" : "var(--danger)")}`
      }
    });
    const tdName = createEl("td", {
      text: check.id.replace(/_/g, " ").toUpperCase(),
      attrs: { style: "font-family: var(--font-mono); font-size: 0.75rem;" }
    });
    const tdDetail = createEl("td", { text: check.detail });

    tr.appendChild(tdStatus);
    tr.appendChild(tdName);
    tr.appendChild(tdDetail);
    table.appendChild(tr);
  }
  return table;
}
