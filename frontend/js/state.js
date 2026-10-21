/**
 * Central State Store & Guarded Transition State Machine
 */

const VALID_PHASES = [
  "empty",
  "image_selected",
  "image_confirmed",
  "validating",
  "accepted",
  "rejected",
  "ambiguous",
  "validation_error",
  "model_selection",
  "analysing",
  "results_ready",
  "inference_error",
];

const INITIAL_STATE = {
  phase: "empty",
  file: null,
  fileMeta: { name: "", size: 0, width: 0, height: 0, type: "" },
  fileFingerprint: "",
  previewURL: null,
  imageId: null,
  validation: null,
  selectedModel: "yolo11n", // default
  modelAvailability: { yolo11n: false, unet: false },
  result: null,
  error: null,
  requestToken: 0,
};

class StateStore {
  constructor() {
    this.state = { ...INITIAL_STATE };
    this.listeners = new Set();
  }

  getState() {
    return this.state;
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  _notify() {
    for (const listener of this.listeners) {
      try {
        listener(this.state);
      } catch (err) {
        console.error("State listener error:", err);
      }
    }
  }

  transition(nextPhase, payload = {}) {
    if (!VALID_PHASES.includes(nextPhase)) {
      console.warn(`[StateStore] Invalid phase requested: ${nextPhase}`);
      return false;
    }

    const prevPhase = this.state.phase;
    console.log(`[StateStore] Transition: ${prevPhase} -> ${nextPhase}`);

    this.state = {
      ...this.state,
      ...payload,
      phase: nextPhase,
    };

    this._notify();
    return true;
  }

  setFile(file, meta, fingerprint, previewURL) {
    // Revoke old object URL if exists
    if (this.state.previewURL && this.state.previewURL !== previewURL) {
      URL.revokeObjectURL(this.state.previewURL);
    }

    this.state = {
      ...this.state,
      file,
      fileMeta: meta,
      fileFingerprint: fingerprint,
      previewURL,
      imageId: null,
      validation: null,
      result: null,
      error: null,
      requestToken: this.state.requestToken + 1,
    };
    this.transition("image_selected");
  }

  reset() {
    if (this.state.previewURL) {
      URL.revokeObjectURL(this.state.previewURL);
    }
    this.state = {
      ...INITIAL_STATE,
      modelAvailability: this.state.modelAvailability, // preserve checked models
      requestToken: this.state.requestToken + 1,
    };
    this.transition("empty");
  }

  incrementToken() {
    this.state.requestToken += 1;
    return this.state.requestToken;
  }
}

export const store = new StateStore();
