# Lunar Boulder and Landslide Detection

> Deep Learning Research Dashboard — YOLO11s Object Detection + U-Net Landslide Segmentation  
> Pure HTML/CSS/JS frontend · Flask backend REST API · Python 3.11

---

## Project Architecture

The backend and frontend are fully decoupled. The frontend communicates with the backend exclusively via REST API endpoints (`/api/*`).

```text
Lunar-Boulder-Detection/
├── backend/
│   ├── app.py                        # Flask application entry point & static file serving
│   ├── routes/
│   │   └── api.py                    # REST endpoints (/api/health, /api/validate, /api/detect, /api/segment)
│   ├── models/
│   │   ├── detection/
│   │   │   └── inference.py          # YOLO11s checkpoint loader & detection runner
│   │   └── segmentation/
│   │       ├── unet_model.py         # U-Net architecture definition
│   │       └── inference.py          # U-Net checkpoint loader & segmentation runner
│   ├── services/
│   │   └── inference_service.py      # Unified model loading & inference coordinator
│   ├── validation/
│   │   ├── classifier.py             # LunarClassifierNet — CNN binary lunar domain classifier
│   │   ├── file_validator.py         # File integrity checks (magic bytes, size, dimensions)
│   │   └── gate.py                   # Three-tier decision gate (accepted / ambiguous / rejected)
│   ├── utils/
│   │   └── __init__.py
│   └── weights/
│       ├── yolo11s_best.pt           # Fine-tuned YOLO11s detection checkpoint (active)
│       ├── yolo11s.pt                # YOLO11s base weights
│       ├── unet_best.pth             # Trained U-Net segmentation checkpoint
│       └── lunar_classifier_best.pth # Trained lunar domain classifier checkpoint
├── frontend/
│   ├── index.html                    # Vanilla HTML5 dashboard (single-page, 4-step flow)
│   ├── css/
│   │   ├── tokens.css                # Design tokens (colours, spacing, typography, radii)
│   │   ├── base.css                  # CSS reset, layout primitives, scrollbars
│   │   ├── components.css            # Reusable UI components (cards, buttons, stepper, frames)
│   │   ├── animations.css            # Keyframe animations & transitions
│   │   └── app.css                   # Page-specific layout (result columns, metrics, charts)
│   ├── js/
│   │   ├── main.js                   # Application bootstrap, step orchestration, event wiring
│   │   ├── api.js                    # REST client (health, validate, detect, segment)
│   │   ├── state.js                  # Centralised application state store
│   │   ├── upload.js                 # Drag-and-drop file upload handler
│   │   ├── validation.js             # Validation step UI & orbital scene controller glue
│   │   ├── scene.js                  # Canvas orbital animation (Earth-Moon scene)
│   │   ├── app.js                    # Results rendering (detection table, segmentation images)
│   │   └── utils.js                  # Shared DOM helpers (querySelector, formatBytes, etc.)
│   └── assets/
│       └── samples/                  # Sample lunar images for manual testing
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## Supported Models

| Model | Task | Checkpoint | Status |
|---|---|---|---|
| **YOLO11s** | Crater, rille, and boulder detection | `weights/yolo11s_best.pt` | Active (mAP50: 0.714) |
| **U-Net** | Landslide pixel segmentation | `weights/unet_best.pth` | Architecture ready / experimental |

**YOLO Detection Classes:**

| ID | Class |
|---|---|
| `0` | crater |
| `1` | rille |
| `2` | boulder |

---

## Installation & Setup

**Requirements:** Python 3.11, pip

```powershell
# Verify Python version
python --version

# Install all dependencies
pip install -r requirements.txt
```

---

## Running the Application

### Option 1 — Unified Mode (Recommended)

Flask serves both the REST API and the frontend from a single process:

```powershell
# From the project root
python -m backend.app
```

Open **http://127.0.0.1:5000** in your browser.

### Option 2 — Decoupled Mode (Separate Servers)

Because CORS is enabled on the backend, the frontend and backend can run on separate ports.

**Terminal 1 — Backend API:**
```powershell
python -m backend.app
```
*Listens at `http://127.0.0.1:5000`*

**Terminal 2 — Frontend Static Server:**
```powershell
cd frontend
python -m http.server 8080
```
*Open `http://127.0.0.1:8080` — the frontend auto-targets the backend at port 5000.*

---

## Lunar Image Validation Gate

Every uploaded image must pass a three-stage server-side validation pipeline before reaching YOLO11s or U-Net.

1. **File Integrity** — Magic byte verification, OpenCV/PIL decodability, 16 MB size cap, and dimension bounds (64x64 to 8192x8192).
2. **Lunar Domain Classifier** — A convolutional `LunarClassifierNet` trained to distinguish lunar surface imagery from non-lunar inputs (Earth landscapes, indoor scenes, diagrams, etc.).
3. **Three-Tier Decision Gate:**

| Outcome | Confidence | Behaviour |
|---|---|---|
| `accepted` | >= 70% | Green — inference proceeds |
| `ambiguous` | 35% – 70% | Amber — inference blocked, user prompted |
| `rejected` | < 35% | Red — request blocked with reason |

> **Anti-bypass enforcement:** Direct calls to `/api/detect` or `/api/segment` re-run validation server-side and return `HTTP 403 Forbidden` if the image does not pass.

---

## REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | `GET` | Readiness status of YOLO, U-Net, and the Validation Gate |
| `/api/validate` | `POST` | Run the lunar validation gate on a submitted image |
| `/api/detect` | `POST` | YOLO11s object detection (enforces validation internally) |
| `/api/segment` | `POST` | U-Net segmentation (enforces validation internally) |

All `POST` endpoints accept `multipart/form-data` with a single `image` field.

### Example — `GET /api/health` Response

```json
{
  "status": "ok",
  "detection": {
    "available": true,
    "checkpoint": "yolo11s_best.pt",
    "classes": { "0": "crater", "1": "rille", "2": "boulder" },
    "num_classes": 3
  },
  "segmentation": {
    "available": true,
    "checkpoint": "unet_best.pth"
  },
  "validation_gate": {
    "gate_enabled": true,
    "classifier": {
      "available": true,
      "checkpoint": "lunar_classifier_best.pth"
    },
    "thresholds": {
      "acceptance": 0.70,
      "ambiguity": 0.35
    }
  }
}
```

### Example — `POST /api/detect` Response

```json
{
  "success": true,
  "detections": [
    {
      "class_id": 2,
      "class_name": "boulder",
      "confidence": 0.873,
      "bbox": [120, 340, 210, 430]
    }
  ],
  "count": 1,
  "inference_ms": 142.3,
  "image_size": [1024, 1024],
  "annotated_image": "<base64-encoded PNG>"
}
```

---

## Notes

- **Research prototype** — outputs are not geologically validated and must not be used for terrain safety certification or operational mission planning.
- The `yolo11s.pt` base weights are retained for reference; the active inference checkpoint is `yolo11s_best.pt`.
- The U-Net segmentation model is experimental. Coverage metrics are indicative only.
