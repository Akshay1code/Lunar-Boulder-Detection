# Lunar Boulder and Landslide Detection

> Deep Learning Research Dashboard — YOLO11n/s Object Detection + U-Net Landslide Segmentation  
> Pure HTML/CSS/JS frontend · Flask backend REST API · Python 3.11

---

## Project Architecture

The codebase is organized into a modular structure where the backend and frontend are completely decoupled. The frontend communicates with the backend solely via REST API endpoints (`/api/*`).

```text
Lunar-Boulder-Detection/
├── backend/
│   ├── app.py                  # Flask application entry point & CORS configuration
│   ├── routes/                 # API endpoints (/api/health, /api/detect, /api/segment)
│   ├── models/
│   │   ├── detection/          # YOLO11n/s inference module & checkpoint loader
│   │   └── segmentation/       # U-Net architecture and segmentation inference
│   ├── preprocessing/          # Canonical & experimental image preprocessing
│   ├── services/               # Unified model loading and inference service
│   ├── utils/                  # Shared utilities (dataset loader, verification)
│   ├── scripts/                # Training, evaluation and prediction CLI scripts
│   └── configs/                # Dataset YAML configs (data_merged, data, etc.)
├── frontend/
│   ├── index.html              # Vanilla HTML5 dashboard
│   ├── css/
│   │   └── style.css           # Premium lunar dark glassmorphic UI stylesheet
│   ├── js/
│   │   └── app.js              # Vanilla JavaScript REST client & UI state manager
│   └── assets/                 # Interface assets, icons, and sample lunar images
├── weights/                    # Trained checkpoints (yolo11s_best.pt, unet_best.pth)
├── datasets/                   # Dataset references and configurations
├── reports/                    # Metrics, plots and evaluation results
├── experiments/                # Research experiments and archives
├── tests/                      # Automated test suite (Pytest for API & pipelines)
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## Supported Models & Features

| Model | Task | Checkpoint | Status |
|---|---|---|---|
| **YOLO11s/n** | Crater, rille, and boulder detection | `weights/yolo11s_best.pt` | Active (mAP50: 0.714) |
| **U-Net** | Landslide segmentation | `weights/unet_best.pth` | Architecture ready / experimental |

- **YOLO Detection Classes**:
  - `0: crater`
  - `1: rille`
  - `2: boulder`

---

## Installation & Setup

1. **Activate Python 3.11 Environment:**
   ```powershell
   python --version
   ```

2. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

---

## Launching the Backend and Frontend

You can run the application either in unified mode (Flask serves the static frontend) or in completely decoupled mode (standalone backend + static frontend server).

### Option 1: Unified Execution (Standard)
Start the Flask backend from either the project root or the `backend/` directory:

```powershell
# From project root:
python backend/app.py

# Or via root wrapper:
python app.py
```
Open your browser at **http://127.0.0.1:5000**.

### Option 2: Completely Decoupled Execution (Separate Servers)
Because CORS is enabled on the backend, you can run the backend and frontend on separate ports or distinct servers:

1. **Terminal 1 — Launch Backend REST API:**
   ```powershell
   python backend/app.py
   ```
   *Runs at `http://127.0.0.1:5000`*

2. **Terminal 2 — Launch Standalone Frontend Web Server:**
   ```powershell
   cd frontend
   python -m http.server 8080
   ```
   *Open browser at `http://127.0.0.1:8080`* (Frontend will communicate seamlessly with backend at `http://127.0.0.1:5000`).

---

## Lunar Image Validation Gate

Uploaded images must pass a strict server-side **Lunar Image Validation Gate** before reaching either YOLO11 or U-Net:

1. **File Integrity Validation**: Verifies magic bytes, decodability via OpenCV/PIL, size limit (16 MB), and minimum/maximum dimension bounds (64x64 to 8192x8192).
2. **Dedicated Binary Lunar Classifier**: Screens images via a convolutional neural network (`LunarClassifierNet`) trained to separate lunar mission surface terrain from non-lunar scenes (electronic diagrams, indoor settings, Earth landscapes).
3. **Conservative Decision Gate**:
   - `accepted` (confidence &ge; 70%): Permits model execution (Green badge).
   - `ambiguous` (35% &le; confidence &lt; 70%): Flags uncertain images, blocks inference until manual confirmation or a clearer image is provided (Amber badge).
   - `rejected` (confidence &lt; 35%): Blocks non-lunar inputs (Red badge).
4. **Anti-Bypass Enforcement**: Direct requests to `/api/detect` or `/api/segment` independently enforce validation and return `HTTP 403 Forbidden` if validation fails.

---

## REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | `GET` | Returns readiness status of YOLO, U-Net, and the Lunar Validation Gate |
| `/api/validate` | `POST` | Dedicated validation gatekeeper. Returns validity, category, confidence, and permission |
| `/api/detect` | `POST` | Upload `image` for YOLO detection (enforces server-side lunar validation) |
| `/api/segment` | `POST` | Upload `image` for U-Net segmentation (enforces server-side lunar validation) |

### Example Health Check Response:
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

---

## Training, Evaluation & Utilities

All training, evaluation, and data processing CLI scripts are maintained under `backend/scripts/`:

```powershell
# Train YOLO detection model
python backend/scripts/train_yolo.py --data backend/configs/data_merged.yaml --model yolo11s.pt

# Evaluate YOLO detection model
python backend/scripts/evaluate_yolo.py --weights weights/yolo11s_best.pt

# Train U-Net segmentation model
python backend/scripts/train_unet.py

# Run standalone image segmentation prediction
python backend/scripts/predict_image.py path/to/lunar_image.jpg
```

---

## Running Automated Tests

Run the backend and API test suite:

```powershell
pytest tests/ -v
```
