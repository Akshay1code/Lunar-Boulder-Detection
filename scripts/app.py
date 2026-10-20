#!/usr/bin/env python3
"""
Lunar-CV Detector - Tkinter desktop app for testing a trained YOLO model
on lunar surface imagery (classes: crater, rille).

Run from the project root:
    python scripts/app.py
"""
from __future__ import annotations

import csv
import queue
import re
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

def get_default_model() -> Path:
    candidates = [
        ROOT / "runs" / "lunar_detection" / "v3" / "weights" / "best.pt",
        ROOT / "runs" / "lunar_detection" / "v2" / "weights" / "best.pt",
        ROOT / "runs" / "lunar_detection" / "baseline" / "weights" / "best.pt",
        ROOT / "weights" / "best.pt",
        ROOT / "weights" / "yolo11s.pt",
        ROOT / "weights" / "yolo11n.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return ROOT / "weights" / "yolo11s.pt"

DEFAULT_MODEL = get_default_model()
DEFAULT_CSV = ROOT / "runs" / "lunar_detection" / "baseline" / "epoch_metrics.csv"
SAMPLES_DIR = ROOT / "assets" / "samples"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {0: "crater", 1: "rille"}

# Dark palette
BG = "#1e1f26"
PANEL = "#272935"
FIELD = "#2f3240"
FG = "#e6e6ea"
MUTED = "#9a9db0"
ACCENT = "#4f8cff"
ACCENT_HOVER = "#6aa0ff"
OK = "#4cc38a"
WARN = "#f0b44c"
ERR = "#ff6b6b"


# --------------------------------------------------------------------------
# Image panel: canvas that scales its image to fit and redraws on resize
# --------------------------------------------------------------------------
class ImagePanel(tk.Canvas):
    def __init__(self, master, placeholder: str):
        super().__init__(master, bg="#14151a", highlightthickness=0)
        self.placeholder = placeholder
        self._img: Image.Image | None = None
        self._tk: ImageTk.PhotoImage | None = None
        self.bind("<Configure>", lambda e: self._redraw())

    def set_image(self, img: Image.Image | None):
        self._img = img
        self._redraw()

    def _redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10:
            return
        if self._img is None:
            self.create_text(w // 2, h // 2, text=self.placeholder, fill=MUTED,
                             font=("Segoe UI", 11), width=w - 40, justify="center")
            return
        iw, ih = self._img.size
        scale = min(w / iw, h / ih)
        size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        self._tk = ImageTk.PhotoImage(self._img.resize(size, Image.LANCZOS))
        self.create_image(w // 2, h // 2, image=self._tk, anchor="center")


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------
class LunarApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lunar-CV Detector")
        self.geometry("1320x820")
        self.minsize(1050, 680)
        self.configure(bg=BG)

        # State
        self.model = None
        self.model_path: Path | None = None
        self.images: list[Path] = []
        self.index = -1
        self.current_path: Path | None = None
        self.original: Image.Image | None = None
        self.annotated: Image.Image | None = None
        self.busy = False
        self.q: queue.Queue = queue.Queue()

        # Tk variables
        self.model_var = tk.StringVar(value=str(DEFAULT_MODEL))
        self.model_state_var = tk.StringVar(value="Not loaded")
        self.image_var = tk.StringVar(value="No image selected")
        self.nav_var = tk.StringVar(value="")
        self.conf_var = tk.DoubleVar(value=0.25)
        self.iou_var = tk.DoubleVar(value=0.45)
        self.conf_lbl = tk.StringVar(value="0.25")
        self.iou_lbl = tk.StringVar(value="0.45")
        self.auto_var = tk.BooleanVar(value=False)
        self.tta_var = tk.BooleanVar(value=False)
        self.total_var = tk.StringVar(value="Detections: -")
        self.classes_var = tk.StringVar(value="")
        self.csv_var = tk.StringVar(value=str(DEFAULT_CSV))
        self.best_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready.")

        self._style()
        self._build()
        self.bind("<Left>", lambda e: self.step(-1))
        self.bind("<Right>", lambda e: self.step(1))
        self.after(100, self._poll)

        # Startup: auto-load default model and metrics
        self.after(300, self._startup)

    # ---------------------------------------------------------------- style
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=FIELD,
                    bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL,
                    troughcolor=FIELD, font=("Segoe UI", 10))
        s.configure("TFrame", background=BG)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("Muted.TLabel", foreground=MUTED)
        s.configure("Head.TLabel", font=("Segoe UI", 11, "bold"))
        s.configure("Big.TLabel", font=("Segoe UI", 13, "bold"))
        s.configure("TLabelframe", background=BG, bordercolor="#3a3d4d")
        s.configure("TLabelframe.Label", background=BG, foreground=ACCENT,
                    font=("Segoe UI", 10, "bold"))
        s.configure("TButton", background=FIELD, foreground=FG, padding=6,
                    borderwidth=0, focusthickness=0)
        s.map("TButton", background=[("active", "#3b3f52"), ("disabled", PANEL)],
              foreground=[("disabled", MUTED)])
        s.configure("Accent.TButton", background=ACCENT, foreground="white",
                    font=("Segoe UI", 10, "bold"))
        s.map("Accent.TButton", background=[("active", ACCENT_HOVER), ("disabled", PANEL)])
        s.configure("TEntry", fieldbackground=FIELD, foreground=FG, insertcolor=FG)
        s.configure("TCheckbutton", background=BG, foreground=FG)
        s.map("TCheckbutton", background=[("active", BG)])
        s.configure("Horizontal.TScale", background=BG)
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(14, 6))
        s.map("TNotebook.Tab", background=[("selected", ACCENT)],
              foreground=[("selected", "white")])
        s.configure("Treeview", background=FIELD, fieldbackground=FIELD,
                    foreground=FG, rowheight=22, borderwidth=0)
        s.configure("Treeview.Heading", background=PANEL, foreground=FG,
                    font=("Segoe UI", 9, "bold"))
        s.map("Treeview", background=[("selected", ACCENT)])

    # ---------------------------------------------------------------- layout
    def _build(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="ns")
        left.configure(width=340)

        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        # Status bar
        bar = tk.Frame(self, bg=PANEL)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.status_lbl = tk.Label(bar, textvariable=self.status_var, bg=PANEL,
                                   fg=FG, anchor="w", padx=10, pady=5,
                                   font=("Segoe UI", 10))
        self.status_lbl.pack(fill="x")

        self._build_left(left)

        nb = ttk.Notebook(right)
        nb.grid(row=0, column=0, sticky="nsew")
        det_tab = ttk.Frame(nb, padding=8)
        met_tab = ttk.Frame(nb, padding=8)
        nb.add(det_tab, text="  Detection  ")
        nb.add(met_tab, text="  Training Metrics  ")
        self._build_detection_tab(det_tab)
        self._build_metrics_tab(met_tab)

    def _build_left(self, p):
        # --- Model
        m = ttk.LabelFrame(p, text="Model", padding=8)
        m.pack(fill="x", pady=(0, 8))
        ttk.Entry(m, textvariable=self.model_var, width=38).pack(fill="x")
        row = ttk.Frame(m)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Browse...", command=self.browse_model).pack(side="left")
        self.load_btn = ttk.Button(row, text="Load Model", style="Accent.TButton",
                                   command=self.load_model)
        self.load_btn.pack(side="right")
        srow = ttk.Frame(m)
        srow.pack(fill="x", pady=(6, 0))
        ttk.Label(srow, text="Status:", style="Muted.TLabel").pack(side="left")
        self.model_state_lbl = tk.Label(srow, textvariable=self.model_state_var,
                                        bg=BG, fg=ERR, font=("Segoe UI", 10, "bold"))
        self.model_state_lbl.pack(side="left", padx=6)

        # --- Image
        i = ttk.LabelFrame(p, text="Image", padding=8)
        i.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(i)
        row.pack(fill="x")
        ttk.Button(row, text="Browse Image", command=self.browse_image).pack(side="left")
        ttk.Button(row, text="Browse Folder", command=self.browse_folder).pack(side="right")
        ttk.Label(i, textvariable=self.image_var, style="Muted.TLabel",
                  wraplength=300).pack(fill="x", pady=(6, 0))
        nav = ttk.Frame(i)
        nav.pack(fill="x", pady=(6, 0))
        self.prev_btn = ttk.Button(nav, text="\u25C0 Prev", width=9,
                                   command=lambda: self.step(-1), state="disabled")
        self.prev_btn.pack(side="left")
        ttk.Label(nav, textvariable=self.nav_var, anchor="center").pack(
            side="left", expand=True, fill="x")
        self.next_btn = ttk.Button(nav, text="Next \u25B6", width=9,
                                   command=lambda: self.step(1), state="disabled")
        self.next_btn.pack(side="right")

        # --- Thresholds
        t = ttk.LabelFrame(p, text="Thresholds", padding=8)
        t.pack(fill="x", pady=(0, 8))
        for label, var, lbl in (("Confidence", self.conf_var, self.conf_lbl),
                                ("IoU", self.iou_var, self.iou_lbl)):
            r = ttk.Frame(t)
            r.pack(fill="x", pady=2)
            ttk.Label(r, text=label, width=11).pack(side="left")
            sc = ttk.Scale(r, from_=0.0, to=1.0, variable=var, orient="horizontal",
                           command=lambda v, vv=var, ll=lbl: ll.set(f"{float(v):.2f}"))
            sc.pack(side="left", fill="x", expand=True, padx=6)
            sc.bind("<ButtonRelease-1>", self._slider_released)
            ttk.Label(r, textvariable=lbl, width=5).pack(side="right")
        ttk.Checkbutton(t, text="Re-run automatically when slider changes",
                        variable=self.auto_var).pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(t, text="Test-time augmentation (slower, often more accurate)",
                        variable=self.tta_var).pack(anchor="w", pady=(2, 0))

        # --- Actions
        a = ttk.Frame(p)
        a.pack(fill="x", pady=(0, 8))
        self.run_btn = ttk.Button(a, text="Run Detection", style="Accent.TButton",
                                  command=self.run_detection)
        self.run_btn.pack(fill="x")
        self.save_btn = ttk.Button(a, text="Save Result", command=self.save_result,
                                   state="disabled")
        self.save_btn.pack(fill="x", pady=(6, 0))

        # --- Results
        r = ttk.LabelFrame(p, text="Results", padding=8)
        r.pack(fill="both", expand=True)
        ttk.Label(r, textvariable=self.total_var, style="Big.TLabel").pack(anchor="w")
        ttk.Label(r, textvariable=self.classes_var, style="Muted.TLabel").pack(
            anchor="w", pady=(2, 6))
        cols = ("n", "cls", "conf", "box")
        tree_frame = ttk.Frame(r)
        tree_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=8)
        for c, text, w in (("n", "#", 35), ("cls", "Class", 70),
                           ("conf", "Conf", 60), ("box", "Box (x1,y1,x2,y2)", 150)):
            self.tree.heading(c, text=text)
            self.tree.column(c, width=w, anchor="w" if c in ("cls", "box") else "center")
        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _build_detection_tab(self, p):
        p.columnconfigure(0, weight=1, uniform="img")
        p.columnconfigure(1, weight=1, uniform="img")
        p.rowconfigure(1, weight=1)
        ttk.Label(p, text="Original", style="Head.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Label(p, text="Detection Result", style="Head.TLabel").grid(row=0, column=1, sticky="w", padx=(8, 0), pady=(0, 4))
        self.orig_panel = ImagePanel(p, "Browse an image or folder to begin")
        self.orig_panel.grid(row=1, column=0, sticky="nsew")
        self.res_panel = ImagePanel(p, "Click \"Run Detection\" to see results")
        self.res_panel.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

    def _build_metrics_tab(self, p):
        p.columnconfigure(0, weight=1)
        p.rowconfigure(2, weight=1)
        top = ttk.Frame(p)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        ttk.Entry(top, textvariable=self.csv_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(top, text="Browse...", command=self.browse_csv).grid(row=0, column=1, padx=6)
        ttk.Button(top, text="Reload", style="Accent.TButton",
                   command=self.load_metrics).grid(row=0, column=2)
        ttk.Label(p, textvariable=self.best_var, style="Big.TLabel").grid(
            row=1, column=0, sticky="w", pady=8)

        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor=BG)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=p)
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")
        self._style_axes()
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- status
    def status(self, msg: str, level: str = "info"):
        color = {"info": FG, "ok": OK, "warn": WARN, "error": ERR}.get(level, FG)
        self.status_var.set(msg)
        self.status_lbl.configure(fg=color)

    def _set_model_state(self, text: str, color: str):
        self.model_state_var.set(text)
        self.model_state_lbl.configure(fg=color)

    # ---------------------------------------------------------------- startup
    def _startup(self):
        if DEFAULT_MODEL.is_file():
            self.load_model()
        else:
            self.status(f"Default model not found at {DEFAULT_MODEL}. Browse for a .pt file.", "warn")
        if DEFAULT_CSV.is_file():
            self.load_metrics()
        else:
            self.best_var.set("epoch_metrics.csv not found - browse for it.")

    # ---------------------------------------------------------------- model
    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO weights",
            initialdir=str(DEFAULT_MODEL.parent if DEFAULT_MODEL.parent.exists() else ROOT),
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)
            self.status("Weights selected. Click \"Load Model\" to load them.")

    def load_model(self):
        if self.busy:
            self.status("Please wait - another task is running.", "warn")
            return
        path = Path(self.model_var.get().strip().strip('"'))
        if not path.is_file():
            self._set_model_state("Not loaded", ERR)
            self.status(f"Model file not found: {path}", "error")
            return
        self.busy = True
        self.load_btn.configure(state="disabled")
        self._set_model_state("Loading...", WARN)
        self.status(f"Loading {path.name} (first load imports ultralytics, may take a moment)...")
        threading.Thread(target=self._load_worker, args=(path,), daemon=True).start()

    def _load_worker(self, path: Path):
        try:
            from ultralytics import YOLO
            model = YOLO(str(path))
            self.q.put(("model_ok", (path, model)))
        except Exception as e:  # noqa: BLE001
            self.q.put(("model_err", f"{type(e).__name__}: {e}"))

    # ---------------------------------------------------------------- images
    def browse_image(self):
        init_dir = str(SAMPLES_DIR if SAMPLES_DIR.exists() else ROOT)
        path = filedialog.askopenfilename(
            title="Select image",
            initialdir=init_dir,
            filetypes=[("Images", " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))),
                       ("All files", "*.*")])
        if not path:
            return
        self.images = [Path(path)]
        self.index = 0
        self._show_current()

    def browse_folder(self):
        init_dir = str(SAMPLES_DIR if SAMPLES_DIR.exists() else ROOT)
        folder = filedialog.askdirectory(title="Select image folder", initialdir=init_dir)
        if not folder:
            return
        files = sorted(p for p in Path(folder).iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not files:
            self.status(f"No supported images found in {folder}", "warn")
            return
        self.images = files
        self.index = 0
        self._show_current()
        self.status(f"Loaded {len(files)} images from folder. Use arrow keys or Prev/Next.", "ok")

    def step(self, delta: int):
        if len(self.images) < 2:
            return
        self.index = (self.index + delta) % len(self.images)
        self._show_current()

    def _show_current(self):
        path = self.images[self.index]
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
        except Exception as e:  # noqa: BLE001
            self.original = None
            self.current_path = None
            self.orig_panel.set_image(None)
            self._clear_results()
            self.status(f"Cannot open image '{path.name}': {e}", "error")
            self._update_nav()
            return
        self.current_path = path
        self.original = img
        self.orig_panel.set_image(img)
        self._clear_results()
        self.image_var.set(f"{path.name}  ({img.width}x{img.height})")
        self._update_nav()
        self.status(f"Loaded {path.name}", "info")

    def _update_nav(self):
        n = len(self.images)
        multi = n > 1
        self.prev_btn.configure(state="normal" if multi else "disabled")
        self.next_btn.configure(state="normal" if multi else "disabled")
        self.nav_var.set(f"{self.index + 1} / {n}" if multi else "")

    def _clear_results(self):
        self.annotated = None
        self.res_panel.set_image(None)
        self.save_btn.configure(state="disabled")
        self.total_var.set("Detections: -")
        self.classes_var.set("")
        self.tree.delete(*self.tree.get_children())

    # ---------------------------------------------------------------- inference
    def _slider_released(self, _event=None):
        if self.auto_var.get() and self.model is not None and self.original is not None:
            self.run_detection()

    def run_detection(self):
        if self.busy:
            self.status("Please wait - another task is running.", "warn")
            return
        if self.model is None:
            self.status("No model loaded. Choose a .pt file and click \"Load Model\".", "error")
            return
        if self.current_path is None or self.original is None:
            self.status("No valid image selected. Use \"Browse Image\" or \"Browse Folder\".", "error")
            return
        conf = round(self.conf_var.get(), 2)
        iou = round(self.iou_var.get(), 2)
        tta = self.tta_var.get()
        self.busy = True
        self.run_btn.configure(state="disabled")
        self.status(f"Running detection (conf={conf:.2f}, IoU={iou:.2f}{', TTA' if tta else ''})...")
        threading.Thread(target=self._detect_worker,
                         args=(self.model, self.current_path, conf, iou, tta),
                         daemon=True).start()

    def _detect_worker(self, model, path: Path, conf: float, iou: float, tta: bool = False):
        try:
            res = model.predict(source=str(path), conf=conf, iou=iou, augment=tta, verbose=False)[0]
            annotated = Image.fromarray(res.plot()[..., ::-1])  # BGR -> RGB
            names = getattr(res, "names", None) or CLASS_NAMES
            dets = []
            for b in res.boxes:
                cid = int(b.cls.item())
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                dets.append((names.get(cid, CLASS_NAMES.get(cid, str(cid))),
                             float(b.conf.item()), (x1, y1, x2, y2)))
            dets.sort(key=lambda d: d[1], reverse=True)
            self.q.put(("det_ok", (path, annotated, dets)))
        except Exception as e:  # noqa: BLE001
            self.q.put(("det_err", f"{type(e).__name__}: {e}"))

    # ---------------------------------------------------------------- save
    def save_result(self):
        if self.annotated is None:
            self.status("Nothing to save yet - run detection first.", "error")
            return
        stem = self.current_path.stem if self.current_path else "result"
        path = filedialog.asksaveasfilename(
            title="Save annotated image", defaultextension=".png",
            initialfile=f"{stem}_detected.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")])
        if not path:
            self.status("Save cancelled.")
            return
        try:
            self.annotated.save(path)
            self.status(f"Saved result to {path}", "ok")
        except Exception as e:  # noqa: BLE001
            self.status(f"Could not save image: {e}", "error")

    # ---------------------------------------------------------------- metrics
    def browse_csv(self):
        path = filedialog.askopenfilename(
            title="Select epoch_metrics.csv",
            initialdir=str(DEFAULT_CSV.parent if DEFAULT_CSV.parent.exists() else ROOT),
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            self.csv_var.set(path)
            self.load_metrics()

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def load_metrics(self):
        path = Path(self.csv_var.get().strip().strip('"'))
        if not path.is_file():
            self.status(f"Metrics CSV not found: {path}", "error")
            return
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                raise ValueError("file has no data rows")
            keys = list(rows[0].keys())
            norm = {k: self._norm(k) for k in keys}

            def find(pred):
                return next((k for k in keys if pred(norm[k])), None)

            c_ep = find(lambda n: n == "epoch" or n.endswith("epoch"))
            c_p = find(lambda n: "precision" in n)
            c_r = find(lambda n: "recall" in n)
            c_m = find(lambda n: "map50" in n and "map5095" not in n)
            missing = [name for name, c in (("precision", c_p), ("recall", c_r),
                                            ("mAP50", c_m)) if c is None]
            if missing:
                raise ValueError(f"missing column(s): {', '.join(missing)} (found: {keys})")

            def val(row, col):
                try:
                    return float(row[col])
                except (TypeError, ValueError):
                    return float("nan")

            epochs = [int(val(r, c_ep)) if c_ep and val(r, c_ep) == val(r, c_ep) else i + 1
                      for i, r in enumerate(rows)]
            prec = [val(r, c_p) for r in rows]
            rec = [val(r, c_r) for r in rows]
            m50 = [val(r, c_m) for r in rows]
        except Exception as e:  # noqa: BLE001
            self.status(f"Could not read metrics CSV: {e}", "error")
            return

        valid = [(m, i) for i, m in enumerate(m50) if m == m]
        if not valid:
            self.status("Metrics CSV has no valid mAP50 values.", "error")
            return
        best_i = max(valid)[1]
        self._plot_metrics(epochs, prec, rec, m50, best_i)
        self.best_var.set(
            f"Best epoch: {epochs[best_i]}   |   mAP50 {m50[best_i]:.3f}   "
            f"P {prec[best_i]:.3f}   R {rec[best_i]:.3f}")
        self.status(f"Loaded {len(epochs)} epochs of metrics from {path.name}", "ok")

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color("#3a3d4d")
        ax.tick_params(colors=MUTED)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)
        ax.grid(True, color="#3a3d4d", alpha=0.5, linewidth=0.6)

    def _plot_metrics(self, ep, prec, rec, m50, best_i):
        ax = self.ax
        ax.clear()
        self._style_axes()
        ax.plot(ep, prec, color="#4f8cff", linewidth=1.8, label="Precision")
        ax.plot(ep, rec, color="#4cc38a", linewidth=1.8, label="Recall")
        ax.plot(ep, m50, color="#f0b44c", linewidth=2.2, label="mAP50")
        ax.axvline(ep[best_i], color=ERR, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.scatter([ep[best_i]], [m50[best_i]], s=110, marker="*", color=ERR,
                   zorder=5, label=f"Best mAP50 (epoch {ep[best_i]})")
        ax.annotate(f"{m50[best_i]:.3f}", (ep[best_i], m50[best_i]),
                    textcoords="offset points", xytext=(-10, 10), color=ERR,
                    fontsize=10, fontweight="bold", ha="right")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Training Progress")
        ax.set_ylim(0, 1.0)
        leg = ax.legend(facecolor=PANEL, edgecolor="#3a3d4d", loc="lower right")
        for t in leg.get_texts():
            t.set_color(FG)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- queue
    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _handle(self, kind, payload):
        if kind == "model_ok":
            path, model = payload
            self.model, self.model_path = model, path
            self.busy = False
            self.load_btn.configure(state="normal")
            self._set_model_state("Loaded", OK)
            self.status(f"Model loaded: {path}", "ok")
        elif kind == "model_err":
            self.busy = False
            self.load_btn.configure(state="normal")
            if self.model is None:
                self._set_model_state("Not loaded", ERR)
            else:
                self._set_model_state("Loaded (previous model)", WARN)
            self.status(f"Failed to load model: {payload}", "error")
        elif kind == "det_ok":
            path, annotated, dets = payload
            self.busy = False
            self.run_btn.configure(state="normal")
            if path != self.current_path:  # user navigated away mid-inference
                self.status("Image changed during detection - run again.", "warn")
                return
            self.annotated = annotated
            self.res_panel.set_image(annotated)
            self.save_btn.configure(state="normal")
            self._show_detections(dets)
            self.status(f"Detection complete: {len(dets)} object(s) found.", "ok")
        elif kind == "det_err":
            self.busy = False
            self.run_btn.configure(state="normal")
            self.status(f"Detection failed: {payload}", "error")

    def _show_detections(self, dets):
        self.tree.delete(*self.tree.get_children())
        mnames = getattr(self.model, "names", None)
        base = list(mnames.values()) if isinstance(mnames, dict) else list(CLASS_NAMES.values())
        counts: dict[str, int] = {name: 0 for name in base}
        for i, (name, conf, box) in enumerate(dets, 1):
            counts[name] = counts.get(name, 0) + 1
            self.tree.insert("", "end", values=(i, name, f"{conf:.3f}",
                                                f"{box[0]}, {box[1]}, {box[2]}, {box[3]}"))
        self.total_var.set(f"Detections: {len(dets)}")
        self.classes_var.set("   ".join(f"{n.capitalize()}s: {c}" for n, c in counts.items()))


if __name__ == "__main__":
    LunarApp().mainloop()