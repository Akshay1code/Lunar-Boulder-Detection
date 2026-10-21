"""
Lunar Boulder and Landslide Detection — Flask Application Entry Point
=====================================================================
Backend entry point. Supports:
- Standalone API server mode (CORS enabled for separate frontend)
- Serving the frontend directly when hosted together
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.routes.api import api_bp
from backend.services.inference_service import InferenceService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FRONTEND_DIR = ROOT / "frontend"


def create_app() -> Flask:
    """Application factory for Flask backend."""
    app = Flask(
        __name__,
        static_folder=str(FRONTEND_DIR),
        static_url_path="",
    )
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB hard limit

    # Security and CORS headers
    @app.after_request
    def apply_security_and_cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' data: blob:; "
            "img-src 'self' data: blob:; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "script-src 'self' 'unsafe-inline'; "
            "connect-src 'self' http://127.0.0.1:5000 http://localhost:5000;"
        )
        return response

    # Register API blueprint
    app.register_blueprint(api_bp)

    # Serve frontend index and assets
    @app.get("/")
    def index():
        if FRONTEND_DIR.is_dir() and (FRONTEND_DIR / "index.html").is_file():
            return send_from_directory(str(FRONTEND_DIR), "index.html")
        return jsonify({
            "message": "Lunar Boulder & Landslide Detection Backend API is running.",
            "health": "/api/health",
            "detect": "/api/detect",
            "segment": "/api/segment",
        })

    @app.get("/<path:filename>")
    def static_proxy(filename: str):
        file_path = FRONTEND_DIR / filename
        if file_path.is_file():
            return send_from_directory(str(FRONTEND_DIR), filename)
        return jsonify({"success": False, "error": f"Asset '{filename}' not found."}), 404

    # Global error handlers
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"success": False, "error": "Upload too large. Maximum 20 MB."}), 413

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"success": False, "error": "Endpoint not found."}), 404

    @app.errorhandler(500)
    def server_error(e):
        log.exception("Unhandled server error")
        return jsonify({"success": False, "error": "Internal server error."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    InferenceService.initialize_models()
    log.info("Starting Lunar Boulder and Landslide Detection Backend...")
    log.info("API server running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
