/**
 * Earth-Moon Orbital Scene Controller
 * Manages canvas starfield, Earth/Moon orbits, and message state.
 */

class SceneController {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.stars = [];
    this.animationFrame = null;
    this.isRunning = false;
    this.isPaused = false;
    this.initStars();
    this.setupListeners();
  }

  initStars(count = 65) {
    this.stars = [];
    const w = this.canvas ? this.canvas.width : 600;
    const h = this.canvas ? this.canvas.height : 260;
    for (let i = 0; i < count; i++) {
      this.stars.push({
        x: Math.random() * w,
        y: Math.random() * h,
        radius: Math.random() * 1.2 + 0.4,
        alpha: Math.random() * 0.7 + 0.3,
        delta: (Math.random() - 0.5) * 0.02,
      });
    }
  }

  setupListeners() {
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        this.pause();
      } else if (this.isRunning) {
        this.resume();
      }
    });
  }

  resize() {
    if (!this.canvas) return;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      this.canvas.width = rect.width;
      this.canvas.height = rect.height;
      this.initStars();
    }
  }

  start(mode = "validation", message = "Scanning orbital trajectory...") {
    this.resize();
    this.setMessage(message);
    this.isRunning = true;
    this.isPaused = false;
    this.render();
  }

  stop() {
    this.isRunning = false;
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  pause() {
    this.isPaused = true;
  }

  resume() {
    if (this.isRunning && this.isPaused) {
      this.isPaused = false;
      this.render();
    }
  }

  setMessage(msg) {
    const el = document.getElementById("sceneCaption");
    if (el) el.textContent = msg;
  }

  render() {
    if (!this.isRunning || this.isPaused || !this.ctx) return;

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw twinkling starfield
    for (const star of this.stars) {
      star.alpha += star.delta;
      if (star.alpha > 0.95 || star.alpha < 0.25) star.delta = -star.delta;

      this.ctx.beginPath();
      this.ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
      this.ctx.fillStyle = `rgba(245, 245, 247, ${star.alpha})`;
      this.ctx.fill();
    }

    this.animationFrame = requestAnimationFrame(() => this.render());
  }
}

let sceneInstance = null;
export function getSceneController() {
  if (!sceneInstance) {
    sceneInstance = new SceneController("sceneCanvas");
  }
  return sceneInstance;
}
