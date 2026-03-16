from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from parking_vision.data.layouts import SlotPrediction, load_layout
from parking_vision.utils.io import ensure_dir
from parking_vision.utils.profiling import Profiler
from parking_vision.utils.video import iter_video_frames, write_video
from parking_vision.utils.visualization import draw_layout_overlay
from parking_vision.web.demo import ensure_demo_gallery, load_eval_summary, model_status_cards
from parking_vision.web.schemas import DemoRequest, StreamRequest
from parking_vision.web.state import AppState


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def summarize_predictions(preds: List[SlotPrediction]) -> dict:
    total = len(preds)
    occupied = sum(1 for p in preds if p.status == "occupied")
    free = sum(1 for p in preds if p.status == "free")
    unknown = sum(1 for p in preds if p.status == "unknown")
    known = occupied + free
    confidences = [float(p.confidence) for p in preds]
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    median_confidence = float(np.median(confidences)) if confidences else 0.0
    low_confidence = sum(1 for score in confidences if score < 0.6)
    unknown_rate = unknown / total if total else 0.0
    coverage = known / total if total else 0.0
    occupancy_rate = occupied / total if total else 0.0
    occupancy_rate_known = occupied / known if known else 0.0
    if total == 0:
        quality_band = "empty"
    elif unknown_rate >= 0.35 or mean_confidence < 0.55:
        quality_band = "low"
    elif unknown_rate >= 0.15 or mean_confidence < 0.72:
        quality_band = "medium"
    else:
        quality_band = "high"
    return {
        "total_slots": total,
        "occupied": occupied,
        "free": free,
        "unknown": unknown,
        "known": known,
        "coverage": coverage,
        "unknown_rate": unknown_rate,
        "occupancy_rate": occupancy_rate,
        "occupancy_rate_known": occupancy_rate_known,
        "mean_confidence": mean_confidence,
        "median_confidence": median_confidence,
        "low_confidence_slots": low_confidence,
        "quality_band": quality_band,
    }


def create_app(config_path: str) -> FastAPI:
    app = FastAPI(title="Parking Vision Workstation", version="0.2.0")
    state = AppState(config_path=config_path)

    root_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(root_dir / "templates"))
    static_dir = root_dir / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    output_dir = ensure_dir(state.cfg["app"]["output_dir"])
    app.mount("/artifacts", StaticFiles(directory=str(output_dir)), name="artifacts")

    eval_dir = PROJECT_ROOT / "runs" / "eval" / "latest"
    if eval_dir.exists():
        app.mount("/report_assets", StaticFiles(directory=str(eval_dir)), name="report_assets")

    profiler = Profiler()

    def dashboard_payload() -> dict:
        demos = ensure_demo_gallery(PROJECT_ROOT, output_dir)
        return {
            "models": list(state.cfg["models"].keys()),
            "default_layout_path": state.cfg["app"]["layout_path"],
            "demo_feed": demos,
            "model_cards": model_status_cards(state.cfg, PROJECT_ROOT),
            "evaluation": load_eval_summary(PROJECT_ROOT),
        }

    def validate_inputs(frame: np.ndarray | None, model_key: str, layout_path: str) -> tuple[str | None, str | None]:
        if model_key not in state.cfg["models"]:
            return f"Unknown model: {model_key}", None
        final_layout = layout_path or state.cfg["app"]["layout_path"]
        if not _resolve_project_path(final_layout).exists():
            return f"Layout file not found: {final_layout}", None
        if frame is None:
            return "Failed to decode input media into an image frame.", None
        return None, final_layout

    def annotate_summary(summary: dict, model_key: str, source_kind: str) -> dict:
        cards = {card["model_key"]: card for card in model_status_cards(state.cfg, PROJECT_ROOT)}
        card = cards.get(model_key, {})
        known = summary.get("known", 0)
        summary["model_label"] = card.get("label", model_key)
        summary["source_kind"] = source_kind
        summary["quality_summary"] = (
            f"{summary['model_label']}: {summary['occupied']} occupied, {summary['free']} free, "
            f"{summary['unknown']} uncertain slots."
        )
        summary["quality_note"] = (
            f"Coverage {summary['coverage'] * 100:.1f}% over all slots, "
            f"mean confidence {summary['mean_confidence'] * 100:.1f}%."
        )
        summary["occupancy_note"] = (
            f"Among confident slots, occupancy is {summary['occupancy_rate_known'] * 100:.1f}%."
            if known
            else "No confident slots in this result."
        )
        return summary

    def run_image_inference(frame: np.ndarray, model_key: str, layout_path: str):
        error, final_layout = validate_inputs(frame, model_key, layout_path)
        if error:
            return {"error": error}
        layout = load_layout(_resolve_project_path(final_layout))
        model = state.get_model(model_key)
        model.reset_state()
        preds, profile = profiler.time_call(model.predict_frame, frame, layout)
        overlay = draw_layout_overlay(frame, layout, preds)

        ts = int(time.time() * 1000)
        save_path = output_dir / f"image_{model_key}_{ts}.jpg"
        cv2.imwrite(str(save_path), overlay)

        summary = summarize_predictions(preds)
        summary.update(
            {
                "model_key": model_key,
                "latency_ms": profile.latency_ms,
                "rss_mb": profile.rss_mb,
                "fps_estimate": float(1000.0 / max(profile.latency_ms, 1e-6)),
                "output_path": f"/artifacts/{save_path.name}",
                "slots": [p.__dict__ for p in preds],
            }
        )
        return annotate_summary(summary, model_key, "image")

    def run_video_inference(video_path: str, model_key: str, layout_path: str, max_frames: int, stride: int):
        if model_key not in state.cfg["models"]:
            return {"error": f"Unknown model: {model_key}"}
        final_layout = layout_path or state.cfg["app"]["layout_path"]
        if not _resolve_project_path(final_layout).exists():
            return {"error": f"Layout file not found: {final_layout}"}
        layout = load_layout(_resolve_project_path(final_layout))
        model = state.get_model(model_key)
        model.reset_state()

        frames_out = []
        last_preds = []
        processed = 0
        t0 = time.perf_counter()
        for _, frame in iter_video_frames(video_path, max_frames=max_frames, stride=stride):
            last_preds = model.predict_frame(frame, layout)
            frames_out.append(draw_layout_overlay(frame, layout, last_preds))
            processed += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0 / max(processed, 1)

        ts = int(time.time() * 1000)
        save_path = output_dir / f"video_{model_key}_{ts}.mp4"
        if frames_out:
            write_video(frames_out, save_path, fps=max(2.0, 12.0 / max(stride, 1)))

        summary = summarize_predictions(last_preds)
        summary.update(
            {
                "model_key": model_key,
                "latency_ms": latency_ms,
                "rss_mb": profiler.rss_mb(),
                "fps_estimate": float(1000.0 / max(latency_ms, 1e-6)),
                "frames_processed": processed,
                "output_path": f"/artifacts/{save_path.name}" if frames_out else None,
                "slots": [p.__dict__ for p in last_preds],
            }
        )
        return annotate_summary(summary, model_key, "video")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        payload = dashboard_payload()
        payload["request"] = request
        return templates.TemplateResponse("index.html", payload)

    @app.get("/dashboard_meta")
    async def dashboard_meta():
        return JSONResponse(dashboard_payload())

    @app.get("/metrics")
    async def metrics():
        payload = []
        eval_summary = load_eval_summary(PROJECT_ROOT)
        eval_map = {row["model_name"]: row for row in eval_summary.get("models", [])}
        for key, model_cfg in state.cfg["models"].items():
            payload.append(
                {
                    "model_key": key,
                    "type": model_cfg["type"],
                    "artifact": model_cfg.get("checkpoint") or model_cfg.get("artifact"),
                    "evaluation": eval_map.get(key),
                }
            )
        return JSONResponse(payload)

    @app.get("/artifacts/{file_path:path}")
    async def get_artifact(file_path: str):
        return FileResponse(output_dir / file_path)

    @app.post("/predict_image")
    async def predict_image(
        image: UploadFile = File(...),
        model_key: str = Form("strong_baseline"),
        layout_path: str = Form(""),
    ):
        raw = np.frombuffer(await image.read(), dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        return JSONResponse(run_image_inference(frame, model_key, layout_path))

    @app.post("/predict_video")
    async def predict_video(
        video: UploadFile = File(...),
        model_key: str = Form("strong_baseline"),
        layout_path: str = Form(""),
        max_frames: int = Form(180),
        stride: int = Form(5),
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name
        return JSONResponse(run_video_inference(tmp_path, model_key, layout_path, max_frames=max_frames, stride=stride))

    @app.post("/predict_stream")
    async def predict_stream(request: StreamRequest):
        return JSONResponse(
            run_video_inference(
                request.rtsp_url,
                request.model_key,
                request.layout_path,
                max_frames=request.max_frames,
                stride=request.stride,
            )
        )

    @app.post("/predict_demo")
    async def predict_demo(request: DemoRequest):
        demos = {item["id"]: item for item in ensure_demo_gallery(PROJECT_ROOT, output_dir)}
        demo = demos.get(request.demo_id)
        if not demo:
            return JSONResponse({"error": f"Unknown demo: {request.demo_id}"}, status_code=404)
        if demo["kind"] == "image":
            frame = cv2.imread(demo["local_path"], cv2.IMREAD_COLOR)
            payload = run_image_inference(frame, request.model_key, demo["layout_path"])
        else:
            payload = run_video_inference(
                demo["local_path"],
                request.model_key,
                demo["layout_path"],
                max_frames=request.max_frames,
                stride=request.stride,
            )
        payload["demo_id"] = demo["id"]
        payload["demo_title"] = demo["title"]
        return JSONResponse(payload)

    return app
