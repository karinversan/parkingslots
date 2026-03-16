from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import joblib
import pandas as pd

from parking_vision.data.layouts import Layout, Slot, save_layout
from parking_vision.utils.io import ensure_dir, load_json, save_json
from parking_vision.utils.video import write_video


MODEL_NOTES = {
    "strong_baseline": {
        "label": "Strong baseline",
        "approach": "MobileNetV3 ROI classifier + temporal smoothing",
        "strength": "Higher accuracy, more robust to glare and shadows",
        "cost": "Heavier runtime",
    },
    "fast_classic": {
        "label": "Fast classic",
        "approach": "Handcrafted ROI features + logistic regression",
        "strength": "Lower latency, CPU-friendly",
        "cost": "Less robust on hard lighting and occlusion",
    },
}


def _norm(p: str | Path) -> str:
    return str(Path(p)).replace("\\", "/")


def _sample_to_layout(sample: dict, layout_path: Path) -> bool:
    image_path = Path(sample["resolved_image_path"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False
    h, w = image.shape[:2]
    polylines = sample.get("parking_spaces", {}).get("polylines", [])
    slots: List[Slot] = []
    for poly in polylines:
        status = str(poly.get("occupancy_status", "")).strip().lower()
        if status not in {"occupied", "free", "not occupied", "vacant", "empty", "0", "1"}:
            continue
        points = poly.get("points") or []
        while isinstance(points, list) and len(points) == 1 and isinstance(points[0], list):
            points = points[0]
        polygon = []
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                continue
            x, y = pt
            if 0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0:
                x = int(round(float(x) * w))
                y = int(round(float(y) * h))
            else:
                x = int(round(float(x)))
                y = int(round(float(y)))
            polygon.append([x, y])
        if len(polygon) < 3:
            continue
        slots.append(Slot(slot_id=str(poly.get("space_id", poly.get("index", len(slots) + 1))), polygon=polygon))
    if not slots:
        return False
    layout = Layout(camera_id=f"demo-{sample.get('source', 'camera')}", image_width=w, image_height=h, slots=slots)
    save_layout(layout, layout_path)
    return True


def _load_samples(samples_path: Path, dataset_root: Path) -> List[dict]:
    payload = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    out = []
    for sample in samples:
        fp = sample.get("filepath")
        if not fp:
            continue
        candidates = [
            dataset_root / fp,
            dataset_root / fp.lstrip("/"),
            dataset_root / Path(fp).name,
        ]
        resolved = None
        for cand in candidates:
            if cand.exists():
                resolved = cand.resolve()
                break
        if resolved is None:
            matches = list(dataset_root.rglob(Path(fp).name))
            if matches:
                resolved = matches[0].resolve()
        if resolved is None:
            continue
        sample = dict(sample)
        sample["resolved_image_path"] = str(resolved)
        out.append(sample)
    return out


def _pick_image_demos(test_df: pd.DataFrame, samples_by_path: Dict[str, dict], limit: int = 6) -> List[dict]:
    chosen = []
    seen = set()
    unique_sources = test_df[["source_image_path", "scene_id", "weather"]].drop_duplicates()
    for row in unique_sources.itertuples(index=False):
        sample = samples_by_path.get(_norm(row.source_image_path))
        if not sample:
            continue
        key = (row.scene_id, row.weather)
        if key in seen and len(chosen) < limit // 2:
            continue
        seen.add(key)
        chosen.append(sample)
        if len(chosen) >= limit:
            break
    return chosen


def _pick_video_groups(test_df: pd.DataFrame, samples_by_path: Dict[str, dict], limit: int = 3) -> List[list[dict]]:
    groups: Dict[tuple, list[dict]] = {}
    unique = test_df[["source_image_path", "scene_id", "group_key", "weather"]].drop_duplicates()
    for row in unique.itertuples(index=False):
        sample = samples_by_path.get(_norm(row.source_image_path))
        if not sample:
            continue
        groups.setdefault((row.scene_id, row.group_key, row.weather), []).append(sample)

    picked = []
    for _, group_samples in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        uniq = {s["resolved_image_path"]: s for s in group_samples}
        frames = sorted(uniq.values(), key=lambda s: Path(s["resolved_image_path"]).stem)
        if len(frames) < 8:
            continue
        picked.append(frames[:18])
        if len(picked) >= limit:
            break
    return picked


def ensure_demo_gallery(project_root: str | Path, output_dir: str | Path) -> List[dict]:
    project_root = Path(project_root)
    output_dir = ensure_dir(output_dir)
    manifest_path = Path(output_dir) / "demo_feed.json"
    media_dir = ensure_dir(Path(output_dir) / "demos" / "media")
    layout_dir = ensure_dir(Path(output_dir) / "demos" / "layouts")

    if manifest_path.exists():
        try:
            demos = load_json(manifest_path)
            valid = True
            for item in demos:
                local_path = Path(item.get("local_path", ""))
                if local_path and not local_path.exists():
                    valid = False
                    break
            if valid:
                return demos
        except Exception:
            pass

    test_manifest = project_root / "data" / "manifests" / "test.csv"
    samples_path = project_root / "data" / "pklot" / "samples.json"
    dataset_root = project_root / "data" / "pklot"
    if not test_manifest.exists() or not samples_path.exists():
        return []

    test_df = pd.read_csv(test_manifest)
    if "source_image_path" not in test_df.columns:
        return []

    samples = _load_samples(samples_path, dataset_root)
    samples_by_path = {_norm(s["resolved_image_path"]): s for s in samples}

    demos: List[dict] = []

    for idx, sample in enumerate(_pick_image_demos(test_df, samples_by_path), start=1):
        media_name = f"image_demo_{idx}.jpg"
        local_media = media_dir / media_name
        shutil.copy2(sample["resolved_image_path"], local_media)
        layout_path = layout_dir / f"image_demo_{idx}.json"
        if not _sample_to_layout(sample, layout_path):
            continue
        weather = sample.get("weather", {})
        weather_label = weather.get("label", "unknown") if isinstance(weather, dict) else str(weather)
        demos.append(
            {
                "id": f"image_demo_{idx}",
                "kind": "image",
                "title": f"{sample.get('source', 'PKLot').upper()} test frame {idx}",
                "subtitle": f"Unseen test image · {weather_label}",
                "scene": str(sample.get("source", "unknown")),
                "weather": str(weather_label).lower(),
                "split": "test",
                "media_url": f"/artifacts/demos/media/{media_name}",
                "poster_url": f"/artifacts/demos/media/{media_name}",
                "local_path": str(local_media.resolve()),
                "layout_path": str(layout_path.resolve()),
            }
        )

    for idx, frames in enumerate(_pick_video_groups(test_df, samples_by_path), start=1):
        local_media = media_dir / f"video_demo_{idx}.mp4"
        poster = media_dir / f"video_demo_{idx}_poster.jpg"
        frame_images = []
        for sample in frames:
            image = cv2.imread(sample["resolved_image_path"], cv2.IMREAD_COLOR)
            if image is not None:
                frame_images.append(image)
        if len(frame_images) < 8:
            continue
        write_video(frame_images, local_media, fps=6.0)
        cv2.imwrite(str(poster), frame_images[0])
        layout_path = layout_dir / f"video_demo_{idx}.json"
        if not _sample_to_layout(frames[0], layout_path):
            continue
        weather = frames[0].get("weather", {})
        weather_label = weather.get("label", "unknown") if isinstance(weather, dict) else str(weather)
        demos.append(
            {
                "id": f"video_demo_{idx}",
                "kind": "video",
                "title": f"{frames[0].get('source', 'PKLot').upper()} short clip {idx}",
                "subtitle": f"Unseen test sequence · {weather_label}",
                "scene": str(frames[0].get("source", "unknown")),
                "weather": str(weather_label).lower(),
                "split": "test",
                "media_url": f"/artifacts/demos/media/{local_media.name}",
                "poster_url": f"/artifacts/demos/media/{poster.name}",
                "local_path": str(local_media.resolve()),
                "layout_path": str(layout_path.resolve()),
            }
        )

    save_json(demos, manifest_path)
    return demos


def load_eval_summary(project_root: str | Path) -> dict:
    project_root = Path(project_root)
    eval_dir = project_root / "runs" / "eval" / "latest"
    summary_path = eval_dir / "summary.csv"
    meta_path = eval_dir / "summary_meta.json"
    if not summary_path.exists():
        return {
            "ready": False,
            "scope": "No held-out evaluation found",
            "models": [],
            "charts": {},
        }
    scope = "Held-out test split from runs/eval/latest"
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            scope = str(meta.get("scope") or scope)
        except Exception:
            pass

    df = pd.read_csv(summary_path)
    chart_urls = {}
    for name in ["quality_metrics.png", "system_metrics.png"]:
        p = eval_dir / name
        if p.exists():
            chart_urls[name.replace(".png", "")] = f"/report_assets/{name}"

    models = []
    for row in df.to_dict(orient="records"):
        row["label"] = MODEL_NOTES.get(row["model_name"], {}).get("label", row["model_name"])
        models.append(row)
    return {
        "ready": True,
        "scope": scope,
        "models": models,
        "charts": chart_urls,
    }


def model_status_cards(cfg: dict, project_root: str | Path) -> List[dict]:
    project_root = Path(project_root)
    eval_summary = load_eval_summary(project_root)
    eval_map = {row["model_name"]: row for row in eval_summary.get("models", [])}
    cards = []
    for model_key, model_cfg in cfg["models"].items():
        artifact = model_cfg.get("checkpoint") or model_cfg.get("artifact")
        artifact_path = project_root / artifact if artifact else None
        run_summary_path = artifact_path.parent / "summary.json" if artifact_path else None
        run_summary = {}
        if run_summary_path and run_summary_path.exists():
            try:
                run_summary = load_json(run_summary_path)
            except Exception:
                run_summary = {}
        artifact_meta = {}
        if artifact_path and artifact_path.exists() and artifact_path.suffix == ".joblib":
            try:
                payload = joblib.load(artifact_path)
                artifact_meta = {
                    "has_scaler": payload.get("scaler") is not None,
                    "has_occupied_threshold": payload.get("occupied_threshold") is not None,
                    "has_unknown_threshold": payload.get("unknown_threshold") is not None,
                    "keys": sorted(payload.keys()),
                }
            except Exception:
                artifact_meta = {}
        note = MODEL_NOTES.get(model_key, {})
        benchmark = eval_map.get(model_key)
        cards.append(
            {
                "model_key": model_key,
                "label": note.get("label", model_key),
                "approach": note.get("approach", model_cfg["type"]),
                "strength": note.get("strength", ""),
                "cost": note.get("cost", ""),
                "artifact_ready": bool(artifact_path and artifact_path.exists()),
                "artifact_path": str(artifact_path) if artifact_path else None,
                "benchmark": benchmark,
                "benchmark_ready": bool(benchmark),
                "run_summary": run_summary,
                "artifact_meta": artifact_meta,
            }
        )
    return cards
