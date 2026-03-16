from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from parking_vision.config import load_yaml
from parking_vision.models.factory import build_model
from parking_vision.models.model_b_classic import HandcraftedFeatureExtractor
from parking_vision.utils.io import ensure_dir, save_dataframe, save_json
from parking_vision.utils.metrics import classification_metrics, compute_confusion
from parking_vision.utils.profiling import Profiler
from parking_vision.utils.visualization import save_confusion_matrix, save_metric_bars


def evaluate_patch_manifest(
    manifest_path: str,
    model_name: str,
    model_type: str,
    config_path: str,
    artifact_path: str | None,
    output_dir: str,
) -> dict:
    df = pd.read_csv(manifest_path)
    out_dir = ensure_dir(output_dir)
    cfg = load_yaml(config_path)

    model = build_model(model_type=model_type, config_path=config_path, artifact_path=artifact_path)
    profiler = Profiler()
    y_true = []
    y_pred = []
    latency_ms = []
    rss_mb = []
    rows = []

    if model_type == "model_a":
        batch_size = int(cfg.get("inference", {}).get("batch_size") or cfg.get("data", {}).get("batch_size") or 32)
        for start in tqdm(range(0, len(df), batch_size), desc=f"Evaluating {model_name}"):
            batch_df = df.iloc[start:start + batch_size]
            images = [cv2.imread(p, cv2.IMREAD_COLOR) for p in batch_df["image_path"].tolist()]
            start_t = time.perf_counter()
            preds = model.predict_patches(images)
            end_t = time.perf_counter()
            batch_latency = ((end_t - start_t) / max(len(images), 1)) * 1000.0
            current_rss = profiler.rss_mb()
            for row, pred in zip(batch_df.itertuples(index=False), preds):
                latency_ms.append(batch_latency)
                rss_mb.append(current_rss)
                y_true.append(row.label)
                y_pred.append(pred.status)
                rows.append(
                    {
                        "image_path": row.image_path,
                        "y_true": row.label,
                        "y_pred": pred.status,
                        "confidence": pred.confidence,
                        "latency_ms": batch_latency,
                        "rss_mb": current_rss,
                    }
                )
    else:
        feature_model = model
        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Evaluating {model_name}"):
            image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
            pred, profile = profiler.time_call(feature_model.predict_patches, [image])
            pred = pred[0]
            latency_ms.append(profile.latency_ms)
            rss_mb.append(profile.rss_mb)
            y_true.append(row.label)
            y_pred.append(pred.status)
            rows.append(
                {
                    "image_path": row.image_path,
                    "y_true": row.label,
                    "y_pred": pred.status,
                    "confidence": pred.confidence,
                    "latency_ms": profile.latency_ms,
                    "rss_mb": profile.rss_mb,
                }
            )

    metrics = classification_metrics(y_true, y_pred)
    cm = compute_confusion(y_true, y_pred)
    predictions_df = pd.DataFrame(rows)
    save_dataframe(predictions_df, Path(out_dir) / f"{model_name}_predictions.csv")
    save_confusion_matrix(cm, Path(out_dir) / f"{model_name}_confusion.png")

    summary = {
        "model_name": model_name,
        "accuracy": metrics.accuracy,
        "accuracy_known": metrics.accuracy_known,
        "coverage": metrics.coverage,
        "unknown_rate": metrics.unknown_rate,
        "precision_macro": metrics.precision_macro,
        "recall_macro": metrics.recall_macro,
        "f1_macro": metrics.f1_macro,
        "precision_free": metrics.precision_free,
        "recall_free": metrics.recall_free,
        "f1_free": metrics.f1_free,
        "precision_occupied": metrics.precision_occupied,
        "recall_occupied": metrics.recall_occupied,
        "f1_occupied": metrics.f1_occupied,
        "latency_ms_mean": float(np.mean(latency_ms)) if latency_ms else 0.0,
        "latency_ms_p95": float(np.percentile(latency_ms, 95)) if latency_ms else 0.0,
        "fps_estimate": float(1000.0 / max(np.mean(latency_ms), 1e-6)) if latency_ms else 0.0,
        "rss_mb_mean": float(np.mean(rss_mb)) if rss_mb else 0.0,
        "num_samples": len(df),
    }
    save_json(summary, Path(out_dir) / f"{model_name}_summary.json")
    return summary


def compare_models(summaries: list[dict], output_dir: str) -> None:
    out_dir = ensure_dir(output_dir)
    metrics_map = {s["model_name"]: s for s in summaries}
    save_metric_bars(
        metrics_map,
        Path(out_dir) / "quality_metrics.png",
        ["accuracy", "accuracy_known", "coverage", "f1_macro"],
    )
    save_metric_bars(metrics_map, Path(out_dir) / "system_metrics.png", ["latency_ms_mean", "fps_estimate", "rss_mb_mean"])
    save_dataframe(pd.DataFrame(summaries), Path(out_dir) / "summary.csv")
