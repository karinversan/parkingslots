from __future__ import annotations

import argparse
import time
from pathlib import Path

from parking_vision.config import load_yaml, save_yaml
from parking_vision.data.prepare import prepare_dataset
from parking_vision.utils.io import ensure_dir, save_json
from parking_vision.utils.logging import configure_logging
from parking_vision.utils.seed import seed_everything


def _device_from_cfg(cfg: dict) -> str:
    import torch

    device_cfg = str(cfg.get("device", "auto")).lower()
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_cfg == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("Requested device=mps, but MPS is not available. Falling back to cpu.", flush=True)
        return "cpu"
    if device_cfg == "cuda" and not torch.cuda.is_available():
        print("Requested device=cuda, but CUDA is not available. Falling back to cpu.", flush=True)
        return "cpu"
    return device_cfg


def _fit_logreg_with_progress(clf, x_train, y_train) -> None:
    import numpy as np
    from tqdm.auto import tqdm

    max_iter = max(int(getattr(clf, "max_iter", 100)), 1)
    clf.set_params(warm_start=True, max_iter=1)
    classes = np.unique(y_train)
    progress = tqdm(range(max_iter), desc=f"Model B: fitting C={clf.C:g}", leave=False, dynamic_ncols=True)
    for step in progress:
        clf.fit(x_train, y_train)
        progress.set_postfix(iter=step + 1)
        current_iter = int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") else step + 1
        if current_iter >= max_iter:
            break
        if hasattr(clf, "classes_"):
            clf.set_params(warm_start=True, max_iter=current_iter + 1)
        else:
            clf.classes_ = classes
    clf.set_params(warm_start=False, max_iter=max_iter)


def download_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = prepare_dataset(args.config)
    print(result)


def prepare_main():
    download_main()


def train_model_a_main():
    import pandas as pd
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from parking_vision.data.datasets import ParkingPatchDataset
    from parking_vision.models.model_a_mobilenet import SlotClassifierModelA
    from parking_vision.training.checkpoints import save_checkpoint
    from parking_vision.training.engine import run_epoch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg.get("seed", 42))
    device = _device_from_cfg(cfg)

    run_root = Path(cfg["paths"]["run_root"])
    run_dir = ensure_dir(run_root / "latest")
    logger = configure_logging(run_dir / "train.log")
    save_yaml(cfg, run_dir / "resolved_config.yaml")

    train_ds = ParkingPatchDataset(cfg["data"]["manifest_train"], image_size=cfg["data"]["image_size"], train=True)
    val_ds = ParkingPatchDataset(cfg["data"]["manifest_val"], image_size=cfg["data"]["image_size"], train=False)

    pin_memory = device.startswith("cuda")
    persistent_workers = cfg["data"].get("num_workers", 4) > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    model = SlotClassifierModelA(
        backbone=cfg["model"].get("backbone", "mobilenet_v3_large"),
        num_classes=cfg["model"].get("num_classes", 3),
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.2),
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.0))
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    logger.info("Starting Model A training on device=%s", device)
    logger.info("train_samples=%d val_samples=%d batch_size=%d", len(train_ds), len(val_ds), cfg["data"]["batch_size"])
    print(f"Model A: device={device} train_samples={len(train_ds)} val_samples={len(val_ds)}", flush=True)

    best_f1 = -1.0
    history = []
    patience = cfg["train"].get("early_stopping_patience", 4)
    bad_epochs = 0
    epochs = int(cfg["train"]["epochs"])

    epoch_progress = tqdm(range(epochs), desc="Model A: epochs", dynamic_ncols=True)
    for epoch in epoch_progress:
        epoch_start = time.time()
        train_result = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            mixed_precision=cfg["train"].get("mixed_precision", False) and device.startswith("cuda"),
            desc=f"Train epoch {epoch + 1}/{epochs}",
        )
        val_result = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            mixed_precision=cfg["train"].get("mixed_precision", False) and device.startswith("cuda"),
            desc=f"Val epoch {epoch + 1}/{epochs}",
        )
        elapsed = time.time() - epoch_start
        logger.info(
            "epoch=%d train_loss=%.4f train_acc=%.4f train_f1=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f time_sec=%.1f",
            epoch + 1,
            train_result.loss,
            train_result.accuracy,
            train_result.f1_macro,
            val_result.loss,
            val_result.accuracy,
            val_result.f1_macro,
            elapsed,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_result.loss,
                "train_accuracy": train_result.accuracy,
                "train_f1_macro": train_result.f1_macro,
                "val_loss": val_result.loss,
                "val_accuracy": val_result.accuracy,
                "val_f1_macro": val_result.f1_macro,
                "epoch_time_sec": elapsed,
            }
        )
        epoch_progress.set_postfix(
            train_f1=f"{train_result.f1_macro:.4f}",
            val_f1=f"{val_result.f1_macro:.4f}",
            best=f"{best_f1 if best_f1 >= 0 else 0.0:.4f}",
            sec=f"{elapsed:.1f}",
        )
        print(
            f"Epoch {epoch + 1}/{epochs}: train_f1={train_result.f1_macro:.4f} val_f1={val_result.f1_macro:.4f} time={elapsed:.1f}s",
            flush=True,
        )

        if val_result.f1_macro > best_f1:
            best_f1 = val_result.f1_macro
            bad_epochs = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_f1_macro": best_f1,
                },
                run_dir / cfg["paths"].get("checkpoint_name", "best.pt"),
            )
            print(f"Saved new best checkpoint with val_f1={best_f1:.4f}", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                print(f"Early stopping at epoch {epoch + 1}", flush=True)
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    save_json({"best_f1_macro": best_f1, "device": device}, run_dir / "summary.json")
    print({"run_dir": str(run_dir), "best_f1_macro": best_f1, "device": device})


def fit_model_b_main():
    import cv2
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from tqdm.auto import tqdm

    from parking_vision.models.model_b_classic import HandcraftedFeatureExtractor

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg.get("seed", 42))
    run_root = Path(cfg["paths"]["run_root"])
    run_dir = ensure_dir(run_root / "latest")
    logger = configure_logging(run_dir / "fit.log")
    save_yaml(cfg, run_dir / "resolved_config.yaml")

    train_df = pd.read_csv(cfg["data"]["manifest_train"])
    val_df = pd.read_csv(cfg["data"]["manifest_val"])
    extractor = HandcraftedFeatureExtractor(image_size=cfg["data"].get("image_size", 160))

    print(f"Model B: train_samples={len(train_df)} val_samples={len(val_df)}", flush=True)
    logger.info("Starting Model B feature extraction train=%d val=%d", len(train_df), len(val_df))

    def make_xy(df: pd.DataFrame, split_name: str):
        xs, ys = [], []
        progress = tqdm(df.itertuples(index=False), total=len(df), desc=f"Model B: {split_name} features", dynamic_ncols=True)
        for row in progress:
            image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read image: {row.image_path}")
            feats = extractor.compute(image)
            xs.append(feats)
            ys.append(1 if row.label == "occupied" else 0)
            if len(xs) % 2000 == 0:
                progress.set_postfix(done=len(xs))
        return np.stack(xs), np.asarray(ys)

    feature_start = time.time()
    x_train, y_train = make_xy(train_df, "train")
    x_val, y_val = make_xy(val_df, "val")
    print(f"Model B: feature extraction finished in {time.time() - feature_start:.1f}s", flush=True)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    threshold_cfg = cfg.get("fit", {})
    occupied_thresholds = list(threshold_cfg.get("occupied_thresholds", [0.45, 0.5, 0.55, 0.6]))
    unknown_thresholds = list(threshold_cfg.get("unknown_thresholds", [0.05, 0.08, 0.12, 0.15]))

    def score_candidate(probs: np.ndarray, occupied_threshold: float, unknown_threshold: float) -> tuple[float, float, float]:
        occupied_prob = probs[:, 1]
        margin = np.abs(occupied_prob - 0.5)
        pred = np.where(occupied_prob >= occupied_threshold, 1, 0)
        known_mask = margin >= unknown_threshold
        coverage = float(known_mask.mean()) if len(known_mask) else 0.0
        if known_mask.any():
            f1_known = float(f1_score(y_val[known_mask], pred[known_mask], average="macro", zero_division=0))
        else:
            f1_known = 0.0
        objective = f1_known * coverage
        return objective, f1_known, coverage

    best_model = None
    best_score = -1.0
    best_f1_known = 0.0
    best_coverage = 0.0
    best_occupied_threshold = 0.5
    best_unknown_threshold = cfg["model"].get("unknown_threshold", 0.15)
    records = []
    c_values = list(cfg["fit"]["c_values"])
    search_progress = tqdm(c_values, desc="Model B: hyperparameter search", dynamic_ncols=True)
    for c in search_progress:
        clf = LogisticRegression(
            C=c,
            class_weight=cfg["fit"].get("class_weight", "balanced"),
            max_iter=cfg["fit"].get("max_iter", 2000),
            solver="saga",
            n_jobs=-1,
            verbose=0,
        )
        fit_start = time.time()
        _fit_logreg_with_progress(clf, x_train_scaled, y_train)
        val_probs = clf.predict_proba(x_val_scaled)
        fit_time = time.time() - fit_start
        local_best = None
        for occupied_threshold in occupied_thresholds:
            for unknown_threshold in unknown_thresholds:
                objective, f1_known, coverage = score_candidate(
                    val_probs,
                    occupied_threshold=occupied_threshold,
                    unknown_threshold=unknown_threshold,
                )
                record = {
                    "C": c,
                    "occupied_threshold": occupied_threshold,
                    "unknown_threshold": unknown_threshold,
                    "val_objective": objective,
                    "val_f1_known": f1_known,
                    "val_coverage": coverage,
                    "fit_time_sec": fit_time,
                }
                records.append(record)
                if local_best is None or objective > local_best["val_objective"]:
                    local_best = record
        logger.info(
            "C=%.3f objective=%.4f f1_known=%.4f coverage=%.4f fit_time_sec=%.1f",
            c,
            local_best["val_objective"],
            local_best["val_f1_known"],
            local_best["val_coverage"],
            fit_time,
        )
        search_progress.set_postfix(
            best=f"{max(best_score, local_best['val_objective']):.4f}",
            current=f"{local_best['val_objective']:.4f}",
        )
        print(
            "Model B: "
            f"C={c} objective={local_best['val_objective']:.4f} "
            f"f1_known={local_best['val_f1_known']:.4f} coverage={local_best['val_coverage']:.4f} "
            f"fit_time={fit_time:.1f}s",
            flush=True,
        )
        if local_best["val_objective"] > best_score:
            best_score = local_best["val_objective"]
            best_model = clf
            best_f1_known = local_best["val_f1_known"]
            best_coverage = local_best["val_coverage"]
            best_occupied_threshold = local_best["occupied_threshold"]
            best_unknown_threshold = local_best["unknown_threshold"]

    artifact_path = run_dir / cfg["paths"].get("artifact_name", "classic_model.joblib")
    joblib.dump(
        {
            "classifier": best_model,
            "scaler": scaler,
            "occupied_threshold": best_occupied_threshold,
            "unknown_threshold": best_unknown_threshold,
            "feature_names": [f"f{i}" for i in range(x_train.shape[1])],
        },
        artifact_path,
    )
    pd.DataFrame(records).to_csv(run_dir / "grid_search.csv", index=False)
    save_json(
        {
            "best_val_objective": best_score,
            "best_val_f1_known": best_f1_known,
            "best_val_coverage": best_coverage,
            "occupied_threshold": best_occupied_threshold,
            "unknown_threshold": best_unknown_threshold,
        },
        run_dir / "summary.json",
    )
    print(
        {
            "run_dir": str(run_dir),
            "best_val_objective": best_score,
            "best_val_f1_known": best_f1_known,
            "best_val_coverage": best_coverage,
            "occupied_threshold": best_occupied_threshold,
            "unknown_threshold": best_unknown_threshold,
            "artifact_path": str(artifact_path),
        }
    )


def evaluate_main():
    from parking_vision.evaluation.report import render_markdown_report
    from parking_vision.evaluation.runner import compare_models, evaluate_patch_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--model-a-config", required=True)
    parser.add_argument("--model-a-checkpoint", required=True)
    parser.add_argument("--model-b-config", required=True)
    parser.add_argument("--model-b-artifact", required=True)
    parser.add_argument("--output-dir", default="runs/eval/latest")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    summaries = []
    summaries.append(
        evaluate_patch_manifest(
            manifest_path=args.test_manifest,
            model_name="strong_baseline",
            model_type="model_a",
            config_path=args.model_a_config,
            artifact_path=args.model_a_checkpoint,
            output_dir=str(out_dir),
        )
    )
    summaries.append(
        evaluate_patch_manifest(
            manifest_path=args.test_manifest,
            model_name="fast_classic",
            model_type="model_b",
            config_path=args.model_b_config,
            artifact_path=args.model_b_artifact,
            output_dir=str(out_dir),
        )
    )
    compare_models(summaries, str(out_dir))
    render_markdown_report(str(Path(out_dir) / "summary.csv"), str(Path(out_dir) / "report.md"))
    print({"output_dir": str(out_dir)})


def web_main():
    import os
    import subprocess
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/app_default.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    env = os.environ.copy()
    env["PARKING_VISION_CONFIG"] = args.config
    app_path = Path(__file__).resolve().with_name("streamlit_app.py")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
    ]
    subprocess.run(cmd, check=True, env=env)


def legacy_web_main():
    import uvicorn

    from parking_vision.web.api import create_app

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/app_default.yaml")
    args = parser.parse_args()
    app = create_app(args.config)
    cfg = load_yaml(args.config)
    uvicorn.run(
        app,
        host=cfg["app"].get("host", "0.0.0.0"),
        port=int(cfg["app"].get("port", 8000)),
        reload=bool(cfg["app"].get("reload", False)),
    )
