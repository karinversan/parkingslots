from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from parking_vision.data.layouts import load_layout
from parking_vision.utils.io import ensure_dir
from parking_vision.utils.profiling import Profiler
from parking_vision.utils.video import iter_video_frames, write_video
from parking_vision.utils.visualization import draw_layout_overlay
from parking_vision.web.demo import ensure_demo_gallery, load_eval_summary, model_status_cards
from parking_vision.web.state import AppState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = os.environ.get("PARKING_VISION_CONFIG", "configs/app_default.yaml")


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def summarize_predictions(preds: list) -> dict:
    total = len(preds)
    occupied = sum(1 for p in preds if p.status == "occupied")
    free = sum(1 for p in preds if p.status == "free")
    unknown = sum(1 for p in preds if p.status == "unknown")
    known = occupied + free
    confidences = [float(p.confidence) for p in preds]
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    low_confidence = sum(1 for score in confidences if score < 0.6)
    coverage = known / total if total else 0.0
    unknown_rate = unknown / total if total else 0.0
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
        "occupancy_rate_known": occupancy_rate_known,
        "mean_confidence": mean_confidence,
        "low_confidence_slots": low_confidence,
        "quality_band": quality_band,
    }


def model_artifact_mode(card: dict) -> str:
    meta = card.get("artifact_meta", {})
    if meta.get("has_scaler") and meta.get("has_occupied_threshold") and meta.get("has_unknown_threshold"):
        return "fully calibrated"
    if meta.get("has_occupied_threshold") or meta.get("has_unknown_threshold"):
        return "threshold-calibrated legacy"
    if card["model_key"] == "fast_classic":
        return "legacy"
    return "default"


def reset_inference_view(show_preview: bool = True) -> None:
    st.session_state["last_result"] = None
    st.session_state["last_comparison"] = None
    st.session_state["hide_demo_preview"] = not show_preview


@st.cache_resource(show_spinner=False)
def get_state(config_path: str) -> AppState:
    return AppState(config_path=config_path)


@st.cache_data(show_spinner=False, ttl=10)
def get_dashboard_meta(config_path: str) -> dict:
    state = get_state(config_path)
    output_dir = ensure_dir(state.cfg["app"]["output_dir"])
    return {
        "demo_feed": ensure_demo_gallery(PROJECT_ROOT, output_dir),
        "model_cards": model_status_cards(state.cfg, PROJECT_ROOT),
        "evaluation": load_eval_summary(PROJECT_ROOT),
        "default_layout_path": state.cfg["app"]["layout_path"],
    }


def run_image_inference(config_path: str, frame: np.ndarray, model_key: str, layout_path: str) -> dict:
    state = get_state(config_path)
    profiler = Profiler()
    output_dir = ensure_dir(state.cfg["app"]["output_dir"])
    layout = load_layout(resolve_project_path(layout_path))
    model = state.get_model(model_key)
    if hasattr(model, "reset_state"):
        model.reset_state()
    preds, profile = profiler.time_call(model.predict_frame, frame, layout)
    overlay = draw_layout_overlay(frame, layout, preds)
    save_path = output_dir / f"streamlit_image_{model_key}_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(save_path), overlay)
    summary = summarize_predictions(preds)
    summary.update(
        {
            "model_key": model_key,
            "latency_ms": profile.latency_ms,
            "rss_mb": profile.rss_mb,
            "fps_estimate": float(1000.0 / max(profile.latency_ms, 1e-6)),
            "output_path": str(save_path),
            "slots": [p.__dict__ for p in preds],
        }
    )
    return summary


def run_video_inference(
    config_path: str,
    video_path: str,
    model_key: str,
    layout_path: str,
    max_frames: int,
    stride: int,
) -> dict:
    state = get_state(config_path)
    profiler = Profiler()
    output_dir = ensure_dir(state.cfg["app"]["output_dir"])
    layout = load_layout(resolve_project_path(layout_path))
    model = state.get_model(model_key)
    if hasattr(model, "reset_state"):
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

    save_path = output_dir / f"streamlit_video_{model_key}_{int(time.time() * 1000)}.mp4"
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
            "output_path": str(save_path) if frames_out else None,
            "slots": [p.__dict__ for p in last_preds],
        }
    )
    return summary


def run_demo(config_path: str, demo: dict, model_key: str) -> dict:
    if demo["kind"] == "image":
        frame = cv2.imread(demo["local_path"], cv2.IMREAD_COLOR)
        return run_image_inference(config_path, frame, model_key, demo["layout_path"])
    return run_video_inference(
        config_path,
        demo["local_path"],
        model_key,
        demo["layout_path"],
        max_frames=120,
        stride=4,
    )


def render_metrics_row(result: dict) -> None:
    cols = st.columns(2)
    cols[0].metric("Occupied", result["occupied"])
    cols[1].metric("Free", result["free"])

    cols = st.columns(2)
    cols[0].metric("Unknown", result["unknown"])
    cols[1].metric("Coverage", f"{result['coverage'] * 100:.1f}%")

    cols = st.columns(2)
    cols[0].metric("Occ. on known", f"{result['occupancy_rate_known'] * 100:.1f}%")
    cols[1].metric("Latency", f"{result['latency_ms']:.2f} ms")

    cols = st.columns(2)
    cols[0].metric("FPS", f"{result['fps_estimate']:.2f}")
    cols[1].metric("Confidence", f"{result['mean_confidence'] * 100:.1f}%")

    cols = st.columns(2)
    cols[0].metric("Low-conf slots", result["low_confidence_slots"])
    cols[1].metric("Quality band", result["quality_band"])


def render_slot_tables(result: dict) -> None:
    slot_df = pd.DataFrame(result["slots"])
    if slot_df.empty:
        st.info("No per-slot output.")
        return
    slot_df["signal"] = np.where(
        slot_df["confidence"] < 0.6,
        "weak",
        np.where(slot_df["confidence"] < 0.8, "moderate", "strong"),
    )
    problems = slot_df[
        (slot_df["status"].isin(["occupied", "unknown"])) | (slot_df["confidence"] < 0.75)
    ].copy()

    tab_all, tab_problems = st.tabs(["All slots", "Flagged slots"])
    with tab_all:
        st.dataframe(
            slot_df[["slot_id", "status", "confidence", "signal"]],
            use_container_width=True,
            height=520,
        )
    with tab_problems:
        if problems.empty:
            st.success("No occupied, unknown, or low-confidence slots in this run.")
        else:
            st.dataframe(
                problems[["slot_id", "status", "confidence", "signal"]],
                use_container_width=True,
                height=420,
            )


def render_result(result: dict, cards: list[dict], show_slot_tables: bool = True) -> None:
    card_map = {card["model_key"]: card for card in cards}
    card = card_map.get(result["model_key"], {})
    st.subheader(f"{card.get('label', result['model_key'])} inference")
    st.caption(
        f"Coverage {result['coverage'] * 100:.1f}% · "
        f"mean confidence {result['mean_confidence'] * 100:.1f}% · "
        f"quality band {result['quality_band']}"
    )
    output_path = result.get("output_path")
    media_col, metrics_col = st.columns([4.3, 1.35], gap="large")
    with media_col:
        if output_path and output_path.endswith(".mp4"):
            st.video(output_path)
        elif output_path:
            st.image(output_path, width=1100)
        else:
            st.info("No renderable output file was produced.")
    with metrics_col:
        render_metrics_row(result)

    if show_slot_tables:
        st.markdown("### Slot results")
        render_slot_tables(result)


def render_comparison_results(comparison: dict[str, dict], cards: list[dict]) -> None:
    card_map = {card["model_key"]: card for card in cards}
    ordered_keys = [card["model_key"] for card in cards if card["model_key"] in comparison]

    st.markdown("### Model comparison")
    st.caption(
        "Верхний ряд показывает визуальный результат инференса для каждой модели. "
        "Нижний ряд показывает только итоговые метрики. Таблицы по слотам для режима сравнения скрыты."
    )

    visual_cols = st.columns(len(ordered_keys), gap="large")
    for col, key in zip(visual_cols, ordered_keys):
        result = comparison[key]
        card = card_map.get(key, {})
        with col:
            with st.container(border=True):
                st.markdown(f"#### {card.get('label', key)}")
                st.caption(
                    f"{card.get('approach', '')} · "
                    f"artifact mode: {model_artifact_mode(card) if card else 'unknown'}"
                )
                output_path = result.get("output_path")
                if output_path and output_path.endswith(".mp4"):
                    st.video(output_path)
                elif output_path:
                    st.image(output_path, use_container_width=True)
                else:
                    st.info("No renderable output file was produced.")

    st.markdown("### Metrics comparison")
    metric_cols = st.columns(len(ordered_keys), gap="large")
    for col, key in zip(metric_cols, ordered_keys):
        result = comparison[key]
        card = card_map.get(key, {})
        with col:
            with st.container(border=True):
                st.markdown(f"#### {card.get('label', key)} metrics")
                render_metrics_row(result)

    comp_df = pd.DataFrame(
        [
            {
                "model": card_map.get(key, {}).get("label", key),
                "occupied": comparison[key]["occupied"],
                "free": comparison[key]["free"],
                "unknown": comparison[key]["unknown"],
                "coverage": round(comparison[key]["coverage"], 4),
                "mean_confidence": round(comparison[key]["mean_confidence"], 4),
                "latency_ms": round(comparison[key]["latency_ms"], 2),
                "fps": round(comparison[key]["fps_estimate"], 2),
            }
            for key in ordered_keys
        ]
    )

    st.markdown("### Compact summary")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


def render_demo_selector(demos: list[dict]) -> dict | None:
    if not demos:
        st.warning("No prepared demos available.")
        return None

    options = {
        f"{item['title']} · {item['kind']} · {item['weather']}": item
        for item in demos
    }
    default_label = list(options.keys())[0]
    selected_label = st.selectbox("Demo sample", list(options.keys()), index=0)
    demo = options.get(selected_label) or options[default_label]

    current_demo_id = demo["id"]
    previous_demo_id = st.session_state.get("selected_demo_id")
    if previous_demo_id != current_demo_id:
        st.session_state["selected_demo_id"] = current_demo_id
        reset_inference_view(show_preview=True)

    if not st.session_state.get("hide_demo_preview", False):
        preview_col, meta_col = st.columns([1.1, 1], gap="large")
        with preview_col:
            preview_path = demo.get("local_path") if demo.get("kind") == "image" else None
            if demo.get("kind") == "video":
                poster_name = Path(str(demo.get("poster_url", ""))).name
                if poster_name:
                    candidate = PROJECT_ROOT / "artifacts" / "web" / "demos" / "media" / poster_name
                    if candidate.exists():
                        preview_path = str(candidate)
            if preview_path and Path(preview_path).exists():
                st.image(preview_path, caption=demo["title"], use_container_width=True)
            else:
                st.info("Preview image is not available for this demo.")
        with meta_col:
            st.markdown(f"### {demo['title']}")
            st.write(demo.get("subtitle", ""))
            info_a, info_b, info_c = st.columns(3)
            info_a.metric("Type", demo["kind"])
            info_b.metric("Scene", demo["scene"])
            info_c.metric("Weather", demo["weather"])
            st.caption(f"Layout: {demo['layout_path']}")
    return demo


def inference_page(config_path: str, meta: dict) -> None:
    cards = meta["model_cards"]
    card_map = {card["label"]: card for card in cards}

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_comparison" not in st.session_state:
        st.session_state["last_comparison"] = None
    if "hide_demo_preview" not in st.session_state:
        st.session_state["hide_demo_preview"] = False

    st.title("Inference")
    st.caption("Pick media, run inference, and inspect the current overlay before looking at global model metrics.")

    with st.sidebar:
        st.subheader("Inference controls")
        selected_label = st.radio("Model", list(card_map.keys()))
        selected_model = card_map[selected_label]["model_key"]
        selected_card = card_map[selected_label]
        source = st.radio("Source", ["Demo", "Image", "Video", "Stream"])
        compare_demo = st.checkbox(
            "Compare models on demo",
            value=False,
            disabled=source != "Demo",
            help="Runs both models on the selected demo and shows side-by-side visual results and separate metric blocks.",
        )
        layout_path = st.text_input("Layout path", value=meta["default_layout_path"])
        st.info(
            f"{selected_card['label']}\n\n"
            f"{selected_card['approach']}\n\n"
            f"Artifact mode: {model_artifact_mode(selected_card)}"
        )
        if selected_card["model_key"] == "fast_classic":
            st.warning(
                "Fast classic is improved, but the current artifact is not a full retrain with scaler yet. "
                "It is a threshold-calibrated legacy classifier, so hard scenes can still degrade."
            )

    previous_model = st.session_state.get("selected_model_key")
    if previous_model != selected_model:
        st.session_state["selected_model_key"] = selected_model
        reset_inference_view(show_preview=True)

    if source == "Demo":
        demo = render_demo_selector(meta["demo_feed"])
        button_label = "Run comparison" if compare_demo else "Run prediction"
        run_pressed = st.button(button_label, type="primary", use_container_width=True)

        if demo and run_pressed:
            st.session_state["hide_demo_preview"] = True

            if compare_demo:
                st.session_state["last_comparison"] = {
                    card["model_key"]: run_demo(config_path, demo, card["model_key"]) for card in cards
                }
                st.session_state["last_result"] = None
            else:
                st.session_state["last_result"] = run_demo(config_path, demo, selected_model)
                st.session_state["last_comparison"] = None

            st.rerun()

    elif source == "Image":
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded and st.button("Predict image", type="primary"):
            st.session_state["hide_demo_preview"] = False
            array = np.frombuffer(uploaded.read(), dtype=np.uint8)
            frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
            st.session_state["last_result"] = run_image_inference(config_path, frame, selected_model, layout_path)
            st.session_state["last_comparison"] = None

    elif source == "Video":
        uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
        cols = st.columns(2)
        max_frames = cols[0].number_input("Max frames", min_value=8, value=180, step=1)
        stride = cols[1].number_input("Stride", min_value=1, value=5, step=1)
        if uploaded and st.button("Predict video", type="primary"):
            st.session_state["hide_demo_preview"] = False
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".mp4") as tmp:
                tmp.write(uploaded.read())
                temp_path = tmp.name
            st.session_state["last_result"] = run_video_inference(
                config_path, temp_path, selected_model, layout_path, int(max_frames), int(stride)
            )
            st.session_state["last_comparison"] = None

    else:
        stream_url = st.text_input("RTSP / stream URL")
        cols = st.columns(2)
        max_frames = cols[0].number_input("Max frames", min_value=8, value=120, step=1, key="stream_frames")
        stride = cols[1].number_input("Stride", min_value=1, value=5, step=1, key="stream_stride")
        if stream_url and st.button("Predict stream", type="primary"):
            st.session_state["hide_demo_preview"] = False
            st.session_state["last_result"] = run_video_inference(
                config_path, stream_url, selected_model, layout_path, int(max_frames), int(stride)
            )
            st.session_state["last_comparison"] = None

    comparison = st.session_state.get("last_comparison")
    result = st.session_state.get("last_result")

    if comparison:
        render_comparison_results(comparison, cards)
    elif result:
        render_result(result, cards, show_slot_tables=True)
    else:
        st.info("Choose a source and run inference. Parking output will appear here first, above everything else.")


def benchmark_page(meta: dict) -> None:
    st.title("Benchmarks")
    evaluation = meta["evaluation"]
    st.caption(evaluation["scope"])
    if not evaluation["models"]:
        st.info("No evaluation summary available.")
        return

    eval_df = pd.DataFrame(evaluation["models"])[
        [
            "label",
            "accuracy",
            "accuracy_known",
            "coverage",
            "unknown_rate",
            "f1_macro",
            "latency_ms_mean",
            "fps_estimate",
            "rss_mb_mean",
            "num_samples",
        ]
    ]
    st.dataframe(eval_df, use_container_width=True, height=220)

    with st.expander("Quality chart", expanded=False):
        chart_path = PROJECT_ROOT / "runs" / "eval" / "latest" / "quality_metrics.png"
        if chart_path.exists():
            st.image(str(chart_path), use_container_width=True)

    with st.expander("System chart", expanded=False):
        chart_path = PROJECT_ROOT / "runs" / "eval" / "latest" / "system_metrics.png"
        if chart_path.exists():
            st.image(str(chart_path), use_container_width=True)


def models_page(meta: dict) -> None:
    st.title("Models")
    st.caption("Global model notes and artifact state live here, not on the main inference screen.")
    cards = meta["model_cards"]

    for card in cards:
        with st.container(border=True):
            st.subheader(card["label"])
            cols = st.columns([1.2, 1, 1])
            cols[0].write(card["approach"])
            cols[1].metric("Artifact", "ready" if card["artifact_ready"] else "missing")
            cols[2].metric("Mode", model_artifact_mode(card))
            st.write(f"Strength: {card['strength']}")
            st.write(f"Trade-off: {card['cost']}")
            benchmark = card.get("benchmark")
            if benchmark:
                st.write(
                    f"Held-out metrics: accuracy {benchmark['accuracy']:.3f}, "
                    f"F1 {benchmark['f1_macro']:.3f}, coverage {benchmark['coverage'] * 100:.1f}%."
                )
            else:
                st.write("Held-out benchmark not available.")
            run_summary = card.get("run_summary") or {}
            if run_summary:
                st.json(run_summary)


def main() -> None:
    st.set_page_config(page_title="Parking Vision Workstation", layout="wide")

    with st.sidebar:
        config_path = st.text_input("Config path", value=DEFAULT_CONFIG)
        page = st.radio("Page", ["Inference", "Benchmarks", "Models"])

    meta = get_dashboard_meta(config_path)

    if page == "Inference":
        inference_page(config_path, meta)
    elif page == "Benchmarks":
        benchmark_page(meta)
    else:
        models_page(meta)


if __name__ == "__main__":
    main()