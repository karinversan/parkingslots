from __future__ import annotations

from pathlib import Path

import pandas as pd

from parking_vision.utils.io import ensure_dir


def render_markdown_report(summary_csv: str, output_path: str) -> None:
    df = pd.read_csv(summary_csv)
    out_path = Path(output_path)
    ensure_dir(out_path.parent)

    lines = [
        "# Parking Vision Workstation Evaluation Report",
        "",
        "## Summary",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | Mean Latency ms | FPS | RSS MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.model_name} | {row.accuracy:.4f} | {row.precision_macro:.4f} | {row.recall_macro:.4f} | "
            f"{row.f1_macro:.4f} | {row.latency_ms_mean:.2f} | {row.fps_estimate:.2f} | {row.rss_mb_mean:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Generated artifacts",
            "",
            "- `quality_metrics.png`",
            "- `system_metrics.png`",
            "- `<model>_confusion.png`",
            "- `<model>_predictions.csv`",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
