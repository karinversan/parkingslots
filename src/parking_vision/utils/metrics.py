from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


LABEL_TO_INT = {"free": 0, "occupied": 1, "unknown": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


@dataclass
class MetricBundle:
    accuracy: float
    accuracy_known: float
    coverage: float
    unknown_rate: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_free: float
    recall_free: float
    f1_free: float
    precision_occupied: float
    recall_occupied: float
    f1_occupied: float


def normalize_labels(values: Iterable[str | int]) -> List[int]:
    normalized = []
    for value in values:
        if isinstance(value, int):
            normalized.append(value)
        else:
            normalized.append(LABEL_TO_INT[str(value)])
    return normalized


def classification_metrics(y_true: Iterable[str | int], y_pred: Iterable[str | int]) -> MetricBundle:
    yt = normalize_labels(y_true)
    yp = normalize_labels(y_pred)
    covered = [pred != LABEL_TO_INT["unknown"] for pred in yp]
    coverage = float(sum(covered) / len(covered)) if covered else 0.0
    known_pairs = [(true, pred) for true, pred in zip(yt, yp) if pred != LABEL_TO_INT["unknown"]]
    if known_pairs:
        yt_known, yp_known = zip(*known_pairs)
        accuracy_known = float(accuracy_score(yt_known, yp_known))
    else:
        accuracy_known = 0.0
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(yt, yp, average="macro", labels=[0, 1], zero_division=0)
    return MetricBundle(
        accuracy=float(accuracy_score(yt, yp)),
        accuracy_known=accuracy_known,
        coverage=coverage,
        unknown_rate=float(1.0 - coverage),
        precision_macro=float(pm),
        recall_macro=float(rm),
        f1_macro=float(fm),
        precision_free=float(p[0]),
        recall_free=float(r[0]),
        f1_free=float(f1[0]),
        precision_occupied=float(p[1]),
        recall_occupied=float(r[1]),
        f1_occupied=float(f1[1]),
    )


def compute_confusion(y_true: Iterable[str | int], y_pred: Iterable[str | int]) -> np.ndarray:
    yt = normalize_labels(y_true)
    yp = normalize_labels(y_pred)
    return confusion_matrix(yt, yp, labels=[0, 1, 2])


def flicker_rate(slot_states: Dict[str, List[str]]) -> float:
    transitions = 0
    observations = 0
    for _, states in slot_states.items():
        if len(states) < 2:
            continue
        observations += len(states) - 1
        transitions += sum(1 for a, b in zip(states[:-1], states[1:]) if a != b)
    return float(transitions / observations) if observations else 0.0
