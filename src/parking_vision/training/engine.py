from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from parking_vision.utils.metrics import classification_metrics


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    f1_macro: float


def _amp_enabled(device: str, mixed_precision: bool) -> bool:
    return mixed_precision and str(device).startswith("cuda")


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device: str = "cpu",
    mixed_precision: bool = False,
    desc: str | None = None,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)
    amp_enabled = _amp_enabled(device, mixed_precision)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total_loss = 0.0
    seen = 0
    y_true = []
    y_pred = []

    progress = tqdm(loader, desc=desc or ("train" if is_train else "eval"), leave=False, dynamic_ncols=True)
    for batch in progress:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = len(images)
        seen += batch_size
        total_loss += float(loss.item()) * batch_size
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(targets.detach().cpu().tolist())

        running_metrics = classification_metrics(y_true, y_pred)
        progress.set_postfix(
            loss=f"{total_loss / max(seen, 1):.4f}",
            acc=f"{running_metrics.accuracy:.4f}",
            f1=f"{running_metrics.f1_macro:.4f}",
            seen=seen,
        )

    metrics = classification_metrics(y_true, y_pred)
    return EpochResult(
        loss=total_loss / max(len(loader.dataset), 1),
        accuracy=metrics.accuracy,
        f1_macro=metrics.f1_macro,
    )
