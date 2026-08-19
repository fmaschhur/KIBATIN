"""Compute target-specific metrics and optional held-out test outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score


VALID_TARGET_HEADS = ("dia", "ind", "tri")
TARGET_INDEX = {"dia": 0, "ind": 1, "tri": 2}


def _unpack_logits(logits):
    """Return diagnosis, indication, and triage logits in fixed head order."""
    if isinstance(logits, dict):
        return logits["dia"], logits["ind"], logits["tri"]
    return logits[0], logits[1], logits[2]


def _metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Return compact classification metrics from logits and integer labels."""
    predictions = np.argmax(logits, axis=1)
    return {
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


def compute_metrics_factory(target_head: str):
    """Build a Hugging Face metric hook for one checkpoint-selection target.

    The returned hook evaluates only the selected head and aliases its
    Matthews correlation coefficient as ``val_metric``.

    Raises:
        ValueError: If ``target_head`` is not ``dia``, ``ind``, or ``tri``.
    """
    if target_head not in VALID_TARGET_HEADS:
        raise ValueError("target_head must be one of: dia, ind, tri.")
    target_index = TARGET_INDEX[target_head]

    def compute_metrics(eval_pred):
        """Compute selected-head validation metrics from trainer predictions."""
        logits, labels = eval_pred
        values = _metrics(
            np.asarray(_unpack_logits(logits)[target_index]),
            np.asarray(labels)[:, target_index],
        )
        return {**values, "val_metric": values["mcc"]}

    return compute_metrics


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Return numerically stable row-wise probabilities for a logits matrix."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _top_indices(row: np.ndarray, count: int = 3) -> list[int]:
    """Return up to ``count`` class indices ordered by descending score."""
    count = min(count, row.shape[0])
    indices = np.argpartition(row, -count)[-count:]
    return [int(index) for index in indices[np.argsort(-row[indices])]]


def write_test_outputs(
    *,
    trainer,
    test_dataset,
    test_ids: list[Any],
    target_head: str,
    id2label: dict[int, str],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Evaluate one target head and write compact held-out test artifacts.

    ``metrics.json`` contains the target, row count, and aggregate metrics.
    ``predictions.csv`` contains row IDs, gold labels, and up to three labels
    with unrounded probabilities in descending order.

    Returns:
        Paths to ``metrics.json`` and ``predictions.csv``.

    Raises:
        ValueError: If the target is invalid or row IDs do not align with labels.
    """
    if target_head not in VALID_TARGET_HEADS:
        raise ValueError("target_head must be one of: dia, ind, tri.")
    predictions, labels, _ = trainer.predict(
        test_dataset=test_dataset, metric_key_prefix="test"
    )
    target_index = TARGET_INDEX[target_head]
    logits = np.asarray(_unpack_logits(predictions)[target_index])
    target_labels = np.asarray(labels)[:, target_index]
    if len(test_ids) != len(target_labels):
        raise ValueError("Test row IDs do not align with predictions.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "target_head": target_head,
                "count": len(target_labels),
                "metrics": _metrics(logits, target_labels),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    probabilities = _softmax(logits)
    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "id",
                "gold_label",
                "top1_label",
                "top1_probability",
                "top2_label",
                "top2_probability",
                "top3_label",
                "top3_probability",
            ]
        )
        for row_id, gold, row in zip(test_ids, target_labels, probabilities):
            top = _top_indices(row)
            ranked: list[Any] = []
            for index in top:
                ranked.extend([id2label[index], float(row[index])])
            while len(ranked) < 6:
                ranked.extend(["", ""])
            writer.writerow([row_id, id2label[int(gold)], *ranked])
    return metrics_path, predictions_path


def get_loss_schedule(target_head: str, increase_per_step: float = 0.1):
    """Build a loss-weight schedule emphasizing one target head.

    Each call increases the selected weight by ``increase_per_step`` and
    decreases each other weight by half that amount. Clipping is performed by
    the trainer after applying this schedule.

    Raises:
        ValueError: If ``target_head`` is not ``dia``, ``ind``, or ``tri``.
    """
    if target_head not in VALID_TARGET_HEADS:
        raise ValueError("target_head must be one of: dia, ind, tri.")
    target_index = TARGET_INDEX[target_head]

    def schedule(weights):
        """Return the next un-clipped diagnosis, indication, and triage weights."""
        return [
            weight + increase_per_step
            if index == target_index
            else weight - increase_per_step / 2
            for index, weight in enumerate(weights)
        ]

    return schedule
