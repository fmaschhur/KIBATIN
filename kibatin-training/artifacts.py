"""Build validated manifests and atomically export deployable artifacts."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch


VALID_TARGET_HEADS = ("dia", "ind", "tri")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write an indented UTF-8 JSON artifact with a trailing newline."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_and_validate_labels(
    *,
    num_labels: list[int],
    id2label: list[dict],
    label2id: list[dict],
) -> tuple[list[dict[int, str]], list[dict[str, int]]]:
    """Return normalized, inverse label maps for the three model heads.

    Diagnosis, indication, and triage maps must have contiguous zero-based
    indices, matching label counts, string values, and at least three labels.

    Raises:
        ValueError: If any count or mapping violates the artifact contract.
    """
    if not all(
        isinstance(value, list) and len(value) == 3
        for value in (num_labels, id2label, label2id)
    ):
        raise ValueError("Artifacts require three task label maps and counts.")
    normalized_id2label: list[dict[int, str]] = []
    normalized_label2id: list[dict[str, int]] = []
    for task_index, count in enumerate(num_labels):
        if isinstance(count, bool) or not isinstance(count, int) or count < 3:
            raise ValueError("Every artifact task must contain at least three labels.")
        try:
            forward = {
                int(index): label for index, label in id2label[task_index].items()
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Artifact label indices must be integers.") from exc
        if sorted(forward) != list(range(count)) or not all(
            isinstance(label, str) for label in forward.values()
        ):
            raise ValueError("Artifact label maps must be contiguous and string-valued.")
        inverse = dict(label2id[task_index])
        if inverse != {label: index for index, label in forward.items()}:
            raise ValueError("Artifact label maps must be exact inverses.")
        normalized_id2label.append(forward)
        normalized_label2id.append(inverse)
    return normalized_id2label, normalized_label2id


def resolved_model_max_length(tokenizer, base_config) -> int:
    """Return a finite tokenizer maximum compatible with the backbone.

    Missing tokenizer limits and Hugging Face sentinel values above 100,000 use
    the backbone maximum instead.

    Raises:
        ValueError: If either effective limit is invalid or incompatible.
    """
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    backbone_max = getattr(base_config, "max_position_embeddings", None)
    if (
        isinstance(backbone_max, bool)
        or not isinstance(backbone_max, int)
        or backbone_max <= 0
    ):
        raise ValueError("Backbone config requires positive max_position_embeddings.")
    if tokenizer_max is None or (
        isinstance(tokenizer_max, int) and tokenizer_max > 100_000
    ):
        tokenizer_max = backbone_max
    if (
        isinstance(tokenizer_max, bool)
        or not isinstance(tokenizer_max, int)
        or tokenizer_max <= 0
        or tokenizer_max > backbone_max
    ):
        raise ValueError("Tokenizer maximum is incompatible with the backbone.")
    return tokenizer_max


def build_model_manifest(*, model_config, data, seed: int) -> dict[str, Any]:
    """Build a validated model manifest from configuration and prepared data.

    The result records the selected target alongside the shared backbone,
    three label maps, numeric architecture, and training seed. Model metadata
    must exactly match the prepared split-wide label sets and numeric width.

    Raises:
        ValueError: If target, labels, backbone, defaults, or widths disagree.
    """
    target_head = getattr(model_config, "target_head", None)
    if target_head not in VALID_TARGET_HEADS:
        raise ValueError("Artifact target_head must be dia, ind, or tri.")
    backbone_config = getattr(model_config, "backbone_config", None)
    if not isinstance(backbone_config, dict) or not backbone_config.get("model_type"):
        raise ValueError(
            "Artifact requires a complete serialized backbone_config."
        )

    tasks = [data.diagnoses, data.indications, data.triage]
    num_labels = [task.num_labels for task in tasks]
    id2label = [task.id2label for task in tasks]
    label2id = [task.label2id for task in tasks]
    normalized_id2label, normalized_label2id = normalize_and_validate_labels(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    config_id2label, config_label2id = normalize_and_validate_labels(
        num_labels=list(getattr(model_config, "_num_labels", [])),
        id2label=list(getattr(model_config, "_id2label", [])),
        label2id=list(getattr(model_config, "_label2id", [])),
    )
    if (
        list(model_config._num_labels) != num_labels
        or config_id2label != normalized_id2label
        or config_label2id != normalized_label2id
    ):
        raise ValueError("Model label metadata does not match the prepared data.")

    numeric_features = data.preprocessing.numeric_features
    expected_width = len(numeric_features) + len(
        data.preprocessing.missingness_indicator_order
    )
    if data.input.train_extra.shape[1] != expected_width:
        raise ValueError("Training numeric width does not match preprocessing.")
    if int(model_config.num_extra_features) != expected_width:
        raise ValueError("Model numeric width does not match preprocessing.")
    if any(not math.isfinite(float(feature.default)) for feature in numeric_features):
        raise ValueError("Numeric preprocessing defaults must be finite.")

    return {
        "target_head": target_head,
        "pretrained_model_name": model_config.pretrained_model_name,
        "training_seed": int(seed),
        "backbone_config": backbone_config,
        "num_labels": num_labels,
        "id2label": normalized_id2label,
        "label2id": normalized_label2id,
        "num_extra_features": int(model_config.num_extra_features),
        "extra_features_scale": int(model_config.extra_features_scale),
    }


def export_artifact(
    *,
    artifact_dir: str | Path,
    model,
    data,
    tokenizer,
    base_config,
    seed: int,
) -> Path:
    """Export one complete model artifact without overwriting existing output.

    Files are written to a temporary sibling directory, checked for the
    required manifests, checkpoint, and tokenizer metadata, then atomically
    moved into place. The temporary directory is removed if any step fails.
    The tokenizer maximum is normalized and persisted as part of the export.

    Returns:
        The resolved target artifact directory.

    Raises:
        FileExistsError: If the target path or symlink already exists.
        ValueError: If tokenizer, model, data, or exported files are invalid.
    """
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    if artifact_dir.exists() or artifact_dir.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite existing artifact: {artifact_dir}"
        )
    if "token_type_ids" not in getattr(tokenizer, "model_input_names", []):
        raise ValueError("Artifact tokenizer must provide token_type_ids.")

    model_max_length = resolved_model_max_length(tokenizer, base_config)
    tokenizer.model_max_length = model_max_length
    manifest = build_model_manifest(model_config=model.config, data=data, seed=seed)
    preprocessing = data.preprocessing.to_manifest(
        model_max_length=model_max_length
    )

    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{artifact_dir.name}.", dir=artifact_dir.parent)
    )
    try:
        _write_json(temporary / "model.json", manifest)
        _write_json(temporary / "preprocessing.json", preprocessing)
        torch.save(model.state_dict(), temporary / "pytorch_model.bin")
        tokenizer.save_pretrained(temporary)
        required = {
            "model.json",
            "preprocessing.json",
            "pytorch_model.bin",
            "tokenizer_config.json",
        }
        missing = sorted(
            name for name in required if not (temporary / name).is_file()
        )
        if missing:
            raise ValueError(f"Artifact export is missing required files: {missing}")
        os.replace(temporary, artifact_dir)
        return artifact_dir
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
