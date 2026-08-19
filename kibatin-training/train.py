"""Authenticate, train, evaluate, and export one target-specialized run."""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from huggingface_hub import get_token

from artifacts import export_artifact, resolved_model_max_length
from data import DataConfig, PreparedData, load_data
from evaluation import compute_metrics_factory, get_loss_schedule, write_test_outputs


VALID_TARGET_HEADS = ("dia", "ind", "tri")
DEFAULT_MODEL_NAME = "GerMedBERT/medbert-512"
GATED_MODEL_NAMES = {DEFAULT_MODEL_NAME}


@dataclass
class TrainingConfig:
    """Hyperparameters and output locations for one three-head training run.

    ``target_head`` selects scheduled loss emphasis and validation MCC; it does
    not remove the other two heads. ``artifact_dir`` is mandatory and must not
    already exist.
    """

    run_dir: str | Path
    artifact_dir: str | Path
    model_name: str = DEFAULT_MODEL_NAME
    target_head: Literal["dia", "ind", "tri"] = "tri"
    seed: int = 42
    batch_size: int = 8
    num_train_epochs: int = 15
    eval_steps: int | None = 500
    patience: int = 12
    lr_bert: float | None = 2e-5
    lr_pooler: float | None = 5e-5
    lr_other: float | None = 2e-4
    warmup_ratio: float = 0.1
    loss_weight_min_max: tuple[float, float] = (0.0, 1.0)
    loss_weights_increase_per_step: float = 0.2
    loss_weight_holdout: int = 1
    loss_weights: list[float] = field(default_factory=lambda: [1 / 3, 1 / 3, 1 / 3])
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    save_total_limit: int = 2


@dataclass
class TrainingResult:
    """Filesystem outputs produced by a completed training run.

    Evaluation paths are ``None`` when no held-out test split was supplied;
    the deployable artifact path is always present after successful training.
    """

    run_dir: Path
    artifact_dir: Path
    metrics_json: Path | None
    predictions_csv: Path | None


def _hf_auth_kwargs() -> tuple[dict[str, str | bool], str | None]:
    """Return Hugging Face auth arguments and their traceable token source.

    ``HF_TOKEN`` takes precedence over a cached Hugging Face login. No
    project-specific secret module or alternative environment variable is read.
    """
    environment_token = os.environ.get("HF_TOKEN")
    if environment_token:
        return {"use_auth_token": environment_token}, "HF_TOKEN"
    if get_token():
        return {"use_auth_token": True}, "Hugging Face cache"
    return {}, None


def _require_hf_auth_for_gated_model(
    model_name: str, token_source: str | None
) -> None:
    """Require an environment or cached token for a known gated backbone.

    Raises:
        RuntimeError: If a gated model is selected without authentication.
    """
    if model_name in GATED_MODEL_NAMES and token_source is None:
        raise RuntimeError(
            f"{model_name} is gated, but this process has no Hugging Face token. "
            "Run `huggingface-cli login` in this environment or export HF_TOKEN."
        )


def _raise_hf_model_access_error(
    model_name: str, token_source: str | None, error: OSError
) -> None:
    """Raise an actionable model-access error while preserving its cause."""
    raise RuntimeError(
        f"Could not access Hugging Face model {model_name}. Token source: "
        f"{token_source or 'none'}. Confirm model access and authentication."
    ) from error


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and every available CUDA device."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _task_values(data: PreparedData):
    """Return aligned dia, ind, and tri metadata used by training components."""
    tasks = [data.diagnoses, data.indications, data.triage]
    return {
        "id2label": [task.id2label for task in tasks],
        "label2id": [task.label2id for task in tasks],
        "num_labels": [task.num_labels for task in tasks],
        "class_weights": [task.class_weights for task in tasks],
        "train_labels": [task.train_labels for task in tasks],
        "validation_labels": [task.val_labels for task in tasks],
        "test_labels": [task.test_labels for task in tasks],
    }


def _assert_no_overlength_sequences(tokenizer, *encoding_sets) -> None:
    """Reject tokenized rows beyond a finite tokenizer maximum.

    Hugging Face sentinel maxima above 100,000 are treated as unspecified.
    Training deliberately does not truncate overlength text.

    Raises:
        ValueError: If any supplied token sequence exceeds the finite maximum.
    """
    maximum = getattr(tokenizer, "model_max_length", 512)
    if maximum is None or maximum > 100_000:
        return
    observed = max(
        (
            len(input_ids)
            for encodings in encoding_sets
            for input_ids in encodings["input_ids"]
        ),
        default=0,
    )
    if observed > maximum:
        raise ValueError(
            f"Found sequence length {observed} > model_max_length {maximum}. "
            "Training does not truncate input."
        )


def train_multitask_bert(data: PreparedData, config: TrainingConfig) -> TrainingResult:
    """Train one target-specialized three-head model and export its artifact.

    The function authenticates for the selected Hugging Face backbone, tokenizes
    every supplied split without truncation, constructs all three heads, moves
    the model explicitly to CUDA, and selects the best checkpoint by the
    target head's validation MCC. Artifact export is mandatory after training;
    compact held-out outputs are written only when test data exists.

    Returns:
        Resolved run, artifact, and optional evaluation output paths.

    Raises:
        RuntimeError: If gated-model authentication or access fails.
        ValueError: If target, tokenizer, sequence length, or architecture
            contracts are invalid.
    """
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        EarlyStoppingCallback,
        IntervalStrategy,
        TrainingArguments,
    )

    from models import (
        CustomModelConfig,
        MultiTaskModel,
        MultiTaskReportDataset,
        WeightLossScheduleCallback,
        WeightedMultiTaskTrainer,
        data_collator_factory,
    )

    if config.target_head not in VALID_TARGET_HEADS:
        raise ValueError("target_head must be one of: dia, ind, tri.")
    _seed_everything(config.seed)
    run_dir = Path(config.run_dir).expanduser().resolve()
    checkpoints_dir = run_dir / "checkpoints"
    evaluation_dir = run_dir / "evaluation"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    values = _task_values(data)
    auth_kwargs, token_source = _hf_auth_kwargs()
    _require_hf_auth_for_gated_model(config.model_name, token_source)
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, **auth_kwargs)
        base_config = AutoConfig.from_pretrained(config.model_name, **auth_kwargs)
    except OSError as exc:
        _raise_hf_model_access_error(config.model_name, token_source, exc)
    if "token_type_ids" not in getattr(tokenizer, "model_input_names", []):
        raise ValueError("The training tokenizer must provide token_type_ids.")

    tokenizer.model_max_length = resolved_model_max_length(tokenizer, base_config)
    train_encodings = tokenizer(
        data.input.train_texts,
        padding=False,
        truncation=False,
        return_token_type_ids=True,
    )
    validation_encodings = tokenizer(
        data.input.val_texts,
        padding=False,
        truncation=False,
        return_token_type_ids=True,
    )
    encoding_sets = [train_encodings, validation_encodings]
    test_encodings = None
    if data.input.test_texts:
        test_encodings = tokenizer(
            data.input.test_texts,
            padding=False,
            truncation=False,
            return_token_type_ids=True,
        )
        encoding_sets.append(test_encodings)
    _assert_no_overlength_sequences(tokenizer, *encoding_sets)

    train_dataset = MultiTaskReportDataset(
        train_encodings,
        values["train_labels"],
        values["label2id"],
        data.input.train_extra,
    )
    validation_dataset = MultiTaskReportDataset(
        validation_encodings,
        values["validation_labels"],
        values["label2id"],
        data.input.val_extra,
    )
    test_dataset = (
        MultiTaskReportDataset(
            test_encodings,
            values["test_labels"],
            values["label2id"],
            data.input.test_extra,
        )
        if test_encodings is not None
        else None
    )

    numeric_width = data.input.train_extra.shape[1]
    hidden_size = getattr(base_config, "hidden_size", 768)
    extra_features_scale = int(hidden_size / numeric_width)
    if extra_features_scale <= 0:
        raise ValueError(
            "Backbone hidden size is smaller than the numeric input width."
        )
    model_config = CustomModelConfig(
        type="MultiTaskModel",
        pretrained_model_name=config.model_name,
        _num_labels=values["num_labels"],
        _id2label=values["id2label"],
        _label2id=values["label2id"],
        num_extra_features=numeric_width,
        extra_features_scale=extra_features_scale,
        backbone_config=base_config.to_dict(),
        target_head=config.target_head,
    )
    try:
        model = MultiTaskModel(
            model_config,
            initialize_backbone_from_pretrained=True,
            **auth_kwargs,
        )
    except OSError as exc:
        _raise_hf_model_access_error(config.model_name, token_source, exc)
    model.to("cuda")

    eval_steps = config.eval_steps or math.ceil(len(train_dataset) / config.batch_size)
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        optim="adamw_torch",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=eval_steps,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=eval_steps,
        save_total_limit=config.save_total_limit,
        save_safetensors=False,
        load_best_model_at_end=True,
        metric_for_best_model="val_metric",
        logging_strategy="steps",
        logging_steps=eval_steps,
        log_level="info",
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = WeightedMultiTaskTrainer(
        class_weights=values["class_weights"],
        loss_schedule_fn=get_loss_schedule(
            config.target_head,
            increase_per_step=config.loss_weights_increase_per_step,
        ),
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_factory(config.target_head),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
        lr_bert=config.lr_bert,
        lr_pooler=config.lr_pooler,
        lr_other=config.lr_other,
        data_collator=data_collator_factory(tokenizer),
        loss_weight_min_max=config.loss_weight_min_max,
        loss_weights=config.loss_weights,
    )
    trainer.add_callback(
        WeightLossScheduleCallback(trainer, holdout=config.loss_weight_holdout)
    )
    trainer.train()

    artifact_dir = export_artifact(
        artifact_dir=config.artifact_dir,
        model=trainer.model,
        data=data,
        tokenizer=tokenizer,
        base_config=base_config,
        seed=config.seed,
    )
    metrics_json = None
    predictions_csv = None
    if test_dataset is not None:
        target_index = {"dia": 0, "ind": 1, "tri": 2}[config.target_head]
        metrics_json, predictions_csv = write_test_outputs(
            trainer=trainer,
            test_dataset=test_dataset,
            test_ids=data.input.test_ids,
            target_head=config.target_head,
            id2label=values["id2label"][target_index],
            output_dir=evaluation_dir,
        )
    return TrainingResult(
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        metrics_json=metrics_json,
        predictions_csv=predictions_csv,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit train, validation, output, and target CLI workflow."""
    parser = argparse.ArgumentParser(
        description="Train and export one target-specialized KIBATIN model."
    )
    parser.add_argument("--target-head", choices=VALID_TARGET_HEADS, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--external-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the processed rules_value field to narrative input.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Load configured splits, train and export, then print produced paths."""
    args = parse_args()
    data = load_data(
        DataConfig(
            train_file=args.train_file,
            validation_file=args.validation_file,
            test_file=args.test_file,
            external_labels=args.external_labels,
        )
    )
    result = train_multitask_bert(
        data,
        TrainingConfig(
            run_dir=args.run_dir,
            artifact_dir=args.artifact_dir,
            model_name=args.model_name,
            target_head=args.target_head,
            seed=args.seed,
        ),
    )
    print(f"Training target: {args.target_head}")
    print(f"Artifact: {result.artifact_dir}")
    if result.metrics_json is not None:
        print(f"Metrics: {result.metrics_json}")
        print(f"Predictions: {result.predictions_csv}")
    else:
        print("No test CSV supplied; skipped final test outputs.")


if __name__ == "__main__":
    main()
