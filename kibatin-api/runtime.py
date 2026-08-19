"""Validate artifacts and build the three offline API inference runtimes.

This module owns preprocessing parity, code-map coverage, strict checkpoint
loading, and startup validation for diagnosis, indication, and triage.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from model import MultiTaskModel


PREPROCESSING_FORMAT_VERSION = 1
PUBLIC_OPTIONS = ("Diagramm", "Indikator", "Triage")
MODE_SPECS = {
    "Diagramm": ("dia", 0, "diagnoses_codes.json"),
    "Indikator": ("ind", 1, "indications_codes.json"),
    "Triage": ("tri", 2, "triage_codes.json"),
}
PUBLIC_TEXT_FIELDS = (
    "notfallgeschehentext",
    "diagnose_text",
    "symptoms",
    "bemerkung",
)
EXPECTED_NUMERIC_FEATURES = (
    "massnahmen_total_nr",
    "ergebnis_schockraum",
    "notaufnahme_messwerte_schmerz",
    "uebergabe_neurologie.glasgowcomascale",
    "uebergabe_motor_schmerzkeine",
    "naca_score_1",
    "naca_score_2",
    "naca_score_3",
    "naca_score_4",
    "naca_score_5",
    "naca_score_6",
    "naca_score_7",
)
EXPECTED_MISSINGNESS_INDICATORS = (
    "massnahmen_total_nr",
    "notaufnahme_messwerte_schmerz",
    "uebergabe_neurologie.glasgowcomascale",
)
API_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = API_DIR.parent / "weights"
DEFAULT_CODE_MAP_DIR = API_DIR / "code_maps"
MODEL_MANIFEST_FIELDS = {
    "target_head",
    "pretrained_model_name",
    "training_seed",
    "backbone_config",
    "num_labels",
    "id2label",
    "label2id",
    "num_extra_features",
    "extra_features_scale",
}
PREPROCESSING_MANIFEST_FIELDS = {
    "format_version",
    "text_fields",
    "rules_value",
    "numeric_features",
    "missingness_indicator_order",
    "tokenizer",
}


@dataclass(frozen=True)
class NumericFeature:
    """Validated numeric feature metadata from a preprocessing artifact.

    Attributes:
        name: Public request key read for the feature.
        default: Training-derived median or fixed zero used when input is missing.
        missing_indicator: Whether the feature contributes a missingness flag.
    """

    name: str
    default: float
    missing_indicator: bool


@dataclass(frozen=True)
class PreprocessingConfig:
    """Validated preprocessing contract shared by training and inference.

    The contract fixes narrative order, optional rules transformation, numeric
    feature defaults and ordering, missingness indicators, and tokenizer length.
    """

    text_fields: tuple[str, ...]
    rules_value_enabled: bool
    numeric_features: tuple[NumericFeature, ...]
    missingness_indicator_order: tuple[str, ...]
    model_max_length: int

    @property
    def feature_width(self) -> int:
        """Return the combined numeric and missingness-indicator width."""
        return len(self.numeric_features) + len(self.missingness_indicator_order)


@dataclass
class ModeRuntime:
    """Loaded model resources for one public prediction option.

    ``internal_task_key`` selects the corresponding ``dia``, ``ind``, or
    ``tri`` logits from a model that always contains all three heads.
    """

    model: MultiTaskModel
    tokenizer: Any
    preprocessing: PreprocessingConfig
    id2labels: dict[int, str]
    code_map: dict[str, str]
    internal_task_key: str
    device: torch.device


@dataclass
class RuntimeAssets:
    """Complete mapping of public options to eagerly loaded runtimes."""

    modes: dict[str, ModeRuntime]


def read_json(path: Path) -> dict[str, Any]:
    """Read a required UTF-8 JSON artifact containing a top-level object.

    Raises:
        ValueError: If the file is missing, invalid JSON, or not an object.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def normalize_and_validate_labels(
    manifest: dict[str, Any],
) -> tuple[list[dict[int, str]], list[dict[str, int]], list[int]]:
    """Return normalized, mutually inverse label maps for all three tasks.

    The manifest must declare diagnosis, indication, and triage maps with
    contiguous zero-based indices, matching counts, and at least three labels
    per task.

    Raises:
        ValueError: If any label count or mapping violates the artifact schema.
    """
    raw_id2label = manifest.get("id2label")
    raw_label2id = manifest.get("label2id")
    raw_num_labels = manifest.get("num_labels")
    if not all(
        isinstance(value, list) and len(value) == 3
        for value in (raw_id2label, raw_label2id, raw_num_labels)
    ):
        raise ValueError("Model artifact requires three label maps and counts.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 3
        for value in raw_num_labels
    ):
        raise ValueError("Every task must contain at least three labels.")

    id2labels: list[dict[int, str]] = []
    label2ids: list[dict[str, int]] = []
    for task_index, count in enumerate(raw_num_labels):
        try:
            id2label = {
                int(index): label
                for index, label in raw_id2label[task_index].items()
            }
            label2id = dict(raw_label2id[task_index])
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Invalid label mapping in model artifact.") from exc
        if sorted(id2label) != list(range(count)) or not all(
            isinstance(label, str) for label in id2label.values()
        ):
            raise ValueError("Model label indices must be contiguous and string-valued.")
        if label2id != {label: index for index, label in id2label.items()}:
            raise ValueError("label2id must be the exact inverse of id2label.")
        id2labels.append(id2label)
        label2ids.append(label2id)
    return id2labels, label2ids, list(raw_num_labels)


def validate_model_manifest(
    manifest: dict[str, Any], *, expected_target: str
) -> None:
    """Validate the exact model-manifest schema and target identity.

    Validation covers target placement, seed and backbone metadata, all three
    label maps, and positive numeric architecture dimensions.

    Raises:
        ValueError: If any required field or value violates the model contract.
    """
    if set(manifest) != MODEL_MANIFEST_FIELDS:
        raise ValueError("Model artifact fields do not match the schema.")
    if manifest.get("target_head") != expected_target:
        raise ValueError("Model target_head does not match its artifact directory.")
    if not isinstance(manifest.get("pretrained_model_name"), str) or not manifest[
        "pretrained_model_name"
    ]:
        raise ValueError("Model artifact requires pretrained_model_name.")
    seed = manifest.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Model artifact requires an integer training_seed.")
    normalize_and_validate_labels(manifest)
    backbone = manifest.get("backbone_config")
    if not isinstance(backbone, dict) or not backbone.get("model_type"):
        raise ValueError("Model artifact requires a complete backbone_config.")
    if isinstance(backbone.get("hidden_size"), bool) or not isinstance(
        backbone.get("hidden_size"), int
    ):
        raise ValueError("Backbone config requires hidden_size.")
    width = manifest.get("num_extra_features")
    scale = manifest.get("extra_features_scale")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (width, scale)
    ) or int(width * scale / 2) <= 0:
        raise ValueError("Invalid extra-feature dimensions in model artifact.")


def load_preprocessing_config(path: Path) -> tuple[PreprocessingConfig, dict[str, Any]]:
    """Load and validate preprocessing metadata against the API contract.

    The accepted schema requires the fixed text and numeric order, three
    training-compatible missingness indicators, finite defaults, no tokenizer
    truncation, and token type IDs.

    Returns:
        The typed preprocessing configuration and its raw manifest.

    Raises:
        ValueError: If the artifact differs from the supported schema.
    """
    manifest = read_json(path)
    if set(manifest) != PREPROCESSING_MANIFEST_FIELDS:
        raise ValueError("Preprocessing fields do not match the declared schema.")
    if manifest.get("format_version") != PREPROCESSING_FORMAT_VERSION:
        raise ValueError("Unsupported preprocessing artifact format.")
    text_fields = tuple(manifest.get("text_fields", ()))
    if text_fields != PUBLIC_TEXT_FIELDS:
        raise ValueError("Preprocessing text order does not match the API contract.")
    rules = manifest.get("rules_value")
    if (
        not isinstance(rules, dict)
        or set(rules) != {"field", "enabled", "format"}
        or rules.get("field") != "rules_value"
        or rules.get("format") != "kibatin_rules_v1"
        or not isinstance(rules.get("enabled"), bool)
    ):
        raise ValueError("Invalid rules_value preprocessing metadata.")

    raw_features = manifest.get("numeric_features")
    if not isinstance(raw_features, list):
        raise ValueError("Preprocessing metadata requires numeric_features.")
    features: list[NumericFeature] = []
    for raw in raw_features:
        if not isinstance(raw, dict):
            raise ValueError("Invalid numeric feature metadata.")
        if set(raw) != {"name", "default", "missing_indicator"}:
            raise ValueError("Invalid numeric feature metadata.")
        name = raw.get("name")
        default = raw.get("default")
        indicator = raw.get("missing_indicator")
        if (
            not isinstance(name, str)
            or isinstance(default, bool)
            or not isinstance(default, (int, float))
            or not math.isfinite(float(default))
            or not isinstance(indicator, bool)
        ):
            raise ValueError("Invalid numeric feature metadata.")
        features.append(NumericFeature(name, float(default), indicator))
    if tuple(feature.name for feature in features) != EXPECTED_NUMERIC_FEATURES:
        raise ValueError("Numeric feature order does not match training.")
    missingness = tuple(manifest.get("missingness_indicator_order", ()))
    if missingness != tuple(
        feature.name for feature in features if feature.missing_indicator
    ) or missingness != EXPECTED_MISSINGNESS_INDICATORS:
        raise ValueError("Missingness indicators do not match training.")
    if any(
        feature.default != 0.0
        for feature in features
        if not feature.missing_indicator
    ):
        raise ValueError("Fixed-default numeric features must use zero.")

    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or set(tokenizer) != {
        "model_max_length",
        "truncation",
        "return_token_type_ids",
    }:
        raise ValueError("Invalid tokenizer preprocessing metadata.")
    model_max_length = (
        tokenizer.get("model_max_length") if isinstance(tokenizer, dict) else None
    )
    if (
        isinstance(model_max_length, bool)
        or not isinstance(model_max_length, int)
        or model_max_length <= 0
        or tokenizer.get("truncation") is not False
        or tokenizer.get("return_token_type_ids") is not True
    ):
        raise ValueError("Invalid tokenizer preprocessing metadata.")
    return (
        PreprocessingConfig(
            text_fields=text_fields,
            rules_value_enabled=rules["enabled"],
            numeric_features=tuple(features),
            missingness_indicator_order=missingness,
            model_max_length=model_max_length,
        ),
        manifest,
    )


def preprocess_rules_value(value: Any) -> str:
    """Return the shared compact representation of an external rules value.

    Non-string and empty values produce an empty string. Each pipe-delimited
    rule is stripped of known source markers, reversed around its semicolon
    fields, and joined with the remaining rules.
    """
    if not isinstance(value, str) or value == "":
        return ""
    return ", ".join(
        ": ".join(
            reversed(
                part.replace("Vital-Parameter; ", "")
                .replace("; Erstbefund", "")
                .replace("; Übergabe", "")
                .split("; ")
            )
        )
        for part in value.split("|")
    )


def build_input_text(data: dict[str, Any], config: PreprocessingConfig) -> str:
    """Build narrative text in the training-defined field and rules order.

    Non-string narrative values become empty fields. When external rules are
    enabled and nonempty, their transformed value follows one ``[SEP]`` token.
    """
    narrative = " ".join(
        value if isinstance((value := data.get(field)), str) else ""
        for field in config.text_fields
    )
    if not config.rules_value_enabled:
        return narrative
    rules = preprocess_rules_value(data.get("rules_value"))
    return narrative + " [SEP] " + rules if rules else narrative


def _is_missing(value: Any) -> bool:
    """Return whether an API numeric value is absent, empty, or NaN."""
    return value is None or value == "" or (
        isinstance(value, float) and math.isnan(value)
    )


def _coerce_numeric(value: Any, *, feature_name: str) -> float:
    """Convert one API feature value to a float with compatible text handling.

    Booleans and case-insensitive boolean strings map to zero or one, and
    decimal commas are accepted.

    Raises:
        ValueError: If the value cannot be converted, naming the failed feature.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
        value = value.replace(",", ".")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for feature {feature_name}.") from exc


def build_numeric_vector(
    data: dict[str, Any], config: PreprocessingConfig
) -> list[float]:
    """Build the ordered numeric vector and append missingness indicators.

    Missing base values use the artifact defaults fitted during training. The
    indicator suffix follows ``missingness_indicator_order`` exactly.
    """
    values: list[float] = []
    missing: dict[str, bool] = {}
    for feature in config.numeric_features:
        raw = data.get(feature.name)
        missing[feature.name] = _is_missing(raw)
        values.append(
            feature.default
            if missing[feature.name]
            else _coerce_numeric(raw, feature_name=feature.name)
        )
    values.extend(
        1.0 if missing[name] else 0.0
        for name in config.missingness_indicator_order
    )
    return values


def preprocess_input(
    data: dict[str, Any],
    tokenizer,
    config: PreprocessingConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Tokenize and move one request's text and numeric features to a device.

    Tokenization never truncates and must return token type IDs. The function
    rejects sequences beyond the artifact maximum and adds a single-row
    ``extra_features`` tensor before moving every tensor to ``device``.

    Raises:
        ValueError: If required token type IDs are absent or input is overlength.
    """
    inputs = tokenizer(
        build_input_text(data, config),
        return_tensors="pt",
        padding=False,
        truncation=False,
        return_token_type_ids=True,
    )
    if "token_type_ids" not in inputs:
        raise ValueError("Tokenizer did not return required token_type_ids.")
    length = int(inputs["input_ids"].shape[-1])
    if length > config.model_max_length:
        raise ValueError(
            f"Input sequence length {length} exceeds model_max_length "
            f"{config.model_max_length}; truncation is disabled."
        )
    inputs["extra_features"] = torch.tensor(
        [build_numeric_vector(data, config)], dtype=torch.float32
    )
    return {key: value.to(device) for key, value in inputs.items()}


def load_code_map(path: Path) -> tuple[dict[str, str], str]:
    """Load a label-to-external-ID mapping and its source marker.

    Identity-fallback artifacts must map every label to itself. This validation
    does not assert that diagnosis or indication mappings are authoritative.

    Raises:
        ValueError: If the artifact values or identity mapping is invalid.
    """
    artifact = read_json(path)
    source = artifact.get("source")
    codes = artifact.get("codes")
    if not isinstance(source, str) or not isinstance(codes, dict) or not all(
        isinstance(label, str) and isinstance(code, str)
        for label, code in codes.items()
    ):
        raise ValueError(f"Invalid label-code artifact: {path}")
    if source == "identity_fallback" and any(
        label != code for label, code in codes.items()
    ):
        raise ValueError("Identity-fallback codes must map labels to themselves.")
    return codes, source


def validate_code_coverage(
    option: str, id2labels: dict[int, str], code_map: dict[str, str]
) -> None:
    """Require an external code-map entry for every label served by an option.

    Raises:
        ValueError: If one or more selected model labels lack a mapping.
    """
    missing = sorted(set(id2labels.values()) - set(code_map))
    if missing:
        raise ValueError(f"Label-code mapping for {option} is missing labels: {missing}")


def validate_checkpoint_heads(
    state_dict: dict[str, torch.Tensor], num_labels: list[int]
) -> None:
    """Validate each checkpoint classification-head width against its labels.

    Raises:
        ValueError: If a required head weight is absent or has incompatible rows.
    """
    expected = {
        "diagnoses_head.1.weight": num_labels[0],
        "indications_head.1.weight": num_labels[1],
        "triage_head.1.weight": num_labels[2],
    }
    for key, rows in expected.items():
        tensor = state_dict.get(key)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or tensor.shape[0] != rows
        ):
            raise ValueError(
                f"Checkpoint head {key} is incompatible with model metadata."
            )


def _shared_model_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the model fields that must match across all three artifacts."""
    return {key: value for key, value in manifest.items() if key != "target_head"}


def resolve_weights_dir() -> Path:
    """Resolve the direct weights root from the environment or API default.

    ``KIBATIN_WEIGHTS_DIR`` takes precedence and must directly contain the
    ``dia``, ``ind``, and ``tri`` artifact directories.
    """
    configured = os.environ.get("KIBATIN_WEIGHTS_DIR")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else DEFAULT_WEIGHTS_DIR.resolve()
    )


def load_runtime(
    weights_dir: str | Path | None = None,
    *,
    device: str | torch.device = "cpu",
    code_map_dir: str | Path | None = None,
) -> RuntimeAssets:
    """Load and validate all three specialized runtimes before readiness.

    Each target is checked against its directory identity and the three
    artifacts must share model and preprocessing contracts. Tokenizers load
    offline, checkpoints load as weights-only state dictionaries with strict
    key and head validation, and models are moved to ``device`` in evaluation
    mode. ``Diagramm``, ``Indikator``, and ``Triage`` select ``dia``, ``ind``,
    and ``tri`` respectively.

    Raises:
        ValueError: If any artifact, shared contract, tokenizer, or code map is
            incomplete or incompatible.
        RuntimeError: If strict PyTorch state loading rejects the checkpoint.
    """
    weights_root = Path(weights_dir or resolve_weights_dir()).expanduser().resolve()
    maps_root = Path(code_map_dir or DEFAULT_CODE_MAP_DIR).expanduser().resolve()
    runtime_device = torch.device(device)
    modes: dict[str, ModeRuntime] = {}
    reference_model: dict[str, Any] | None = None
    reference_preprocessing: dict[str, Any] | None = None

    for option in PUBLIC_OPTIONS:
        target, task_index, code_map_name = MODE_SPECS[option]
        model_dir = weights_root / target
        manifest = read_json(model_dir / "model.json")
        validate_model_manifest(manifest, expected_target=target)
        id2labels, _, num_labels = normalize_and_validate_labels(manifest)
        preprocessing, preprocessing_manifest = load_preprocessing_config(
            model_dir / "preprocessing.json"
        )
        if preprocessing.feature_width != manifest["num_extra_features"]:
            raise ValueError("Preprocessing width does not match model metadata.")
        backbone_max = manifest["backbone_config"].get("max_position_embeddings")
        if not isinstance(backbone_max, int) or preprocessing.model_max_length > backbone_max:
            raise ValueError("Tokenizer maximum exceeds the backbone configuration.")
        shared = _shared_model_contract(manifest)
        if reference_model is None:
            reference_model = shared
            reference_preprocessing = preprocessing_manifest
        elif shared != reference_model or preprocessing_manifest != reference_preprocessing:
            raise ValueError(
                "The three artifacts do not share one model/preprocessing contract."
            )

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        if int(tokenizer.model_max_length) != preprocessing.model_max_length:
            raise ValueError("Saved tokenizer maximum does not match preprocessing metadata.")
        if "token_type_ids" not in tokenizer.model_input_names:
            raise ValueError("Saved tokenizer does not provide token_type_ids.")
        checkpoint = model_dir / "pytorch_model.bin"
        try:
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except FileNotFoundError as exc:
            raise ValueError(f"Required checkpoint is missing: {checkpoint}") from exc
        if not isinstance(state_dict, dict):
            raise ValueError("Model checkpoint must contain a state dictionary.")
        validate_checkpoint_heads(state_dict, num_labels)
        model = MultiTaskModel(manifest)
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("Strict checkpoint loading reported incompatible keys.")
        model.to(runtime_device)
        model.eval()

        code_map, _ = load_code_map(maps_root / code_map_name)
        selected_labels = id2labels[task_index]
        validate_code_coverage(option, selected_labels, code_map)
        modes[option] = ModeRuntime(
            model=model,
            tokenizer=tokenizer,
            preprocessing=preprocessing,
            id2labels=selected_labels,
            code_map=code_map,
            internal_task_key=target,
            device=runtime_device,
        )
    return RuntimeAssets(modes=modes)
