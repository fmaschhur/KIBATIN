"""Prepare fixed-schema CSV splits for three-head KIBATIN training.

This module owns split filtering, narrative construction, numeric defaults and
missingness, and label metadata shared with exported API artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

DefaultValue = str | int | float
PREPROCESSING_FORMAT_VERSION = 1
PUBLIC_TEXT_FIELDS: tuple[str, ...] = (
    "notfallgeschehentext",
    "diagnose_text",
    "symptoms",
    "bemerkung",
)
CSV_TEXT_FIELDS: tuple[str, ...] = (
    "notfallgeschehen.notfallgeschehentext",
    "erstdiagnose_verletzungen.diagnose_text",
    "erstbefund_messwerte.symptoms",
    "alarmierung.bemerkung",
)
LABEL_FIELDS: tuple[str, ...] = (
    "label_color",
    "label_diagnose",
    "label_indication",
)
ROW_ID_FIELD = "master.mdeid"


@dataclass(frozen=True)
class FeatureSpec:
    """Unresolved numeric feature declaration.

    Attributes:
        name: CSV and public-API feature key.
        default: Fixed value or ``MEDIAN`` training-fit marker.
    """

    name: str
    default: DefaultValue


@dataclass(frozen=True)
class ResolvedFeatureSpec:
    """Numeric feature contract with a concrete inference default.

    Attributes:
        name: CSV and public-API feature key.
        default: Training-derived median or fixed numeric fallback.
        missing_indicator: Whether to append a missingness flag for the feature.
    """

    name: str
    default: float
    missing_indicator: bool


@dataclass(frozen=True)
class PreprocessingSpec:
    """Serializable text, rules, numeric, and tokenizer preprocessing contract."""

    format_version: int
    text_fields: tuple[str, ...]
    numeric_features: tuple[ResolvedFeatureSpec, ...]
    missingness_indicator_order: tuple[str, ...]
    rules_value_enabled: bool

    def to_manifest(self, *, model_max_length: int) -> dict[str, Any]:
        """Return preprocessing metadata for strict API reconstruction.

        The manifest fixes field ordering and defaults, disables truncation,
        and requires token type IDs at training and inference time.
        """
        return {
            "format_version": self.format_version,
            "text_fields": list(self.text_fields),
            "rules_value": {
                "field": "rules_value",
                "enabled": self.rules_value_enabled,
                "format": "kibatin_rules_v1",
            },
            "numeric_features": [
                {
                    "name": feature.name,
                    "default": feature.default,
                    "missing_indicator": feature.missing_indicator,
                }
                for feature in self.numeric_features
            ],
            "missingness_indicator_order": list(self.missingness_indicator_order),
            "tokenizer": {
                "model_max_length": int(model_max_length),
                "truncation": False,
                "return_token_type_ids": True,
            },
        }


EXTRA_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("massnahmen_total_nr", "MEDIAN"),
    FeatureSpec("ergebnis_schockraum", 0),
    FeatureSpec("notaufnahme_messwerte_schmerz", "MEDIAN"),
    FeatureSpec("uebergabe_neurologie.glasgowcomascale", "MEDIAN"),
    FeatureSpec("uebergabe_motor_schmerzkeine", 0),
    FeatureSpec("naca_score_1", 0),
    FeatureSpec("naca_score_2", 0),
    FeatureSpec("naca_score_3", 0),
    FeatureSpec("naca_score_4", 0),
    FeatureSpec("naca_score_5", 0),
    FeatureSpec("naca_score_6", 0),
    FeatureSpec("naca_score_7", 0),
)


@dataclass(frozen=True)
class DataConfig:
    """Paths and rules-input setting for explicit dataset splits.

    Training and validation paths are required; a held-out test path is
    optional and is never inferred from either required split.
    """

    train_file: str | Path
    validation_file: str | Path
    test_file: str | Path | None = None
    external_labels: bool = True


@dataclass
class InputData:
    """Prepared texts, row IDs, and numeric matrices for each supplied split.

    Numeric columns contain the twelve base features followed by the three
    training-defined missingness indicators. ``test_extra`` is ``None`` when
    no test split was supplied.
    """

    train_texts: list[str]
    val_texts: list[str]
    test_texts: list[str]
    train_ids: list[Any]
    val_ids: list[Any]
    test_ids: list[Any]
    train_extra: np.ndarray
    val_extra: np.ndarray
    test_extra: np.ndarray | None


@dataclass
class TaskData:
    """Labels and training weights for one classification head.

    Label maps cover all supplied splits, while ``class_weights`` are computed
    solely from labels present in the filtered training split.
    """

    train_labels: list[str]
    val_labels: list[str]
    test_labels: list[str]
    id2label: dict[int, str]
    label2id: dict[str, int]
    num_labels: int
    mode: str
    class_weights: np.ndarray


@dataclass
class PreparedData:
    """Complete inputs, task metadata, and exportable preprocessing contract."""

    input: InputData
    diagnoses: TaskData
    indications: TaskData
    triage: TaskData
    preprocessing: PreprocessingSpec


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


def _is_missing(value: Any) -> bool:
    """Return whether a training value is absent, empty, or pandas-missing."""
    return (
        value is None
        or (isinstance(value, str) and value == "")
        or bool(pd.isna(value))
    )


def _coerce_numeric(value: Any) -> float:
    """Convert a numeric, boolean, or decimal-comma string to a float."""
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
        value = value.replace(",", ".")
    return float(value)


def resolve_extra_feature_defaults(
    values: np.ndarray,
    feature_specs: tuple[FeatureSpec, ...] = EXTRA_FEATURES,
) -> tuple[ResolvedFeatureSpec, ...]:
    """Fit numeric defaults from filtered training values.

    ``MEDIAN`` features receive the median of their observed training values
    and a missingness indicator; an entirely missing feature falls back to
    ``-1.0``. Fixed-default features retain their declared numeric value.
    """
    resolved: list[ResolvedFeatureSpec] = []
    for index, feature in enumerate(feature_specs):
        if feature.default == "MEDIAN":
            observed = [
                _coerce_numeric(value)
                for value in values[:, index]
                if not _is_missing(value)
            ]
            default = float(np.median(observed)) if observed else -1.0
            missing_indicator = True
        else:
            default = _coerce_numeric(feature.default)
            missing_indicator = False
        resolved.append(ResolvedFeatureSpec(feature.name, default, missing_indicator))
    return tuple(resolved)


def build_numeric_matrix(
    values: np.ndarray,
    resolved_features: tuple[ResolvedFeatureSpec, ...],
) -> np.ndarray:
    """Build base numeric columns followed by ordered missingness indicators.

    Missing cells use the resolved feature default. Indicator columns follow
    the order of features whose ``missing_indicator`` flag is true.

    Raises:
        ValueError: If the input is not a two-dimensional matrix of the
            declared base-feature width.
    """
    if values.ndim != 2 or values.shape[1] != len(resolved_features):
        raise ValueError("Numeric input width does not match the feature contract.")
    rows = [
        [
            (
                feature.default
                if _is_missing(values[row, column])
                else _coerce_numeric(values[row, column])
            )
            for column, feature in enumerate(resolved_features)
        ]
        for row in range(values.shape[0])
    ]
    base = np.asarray(rows, dtype=float).reshape(
        values.shape[0], len(resolved_features)
    )
    indicators = [
        np.asarray(
            [
                1.0 if _is_missing(values[row, index]) else 0.0
                for row in range(values.shape[0])
            ]
        ).reshape(-1, 1)
        for index, feature in enumerate(resolved_features)
        if feature.missing_indicator
    ]
    return np.concatenate([base, *indicators], axis=1) if indicators else base


def _prepare_frame(frame: pd.DataFrame, *, external_labels: bool) -> pd.DataFrame:
    """Select, normalize, and filter one supplied CSV split.

    The returned copy contains only supported fields. It excludes blue triage,
    ``ZNA`` or ``GC`` indications, missing labels, and labels occurring fewer
    than twice within that split.

    Raises:
        ValueError: If required columns are absent or filtering removes all rows.
    """
    numeric_names = [feature.name for feature in EXTRA_FEATURES]
    columns = [*LABEL_FIELDS, ROW_ID_FIELD, *numeric_names, *CSV_TEXT_FIELDS]
    if external_labels:
        columns.append("rules_value")
    missing_columns = sorted(set(columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {missing_columns}")

    prepared = frame.loc[:, columns].copy()
    prepared.loc[:, numeric_names] = prepared.loc[:, numeric_names].replace(
        {"true": "1", "false": "0", "True": "1", "False": "0"}
    )
    prepared.loc[:, numeric_names] = prepared.loc[:, numeric_names].apply(
        lambda column: column.astype("string").str.replace(",", ".", regex=False)
    )
    prepared = prepared[~prepared["label_indication"].isin(["ZNA", "GC"])]
    prepared = prepared[prepared["label_color"] != "blau"]
    for label_field in LABEL_FIELDS:
        prepared = prepared.groupby(label_field).filter(lambda group: len(group) > 1)
    prepared = prepared.dropna(subset=list(LABEL_FIELDS))
    if prepared.empty:
        raise ValueError("No rows remain after applying the training filters.")
    return prepared


def _build_texts(frame: pd.DataFrame, *, external_labels: bool) -> list[str]:
    """Build narratives in fixed field order with optional transformed rules."""
    texts = [
        " ".join(value if isinstance(value, str) else "" for value in row)
        for row in frame.loc[:, CSV_TEXT_FIELDS].itertuples(index=False, name=None)
    ]
    if not external_labels:
        return texts
    external = [preprocess_rules_value(value) for value in frame["rules_value"]]
    return [
        text + " [SEP] " + rules if rules else text
        for text, rules in zip(texts, external)
    ]


def _split_values(
    frame: pd.DataFrame,
    *,
    external_labels: bool,
    resolved_features: tuple[ResolvedFeatureSpec, ...],
) -> tuple[list[str], np.ndarray, list[Any], np.ndarray]:
    """Extract texts, labels, row IDs, and numeric inputs from one split.

    Label columns are returned in indication, diagnosis, and triage order so
    task construction can select indices 0, 1, and 2 explicitly.
    """
    labels = frame[["label_indication", "label_diagnose", "label_color"]].to_numpy()
    numeric_names = [feature.name for feature in EXTRA_FEATURES]
    return (
        _build_texts(frame, external_labels=external_labels),
        labels,
        frame[ROW_ID_FIELD].tolist(),
        build_numeric_matrix(frame[numeric_names].to_numpy(), resolved_features),
    )


def _task_labels(labels: np.ndarray, index: int) -> list[str]:
    """Return one label column as a list, or an empty list for no rows."""
    return labels[:, index].tolist() if labels.size else []


def _class_weights(label_set: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    """Return balanced training weights aligned to the split-wide label set.

    Labels absent from training retain weight one, even when introduced by a
    validation or test split.
    """
    present = np.intersect1d(label_set, np.unique(train_labels))
    weights = np.ones(len(label_set), dtype=float)
    if not present.size:
        return weights
    computed = compute_class_weight("balanced", classes=present, y=train_labels)
    by_label = dict(zip(present, computed))
    return np.asarray([by_label.get(label, 1.0) for label in label_set])


def _task_data(
    mode: str,
    index: int,
    label_set: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
) -> TaskData:
    """Build label lists, inverse maps, and training weights for one head."""
    id2label = {idx: label for idx, label in enumerate(label_set)}
    return TaskData(
        train_labels=_task_labels(train_labels, index),
        val_labels=_task_labels(validation_labels, index),
        test_labels=_task_labels(test_labels, index),
        id2label=id2label,
        label2id={label: idx for idx, label in id2label.items()},
        num_labels=len(label_set),
        mode=mode,
        class_weights=_class_weights(label_set, train_labels[:, index]),
    )


def prepare_data(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame | None = None,
    *,
    external_labels: bool = True,
) -> PreparedData:
    """Prepare explicit training, validation, and optional test splits.

    Each supplied split is filtered independently. Numeric medians are fitted
    only on filtered training rows and reused elsewhere, while label maps cover
    the union of all supplied splits. Diagnosis, indication, and triage use
    label-array indices 1, 0, and 2 respectively.
    """
    train = _prepare_frame(train_frame, external_labels=external_labels)
    validation = _prepare_frame(validation_frame, external_labels=external_labels)
    test = (
        _prepare_frame(test_frame, external_labels=external_labels)
        if test_frame is not None
        else None
    )

    numeric_names = [feature.name for feature in EXTRA_FEATURES]
    resolved_features = resolve_extra_feature_defaults(train[numeric_names].to_numpy())
    train_texts, train_labels, train_ids, train_extra = _split_values(
        train, external_labels=external_labels, resolved_features=resolved_features
    )
    val_texts, val_labels, val_ids, val_extra = _split_values(
        validation,
        external_labels=external_labels,
        resolved_features=resolved_features,
    )
    if test is None:
        test_texts: list[str] = []
        test_labels = np.empty((0, 3), dtype=object)
        test_ids: list[Any] = []
        test_extra = None
    else:
        test_texts, test_labels, test_ids, test_extra = _split_values(
            test,
            external_labels=external_labels,
            resolved_features=resolved_features,
        )

    label_sources = [train_labels, val_labels]
    if test_labels.size:
        label_sources.append(test_labels)
    all_labels = np.concatenate(label_sources)
    indication_set = np.unique(all_labels[:, 0])
    diagnosis_set = np.unique(all_labels[:, 1])
    triage_set = np.unique(all_labels[:, 2])

    return PreparedData(
        input=InputData(
            train_texts=train_texts,
            val_texts=val_texts,
            test_texts=test_texts,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            train_extra=train_extra,
            val_extra=val_extra,
            test_extra=test_extra,
        ),
        diagnoses=_task_data(
            "dia", 1, diagnosis_set, train_labels, val_labels, test_labels
        ),
        indications=_task_data(
            "ind", 0, indication_set, train_labels, val_labels, test_labels
        ),
        triage=_task_data("tri", 2, triage_set, train_labels, val_labels, test_labels),
        preprocessing=PreprocessingSpec(
            format_version=PREPROCESSING_FORMAT_VERSION,
            text_fields=PUBLIC_TEXT_FIELDS,
            numeric_features=resolved_features,
            missingness_indicator_order=tuple(
                feature.name
                for feature in resolved_features
                if feature.missing_indicator
            ),
            rules_value_enabled=external_labels,
        ),
    )


def load_data(config: DataConfig) -> PreparedData:
    """Read configured CSV splits as strings and return prepared training data."""

    def read(path):
        return pd.read_csv(path, dtype="string")

    return prepare_data(
        read(config.train_file),
        read(config.validation_file),
        read(config.test_file) if config.test_file is not None else None,
        external_labels=config.external_labels,
    )
