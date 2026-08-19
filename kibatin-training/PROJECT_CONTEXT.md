# KIBATIN training context

## Module ownership

- `data.py` owns the fixed CSV schema, filtering, narrative construction,
  numeric coercion, training-derived medians, missingness indicators, and label
  maps for explicit training/validation/optional-test splits.
- `models.py` owns the serialized three-head architecture and Hugging Face
  dataset, trainer, optimizer, dynamic collator, and target loss schedule hooks.
- `evaluation.py` owns selected-target validation MCC and the optional compact
  test `metrics.json` plus `predictions.csv`.
- `artifacts.py` owns model manifests and refusal-to-overwrite atomic export.
- `train.py` owns Hugging Face authentication, orchestration, the public CLI,
  best-checkpoint loading, and mandatory artifact export.

## Invariants

Each run contains diagnosis, indication, and triage heads. The selected target
changes its loss schedule and the MCC used for best-checkpoint selection.

Narrative field order, rules transformation, twelve numeric features, three
missingness indicators, training-only medians, no truncation, and token type IDs
are runtime contracts. Change them only together with `kibatin-api/runtime.py`
and the parity tests.

The exporter writes `model.json`, `preprocessing.json`, `pytorch_model.bin`, and
local tokenizer files to one target directory. The three deployed artifacts
must share seed, backbone, labels, numeric architecture, and preprocessing, and
their `target_head` must match `dia`, `ind`, or `tri` respectively.

Training needs Python 3.11, `pyproject.toml`, and `uv.lock`. The ignored local
CSV and secret files are development inputs and must not be distributed.
