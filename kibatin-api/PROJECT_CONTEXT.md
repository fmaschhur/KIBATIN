# KIBATIN API context

## Module ownership

- `model.py` is the inference-only copy of the serialized three-head training
  architecture. It keeps the API deployable without the training project.
- `runtime.py` owns artifact validation, offline tokenizer/checkpoint loading,
  preprocessing, code-map coverage, and the three specialized runtime objects.
- `inference.py` eagerly constructs the Flask app, routes each public option to
  its target artifact, and formats top-three responses.
- `code_maps/` is API-owned mapping data. Diagnosis/indication sources are
  unverified; triage is an explicitly non-authoritative identity fallback.

## Runtime invariants

`KIBATIN_WEIGHTS_DIR` directly contains `dia`, `ind`, and `tri`. Their target
identities must match their directories and their seed, backbone, all three
label maps, numeric architecture, and preprocessing manifests must be equal.

Tokenizers load with `local_files_only=True`. Checkpoints load with
`weights_only=True`, their head rows are checked against label counts, and state
dictionaries load with `strict=True`. The API validates the fixed text order,
rules transformation, numeric feature/default order, three missingness
indicators, tokenizer maximum, no-truncation setting, and token type IDs before
readiness.

`Diagramm`, `Indikator`, and `Triage` route to `dia`, `ind`, and `tri`. Caught
application failures intentionally remain HTTP 200 error objects; routing and
method failures remain ordinary 404/405 responses.
