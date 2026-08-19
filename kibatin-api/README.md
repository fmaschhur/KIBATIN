# KIBATIN API

The API eagerly loads three model artifacts from one direct weights root and
serves diagnosis, indication, and triage predictions. Startup fails before the
application becomes ready if any artifact or mapping contract is incompatible.

## Weights layout

Point `KIBATIN_WEIGHTS_DIR` at:

```text
$KIBATIN_WEIGHTS_DIR/
├── dia/
│   ├── model.json
│   ├── preprocessing.json
│   ├── pytorch_model.bin
│   └── tokenizer files
├── ind/
│   └── same files
└── tri/
    └── same files
```

All files are loaded locally. The loader checks target identity, inverse label
maps, shared seed/backbone/label/numeric/preprocessing
contracts, tokenizer availability and maximum length, checkpoint head shapes,
strict state-dictionary compatibility, and code-map coverage.

## Code maps

Files in `code_maps/` map the exact labels stored in the model artifacts to the
string IDs returned by the API. Each file has this structure:

```json
{
  "codes": {
    "exact model label": "external-code"
  },
  "source": "approved-registry-2026-07"
}
```

Every label selected from the corresponding artifact must appear in `codes`;
otherwise API startup fails. For an `identity_fallback`, every code must equal
its label.

These codes are reference identifiers used by the overseeing program that
calls this API; they are not intrinsic model labels. They may therefore change
as that external integration evolves. When they do, update and revalidate the
corresponding code map while keeping its label keys aligned with the deployed
model artifact.

Before production use, the responsible clinical or data-governance owner must
review and approve each map. The `source` value should identify its traceable
authority or provenance, and maps must be reviewed whenever a model label set
changes. The bundled diagnosis and indication maps remain marked
`legacy_unverified` because their governance status is unknown. The bundled
triage map is an `identity_fallback` intended for development, not an
authoritative clinical mapping.

## Run locally

Create a Python 3.11 environment, install `requirements.txt`, and set the direct
root:

```bash
python -m pip install -r requirements.txt
export KIBATIN_WEIGHTS_DIR=/absolute/path/to/weights
gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 1 --timeout 180 inference:app
```

Readiness is available only after all three models load:

```bash
curl http://127.0.0.1:5000/healthz
# {"status":"ready"}
```

## Predictions

`POST /predict` accepts `Diagramm`, `Indikator`, or `Triage`. Extra JSON fields
are ignored. The response contains the top three descending results with
`Name`, `ID`, and four-decimal `Prediction` values.

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "option": "Triage",
    "notfallgeschehentext": "Beispiel",
    "diagnose_text": "",
    "symptoms": "",
    "bemerkung": "",
    "rules_value": "",
    "massnahmen_total_nr": 0,
    "ergebnis_schockraum": 0,
    "notaufnahme_messwerte_schmerz": 0,
    "uebergabe_neurologie.glasgowcomascale": 15,
    "uebergabe_motor_schmerzkeine": 0,
    "naca_score_1": 0,
    "naca_score_2": 0,
    "naca_score_3": 0,
    "naca_score_4": 0,
    "naca_score_5": 0,
    "naca_score_6": 0,
    "naca_score_7": 0
  }'
```

For compatibility, invalid options and caught input/inference failures return
HTTP 200 with `{"error":"..."}`. Unknown paths and unsupported methods retain
ordinary Flask 404/405 responses.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Docker

```bash
docker build -t kibatin-api .
docker run --rm -p 5000:5000 \
  --mount type=bind,src="$WEIGHTS_ROOT",dst=/app/weights,readonly \
  kibatin-api
```

The image runs one worker/thread because each worker owns all three models. Its
health check calls `/healthz`, and the mapping files are built into the image.
