# KIBATIN training

This project trains one target-specialized three-head model per run and always
exports a self-contained artifact. Run it once for each target:
`dia`, `ind`, and `tri`.

## Setup

Training requires:

- Python 3.11;
- [uv](https://docs.astral.sh/uv/);
- a CUDA-capable NVIDIA GPU; and
- a CUDA-enabled PyTorch environment in which `torch.cuda.is_available()` is
  `True`.

```bash
uv sync
```

The default backbone is gated. Accept its Hugging Face terms, then either run
`huggingface-cli login` or export `HF_TOKEN`. A different backbone can be set
with `--model-name`.

## Input CSVs

Training and validation CSVs are required; the test CSV is optional. They use
the same schema. The implementation reads only:

- `master.mdeid`;
- `label_color`, `label_diagnose`, and `label_indication`;
- `notfallgeschehen.notfallgeschehentext`,
  `erstdiagnose_verletzungen.diagnose_text`,
  `erstbefund_messwerte.symptoms`, and `alarmierung.bemerkung`;
- the twelve numeric feature names declared in `data.py`;
- `rules_value`, unless `--no-external-labels` is supplied.

Rows with blue triage, `ZNA`/`GC` indication, missing labels, or a label that
occurs fewer than twice within its supplied split are filtered. Numeric medians
are learned only from the training CSV and reused for validation, test, and API
inference. Text is never truncated.

### Batch Size, Epochs & Validation Step Size
The current training defaults use a batch size of 8, run for 15 epochs, and
perform validation every 500 optimizer steps. In the documented single-process
setup, the final incomplete batch is discarded, so the training split must
contain at least 8 rows after filtering to produce any training batch. At least
272 filtered rows are needed to reach the first scheduled validation within 15
epochs; with 8–271 rows, training can run, but validation and best-checkpoint
selection do not occur as intended.

## Train all three targets

Set the input and output paths to absolute paths before running training:

| Variable | Use | Example value |
| --- | --- | --- |
| `TRAIN_CSV` | Training CSV | `/data/kibatin/train.csv` |
| `VAL_CSV` | Validation CSV | `/data/kibatin/validation.csv` |
| `TEST_CSV` | Optional held-out test CSV | `/data/kibatin/test.csv` |
| `RUN_ROOT` | Checkpoints and evaluation outputs | `/data/kibatin/runs` |
| `WEIGHTS_ROOT` | Deployable artifact root | `/data/kibatin/weights` |

For example:

```bash
export TRAIN_CSV=/data/kibatin/train.csv
export VAL_CSV=/data/kibatin/validation.csv
export TEST_CSV=/data/kibatin/test.csv
export RUN_ROOT=/data/kibatin/runs
export WEIGHTS_ROOT=/data/kibatin/weights
```

`RUN_ROOT` and `WEIGHTS_ROOT` must be separate directories. The target artifact
directories `$WEIGHTS_ROOT/dia`, `$WEIGHTS_ROOT/ind`, and `$WEIGHTS_ROOT/tri`
must not already exist.

```bash
for TARGET in dia ind tri; do
  uv run python train.py \
    --target-head "$TARGET" \
    --seed 42 \
    --train-file "$TRAIN_CSV" \
    --validation-file "$VAL_CSV" \
    --test-file "$TEST_CSV" \
    --run-dir "$RUN_ROOT/$TARGET" \
    --artifact-dir "$WEIGHTS_ROOT/$TARGET"
done
```

When no held-out test CSV is available, do not set `TEST_CSV` and remove the
`--test-file "$TEST_CSV"` line from the command. Export is mandatory, and an
existing artifact directory is never overwritten. Each run still trains all
three heads; `--target-head` chooses the loss schedule and validation MCC used
for best-checkpoint selection.

With a test CSV, the run writes `evaluation/metrics.json` and
`evaluation/predictions.csv`. The artifact root is:

```text
$WEIGHTS_ROOT/
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

Deploy all three newly generated directories together.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

The suite covers filtering and preprocessing, target metrics and loss
scheduling, atomic export, strict API loading, and training/API parity with a
tiny offline BERT fixture. It does not execute CUDA training or validate a full
GPU training run.
