#!/usr/bin/env bash
# Example long-run trainer wrapper. Paths are relative to repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
OUT="${OUT:-runs/shelf_300m}"
CONFIG="${CONFIG:-configs/shelf_300m.json}"
SPM="${SPM:-vocab/out/spm_zh_24576.model}"
DEVICE="${DEVICE:-cuda}"
BATCH="${BATCH:-4}"
SEQ="${SEQ:-192}"
HOURS="${HOURS:-24}"
SEED="${SEED:-42}"

mkdir -p "$OUT" runs/logs

DATA=(
  data/seed_zh.jsonl
  data/samples/extra_zh_short.sample.jsonl
  data/samples/knowledge_zh.sample.jsonl
  data/samples/reasoning_zh.sample.jsonl
  data/samples/opensource_firefly.sample.jsonl
  data/samples/opensource_belle_general.sample.jsonl
  data/samples/opensource_belle_school_math.sample.jsonl
)
# Add your full corpora here, e.g.:
# DATA+=( /path/to/firefly.jsonl /path/to/belle_general.jsonl )

RESUME_ARGS=()
if [[ -f "$OUT/model.pt" ]]; then
  RESUME_ARGS=(--resume "$OUT/model.pt")
fi

exec "$PY" -u train_v2.py \
  --config "$CONFIG" \
  --spm "$SPM" \
  --data "${DATA[@]}" \
  --out "$OUT" \
  --hours "$HOURS" \
  --device "$DEVICE" \
  --batch "$BATCH" \
  --seq "$SEQ" \
  --lr 2e-4 \
  --limit 0 \
  --eval-every 100 \
  --save-every 500 \
  --log-every 10 \
  --seed "$SEED" \
  --tag v2_300m_example \
  "${RESUME_ARGS[@]}"
