# ShelfLM v2 — Solo Scale-Up Design Notes

Design notes that accompanied the v2 tree. For a short public overview see `README.md`
and `ARCHITECTURE_NOTES.md`.

## Goals

Move from a small char-level prototype toward a **standalone** mid-size Chinese LM:

| Item | Earlier prototype | v2 (this tree) |
|------|-------------------|----------------|
| Vocab | Char-level ~7.7k | SentencePiece BPE **24k** (16k–32k OK) |
| FactTable | 65k × d_model ≈ large share of params | 8k × 128 + projection ≈ **~1M (&lt;1%)** |
| Total params | ~90M | **~300M** primary; 1B roadmap config |
| Sequence mix | StateMix only | StateMix + **causal Attn every 4 layers** |
| Data | Instruction SFT-heavy | Pretrain entry + firefly/belle/math SFT |

## Stages

1. **Pretrain (PT)** — plain Chinese text streams (`data/stream_pretrain.py` + local wiki/web text); lower valid CE / fluent continuation.
2. **SFT** — open instruction jsonl + `data/seed_zh.jsonl`; format `用户：…\n助手：…`; causal next-token shift.
3. **Optional preference** — not implemented; placeholder for DPO/ORPO later.

## Configs

- `configs/shelf_300m.json` — single-GPU primary target.
- `configs/shelf_1b.json` — roadmap; needs more VRAM.
- `configs/shelf_tiny.json` — CPU / smoke.

```bash
python scripts/count_params.py --config configs/shelf_300m.json
```

Expected order of magnitude: embed ~25M, fact ~1M, blocks ~270M; Fact share drops from ~46% → &lt;1%, capacity moved into `d_model` / `n_layers`.

## Sequence modeling trade-off

| Option | Upside | Cost | Choice here |
|--------|--------|------|-------------|
| StateMix only | O(T) fixed state | Weak nonlocal mix | Default layer body |
| Full-layer Attention | Higher quality ceiling | Quadratic every layer | Not used |
| **Sparse Attention** | Cheap nonlocal boost | Mild VRAM bump | **`attn_every=4`** |
| Stronger StateMix gates | No extra layers | Limited gain | `state_gate_bias>0` |

## FactTable

Default `fact_mode=lowrank`:

- Rows 65536 → **8192**
- Dim `d_model` → **`fact_dim=128` + Linear**
- Optional `sparse_gate` (token gate)
- Optional `external` retrieval stub (`set_external(fn)`)

## Vocab

```bash
python data/prepare_corpus.py \
  --inputs data/seed_zh.jsonl data/samples/opensource_firefly.sample.jsonl \
  --out-txt runs/corpus_zh.txt --limit 150000

python vocab/train_spm.py --corpus runs/corpus_zh.txt --out-dir vocab/out --vocab-size 24576
python vocab/export_spm.py --model vocab/out/spm_zh_24576.model --out-json vocab/out/export.json
```

## Short compare / smoke

```bash
python scripts/compare_short.py \
  --corpus runs/corpus_zh.txt \
  --spm vocab/out/spm_zh_24576.model \
  --out runs/compare_short --steps 150 --device cpu

python train_v2.py \
  --config configs/shelf_tiny.json \
  --spm vocab/out/spm_zh_24576.model \
  --data data/seed_zh.jsonl \
  --out runs/tiny_smoke --steps 100 --device cpu
```

Do **not** warm-start v2 from older char-level checkpoints (vocab + Fact layout incompatible).

## File map

```
model_v2.py          # ShelfLM / StateMix / FactTable / ShelfMoE + Attn
train_v2.py
configs/shelf_{300m,1b,tiny}.json
vocab/train_spm.py, export_spm.py
data/prepare_corpus.py, stream_pretrain.py
scripts/count_params.py, compare_short.py, chat_smoke_cpu.py
examples/run_train_example.sh
```
