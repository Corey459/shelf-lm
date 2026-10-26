# ShelfLM v2 — Architecture Notes

Objective, factual summary of the custom pieces in `model_v2.py` for README / paper-style use.
This is not a marketing document.

## 1. Overall stack

ShelfLM v2 is a decoder-only language model. A forward pass:

1. Token embedding (`nn.Embedding`, vocab typically 24 576 via SentencePiece BPE).
2. **FactTable** lookup mixed into the embedding (optional compressed memory).
3. `n_layers` residual **Block**s. Each block always has **StateMix** + **ShelfMoE**;
   every `attn_every` layers also inserts a **CausalSelfAttn** (scaled dot-product, causal).
4. Final RMSNorm + tied (optional) LM head.

Default ~300M config (`configs/shelf_300m.json`): `d_model=1024`, `n_layers=18`,
`n_heads=16`, `attn_every=4`, FactTable `8192×128` low-rank, ShelfMoE with 4 shelves × 8 experts.

## 2. StateMix (default sequence mixer)

Unlike a standard Transformer layer that mixes tokens only via attention,
StateMix is a **gated diagonal recurrence** with fixed-size per-head state:

- Project RMSNorm(x) to Q/K/V.
- Per-head decay gate `a_t = σ(Linear(x))` (bias `state_gate_bias` encourages longer memory at init).
- Parallel chunked scan: `s_t = a_t · s_{t-1} + (k_t ⊙ v_t)`.
- Output `(q ⊙ s)` then linear projection.

Complexity is linear in sequence length for the scan path (state size fixed),
so long context is cheaper than full attention on every layer. Non-local mixing
is weaker than MHA, which is why sparse attention is interleaved.

## 3. Interleaved causal attention

`CausalSelfAttn` uses `scaled_dot_product_attention(..., is_causal=True)`.
It is **not** present on every layer: with `attn_every=4`, attention runs on
layers 4, 8, 12, 16 (1-based). Design intent: keep StateMix as the architectural
backbone; spend a few quadratic layers to recover long-range composition.

## 4. FactTable (compressed fact memory)

v1 used a large `Embedding(65536, d_model)` fact store (~tens of millions of params,
large share of a ~90M model). v2 defaults to **low-rank**:

- `Embedding(fact_rows, fact_dim)` with `fact_rows=8192`, `fact_dim=128`, plus `Linear → d_model`.
- Indexing is a cheap bigram hash of `(prev_token, token) % fact_rows`.
- Modes: `lowrank` (default), `sparse_gate` (sigmoid gate from token emb),
  `external` (zero stub / pluggable retrieval via `set_external(fn)`).

Role: inject a small, learnable memory feature into the token stream without
dominating the parameter budget (target &lt;1% of total params vs ~46% in v1).

## 5. ShelfMoE

Feed-forward side is **not** a dense FFN alone:

- Shared SwiGLU FFN (`w1`/`w2`/`w3`).
- Two-level routing: pick a **shelf**, then top-k **experts** constrained to that shelf.
- Each expert contributes a **low-rank delta** `A[e] @ B[e]` (rank `r`) added to the shared FFN output.

Auxiliary loss encourages shelf/expert balance and shelf “stickiness” across time
(see `auxiliary_loss` in `model_v2.py`).

## 6. Differences vs a standard Transformer (GPT-style)

| Piece | Standard Transformer | ShelfLM v2 |
|-------|----------------------|------------|
| Token mix per layer | Full causal MHA | StateMix always; MHA every N layers |
| Context cost | O(T²) every layer | O(T) scan + sparse O(T²) layers |
| FFN | Dense MLP / optional MoE | Shared SwiGLU + shelf→expert low-rank MoE |
| Extra memory | Usually none | FactTable (hashed bigram → low-dim table) |
| Norm | LayerNorm / RMSNorm | RMSNorm |
| Vocab (this tree) | Often large BPE | SentencePiece BPE 16k–32k (shipped 24 576) |

## 7. Training surface

- `train_v2.py`: causal next-token CE + MoE aux loss; hours/steps supervisor-friendly;
  resume only v2 checkpoints (vocab / Fact layout incompatible with older char-level v1).
- `vocab/train_spm.py` / `export_spm.py`: build and wrap SentencePiece.
- `data/prepare_corpus.py`, `data/stream_pretrain.py`: local jsonl → text; optional HF mirror streams.
- Config JSON files are plain `ShelfConfigV2` fields.

## 8. What this release is / is not

**Is:** reproducible model definition, trainer, SPM tooling, configs, and small sample data.

**Is not:** trained weight dumps (`.pt` / `.ckpt`), private domain corpora, or deployment /
ops scripts tied to any specific lab host.
