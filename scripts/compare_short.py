#!/usr/bin/env python3
"""Short CE compare: v1 char-level tiny vs v2 SP+shrunk-Fact tiny.

Runs on CPU by default so 24h GPU jobs are untouched.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

SHELF = Path(__file__).resolve().parents[2]
V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHELF))
sys.path.insert(0, str(V2))
sys.path.insert(0, str(V2 / "vocab"))

from model import ShelfConfig as ShelfConfigV1  # noqa: E402
from model import ShelfLM as ShelfLMV1  # noqa: E402
from model_v2 import ShelfConfigV2, ShelfLM as ShelfLMV2, count_parameters_breakdown  # noqa: E402


def build_char_vocab(texts: list[str], vocab_size: int) -> tuple[dict[str, int], list[str]]:
    counts: dict[str, int] = {}
    for t in texts:
        for ch in t:
            counts[ch] = counts.get(ch, 0) + 1
    specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
    chars = sorted(counts, key=counts.get, reverse=True)
    itos = specials + [c for c in chars if c not in specials][: max(0, vocab_size - len(specials))]
    stoi = {c: i for i, c in enumerate(itos)}
    return stoi, itos


def encode_char(text: str, stoi: dict[str, int], seq: int) -> tuple[list[int], list[int]]:
    unk = stoi.get("<unk>", 1)
    bos, eos, pad = stoi["<bos>"], stoi["<eos>"], stoi["<pad>"]
    ids = [bos] + [stoi.get(ch, unk) for ch in text] + [eos]
    if len(ids) > seq + 1:
        ids = ids[: seq + 1]
    else:
        ids = ids + [pad] * (seq + 1 - len(ids))
    x, y = ids[:-1], ids[1:]
    y = [(-100 if t == pad else t) for t in y]
    return x, y


def batch_char(texts, stoi, seq, batch, device):
    xs, ys = [], []
    for _ in range(batch):
        x, y = encode_char(random.choice(texts), stoi, seq)
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, device=device), torch.tensor(ys, device=device)


def train_loop(model, make_batch, steps, lr, device, name):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    hist = []
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = make_batch()
        logits, _ = model(x)
        ce = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        if not torch.isfinite(ce):
            continue
        opt.zero_grad(set_to_none=True)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 25 == 0 or step == steps:
            # quick valid: same distribution fresh batches
            model.eval()
            with torch.no_grad():
                vals = []
                for _ in range(4):
                    vx, vy = make_batch()
                    vlogits, _ = model(vx)
                    vals.append(float(F.cross_entropy(vlogits.flatten(0, 1), vy.flatten(), ignore_index=-100).item()))
            model.train()
            vce = sum(vals) / len(vals)
            hist.append({"step": step, "train_ce": float(ce.item()), "valid_ce": vce})
            print(f"[{name}] step={step} train_ce={ce.item():.3f} valid_ce={vce:.3f}", flush=True)
    return hist, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--spm", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from export_spm import SpmTokenizer

    random.seed(3)
    torch.manual_seed(3)
    device = torch.device(args.device)
    texts = [ln.strip() for ln in args.corpus.read_text(encoding="utf-8", errors="ignore").splitlines() if len(ln.strip()) >= 4][: args.limit]
    if len(texts) < 64:
        raise SystemExit("corpus too small")

    # --- v1 char tiny (mimic old fact-heavy shape at small scale) ---
    stoi, itos = build_char_vocab(texts, vocab_size=2048)
    cfg1 = ShelfConfigV1(
        vocab_size=len(itos),
        d_model=256,
        n_layers=4,
        n_heads=4,
        n_shelves=2,
        experts_per_shelf=4,
        top_k=2,
        rank=16,
        ffn_mult=2,
        fact_rows=8192,  # still fact-heavy proportionally
        tie_embeddings=True,
    )
    m1 = ShelfLMV1(cfg1).to(device)
    p1 = sum(p.numel() for p in m1.parameters())
    fact1 = m1.fact.table.weight.numel()

    # --- v2 SP + shrunk fact ---
    tok = SpmTokenizer(args.spm)
    cfg2 = ShelfConfigV2(
        vocab_size=tok.vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=4,
        n_shelves=2,
        experts_per_shelf=4,
        top_k=2,
        rank=16,
        ffn_mult=2,
        fact_rows=1024,
        fact_dim=32,
        fact_mode="lowrank",
        attn_every=2,
        tie_embeddings=True,
        state_gate_bias=1.0,
    )
    m2 = ShelfLMV2(cfg2).to(device)
    br2 = count_parameters_breakdown(m2)

    def make1():
        return batch_char(texts, stoi, args.seq, args.batch, device)

    def make2():
        xs, ys = [], []
        for _ in range(args.batch):
            ids = tok.encode(random.choice(texts), add_bos=True, add_eos=True)
            pad = tok.pad_id if tok.pad_id >= 0 else 0
            if len(ids) > args.seq + 1:
                start = random.randint(0, len(ids) - args.seq - 1)
                ids = ids[start : start + args.seq + 1]
            else:
                ids = ids + [pad] * (args.seq + 1 - len(ids))
            x, y = ids[:-1], ids[1:]
            y = [(-100 if t == pad else t) for t in y]
            xs.append(x)
            ys.append(y)
        return torch.tensor(xs, device=device), torch.tensor(ys, device=device)

    print(f"v1_char params={p1/1e6:.2f}M fact={fact1/1e6:.2f}M ({100*fact1/p1:.1f}%) vocab={len(itos)}", flush=True)
    print(f"v2_spm params={br2['total']/1e6:.2f}M fact={br2['fact']/1e6:.2f}M ({100*br2['fact']/br2['total']:.1f}%) vocab={tok.vocab_size} attn_layers={br2['n_attn_layers']}", flush=True)

    h1, t1 = train_loop(m1, make1, args.steps, 3e-4, device, "v1_char")
    h2, t2 = train_loop(m2, make2, args.steps, 3e-4, device, "v2_spm")

    result = {
        "v1_char": {
            "params": p1,
            "fact": fact1,
            "fact_pct": 100 * fact1 / p1,
            "vocab": len(itos),
            "history": h1,
            "elapsed_sec": t1,
            "final_valid_ce": h1[-1]["valid_ce"] if h1 else None,
        },
        "v2_spm": {
            "params": br2["total"],
            "fact": br2["fact"],
            "fact_pct": 100 * br2["fact"] / br2["total"],
            "vocab": tok.vocab_size,
            "breakdown": br2,
            "history": h2,
            "elapsed_sec": t2,
            "final_valid_ce": h2[-1]["valid_ce"] if h2 else None,
        },
        "note": "Same corpus/steps/batch/seq; tokenizations differ so CE not strictly commensurate, but directionally informative.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    outp = args.out / "compare_short.json"
    outp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "v1_final_valid_ce": result["v1_char"]["final_valid_ce"],
        "v2_final_valid_ce": result["v2_spm"]["final_valid_ce"],
        "v1_fact_pct": result["v1_char"]["fact_pct"],
        "v2_fact_pct": result["v2_spm"]["fact_pct"],
        "out": str(outp),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
