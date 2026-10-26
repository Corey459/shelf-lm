#!/usr/bin/env python3
"""ShelfLM v2 trainer (pretrain or SFT). Random-init by default; resume only v2 out ckpt.

Long-run: --hours and/or large --steps, periodic --save-every, no early stop.
Do NOT load v1 shelf-chat-* / train_24h checkpoints (vocab/Fact incompatible).
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vocab"))

from model_v2 import (  # noqa: E402
    ShelfConfigV2,
    ShelfLM,
    auxiliary_loss,
    count_parameters_breakdown,
    describe,
)
from export_spm import SpmTokenizer  # noqa: E402


def load_texts(paths: list[Path], limit: int) -> list[str]:
    texts: list[str] = []
    for path in paths:
        if path.suffix.lower() == ".txt":
            with path.open(encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if len(line) >= 4:
                        texts.append(line)
                        if 0 < limit <= len(texts):
                            return texts
            continue
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    user = (row.get("user") or row.get("instruction") or "").strip()
                    extra = (row.get("input") or "").strip()
                    assistant = (row.get("assistant") or row.get("output") or "").strip()
                    text = (row.get("text") or "").strip()
                    if text:
                        texts.append(text)
                    elif user and assistant:
                        if extra:
                            user = f"{user}\n{extra}"
                        texts.append(f"用户：{user}\n助手：{assistant}")
                    if 0 < limit <= len(texts):
                        return texts
    if not texts:
        raise SystemExit("no texts loaded")
    return texts


def encode_pack(texts: list[str], tok: SpmTokenizer, seq: int, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for _ in range(batch):
        t = random.choice(texts)
        ids = tok.encode(t, add_bos=True, add_eos=True)
        if len(ids) < 3:
            ids = tok.encode("你好", add_bos=True, add_eos=True)
        if len(ids) > seq + 1:
            start = random.randint(0, len(ids) - seq - 1)
            ids = ids[start : start + seq + 1]
        else:
            pad = [tok.pad_id if tok.pad_id >= 0 else 0] * (seq + 1 - len(ids))
            ids = ids + pad
        x = ids[:-1]
        y = ids[1:]
        y = [(-100 if (tok.pad_id >= 0 and t == tok.pad_id) else t) for t in y]
        xs.append(x)
        ys.append(y)
    return (
        torch.tensor(xs, dtype=torch.long, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
    )


@torch.no_grad()
def valid_ce(model, texts, tok, seq, device, batches=8, batch=4) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = encode_pack(texts, tok, seq, batch, device)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        losses.append(float(ce.item()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def save_ckpt(path: Path, model, cfg, spm: str, step: int, history, opt=None, extra=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg.to_dict(),
        "model": model.state_dict(),
        "spm": spm,
        "step": step,
        "history": history[-200:],
        "elapsed_sec": None,
    }
    if opt is not None:
        payload["optim"] = opt.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--spm", type=Path, required=True)
    ap.add_argument("--data", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=200, help="max steps; ignored if --hours set (then soft cap 10^12)")
    ap.add_argument("--hours", type=float, default=0.0, help="if >0, train until wall clock hours elapse (no early stop)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=20000, help="0 = load all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--valid-frac", type=float, default=0.05)
    ap.add_argument("--resume", type=Path, default=None, help="resume v2 ckpt under this out only; never v1")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    # Refuse accidental v1 paths
    if args.resume is not None:
        rs = str(args.resume)
        bad = ("shelf-chat-", "train_24h", "shelf-chat-hp24", "shelf-chat-corey24")
        if any(b in rs for b in bad) and "shelf_v2" not in rs:
            raise SystemExit(f"refuse v1 ckpt resume: {args.resume}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    tok = SpmTokenizer(args.spm)
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    raw["vocab_size"] = tok.vocab_size
    cfg = ShelfConfigV2.from_dict(raw)
    model = ShelfLM(cfg).to(device)
    br = count_parameters_breakdown(model)
    limit = args.limit if args.limit > 0 else 10**9
    texts = load_texts(args.data, limit)
    random.shuffle(texts)
    n_val = max(32, int(len(texts) * args.valid_frac))
    valid, train = texts[:n_val], texts[n_val:]
    if not train:
        train = texts
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_meta.json").write_text(
        json.dumps(
            {
                "cfg": cfg.to_dict(),
                "breakdown": br,
                "params_M": br["total"] / 1e6,
                "spm": str(args.spm),
                "n_train": len(train),
                "n_valid": len(valid),
                "device": str(device),
                "tag": args.tag,
                "hours": args.hours,
                "from_scratch": args.resume is None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"v2 {describe(cfg)} params={br['total']/1e6:.2f}M fact%={100*br['fact']/br['total']:.1f} "
        f"device={device} train={len(train)} valid={len(valid)}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    start_step = 0
    history = []
    if args.resume is not None and args.resume.is_file():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            try:
                opt.load_state_dict(ckpt["optim"])
            except Exception as e:
                print(f"optim resume skip: {e}", flush=True)
        start_step = int(ckpt.get("step", 0))
        history = list(ckpt.get("history") or [])
        print(f"resumed v2 step={start_step} from {args.resume}", flush=True)
    else:
        print("random init from scratch (no v1 load)", flush=True)

    max_steps = args.steps
    if args.hours and args.hours > 0:
        max_steps = 10**12
    end_t = time.time() + args.hours * 3600 if args.hours and args.hours > 0 else None
    t0 = time.time()
    status_path = args.out / "status.json"
    log_every = max(1, args.log_every)

    step = start_step
    model.train()
    while True:
        step += 1
        if step > max_steps:
            break
        if end_t is not None and time.time() >= end_t:
            print(f"hours budget done at step={step}", flush=True)
            break

        x, y = encode_pack(train, tok, args.seq, args.batch, device)
        use_amp = device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, aux = model(x)
            ce = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
            aux_v = auxiliary_loss(aux, cfg.n_shelves, cfg.n_experts)
            loss = ce + 0.01 * aux_v
        if not torch.isfinite(loss):
            print(f"non-finite step {step}, skip", flush=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % log_every == 0:
            elapsed_h = (time.time() - t0) / 3600
            remain = ""
            if end_t is not None:
                remain = f" remain_h={(end_t - time.time())/3600:.2f}"
            print(f"step={step} ce={ce.item():.3f} aux={aux_v.item():.3f} elapsed_h={elapsed_h:.3f}{remain}", flush=True)

        if step % args.eval_every == 0:
            vce = valid_ce(model, valid, tok, args.seq, device)
            history.append({"step": step, "train_ce": float(ce.item()), "valid_ce": vce, "time": time.strftime("%F %T")})
            print(f"eval step={step} train_ce={ce.item():.3f} valid_ce={vce:.3f}", flush=True)
            status_path.write_text(
                json.dumps(
                    {
                        "tag": args.tag,
                        "step": step,
                        "train_ce": float(ce.item()),
                        "valid_ce": vce,
                        "elapsed_h": (time.time() - t0) / 3600,
                        "time": time.strftime("%F %T"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        if args.save_every > 0 and step % args.save_every == 0:
            save_ckpt(
                args.out / "model.pt",
                model,
                cfg,
                str(args.spm),
                step,
                history,
                opt=opt,
                extra={"elapsed_sec": time.time() - t0},
            )
            (args.out / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"saved {args.out / 'model.pt'} step={step}", flush=True)

    # final save
    save_ckpt(
        args.out / "model.pt",
        model,
        cfg,
        str(args.spm),
        step,
        history,
        opt=opt,
        extra={"elapsed_sec": time.time() - t0},
    )
    (args.out / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done step={step} elapsed={time.time()-t0:.1f}s -> {args.out / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
