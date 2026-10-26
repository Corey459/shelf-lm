
#!/usr/bin/env python3
"""CPU-only chat smoke for shelf_v2 ckpt; does not touch CUDA / training GPU."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vocab"))
from model_v2 import ShelfConfigV2, ShelfLM
from export_spm import SpmTokenizer

def generate(model, tok, prompt: str, max_new: int = 80, temperature: float = 0.8, top_p: float = 0.9):
    model.eval()
    ids = tok.encode(prompt, add_bos=True, add_eos=False)
    if not ids:
        ids = [tok.bos_id if tok.bos_id >= 0 else 0]
    x = torch.tensor([ids], dtype=torch.long)
    eos = tok.eos_id
    with torch.no_grad():
        for _ in range(max_new):
            logits, _ = model(x)
            logits = logits[0, -1].float() / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
                next_id = sorted_idx[torch.multinomial(sorted_probs, 1)].item()
            else:
                next_id = torch.argmax(probs).item()
            ids.append(next_id)
            if eos is not None and eos >= 0 and next_id == eos:
                break
            x = torch.tensor([ids], dtype=torch.long)
            # keep context bounded
            if x.size(1) > 384:
                x = x[:, -384:]
                ids = ids[-384:]
    # decode only continuation after prompt tokens length
    prompt_n = len(tok.encode(prompt, add_bos=True, add_eos=False))
    cont = ids[prompt_n:]
    return tok.decode(cont), tok.decode(ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--spm", type=Path, required=True)
    ap.add_argument("--max-new", type=int, default=96)
    args = ap.parse_args()

    cfg = ShelfConfigV2.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    tok = SpmTokenizer(args.spm)
    model = ShelfLM(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    step = ckpt.get("step")
    print(f"LOADED step={step} device=cpu params~{sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    prompts = [
        "用户：你好，请用一句话介绍你自己。\n助手：",
        "用户：1+1等于多少？只回答数字。\n助手：",
        "用户：用两句话说明什么是仓库货架管理。\n助手：",
        "你好，今天天气怎么样？",
    ]
    for i, p in enumerate(prompts, 1):
        cont, full = generate(model, tok, p, max_new=args.max_new)
        print(f"\n===== PROMPT {i} =====", flush=True)
        print(p, flush=True)
        print("--- continuation ---", flush=True)
        print(cont.replace("\n", "\\n"), flush=True)

if __name__ == "__main__":
    main()
