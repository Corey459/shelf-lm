#!/usr/bin/env python3
"""Build a plain-text corpus for SentencePiece + optional pretrain shards.

Pulls from existing shelf-lm jsonl/json and optionally streams HF mirror datasets.
Does NOT touch train_24h outputs.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def iter_pairs(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield row
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            yield row


def row_to_text(row: dict) -> str | None:
    # instruction / chat
    user = (row.get("user") or row.get("instruction") or row.get("prompt") or "").strip()
    extra = (row.get("input") or "").strip()
    assistant = (row.get("assistant") or row.get("output") or row.get("response") or "").strip()
    text = (row.get("text") or row.get("content") or "").strip()
    if text:
        return text
    if user and assistant:
        if extra:
            user = f"{user}\n{extra}"
        return f"用户：{user}\n助手：{assistant}"
    if user:
        return user
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-txt", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, default=None, help="optional chat/pretrain mixed jsonl")
    ap.add_argument("--limit", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    jsonl_f = args.out_jsonl.open("w", encoding="utf-8") if args.out_jsonl else None
    with args.out_txt.open("w", encoding="utf-8") as out:
        for path in args.inputs:
            if not path.exists():
                print(f"skip missing {path}", flush=True)
                continue
            for row in iter_pairs(path):
                text = row_to_text(row)
                if not text or len(text) < 4:
                    continue
                # keep lines manageable for SPM
                text = text.replace("\n", " ").strip()[:2000]
                out.write(text + "\n")
                if jsonl_f is not None:
                    kind = "sft" if ("assistant" in row or "output" in row) else "pt"
                    jsonl_f.write(json.dumps({"text": text, "kind": kind}, ensure_ascii=False) + "\n")
                n += 1
                if n >= args.limit:
                    break
            if n >= args.limit:
                break
    if jsonl_f:
        jsonl_f.close()
    print(f"wrote {n} lines -> {args.out_txt}", flush=True)


if __name__ == "__main__":
    main()
