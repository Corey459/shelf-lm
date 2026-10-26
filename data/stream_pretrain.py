#!/usr/bin/env python3
"""Streaming Chinese pretrain / instruction mix via hf-mirror.

Stage plan (see README_SOLO.md):
  1) pretrain (causal LM on plain text)
  2) SFT (existing firefly/belle/math + seed)
  3) optional preference

This module only provides iterators + a download helper; it does not start
long GPU jobs while train_24h owns the card.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator


# Prefer China mirror; user env can override.
DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# Lightweight, known-good Chinese sources (instruction + wiki-ish).
PRETRAIN_SOURCES = [
    {
        "name": "wikipedia_zh_snippet",
        "path": "pleisto/wikipedia-cn-20230720-filtered",
        "split": "train",
        "text_field": "completion",
        "note": "Chinese Wikipedia filtered; use as PT stream",
    },
    {
        "name": "sky_pile_sample",
        "path": "Skywork/SkyPile-150B",
        "split": "train",
        "text_field": "text",
        "note": "Huge; only stream a few shards for SP/corpus bootstrap",
    },
]

SFT_SOURCES = [
    {
        "name": "belle_1m",
        "path": "BelleGroup/train_1M_CN",
        "split": "train",
        "map": "belle",
    },
    {
        "name": "firefly",
        "path": "YeungNLP/firefly-train-1.1M",
        "split": "train",
        "map": "firefly",
    },
    {
        "name": "belle_math",
        "path": "BelleGroup/school_math_0.25M",
        "split": "train",
        "map": "belle",
    },
]


def _ensure_hf_endpoint() -> None:
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_ENDPOINT)
    # datasets / huggingface_hub honor HF_ENDPOINT for mirror


def iter_local_jsonl(path: Path, text_field: str = "text") -> Iterator[str]:
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = row.get(text_field) or row.get("content") or ""
            if t:
                yield t


def stream_hf(path: str, split: str, text_field: str, limit: int) -> Iterator[str]:
    _ensure_hf_endpoint()
    from datasets import load_dataset

    ds = load_dataset(path, split=split, streaming=True)
    n = 0
    for row in ds:
        text = row.get(text_field) or ""
        if not text and "input" in row and "target" in row:
            text = f"{row['input']}\n{row['target']}"
        if not text:
            continue
        yield str(text).replace("\n", " ").strip()
        n += 1
        if n >= limit:
            break


def download_sample(out: Path, source_key: str, limit: int) -> None:
    meta = {s["name"]: s for s in PRETRAIN_SOURCES + SFT_SOURCES}
    if source_key not in meta:
        raise SystemExit(f"unknown source {source_key}; choose from {list(meta)}")
    src = meta[source_key]
    field = src.get("text_field", "text")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for text in stream_hf(src["path"], src.get("split", "train"), field, limit):
            f.write(json.dumps({"text": text, "source": source_key, "kind": "pt" if source_key in {p['name'] for p in PRETRAIN_SOURCES} else "sft"}, ensure_ascii=False) + "\n")
            n += 1
    print(f"downloaded {n} rows from {src['path']} -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download", default=None, help="source name to sample via hf-mirror")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=5000)
    args = ap.parse_args()
    if args.list:
        print("PRETRAIN:", json.dumps(PRETRAIN_SOURCES, ensure_ascii=False, indent=2))
        print("SFT:", json.dumps(SFT_SOURCES, ensure_ascii=False, indent=2))
        print(f"HF_ENDPOINT={DEFAULT_ENDPOINT}")
        return
    if args.download:
        if args.out is None:
            raise SystemExit("--out required with --download")
        download_sample(args.out, args.download, args.limit)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
