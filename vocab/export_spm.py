#!/usr/bin/env python3
"""Export SentencePiece piece list / encode-decode helpers for ShelfLM v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm


class SpmTokenizer:
    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos and self.bos_id >= 0:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id >= 0:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    def export_itos(self) -> list[str]:
        return [self.sp.id_to_piece(i) for i in range(self.vocab_size)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--demo", default="你好，请做个自我介绍。1+1等于多少？")
    args = ap.parse_args()

    tok = SpmTokenizer(args.model)
    ids = tok.encode(args.demo, add_eos=True)
    blob = {
        "model": str(args.model),
        "vocab_size": tok.vocab_size,
        "unk_id": tok.unk_id,
        "bos_id": tok.bos_id,
        "eos_id": tok.eos_id,
        "pad_id": tok.pad_id,
        "demo_text": args.demo,
        "demo_ids": ids,
        "demo_decode": tok.decode(ids),
        "itos_head": tok.export_itos()[:64],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
    # also dump full itos for debugging / parity with v1 loaders
    itos_path = args.out_json.with_suffix(".itos.json")
    itos_path.write_text(json.dumps({"itos": tok.export_itos()}, ensure_ascii=False), encoding="utf-8")
    print(f"vocab_size={tok.vocab_size} demo_ids={len(ids)} wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
