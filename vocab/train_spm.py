#!/usr/bin/env python3
"""Train a Chinese-first SentencePiece model (BPE/Unigram).

Example:
  python train_spm.py --corpus /path/to/corpus.txt --out-dir ./vocab --vocab-size 24576
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True, help="UTF-8 text, one doc per line or free text")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--vocab-size", type=int, default=24576)
    ap.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    ap.add_argument("--character-coverage", type=float, default=0.9995)
    ap.add_argument("--input-sentence-size", type=int, default=5_000_000)
    ap.add_argument("--max-sentence-length", type=int, default=8192)
    ap.add_argument("--num-threads", type=int, default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_dir / f"spm_zh_{args.vocab_size}"
    spm.SentencePieceTrainer.train(
        input=str(args.corpus),
        model_prefix=str(prefix),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
        max_sentence_length=args.max_sentence_length,
        num_threads=args.num_threads,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=["用户：", "助手：", "<sft>", "<pt>"],
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
    )
    meta = {
        "prefix": str(prefix),
        "model": str(prefix) + ".model",
        "vocab": str(prefix) + ".vocab",
        "vocab_size": args.vocab_size,
        "model_type": args.model_type,
        "corpus": str(args.corpus),
        "special": {"unk": 0, "bos": 1, "eos": 2, "pad": 3},
    }
    (args.out_dir / "spm_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
