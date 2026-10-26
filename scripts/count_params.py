#!/usr/bin/env python3
"""Print parameter breakdown for a ShelfLM v2 config."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_v2 import ShelfConfigV2, ShelfLM, count_parameters_breakdown, describe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = ShelfConfigV2.from_dict(raw)
    model = ShelfLM(cfg)
    br = count_parameters_breakdown(model)
    print(describe(cfg))
    for k, v in br.items():
        if k == "n_attn_layers":
            print(f"  {k}: {v}")
        else:
            pct = 100.0 * v / br["total"] if br["total"] else 0.0
            print(f"  {k}: {v:,} ({v/1e6:.2f}M, {pct:.1f}%)")
    out = args.config.with_suffix(".params.json")
    payload = {"config": cfg.to_dict(), "breakdown": br, "params_M": br["total"] / 1e6}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
