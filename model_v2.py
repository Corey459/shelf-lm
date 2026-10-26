"""ShelfLM v2 — independent-scale architecture.

Changes vs v1 (90M char-level):
1. SentencePiece / BPE vocab (16k–32k), not char-level.
2. FactTable shrink: low-dim rows + projection (+ optional soft gate / external stub).
3. Interleaved causal attention every `attn_every` layers (StateMix remains default mixer).
4. Config target ~300M total params for single RTX 5090 train / 5060 infer.

Architecture spirit kept: ShelfLM / StateMix / FactTable / ShelfMoE.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ShelfConfigV2:
    vocab_size: int = 24576
    d_model: int = 1024
    n_layers: int = 18
    n_heads: int = 16
    n_shelves: int = 4
    experts_per_shelf: int = 8
    top_k: int = 2
    rank: int = 64
    ffn_mult: int = 2
    # FactTable: old 65k×d_model ≈ 42M; new default 8k×128 + proj ≈ 1.2M
    fact_rows: int = 8192
    fact_dim: int = 128
    fact_mode: str = "lowrank"  # lowrank | sparse_gate | external
    # Sequence: insert causal MHA every N layers (0 = StateMix only)
    attn_every: int = 4
    tie_embeddings: bool = True
    # StateMix: positive bias → slower decay → longer memory at init
    state_gate_bias: float = 1.5
    dropout: float = 0.0

    @property
    def n_experts(self) -> int:
        return self.n_shelves * self.experts_per_shelf

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ShelfConfigV2":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in names})


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        y = x.float() * torch.rsqrt(var + self.eps)
        return (y * self.weight).to(dtype=x.dtype)


def parallel_decay_scan(u: torch.Tensor, a: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """s_t = a_t * s_{t-1} + u_t, s_0 = 0. Chunked to avoid cumprod underflow."""
    u32 = u.float()
    a32 = a.float().clamp(1e-4, 1.0 - 1e-4)
    length = u32.shape[1]
    state = u32.new_zeros(u32.shape[0], u32.shape[2], u32.shape[3])
    outputs = []
    for start in range(0, length, chunk):
        u_chunk = u32[:, start : start + chunk]
        a_chunk = a32[:, start : start + chunk]
        steps = []
        for index in range(u_chunk.shape[1]):
            state = a_chunk[:, index] * state + u_chunk[:, index]
            steps.append(state)
        outputs.append(torch.stack(steps, dim=1))
    scanned = torch.cat(outputs, dim=1)
    if scanned.shape[1] != length:
        scanned = scanned[:, :length]
    return scanned.to(dtype=u.dtype)


class StateMix(nn.Module):
    """Gated diagonal recurrence; fixed-size context state."""

    def __init__(self, cfg: ShelfConfigV2):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.norm = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.decay = nn.Linear(d, cfg.n_heads, bias=True)
        nn.init.constant_(self.decay.bias, cfg.state_gate_bias)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        b, t, _ = q.shape
        heads, width = self.n_heads, self.d_head
        q = q.view(b, t, heads, width)
        k = k.view(b, t, heads, width)
        v = v.view(b, t, heads, width)
        with torch.autocast(device_type=h.device.type, enabled=False):
            gate = torch.sigmoid(self.decay(h.float())).unsqueeze(-1)
            state = parallel_decay_scan(k.float() * v.float(), gate)
        y = (q * state).reshape(b, t, -1)
        return self.out(y)


class CausalSelfAttn(nn.Module):
    """Sparse nonlocal mixer inserted every attn_every layers."""

    def __init__(self, cfg: ShelfConfigV2):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.norm = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        b, t, _ = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        # scaled_dot_product_attention uses flash/mem-efficient when available
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.out(y)


class FactTable(nn.Module):
    """Compressed fact memory.

    Modes:
      - lowrank: Embedding(rows, fact_dim) + Linear → d_model  (default, tiny params)
      - sparse_gate: same + sigmoid gate from token embedding (only inject when useful)
      - external: zero stub; call `set_external(fn)` for retrieval later
    """

    def __init__(self, cfg: ShelfConfigV2):
        super().__init__()
        self.mode = cfg.fact_mode
        self.n_rows = cfg.fact_rows
        self.d_model = cfg.d_model
        if self.mode == "external":
            self.table = None
            self.proj = None
            self.gate = None
            self._external = None
        else:
            self.table = nn.Embedding(cfg.fact_rows, cfg.fact_dim)
            self.proj = nn.Linear(cfg.fact_dim, cfg.d_model, bias=False)
            nn.init.normal_(self.table.weight, std=0.02)
            nn.init.zeros_(self.proj.weight)
            if self.mode == "sparse_gate":
                self.gate = nn.Linear(cfg.d_model, 1, bias=True)
                nn.init.constant_(self.gate.bias, -2.0)  # start mostly off
            else:
                self.gate = None
            self._external = None

    def set_external(self, fn) -> None:
        """fn(token_ids) -> [B,T,d_model] retrieval features."""
        self._external = fn

    def forward(self, token_ids: torch.Tensor, tok_emb: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "external":
            if self._external is not None:
                return self._external(token_ids)
            return token_ids.new_zeros(*token_ids.shape, self.d_model, dtype=torch.float32).to(
                dtype=tok_emb.dtype if tok_emb is not None else torch.float32
            )
        prev = F.pad(token_ids[:, :-1], (1, 0), value=0)
        idx = (prev * 31337 + token_ids) % self.n_rows
        feat = self.proj(self.table(idx))
        if self.gate is not None and tok_emb is not None:
            g = torch.sigmoid(self.gate(tok_emb))
            feat = feat * g
        return feat


class ShelfMoE(nn.Module):
    """Shared SwiGLU FFN + low-rank expert deltas; two-level shelf→expert routing."""

    def __init__(self, cfg: ShelfConfigV2):
        super().__init__()
        d = cfg.d_model
        hidden = int(cfg.d_model * cfg.ffn_mult)
        self.cfg = cfg
        self.norm = RMSNorm(d)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)
        self.shelf_router = nn.Linear(d, cfg.n_shelves, bias=False)
        self.expert_router = nn.Linear(d, cfg.n_experts, bias=False)
        self.A = nn.Parameter(torch.empty(cfg.n_experts, d, cfg.rank))
        self.B = nn.Parameter(torch.zeros(cfg.n_experts, cfg.rank, d))
        nn.init.normal_(self.A, std=0.02)
        shelf_of = torch.arange(cfg.n_experts) // cfg.experts_per_shelf
        self.register_buffer("shelf_of", shelf_of, persistent=False)

    def shared(self, h: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(h)) * self.w2(h))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.norm(x)
        b, t, d = h.shape
        shelf_logits = self.shelf_router(h)
        shelf_prob = F.softmax(shelf_logits, dim=-1)
        shelf_id = shelf_logits.argmax(dim=-1)

        expert_logits = self.expert_router(h)
        on_shelf = self.shelf_of.view(1, 1, -1) == shelf_id.unsqueeze(-1)
        expert_logits = expert_logits.masked_fill(~on_shelf, -1e4)
        top_val, top_idx = torch.topk(expert_logits, self.cfg.top_k, dim=-1)
        top_w = F.softmax(top_val, dim=-1)

        flat_h = h.reshape(-1, d)
        flat_idx = top_idx.reshape(-1, self.cfg.top_k)
        flat_w = top_w.reshape(-1, self.cfg.top_k)
        delta = flat_h.new_zeros(flat_h.shape)
        for slot in range(self.cfg.top_k):
            idx = flat_idx[:, slot]
            mid = torch.einsum("nd,ndr->nr", flat_h, self.A[idx])
            out = torch.einsum("nr,nrd->nd", mid, self.B[idx])
            delta = delta + flat_w[:, slot : slot + 1] * out

        y = self.shared(h) + delta.view(b, t, d)
        stats = {
            "shelf_prob": shelf_prob,
            "expert_prob": F.softmax(expert_logits, dim=-1),
            "shelf_id": shelf_id,
        }
        return y, stats


class Block(nn.Module):
    def __init__(self, cfg: ShelfConfigV2, use_attn: bool):
        super().__init__()
        self.state = StateMix(cfg)
        self.attn = CausalSelfAttn(cfg) if use_attn else None
        self.moe = ShelfMoE(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = x + self.state(x)
        if self.attn is not None:
            x = x + self.attn(x)
        delta, stats = self.moe(x)
        return x + delta, stats


class ShelfLM(nn.Module):
    def __init__(self, cfg: ShelfConfigV2):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.fact = FactTable(cfg)
        blocks = []
        for i in range(cfg.n_layers):
            # layers 0-indexed: attn on layers where (i+1) % attn_every == 0
            use_attn = cfg.attn_every > 0 and ((i + 1) % cfg.attn_every == 0)
            blocks.append(Block(cfg, use_attn=use_attn))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.tok.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tok = self.tok(token_ids)
        x = tok + self.fact(token_ids, tok_emb=tok)
        shelf_probs, expert_probs, shelf_ids = [], [], []
        for block in self.blocks:
            x, stats = block(x)
            shelf_probs.append(stats["shelf_prob"])
            expert_probs.append(stats["expert_prob"])
            shelf_ids.append(stats["shelf_id"])
        logits = self.head(self.norm(x))
        aux = {
            "shelf_prob": torch.stack(shelf_probs, dim=0),
            "expert_prob": torch.stack(expert_probs, dim=0),
            "shelf_id": torch.stack(shelf_ids, dim=0),
        }
        return logits, aux


def auxiliary_loss(aux: dict[str, torch.Tensor], n_shelves: int, n_experts: int) -> torch.Tensor:
    shelf = aux["shelf_prob"]
    expert = aux["expert_prob"]
    balance_s = n_shelves * shelf.mean(dim=(1, 2)).pow(2).sum(dim=-1).mean()
    balance_e = n_experts * expert.mean(dim=(1, 2)).pow(2).sum(dim=-1).mean()
    left = shelf[:, :, :-1, :]
    right = shelf[:, :, 1:, :]
    stick = (1.0 - F.cosine_similarity(left, right, dim=-1)).mean()
    return balance_s + 0.1 * balance_e + 0.05 * stick


@torch.no_grad()
def generate(model: ShelfLM, prompt: torch.Tensor, steps: int, temperature: float = 0.8) -> torch.Tensor:
    model.eval()
    out = prompt
    for _ in range(steps):
        logits, _ = model(out[:, -1024:])
        nxt = logits[:, -1, :]
        if temperature <= 0:
            idx = nxt.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(nxt / temperature, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, idx], dim=1)
    return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_parameters_breakdown(model: ShelfLM) -> dict[str, int]:
    tok = sum(p.numel() for p in model.tok.parameters())
    fact = sum(p.numel() for p in model.fact.parameters()) if model.fact.table is not None else 0
    blocks = sum(p.numel() for p in model.blocks.parameters())
    state = sum(p.numel() for b in model.blocks for p in b.state.parameters())
    attn = sum(p.numel() for b in model.blocks if b.attn is not None for p in b.attn.parameters())
    moe = sum(p.numel() for b in model.blocks for p in b.moe.parameters())
    head = 0 if model.cfg.tie_embeddings else sum(p.numel() for p in model.head.parameters())
    norm = sum(p.numel() for p in model.norm.parameters())
    total = count_parameters(model)
    return {
        "total": total,
        "embed": tok,
        "fact": fact,
        "blocks": blocks,
        "state": state,
        "attn": attn,
        "moe": moe,
        "head": head,
        "norm": norm,
        "n_attn_layers": sum(1 for b in model.blocks if b.attn is not None),
    }


def describe(cfg: ShelfConfigV2) -> str:
    return (
        f"d={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} "
        f"shelves={cfg.n_shelves} exp/shelf={cfg.experts_per_shelf} "
        f"top_k={cfg.top_k} rank={cfg.rank} fact={cfg.fact_rows}x{cfg.fact_dim}/{cfg.fact_mode} "
        f"attn_every={cfg.attn_every} vocab={cfg.vocab_size}"
    )
