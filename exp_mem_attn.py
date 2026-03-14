"""
exp_mem_attn.py — Memory-augmented Plücker attention

Combines the transversal memory system with transformer attention:
- Standard Q·K attention over the sequence (intra-sequence)
- Plücker incidence scoring against an external Gram memory bank (inter-memory)
- Online accumulation: tokens write Plücker lines into memory during forward pass

The Gram memory stores relational structure as M = Σ pᵢ⊗pᵢ (6×6 matrices).
Query lines from the current token are scored against memory via p·(J6·M·J6ᵀ)·p,
which measures how well the query line fits the stored relational pattern.

Two modes tested:
  1. Static memory: pre-built from training data, frozen during inference
  2. Online memory: accumulated during the forward pass (option 3 from ideas.md)

Usage:
  python exp_mem_attn.py static    # static external memory
  python exp_mem_attn.py online    # online accumulation
  python exp_mem_attn.py both      # compare all variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import pyarrow.parquet as pq
import time
import math
import sys
from pathlib import Path
from itertools import combinations

# ── Config ───────────────────────────────────────────────────────────────────

class Config:
    d_model = 192
    n_heads = 6
    n_layers = 4
    dropout = 0.1
    seq_len = 128
    batch_size = 64
    n_epochs = 10
    lr = 3e-4
    grad_clip = 1.0
    n_mem_slots = 32       # number of Gram memory slots
    mem_decay = 0.99       # exponential decay for online memory
    data_dir = Path("data/wikitext")

# ── Plücker primitives (same as exp_lm_variants.py) ─────────────────────────

_PAIRS = list(combinations(range(4), 2))

_J6 = torch.tensor([
    [0,0,0,0,0,1],[0,0,0,0,-1,0],[0,0,0,1,0,0],
    [0,0,1,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0],
], dtype=torch.float32)

def exterior(p1, p2):
    """(..., 4) × (..., 4) → (..., 6) normalized Plücker lines."""
    parts = [p1[...,i]*p2[...,j] - p1[...,j]*p2[...,i] for i,j in _PAIRS]
    L = torch.stack(parts, dim=-1)
    return L / L.norm(dim=-1, keepdim=True).clamp(min=1e-12)

def gram_score(lines, gram):
    """
    Score lines against a Gram matrix via Plücker inner product.
    lines: (..., 6), gram: (..., 6, 6)
    Returns (...) scalar scores.
    """
    # score = lines @ J6 @ gram @ J6.T @ lines.T (per-element)
    # Simplified: score = (lines @ J6 @ gram) . (J6 @ lines)
    J = _J6.to(lines.device)
    Jlines = lines @ J.T          # (..., 6)
    scored = Jlines @ gram        # (..., 6)
    return (scored * Jlines).sum(dim=-1)

# ── Standard attention (baseline) ────────────────────────────────────────────

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = self.drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)

# ── Memory-augmented attention (static external memory) ──────────────────────

class MemoryAugmentedAttention(nn.Module):
    """
    Standard Q·K attention + Plücker memory read.

    Each head projects tokens to Plücker lines and scores them against
    an external Gram memory bank. The memory scores are added as a bias
    to the attention logits (like hybrid v4) AND used to gate a memory
    value that's added to the output.

    Memory bank: (n_slots, 6, 6) tensor of Gram matrices, one per "concept".
    Each token's query line is scored against all slots. The best-matching
    slot's value is mixed into the output.
    """
    def __init__(self, d_model, n_heads, n_mem_slots, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_mem_slots = n_mem_slots
        self.scale = self.d_head ** -0.5

        # Standard Q/K/V
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # Plücker projections for memory queries (per head)
        self.W1_mem = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_mem = nn.Linear(d_model, 4 * n_heads, bias=False)

        # Memory bank: learnable Gram matrices and values
        # Each slot stores a 6×6 Gram matrix and a d_model-dim value
        self.mem_grams = nn.Parameter(torch.randn(n_mem_slots, 6, 6) * 0.01)
        self.mem_values = nn.Parameter(torch.randn(n_mem_slots, d_model) * 0.02)

        # Memory gate: controls how much memory contributes
        self.mem_gate = nn.Linear(d_model, n_heads)

        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('J6', _J6)

    def forward(self, x, **kwargs):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        # === Standard attention ===
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = self.drop(F.softmax(attn, dim=-1))
        seq_out = (attn @ v).transpose(1, 2).reshape(B, T, D)

        # === Memory read via Plücker scoring ===
        # Project tokens to lines: (B, T, H, 6)
        p1 = self.W1_mem(x).reshape(B, T, H, 4)
        p2 = self.W2_mem(x).reshape(B, T, H, 4)
        lines = exterior(p1, p2)  # (B, T, H, 6)

        # Score each line against each memory Gram matrix
        # lines: (B, T, H, 6), mem_grams: (S, 6, 6)
        # Make Gram matrices symmetric positive semi-definite
        G = self.mem_grams
        G_sym = (G + G.transpose(-1, -2)) / 2  # (S, 6, 6)

        # Score: for each (b, t, h), compute line @ J6 @ G_s @ J6.T @ line
        J = self.J6  # (6, 6)
        Jlines = torch.einsum('bthi,ij->bthj', lines, J)  # (B, T, H, 6)
        # (B, T, H, 6) @ (S, 6, 6) -> (B, T, H, S, 6) -> dot with Jlines -> (B, T, H, S)
        scored = torch.einsum('bthi,sij,bthj->bths', Jlines, G_sym, Jlines)  # (B, T, H, S)

        # Softmax over slots to get memory attention weights
        mem_attn = F.softmax(scored, dim=-1)  # (B, T, H, S)

        # Read from memory values: (B, T, H, S) @ (S, D) -> (B, T, H, D)
        # Then reduce over heads by reshaping
        mem_read = torch.einsum('bths,sd->bthd', mem_attn, self.mem_values)  # (B, T, H, D)
        # Average over heads
        mem_read = mem_read.mean(dim=2)  # (B, T, D)

        # Gate: how much memory to mix in
        gate = torch.sigmoid(self.mem_gate(x))  # (B, T, H)
        gate = gate.mean(dim=-1, keepdim=True)  # (B, T, 1)

        # Combine sequence attention output + gated memory read
        combined = seq_out + gate * mem_read

        return self.out(combined)

# ── Online memory accumulation attention ─────────────────────────────────────

class OnlineMemoryAttention(nn.Module):
    """
    Standard attention + online Gram memory that accumulates during the
    forward pass.

    As tokens are processed, consecutive pairs are encoded as Plücker lines
    and accumulated into a running Gram matrix. The current token's query
    line is scored against this accumulated memory.

    Memory-efficient implementation: instead of materializing (B,T,H,6,6)
    Gram matrices at every position, we compute the Plücker incidence
    between read lines and all past write lines using causal masking on
    the (T,T) incidence matrix — same memory footprint as standard attention.

    The score at position t is: Σ_{s<t} (read_t · J6 · write_s)²
    This is equivalent to read_t @ J6 @ M_t @ J6.T @ read_t where
    M_t = Σ_{s<t} write_s ⊗ write_s, but computed without building M.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, decay=0.99):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.decay = decay

        # Standard Q/K/V
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # Plücker projections for memory write/read (per head)
        self.W1_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W1_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_read = nn.Linear(d_model, 4 * n_heads, bias=False)

        # Memory value projection (what gets read out)
        self.mem_value = nn.Linear(d_model, d_model)
        # Gate
        self.mem_gate = nn.Linear(d_model, n_heads)
        # Scale for memory score
        self.mem_scale = nn.Parameter(torch.full((n_heads,), 0.1))

        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('J6', _J6)

    def forward(self, x, **kwargs):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        # === Standard attention ===
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        std_attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        std_attn = std_attn.masked_fill(mask, float('-inf'))
        std_attn = self.drop(F.softmax(std_attn, dim=-1))
        seq_out = (std_attn @ v).transpose(1, 2).reshape(B, T, D)

        # === Online memory via Plücker incidence ===
        # Write lines from bigrams [x_{t-1}, x_t]
        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1]], dim=1)
        w1 = self.W1_write(x_prev).reshape(B, T, H, 4)
        w2 = self.W2_write(x).reshape(B, T, H, 4)
        write_lines = exterior(w1, w2)  # (B, T, H, 6)

        # Read lines from current token
        r1 = self.W1_read(x).reshape(B, T, H, 4)
        r2 = self.W2_read(x).reshape(B, T, H, 4)
        read_lines = exterior(r1, r2)  # (B, T, H, 6)

        # Compute Plücker incidence matrix: (B, H, T_read, T_write)
        # incidence[t,s] = read_t · J6 · write_s
        J = self.J6
        J_write = torch.einsum('bthi,ij->bthj', write_lines, J)  # (B, T, H, 6)
        # Reshape for batched matmul: (B, H, T, 6)
        read_h = read_lines.permute(0, 2, 1, 3)    # (B, H, T, 6)
        Jwrite_h = J_write.permute(0, 2, 1, 3)     # (B, H, T, 6)

        # Incidence matrix: (B, H, T, T) — same shape as standard attention
        incidence = read_h @ Jwrite_h.transpose(-1, -2)  # (B, H, T, T)

        # Square it — Gram energy is sum of squared incidences
        incidence_sq = incidence ** 2

        # Causal mask: position t only sees s < t (strictly causal)
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=0).bool()
        incidence_sq = incidence_sq.masked_fill(causal, 0.0)

        # Memory score per position: sum of squared incidences with past
        mem_score = incidence_sq.sum(dim=-1)  # (B, H, T)

        # Memory value: gated by score
        mem_val = self.mem_value(x)  # (B, T, D)
        gate = torch.sigmoid(self.mem_gate(x))  # (B, T, H)
        scale = self.mem_scale.reshape(1, H, 1)
        mem_score_t = mem_score.permute(0, 2, 1)  # (B, T, H)
        gated = torch.sigmoid(mem_score_t * scale.permute(0, 2, 1)) * gate  # (B, T, H)
        gated = gated.mean(dim=-1, keepdim=True)  # (B, T, 1)

        combined = seq_out + gated * mem_val

        return self.out(combined)

# ── Transformer + LM ────────────────────────────────────────────────────────

ATTN_CLASSES = {
    "standard": lambda d, h, **kw: StandardAttention(d, h, kw.get('dropout', 0.1)),
    "static_mem": lambda d, h, **kw: MemoryAugmentedAttention(
        d, h, kw.get('n_mem_slots', 32), kw.get('dropout', 0.1)),
    "online_mem": lambda d, h, **kw: OnlineMemoryAttention(
        d, h, kw.get('dropout', 0.1), kw.get('decay', 0.99)),
}

class Block(nn.Module):
    def __init__(self, d_model, n_heads, attn_type, dropout=0.1, **attn_kw):
        super().__init__()
        self.attn = ATTN_CLASSES[attn_type](d_model, n_heads, dropout=dropout, **attn_kw)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class LM(nn.Module):
    def __init__(self, vocab_size, cfg, attn_type, **attn_kw):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.Sequential(*[
            Block(cfg.d_model, cfg.n_heads, attn_type, cfg.dropout, **attn_kw)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        return self.head(self.ln_f(self.blocks(x)))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# ── Data ─────────────────────────────────────────────────────────────────────

def load_tokens(split, data_dir, enc):
    fname = {"train": "train.parquet", "val": "val.parquet", "test": "test.parquet"}
    table = pq.read_table(data_dir / fname[split])
    text = "\n".join(t for t in table['text'].to_pylist() if t.strip())
    return torch.tensor(enc.encode(text, allowed_special=set()), dtype=torch.long)

def make_batches(data, seq_len, batch_size):
    n = len(data) - 1
    n_seqs = (n // seq_len // batch_size) * batch_size
    data = data[:n_seqs * seq_len + 1]
    x = data[:-1].reshape(n_seqs, seq_len)
    y = data[1:].reshape(n_seqs, seq_len)
    perm = torch.randperm(n_seqs)
    return [(x[perm[i:i+batch_size]], y[perm[i:i+batch_size]])
            for i in range(0, n_seqs, batch_size)
            if i + batch_size <= n_seqs]

# ── Training ─────────────────────────────────────────────────────────────────

def train_one(attn_type, cfg, device, **attn_kw):
    enc = tiktoken.get_encoding("gpt2")
    train_data = load_tokens("train", cfg.data_dir, enc)
    val_data = load_tokens("val", cfg.data_dir, enc)

    model = LM(enc.n_vocab, cfg, attn_type, **attn_kw).to(device)
    print(f"  {attn_type}: {model.count_params():,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    losses, ppls = [], []

    for ep in range(cfg.n_epochs):
        t0 = time.time()
        model.train()
        batches = make_batches(train_data, cfg.seq_len, cfg.batch_size)
        ep_loss = 0.0
        for xb, yb in batches:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb).view(-1, enc.n_vocab), yb.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); ep_loss += loss.item()
        avg = ep_loss / len(batches); losses.append(avg)

        model.eval()
        with torch.no_grad():
            vb = make_batches(val_data, cfg.seq_len, cfg.batch_size)
            vl = sum(F.cross_entropy(model(x.to(device)).view(-1, enc.n_vocab),
                     y.to(device).view(-1)).item() for x, y in vb) / len(vb)
        ppl = math.exp(min(vl, 20)); ppls.append(ppl)

        print(f"    ep {ep+1:2d}: loss={avg:.3f}  ppl={ppl:.1f}  ({time.time()-t0:.1f}s)")

    # Save
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "type": attn_type,
                "losses": losses, "ppls": ppls, "vocab": enc.n_vocab},
               f"checkpoints/lm_{attn_type}.pt")
    return model, losses, ppls

# ── Generation ───────────────────────────────────────────────────────────────

def generate(model, enc, prompt, max_tok=60, temp=0.8, top_k=40):
    model.eval()
    dev = next(model.parameters()).device
    idx = torch.tensor([enc.encode(prompt, allowed_special=set())[-model.cfg.seq_len:]],
                       dtype=torch.long, device=dev)
    with torch.no_grad():
        for _ in range(max_tok):
            logits = model(idx[:, -model.cfg.seq_len:])[:, -1] / temp
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    return enc.decode(idx[0].tolist())

# ── Main ─────────────────────────────────────────────────────────────────────

PROMPTS = [
    "The president of the United States",
    "In the year 1945 ,",
    "The cat sat on the",
    "Scientists discovered that",
    "The album was released in",
]

def run_variant(variant, cfg, device):
    print(f"\n{'='*60}")
    print(f"  VARIANT: {variant} (vs standard baseline)")
    print(f"{'='*60}")

    attn_kw = {}
    if variant == "static_mem":
        attn_kw = {"n_mem_slots": cfg.n_mem_slots}
    elif variant == "online_mem":
        attn_kw = {"decay": cfg.mem_decay}

    print("\n  Training standard baseline...")
    std, std_l, std_p = train_one("standard", cfg, device)

    print(f"\n  Training {variant}...")
    var, var_l, var_p = train_one(variant, cfg, device, **attn_kw)

    # Move to CPU for generation to avoid MPS issues
    std = std.cpu()
    var = var.cpu()

    enc = tiktoken.get_encoding("gpt2")

    print(f"\n  {'':20s} {'Standard':>10s} {variant:>10s}")
    print(f"  {'-'*42}")
    print(f"  {'Params':20s} {std.count_params():>10,d} {var.count_params():>10,d}")
    print(f"  {'Best val PPL':20s} {min(std_p):>10.1f} {min(var_p):>10.1f}")
    print(f"  {'Final val PPL':20s} {std_p[-1]:>10.1f} {var_p[-1]:>10.1f}")

    # Report memory-specific parameters
    if variant == "online_mem":
        for name, mod in var.named_modules():
            if isinstance(mod, OnlineMemoryAttention):
                mem_params = sum(p.numel() for n, p in mod.named_parameters()
                               if 'write' in n or 'read' in n or 'mem_' in n)
                print(f"  {'Memory params':20s} {'':>10s} {mem_params:>10,d}")
                break
    elif variant == "static_mem":
        for name, mod in var.named_modules():
            if isinstance(mod, MemoryAugmentedAttention):
                mem_params = sum(p.numel() for n, p in mod.named_parameters()
                               if 'mem_' in n or 'W1_mem' in n or 'W2_mem' in n)
                print(f"  {'Memory params':20s} {'':>10s} {mem_params:>10,d}")
                break

    print("\n  SAMPLES:")
    for prompt in PROMPTS:
        print(f"\n  PROMPT: \"{prompt}\"")
        print(f"    STD: {generate(std, enc, prompt)}")
        print(f"    {variant.upper()}: {generate(var, enc, prompt)}")

    return std_p, var_p

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = Config()

    variants = sys.argv[1:] if len(sys.argv) > 1 else ["both"]
    if "both" in variants:
        variants = ["static_mem", "online_mem"]

    results = {}
    for v in variants:
        if v not in ATTN_CLASSES or v == "standard":
            print(f"Unknown variant: {v}")
            continue
        std_p, var_p = run_variant(v, cfg, device)
        results[v] = (min(std_p), min(var_p))

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for v, (sp, vp) in results.items():
            ratio = vp / sp
            print(f"  {v:12s}: PPL {vp:.1f} vs baseline {sp:.1f} ({ratio:.2f}x)")
