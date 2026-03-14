"""
exp_lm_variants.py — Three Plücker attention fixes, tested on WikiText-2

Usage:
  python exp_lm_variants.py kernel   # V2: asymmetric Q/K lines, signed inner product
  python exp_lm_variants.py bigram   # V3: query-lines from token pairs
  python exp_lm_variants.py hybrid   # V4: standard attention + Plücker bias
  python exp_lm_variants.py all      # run all three sequentially
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
    data_dir = Path("data/wikitext")

# ── Plücker primitives ──────────────────────────────────────────────────────

_PAIRS = list(combinations(range(4), 2))
_PI = [i for i,j in _PAIRS]
_PJ = [j for i,j in _PAIRS]

_J6 = torch.tensor([
    [0,0,0,0,0,1],[0,0,0,0,-1,0],[0,0,0,1,0,0],
    [0,0,1,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0],
], dtype=torch.float32)

def exterior(p1, p2):
    """(..., 4) × (..., 4) → (..., 6) normalized Plücker lines."""
    parts = [p1[...,i]*p2[...,j] - p1[...,j]*p2[...,i] for i,j in _PAIRS]
    L = torch.stack(parts, dim=-1)
    return L / L.norm(dim=-1, keepdim=True).clamp(min=1e-12)

# ── Standard attention (shared baseline) ─────────────────────────────────────

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
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

# ── V2: Plücker Kernel Attention ─────────────────────────────────────────────

class PluckerKernelAttention(nn.Module):
    """
    Asymmetric Q/K lines. Each token produces a query-line and key-line
    from separate projections. Attention logit = signed plucker_inner / sqrt(6).
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 6.0 ** -0.5

        # Query projections: token → 2 points in R⁴ → query line
        self.W1_q = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_q = nn.Linear(d_model, 4 * n_heads, bias=False)
        # Key projections: token → 2 points in R⁴ → key line
        self.W1_k = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_k = nn.Linear(d_model, 4 * n_heads, bias=False)

        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('J6', _J6)

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads

        # Query lines: (B, T, H, 6)
        q1 = self.W1_q(x).reshape(B, T, H, 4)
        q2 = self.W2_q(x).reshape(B, T, H, 4)
        L_q = exterior(q1, q2)  # (B, T, H, 6)

        # Key lines: (B, T, H, 6)
        k1 = self.W1_k(x).reshape(B, T, H, 4)
        k2 = self.W2_k(x).reshape(B, T, H, 4)
        L_k = exterior(k1, k2)  # (B, T, H, 6)

        # Attention via signed Plücker inner product
        L_q_h = L_q.permute(0, 2, 1, 3)  # (B, H, T, 6)
        JL_k = (L_k @ self.J6.T).permute(0, 2, 1, 3)  # (B, H, T, 6)
        attn = (L_q_h @ JL_k.transpose(-1, -2)) * self.scale  # (B, H, T, T)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = self.drop(F.softmax(attn, dim=-1))

        v = self.v_proj(x).reshape(B, T, H, self.d_head).permute(0, 2, 1, 3)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)

# ── V3: Plücker Bigram Attention ─────────────────────────────────────────────

class PluckerBigramAttention(nn.Module):
    """
    Query-lines from consecutive token pairs (bigrams).
    Key-lines from single tokens. Captures local context in geometry.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 6.0 ** -0.5

        # Query: from bigram [x_i; x_{i-1}] → line
        self.W1_q = nn.Linear(2 * d_model, 4 * n_heads, bias=False)
        self.W2_q = nn.Linear(2 * d_model, 4 * n_heads, bias=False)
        # Key: from single token → line
        self.W1_k = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_k = nn.Linear(d_model, 4 * n_heads, bias=False)

        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('J6', _J6)

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads

        # Build bigram input: [x_i; x_{i-1}], zero-padded at position 0
        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1]], dim=1)
        x_bigram = torch.cat([x, x_prev], dim=-1)  # (B, T, 2D)

        # Query lines from bigrams
        q1 = self.W1_q(x_bigram).reshape(B, T, H, 4)
        q2 = self.W2_q(x_bigram).reshape(B, T, H, 4)
        L_q = exterior(q1, q2)

        # Key lines from single tokens
        k1 = self.W1_k(x).reshape(B, T, H, 4)
        k2 = self.W2_k(x).reshape(B, T, H, 4)
        L_k = exterior(k1, k2)

        # Signed Plücker inner product
        L_q_h = L_q.permute(0, 2, 1, 3)
        JL_k = (L_k @ self.J6.T).permute(0, 2, 1, 3)
        attn = (L_q_h @ JL_k.transpose(-1, -2)) * self.scale

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = self.drop(F.softmax(attn, dim=-1))

        v = self.v_proj(x).reshape(B, T, H, self.d_head).permute(0, 2, 1, 3)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)

# ── V4: Hybrid Standard + Plücker Attention ──────────────────────────────────

class HybridAttention(nn.Module):
    """
    Standard Q·K attention + additive Plücker incidence bias.
    Tests whether geometric information is complementary.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Standard Q/K/V
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # Plücker bias: token → line, pairwise incidence added to logits
        self.W1_p = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_p = nn.Linear(d_model, 4 * n_heads, bias=False)
        # Learnable scale per head (initialized small)
        self.plucker_scale = nn.Parameter(torch.full((n_heads,), 0.1))

        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('J6', _J6)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        # Standard Q·K
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        std_attn = (q @ k.transpose(-1, -2)) * self.scale  # (B, H, T, T)

        # Plücker bias
        p1 = self.W1_p(x).reshape(B, T, H, 4)
        p2 = self.W2_p(x).reshape(B, T, H, 4)
        lines = exterior(p1, p2)  # (B, T, H, 6)
        lines_h = lines.permute(0, 2, 1, 3)  # (B, H, T, 6)
        Jlines = lines_h @ self.J6.T
        plucker_bias = lines_h @ Jlines.transpose(-1, -2)  # (B, H, T, T)

        # Scale per head and add
        scale = self.plucker_scale.reshape(1, H, 1, 1)
        attn = std_attn + plucker_bias * scale

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = self.drop(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)

# ── Transformer + LM ────────────────────────────────────────────────────────

ATTN_CLASSES = {
    "standard": StandardAttention,
    "kernel": PluckerKernelAttention,
    "bigram": PluckerBigramAttention,
    "hybrid": HybridAttention,
}

class Block(nn.Module):
    def __init__(self, d_model, n_heads, attn_type, dropout=0.1):
        super().__init__()
        self.attn = ATTN_CLASSES[attn_type](d_model, n_heads, dropout)
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
    def __init__(self, vocab_size, cfg, attn_type):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.Sequential(*[
            Block(cfg.d_model, cfg.n_heads, attn_type, cfg.dropout)
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

def train_one(attn_type, cfg, device):
    enc = tiktoken.get_encoding("gpt2")
    train_data = load_tokens("train", cfg.data_dir, enc)
    val_data = load_tokens("val", cfg.data_dir, enc)

    model = LM(enc.n_vocab, cfg, attn_type).to(device)
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

    print("\n  Training standard baseline...")
    std, std_l, std_p = train_one("standard", cfg, device)

    print(f"\n  Training {variant}...")
    var, var_l, var_p = train_one(variant, cfg, device)

    enc = tiktoken.get_encoding("gpt2")

    print(f"\n  {'':20s} {'Standard':>10s} {variant:>10s}")
    print(f"  {'-'*42}")
    print(f"  {'Params':20s} {std.count_params():>10,d} {var.count_params():>10,d}")
    print(f"  {'Best val PPL':20s} {min(std_p):>10.1f} {min(var_p):>10.1f}")
    print(f"  {'Final val PPL':20s} {std_p[-1]:>10.1f} {var_p[-1]:>10.1f}")

    print("\n  SAMPLES:")
    for prompt in PROMPTS:
        print(f"\n  PROMPT: \"{prompt}\"")
        print(f"    STD: {generate(std, enc, prompt)}")
        print(f"    {variant.upper()}: {generate(var, enc, prompt)}")

    return std_p, var_p

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = Config()

    variants = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    if "all" in variants:
        variants = ["kernel", "bigram", "hybrid"]

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
