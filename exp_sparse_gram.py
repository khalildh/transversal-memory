"""
exp_sparse_gram.py — Can Plücker Gram eigenstructure help standard attention?

Synthetic induction head experiment. The model sees:
    [random..., a, b, random..., a, b, ???]
and must predict the token that followed the FIRST (a,b) occurrence.

This requires an induction head: match current context to a past pattern,
then copy the next token. Standard attention learns this via Q·K matching.

We test whether adding a Plücker Gram bias helps attention find the right
positions faster, with the O(T) eigen_gram trick providing the signal.

Three variants:
  1. standard  — vanilla Q·K attention
  2. gram_bias — Q·K + additive Gram-incidence bias (O(T²), oracle signal quality)
  3. eigen_bias — Q·K + eigenstructure-approximated bias (O(T·k), the efficient version)

If gram_bias >> standard, the geometric signal helps routing.
If eigen_bias ≈ gram_bias, the low-rank eigenapproximation preserves the signal.

Usage:
  python exp_sparse_gram.py              # run all three
  python exp_sparse_gram.py standard     # single variant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from itertools import combinations

# ── Config ───────────────────────────────────────────────────────────────────

VOCAB = 64          # small vocab for tractability
SEQ_LEN = 48        # sequence length
D_MODEL = 64        # model dimension
N_HEADS = 4         # attention heads
N_LAYERS = 2        # transformer layers
BATCH = 128         # batch size
N_STEPS = 3000      # training steps
LR = 3e-4           # learning rate
PATTERN_LEN = 2     # length of the repeated pattern (bigram)
MIN_GAP = 4         # min distance between first and second pattern occurrence
EVAL_EVERY = 200    # evaluation interval

# ── Data generation ──────────────────────────────────────────────────────────

def make_induction_batch(batch_size, seq_len, vocab, pattern_len=2, min_gap=4):
    """Generate sequences with a repeated pattern for induction head testing.

    Each sequence: [random..., a, b, random..., a, b, target, random...]
    Target: the token that followed the FIRST (a, b) occurrence.

    Returns:
        x: (batch, seq_len) input tokens
        y: (batch, seq_len) target tokens (-100 for non-induction positions)
        induction_pos: (batch,) position where the model must predict
    """
    x = torch.randint(0, vocab, (batch_size, seq_len))
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long)  # ignore most positions

    induction_pos = torch.zeros(batch_size, dtype=torch.long)

    for i in range(batch_size):
        # Choose pattern tokens
        pattern = torch.randint(0, vocab, (pattern_len,))

        # Choose positions: first pattern at pos1, second at pos2
        max_pos1 = seq_len - pattern_len - min_gap - pattern_len - 1
        if max_pos1 < 1:
            max_pos1 = 1
        pos1 = torch.randint(1, max_pos1 + 1, (1,)).item()

        min_pos2 = pos1 + pattern_len + min_gap
        max_pos2 = seq_len - pattern_len - 1
        if min_pos2 > max_pos2:
            min_pos2 = max_pos2
        pos2 = torch.randint(min_pos2, max_pos2 + 1, (1,)).item()

        # Insert pattern at pos1: x[pos1:pos1+pattern_len] = pattern
        x[i, pos1:pos1 + pattern_len] = pattern

        # The "answer" token right after the first pattern
        answer = torch.randint(0, vocab, (1,)).item()
        x[i, pos1 + pattern_len] = answer

        # Insert same pattern at pos2
        x[i, pos2:pos2 + pattern_len] = pattern

        # Target: predict 'answer' at position pos2 + pattern_len - 1
        # (after seeing the full second pattern)
        target_pos = pos2 + pattern_len - 1
        y[i, target_pos] = answer
        induction_pos[i] = target_pos

    return x, y, induction_pos


# ── Plücker primitives ───────────────────────────────────────────────────────

_PAIRS = list(combinations(range(4), 2))

_J6 = torch.tensor([
    [0,0,0,0,0,1],[0,0,0,0,-1,0],[0,0,0,1,0,0],
    [0,0,1,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0],
], dtype=torch.float32)

def exterior(p1, p2):
    parts = [p1[...,i]*p2[...,j] - p1[...,j]*p2[...,i] for i,j in _PAIRS]
    L = torch.stack(parts, dim=-1)
    return L / L.norm(dim=-1, keepdim=True).clamp(min=1e-12)


# ── Attention variants ───────────────────────────────────────────────────────

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class GramBiasAttention(nn.Module):
    """Standard attention + incidence² bias from Plücker geometry.

    Computes the full (T,T) incidence matrix and adds it as a bias
    to the standard Q·K logits. This is O(T²) — it tests whether
    the geometric SIGNAL helps, before we optimize for efficiency.
    """
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.W1_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W1_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.bias_scale = nn.Parameter(torch.full((n_heads,), 0.1))
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer('J6', _J6)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        # Standard Q·K
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        std_logits = (q @ k.transpose(-1, -2)) * self.scale  # (B, H, T, T)

        # Plücker incidence bias
        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1]], dim=1)
        w1 = self.W1_write(x_prev).reshape(B, T, H, 4)
        w2 = self.W2_write(x).reshape(B, T, H, 4)
        write_lines = exterior(w1, w2)
        r1 = self.W1_read(x).reshape(B, T, H, 4)
        r2 = self.W2_read(x).reshape(B, T, H, 4)
        read_lines = exterior(r1, r2)

        J = self.J6
        J_write = torch.einsum('bthi,ij->bthj', write_lines, J)
        read_h = read_lines.permute(0, 2, 1, 3)
        Jwrite_h = J_write.permute(0, 2, 1, 3)
        incidence = read_h @ Jwrite_h.transpose(-1, -2)  # (B, H, T, T)

        # Use |incidence| as additive bias (preserves sign structure)
        scale = self.bias_scale.reshape(1, H, 1, 1)
        bias = incidence.abs() * scale

        # Combined logits
        logits = std_logits + bias
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        logits = logits.masked_fill(mask, float('-inf'))
        attn = F.softmax(logits, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class EigenBiasAttention(nn.Module):
    """Standard attention + Gram eigenstructure bias — the O(T·k) version.

    Instead of the full (T,T) incidence matrix, uses the Gram's top-k
    eigenvectors to build a low-rank approximation of the incidence:

      incidence[t,s] ≈ Σ_i (read_t · V_i)(write_s · V_i)
                      = R_proj @ W_proj.T      (rank-k matmul)

    where V_i are the Gram's top eigenvectors.

    The outer matmul is still O(T²·k), but the key insight is that the
    eigenvectors ADAPT to the data: they rotate to capture the dominant
    relational patterns. This is a learned, data-dependent approximation.

    More importantly, the eigenprojections enable BUCKETING: sort keys by
    their first eigenprojection, and queries attend only to nearby keys.
    This experiment tests the signal quality before implementing bucketing.
    """
    def __init__(self, d_model, n_heads, dropout=0.0, n_eigen=3):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.n_eigen = n_eigen
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.W1_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_write = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W1_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.W2_read = nn.Linear(d_model, 4 * n_heads, bias=False)
        self.bias_scale = nn.Parameter(torch.full((n_heads,), 0.1))
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer('J6', _J6)

    def _compute_lines(self, x):
        """Compute Plücker write/read lines and their Hodge duals."""
        B, T, D = x.shape
        H = self.n_heads

        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1]], dim=1)
        w1 = self.W1_write(x_prev).reshape(B, T, H, 4)
        w2 = self.W2_write(x).reshape(B, T, H, 4)
        write_lines = exterior(w1, w2)
        r1 = self.W1_read(x).reshape(B, T, H, 4)
        r2 = self.W2_read(x).reshape(B, T, H, 4)
        read_lines = exterior(r1, r2)

        J = self.J6
        Jwrite = torch.einsum('bthi,ij->bthj', write_lines, J)
        Jw = Jwrite.permute(0, 2, 1, 3)  # (B, H, T, 6)
        rd = read_lines.permute(0, 2, 1, 3)  # (B, H, T, 6)

        return Jw, rd

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        # Standard Q·K
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, kk, v = qkv[0], qkv[1], qkv[2]
        std_logits = (q @ kk.transpose(-1, -2)) * self.scale

        Jw, rd = self._compute_lines(x)

        # ── Causal Gram-mediated incidence ──
        # bias[t,s] = rd[t] · M_t · Jw[s]  where M_t = Σ_{r≤t} Jw[r]⊗Jw[r]
        #
        # Rewrite: rd[t] · M_t = rd[t] · (Σ_{r≤t} Jw[r]⊗Jw[r])
        #        = Σ_{r≤t} (rd[t]·Jw[r]) Jw[r]
        #
        # This is the same as: (rd @ Jw.T) gives pairwise dots (B,H,T,T),
        # then cumsum along the key dim and dot with Jw again.
        #
        # More efficiently: rd_M[t] = rd[t] @ M_t  via cumulative outer products
        # M_t: (B, H, T, 6, 6) via cumsum of outer products
        outer = Jw.unsqueeze(-1) * Jw.unsqueeze(-2)  # (B, H, T, 6, 6)
        M_causal = outer.cumsum(dim=2)                # (B, H, T, 6, 6)

        # rd_M[t] = rd[t] @ M_t → (B, H, T, 6)
        rd_M = torch.einsum('bhti,bhtij->bhtj', rd, M_causal)

        # bias[t,s] = rd_M[t] · Jw[s] → (B, H, T, T)
        bias_raw = torch.einsum('bhti,bhsi->bhts', rd_M, Jw)

        # Add as bias
        scale = self.bias_scale.reshape(1, H, 1, 1)
        bias = bias_raw.abs() * scale

        logits = std_logits + bias
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        logits = logits.masked_fill(mask, float('-inf'))
        attn = F.softmax(logits, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class XSAEigenBiasAttention(EigenBiasAttention):
    """Causal Gram bias + XSA exclusion: project out self-write direction from read lines.

    Before computing the Gram-mediated incidence, each read line has its
    component along the corresponding write line removed. This forces the
    geometric bias to capture ONLY cross-token relational structure — the
    self-information is already in the residual stream for the FFN.

    Same param count as eigen_bias. Only the read projection changes.
    Uses the same causal cumulative Gram as EigenBiasAttention.
    """

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        # Standard Q·K
        qkv = self.qkv(x).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, kk, v = qkv[0], qkv[1], qkv[2]
        std_logits = (q @ kk.transpose(-1, -2)) * self.scale

        Jw, rd = self._compute_lines(x)

        # ── XSA exclusion: project out self-write direction from read lines ──
        dot = (rd * Jw).sum(dim=-1, keepdim=True)       # (B, H, T, 1)
        norm_sq = (Jw * Jw).sum(dim=-1, keepdim=True).clamp(min=1e-12)
        rd_excl = rd - (dot / norm_sq) * Jw              # (B, H, T, 6)

        # ── Causal Gram-mediated incidence (with excluded read lines) ──
        outer = Jw.unsqueeze(-1) * Jw.unsqueeze(-2)
        M_causal = outer.cumsum(dim=2)

        rd_M = torch.einsum('bhti,bhtij->bhtj', rd_excl, M_causal)
        bias_raw = torch.einsum('bhti,bhsi->bhts', rd_M, Jw)

        scale = self.bias_scale.reshape(1, H, 1, 1)
        bias = bias_raw.abs() * scale

        logits = std_logits + bias
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        logits = logits.masked_fill(mask, float('-inf'))
        attn = F.softmax(logits, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


# ── Transformer ──────────────────────────────────────────────────────────────

ATTN_MAP = {
    "standard": StandardAttention,
    "gram_bias": GramBiasAttention,
    "eigen_bias": EigenBiasAttention,
    "xsa_eigen_bias": XSAEigenBiasAttention,
}

class Block(nn.Module):
    def __init__(self, d_model, n_heads, attn_type):
        super().__init__()
        self.attn = ATTN_MAP[attn_type](d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class SmallLM(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, seq_len, attn_type):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, attn_type) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
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
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


# ── Training ─────────────────────────────────────────────────────────────────

def evaluate(model, n_batches=20):
    """Evaluate induction accuracy: fraction of correct predictions at target positions."""
    model.eval()
    dev = next(model.parameters()).device
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y, ind_pos = make_induction_batch(BATCH, SEQ_LEN, VOCAB, PATTERN_LEN, MIN_GAP)
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            for i in range(BATCH):
                pos = ind_pos[i].item()
                if y[i, pos] != -100:
                    total += 1
                    if preds[i, pos] == y[i, pos]:
                        correct += 1
    return correct / max(total, 1)


def train_variant(attn_type, device):
    """Train a variant and return (accuracy_history, final_accuracy, time)."""
    model = SmallLM(VOCAB, D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN, attn_type).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  {attn_type}: {params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    accs = []
    t0 = time.time()

    for step in range(1, N_STEPS + 1):
        model.train()
        x, y, _ = make_induction_batch(BATCH, SEQ_LEN, VOCAB, PATTERN_LEN, MIN_GAP)
        x, y = x.to(device), y.to(device)
        logits = model(x).view(-1, VOCAB)
        loss = F.cross_entropy(logits, y.view(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            acc = evaluate(model)
            accs.append((step, acc))
            elapsed = time.time() - t0
            print(f"    step {step:4d}: loss={loss.item():.3f}  acc={acc:.3f}  ({elapsed:.1f}s)")

    final_acc = evaluate(model, n_batches=50)
    total_time = time.time() - t0
    return accs, final_acc, total_time


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    variants = sys.argv[1:] if len(sys.argv) > 1 else ["standard", "gram_bias", "eigen_bias", "xsa_eigen_bias"]
    results = {}

    for v in variants:
        if v not in ATTN_MAP:
            print(f"Unknown variant: {v}. Choose from: {list(ATTN_MAP.keys())}")
            continue
        print(f"\n{'='*60}")
        print(f"  VARIANT: {v}")
        print(f"  Task: induction head (vocab={VOCAB}, seq={SEQ_LEN}, pattern={PATTERN_LEN})")
        print(f"{'='*60}")

        torch.manual_seed(42)
        accs, final_acc, elapsed = train_variant(v, device)
        results[v] = (accs, final_acc, elapsed)

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  SUMMARY — Induction Head Accuracy")
        print(f"{'='*60}")
        print(f"  {'Variant':15s} {'Final Acc':>10s} {'Time':>8s} {'Params':>8s}")
        print(f"  {'-'*45}")
        for v, (accs, final_acc, elapsed) in results.items():
            model = SmallLM(VOCAB, D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN, v)
            params = sum(p.numel() for p in model.parameters())
            print(f"  {v:15s} {final_acc:10.3f} {elapsed:7.1f}s {params:8,}")

        # Learning curve comparison
        print(f"\n  Learning curves (accuracy at each checkpoint):")
        all_steps = sorted(set(s for v in results for s, _ in results[v][0]))
        header = f"  {'Step':>6s}"
        for v in results:
            header += f" {v:>12s}"
        print(header)
        for step in all_steps:
            row = f"  {step:6d}"
            for v in results:
                acc_at_step = [a for s, a in results[v][0] if s == step]
                if acc_at_step:
                    row += f" {acc_at_step[0]:12.3f}"
                else:
                    row += f" {'—':>12s}"
            print(row)

if __name__ == "__main__":
    main()
