"""
test_xsa_gram.py — Synthetic benchmark comparing online_mem vs xsa_online_mem

Uses a simple synthetic language modeling task (repeated patterns + noise)
to verify that ExclusiveOnlineMemoryAttention trains correctly and to
compare its learning dynamics against the standard OnlineMemoryAttention.

No external data or tokenizer needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

from exp_mem_attn import (
    StandardAttention,
    OnlineMemoryAttention,
    ExclusiveOnlineMemoryAttention,
)

# ── Config ────────────────────────────────────────────────────────────────────

VOCAB_SIZE = 256
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
SEQ_LEN = 64
BATCH_SIZE = 32
N_EPOCHS = 15
LR = 3e-4
DEVICE = "cpu"

# ── Synthetic data: patterned sequences ──────────────────────────────────────

def make_data(n_seqs, seq_len, vocab_size):
    """
    Generate sequences with learnable patterns:
    - Repeated bigrams (token A is often followed by token B)
    - Periodic structure (every K tokens, a special marker appears)
    - Some noise to prevent memorization

    This tests whether attention can learn contextual patterns,
    which is exactly where XSA should help.
    """
    data = torch.zeros(n_seqs, seq_len + 1, dtype=torch.long)

    # Create bigram transition table (each token has a preferred successor)
    # Use a limited "active" vocab for denser patterns
    active_vocab = min(vocab_size, 64)
    bigram_table = torch.randint(0, active_vocab, (active_vocab,))

    for i in range(n_seqs):
        # Start with a random token
        data[i, 0] = torch.randint(0, active_vocab, (1,))
        for t in range(1, seq_len + 1):
            r = torch.rand(1).item()
            if r < 0.6:
                # Follow bigram pattern (60%)
                data[i, t] = bigram_table[data[i, t-1]]
            elif r < 0.8:
                # Copy from ~8 positions back (20%) — needs context
                src = max(0, t - 8)
                data[i, t] = data[i, src]
            else:
                # Random noise (20%)
                data[i, t] = torch.randint(0, active_vocab, (1,))

    x = data[:, :-1]
    y = data[:, 1:]
    return x, y


# ── Simple LM ────────────────────────────────────────────────────────────────

class SimpleLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, seq_len, attn_class):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'attn': attn_class(d_model, n_heads),
                'ln2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model), nn.GELU(),
                    nn.Linear(4 * d_model, d_model), nn.Dropout(0.1),
                ),
            }))

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.seq_len = seq_len

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = x + block['attn'](block['ln1'](x))
            x = x + block['ffn'](block['ln2'](x))
        return self.head(self.ln_f(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Training loop ────────────────────────────────────────────────────────────

def train_variant(name, attn_class, train_x, train_y, val_x, val_y):
    model = SimpleLM(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN, attn_class)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    n_params = model.count_params()
    print(f"  {name}: {n_params:,} params")

    ppls = []
    for ep in range(N_EPOCHS):
        t0 = time.time()
        model.train()
        ep_loss = 0.0
        n_batches = 0

        perm = torch.randperm(train_x.size(0))
        for i in range(0, train_x.size(0) - BATCH_SIZE + 1, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb = train_x[idx].to(DEVICE)
            yb = train_y[idx].to(DEVICE)

            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += loss.item()
            n_batches += 1

        avg_loss = ep_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = []
            val_losses = []
            for i in range(0, val_x.size(0) - BATCH_SIZE + 1, BATCH_SIZE):
                xb = val_x[i:i+BATCH_SIZE].to(DEVICE)
                yb = val_y[i:i+BATCH_SIZE].to(DEVICE)
                logits = model(xb)
                vl = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
                val_losses.append(vl.item())
            val_loss = sum(val_losses) / max(len(val_losses), 1)

        ppl = math.exp(min(val_loss, 20))
        ppls.append(ppl)
        dt = time.time() - t0
        print(f"    ep {ep+1:2d}: train={avg_loss:.3f}  val={val_loss:.3f}  ppl={ppl:.1f}  ({dt:.1f}s)")

    return ppls


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    print("Generating synthetic data...")
    train_x, train_y = make_data(2048, SEQ_LEN, VOCAB_SIZE)
    val_x, val_y = make_data(256, SEQ_LEN, VOCAB_SIZE)
    print(f"  Train: {train_x.shape}, Val: {val_x.shape}")

    variants = {
        "standard": lambda d, h: StandardAttention(d, h, dropout=0.1),
        "online_mem": lambda d, h: OnlineMemoryAttention(d, h, dropout=0.1, decay=0.99),
        "xsa_online_mem": lambda d, h: ExclusiveOnlineMemoryAttention(d, h, dropout=0.1, decay=0.99),
    }

    results = {}
    for name, attn_cls in variants.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        torch.manual_seed(42)  # same init for fair comparison
        ppls = train_variant(name, attn_cls, train_x, train_y, val_x, val_y)
        results[name] = ppls

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"  {'Variant':20s} {'Best PPL':>10s} {'Final PPL':>10s}")
    print(f"  {'-'*42}")
    for name, ppls in results.items():
        print(f"  {name:20s} {min(ppls):>10.1f} {ppls[-1]:>10.1f}")

    std_best = min(results["standard"])
    for name in ["online_mem", "xsa_online_mem"]:
        best = min(results[name])
        delta = std_best - best
        print(f"\n  {name} vs standard: {delta:+.1f} PPL ({'better' if delta > 0 else 'worse'})")
