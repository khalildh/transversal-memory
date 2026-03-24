"""
arc_tables_wrapper.py — Python ctypes wrapper for arc_tables.so

Provides drop-in replacements for the hot-path functions in exp_arc_fast_solve.py:
  - precompute_cell_embeddings_c()
  - build_score_tables_c()
  - build_hist_score_tables_c()

Usage:
    from arc_tables_wrapper import precompute_cell_embeddings_c, build_score_tables_c

Compile first:
    make -f Makefile arc_tables.so
  or:
    cc -O3 -march=native -shared -fPIC -o arc_tables.so arc_tables.c -lm
"""

import ctypes
import os
import sys
import numpy as np

# ── Load shared library ──────────────────────────────────────────────────────

_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_dir, "arc_tables.so")

if not os.path.exists(_lib_path):
    raise RuntimeError(
        f"arc_tables.so not found at {_lib_path}. "
        "Compile with: cc -O3 -march=native -shared -fPIC -o arc_tables.so arc_tables.c -lm"
    )

_lib = ctypes.CDLL(_lib_path)

# ── C function signatures ────────────────────────────────────────────────────

_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_float_p = ctypes.POINTER(ctypes.c_float)

# int precompute_cell_embeddings(int emb_type, const int *inp, const int *out_grid,
#     const int *used_colors, int nc, int H, int W, float *out_embs)
_lib.precompute_cell_embeddings.restype = ctypes.c_int
_lib.precompute_cell_embeddings.argtypes = [
    ctypes.c_int, _c_int_p, _c_int_p, _c_int_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, _c_float_p,
]

# void build_score_tables(const float *embs, int dim,
#     const int *adj_pairs, int n_adj,
#     const float *W1, const float *W2,
#     const float *JTm, int n_trans,
#     int nc, int H, int W, float *out_scores)
_lib.build_score_tables.restype = None
_lib.build_score_tables.argtypes = [
    _c_float_p, ctypes.c_int,
    _c_int_p, ctypes.c_int,
    _c_float_p, _c_float_p,
    _c_float_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    _c_float_p,
]

# void build_hist_score_tables(const float *diff, const int *inp,
#     const int *used_colors, const int *adj_pairs, int n_adj,
#     int nc, int H, int W,
#     const float *W1, const float *W2,
#     const float *JTm, int n_trans,
#     float *out_scores)
_lib.build_hist_score_tables.restype = None
_lib.build_hist_score_tables.argtypes = [
    _c_float_p, _c_int_p, _c_int_p, _c_int_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    _c_float_p, _c_float_p,
    _c_float_p, ctypes.c_int,
    _c_float_p,
]

# int get_embedding_dim(int emb_type)
_lib.get_embedding_dim.restype = ctypes.c_int
_lib.get_embedding_dim.argtypes = [ctypes.c_int]

# ── Embedding name → type ID mapping ────────────────────────────────────────

EMB_NAME_TO_ID = {
    'hist_color': 0,
    'color_only': 1,
    'pos_color': 2,
    'all': 3,
    'row_feat': 4,
    'col_feat': 5,
    'color_count': 6,
    'diagonal': 7,
}


# ── Helper: ensure contiguous C-order float32/int32 ─────────────────────────

def _f32(arr):
    """Ensure float32 C-contiguous array."""
    return np.ascontiguousarray(arr, dtype=np.float32)

def _i32(arr):
    """Ensure int32 C-contiguous array."""
    return np.ascontiguousarray(arr, dtype=np.int32)

def _ptr_f(arr):
    return arr.ctypes.data_as(_c_float_p)

def _ptr_i(arr):
    return arr.ctypes.data_as(_c_int_p)


# ── Public API ───────────────────────────────────────────────────────────────

def get_embedding_dim(emb_type):
    """Return embedding dimension for a given type ID or name."""
    if isinstance(emb_type, str):
        emb_type = EMB_NAME_TO_ID[emb_type]
    return _lib.get_embedding_dim(emb_type)


def precompute_cell_embeddings_c(emb_type, inp, used_colors, H, W, out_grid=None):
    """
    Compute embeddings for all (position, candidate_color) combos.

    Parameters:
        emb_type   : int or str — embedding type ID or name
        inp        : (H, W) int array — input grid
        used_colors: list/array of int — candidate color indices
        H, W       : grid dimensions
        out_grid   : (H, W) int array — output grid proxy (default: inp)

    Returns:
        embs: (H*W, nc, dim) float32 array
    """
    if isinstance(emb_type, str):
        emb_type = EMB_NAME_TO_ID[emb_type]

    dim = _lib.get_embedding_dim(emb_type)
    if dim < 0:
        raise ValueError(f"Invalid embedding type: {emb_type}")

    inp_flat = _i32(inp.flatten())
    if out_grid is None:
        out_flat = inp_flat
    else:
        out_flat = _i32(out_grid.flatten())
    uc = _i32(np.array(used_colors))
    nc = len(used_colors)

    out_embs = np.zeros((H * W * nc * dim,), dtype=np.float32)
    _lib.precompute_cell_embeddings(
        emb_type, _ptr_i(inp_flat), _ptr_i(out_flat), _ptr_i(uc),
        nc, H, W, _ptr_f(out_embs),
    )
    return out_embs.reshape(H * W, nc, dim)


def build_score_tables_c(embs, adj_pairs, W1, W2, JTm, nc, H, W):
    """
    Build (n_adj, nc, nc) score tables from precomputed embeddings.

    Parameters:
        embs      : (H*W, nc, dim) float32 — precomputed embeddings
        adj_pairs : list of (r1,c1,r2,c2) tuples — adjacency pairs
        W1, W2    : (4, 2*dim) float32 — projection matrices
        JTm       : (6, n_trans) float32 — J6 @ transversals.T
        nc        : int — number of candidate colors
        H, W      : int — grid dimensions

    Returns:
        scores: (n_adj, nc, nc) float32 array
    """
    dim = embs.shape[2]
    n_adj = len(adj_pairs)
    n_trans = JTm.shape[1]

    embs_c = _f32(embs.reshape(-1))
    adj_c = _i32(np.array(adj_pairs).reshape(-1))
    W1_c = _f32(W1)
    W2_c = _f32(W2)
    JTm_c = _f32(JTm)
    out = np.zeros(n_adj * nc * nc, dtype=np.float32)

    _lib.build_score_tables(
        _ptr_f(embs_c), dim,
        _ptr_i(adj_c), n_adj,
        _ptr_f(W1_c), _ptr_f(W2_c),
        _ptr_f(JTm_c), n_trans,
        nc, H, W,
        _ptr_f(out),
    )
    return out.reshape(n_adj, nc, nc)


def build_hist_score_tables_c(diff, inp, used_colors, adj_pairs,
                               nc, H, W, W1, W2, JTm, n_trans,
                               out_scores=None):
    """
    Accumulate hist_color scores for a single histogram diff into out_scores.

    Parameters:
        diff        : (10,) float32 — histogram difference vector
        inp         : (H, W) int — input grid
        used_colors : list/array of int — candidate colors
        adj_pairs   : list of (r1,c1,r2,c2) — adjacency pairs
        nc          : int
        H, W        : int
        W1, W2      : (4, 60) float32 — projection matrices
        JTm         : (6, n_trans) float32
        n_trans     : int
        out_scores  : (n_adj, nc, nc) float32 — ACCUMULATED output (optional)

    Returns:
        out_scores: (n_adj, nc, nc) float32
    """
    n_adj = len(adj_pairs)
    if out_scores is None:
        out_scores = np.zeros((n_adj, nc, nc), dtype=np.float32)

    diff_c = _f32(diff)
    inp_flat = _i32(inp.flatten())
    uc = _i32(np.array(used_colors))
    adj_c = _i32(np.array(adj_pairs).reshape(-1))
    W1_c = _f32(W1)
    W2_c = _f32(W2)
    JTm_c = _f32(JTm)
    out_flat = _f32(out_scores.reshape(-1))

    _lib.build_hist_score_tables(
        _ptr_f(diff_c), _ptr_i(inp_flat), _ptr_i(uc),
        _ptr_i(adj_c), n_adj,
        nc, H, W,
        _ptr_f(W1_c), _ptr_f(W2_c),
        _ptr_f(JTm_c), n_trans,
        _ptr_f(out_flat),
    )
    return out_flat.reshape(n_adj, nc, nc)
