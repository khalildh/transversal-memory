/*
 * arc_tables.c — C implementation of ARC Plücker solver table-building pipeline.
 *
 * Implements:
 *   1. precompute_cell_embeddings() for all 8 embedding types
 *   2. build_score_tables() — vectorized Plücker line + scoring
 *   3. build_hist_tables() — per-histogram variant of the above
 *
 * Compile:
 *   cc -O3 -march=native -shared -fPIC -o arc_tables.so arc_tables.c -lm
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Constants ─────────────────────────────────────────────────────────────── */

#define N_COLORS 10
#define PLUCKER_DIM 6

/* Plücker index pairs: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) */
static const int PL_I[6] = {0, 1, 0, 1, 0, 2};
static const int PL_J[6] = {1, 2, 3, 2, 3, 3};

/* Wait — let me get the pairs right from the Python code:
   pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
   So PL_I = {0,0,0,1,1,2}, PL_J = {1,2,3,2,3,3} */

/* Embedding type IDs */
#define EMB_HIST_COLOR  0
#define EMB_COLOR_ONLY  1
#define EMB_POS_COLOR   2
#define EMB_ALL         3
#define EMB_ROW_FEAT    4
#define EMB_COL_FEAT    5
#define EMB_COLOR_COUNT 6
#define EMB_DIAGONAL    7

/* Embedding dimensions */
static const int EMB_DIMS[8] = {30, 20, 22, 42, 44, 42, 24, 26};

/* ── Helpers ───────────────────────────────────────────────────────────────── */

static void one_hot(float *dst, int idx) {
    memset(dst, 0, N_COLORS * sizeof(float));
    if (idx >= 0 && idx < N_COLORS)
        dst[idx] = 1.0f;
}

/* Count occurrences of value v in array of length n */
static int count_val(const int *arr, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; i++)
        if (arr[i] == v) c++;
    return c;
}

/* Count unique values in array */
static int count_unique(const int *arr, int n) {
    int seen[N_COLORS] = {0};
    int u = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] >= 0 && arr[i] < N_COLORS && !seen[arr[i]]) {
            seen[arr[i]] = 1;
            u++;
        }
    }
    return u;
}

/* ── Embedding functions ───────────────────────────────────────────────────── */

/*
 * All embedding functions write into `out` and return the dimension.
 * Parameters:
 *   r, c       — row, col position in grid
 *   in_c       — input color at (r,c)
 *   out_c      — candidate output color
 *   inp        — input grid (H*W flat, row-major)
 *   out_grid   — output grid (same layout; for test, == inp as proxy)
 *   H, W       — grid dimensions
 */

static int emb_hist_color(float *out, int r, int c, int in_c, int out_c,
                           const int *inp, const int *out_grid, int H, int W) {
    int sz = H * W;
    float inv_sz = 1.0f / (sz > 0 ? sz : 1);
    one_hot(out, in_c);
    one_hot(out + 10, out_c);
    for (int i = 0; i < N_COLORS; i++) {
        int cnt_out = count_val(out_grid, sz, i);
        int cnt_inp = count_val(inp, sz, i);
        out[20 + i] = (float)(cnt_out - cnt_inp) * inv_sz;
    }
    return 30;
}

static int emb_color_only(float *out, int r, int c, int in_c, int out_c,
                            const int *inp, const int *out_grid, int H, int W) {
    one_hot(out, in_c);
    one_hot(out + 10, out_c);
    return 20;
}

static int emb_pos_color(float *out, int r, int c, int in_c, int out_c,
                           const int *inp, const int *out_grid, int H, int W) {
    out[0] = (H > 1) ? (float)r / (H - 1) : 0.0f;
    out[1] = (W > 1) ? (float)c / (W - 1) : 0.0f;
    one_hot(out + 2, in_c);
    one_hot(out + 12, out_c);
    return 22;
}

static int emb_all(float *out, int r, int c, int in_c, int out_c,
                    const int *inp, const int *out_grid, int H, int W) {
    int sz = H * W;
    float inv_sz = 1.0f / (sz > 0 ? sz : 1);
    out[0] = (H > 1) ? (float)r / (H - 1) : 0.0f;
    out[1] = (W > 1) ? (float)c / (W - 1) : 0.0f;
    one_hot(out + 2, in_c);
    one_hot(out + 12, out_c);
    for (int i = 0; i < N_COLORS; i++)
        out[22 + i] = (float)count_val(inp, sz, i) * inv_sz;
    for (int i = 0; i < N_COLORS; i++)
        out[32 + i] = (float)count_val(out_grid, sz, i) * inv_sz;
    return 42;
}

static int emb_row_features(float *out, int r, int c, int in_c, int out_c,
                              const int *inp, const int *out_grid, int H, int W) {
    one_hot(out, in_c);
    one_hot(out + 10, out_c);
    /* in_rh: row histogram of input row r */
    const int *in_row = inp + r * W;
    const int *out_row = out_grid + r * W;
    float inv_w = (W > 0) ? 1.0f / W : 0.0f;
    for (int i = 0; i < N_COLORS; i++)
        out[20 + i] = (float)count_val(in_row, W, i) * inv_w;
    int in_row_unique = count_unique(in_row, W);
    out[30] = (in_row_unique == 1) ? 1.0f : 0.0f;
    out[31] = (W > 0) ? (float)in_row_unique / W : 0.0f;
    for (int i = 0; i < N_COLORS; i++)
        out[32 + i] = (float)count_val(out_row, W, i) * inv_w;
    int out_row_unique = count_unique(out_row, W);
    out[42] = (out_row_unique == 1) ? 1.0f : 0.0f;
    out[43] = (W > 0) ? (float)out_row_unique / W : 0.0f;
    return 44;
}

static int emb_col_features(float *out, int r, int c, int in_c, int out_c,
                              const int *inp, const int *out_grid, int H, int W) {
    one_hot(out, in_c);
    one_hot(out + 10, out_c);
    /* Column data: extract column c from flat grid */
    float inv_h = (H > 0) ? 1.0f / H : 0.0f;
    int col_buf[64]; /* max grid height */
    for (int i = 0; i < H; i++) col_buf[i] = inp[i * W + c];
    for (int i = 0; i < N_COLORS; i++)
        out[20 + i] = (float)count_val(col_buf, H, i) * inv_h;
    out[30] = (count_unique(col_buf, H) == 1) ? 1.0f : 0.0f;
    for (int i = 0; i < H; i++) col_buf[i] = out_grid[i * W + c];
    for (int i = 0; i < N_COLORS; i++)
        out[31 + i] = (float)count_val(col_buf, H, i) * inv_h;
    out[41] = (count_unique(col_buf, H) == 1) ? 1.0f : 0.0f;
    return 42;
}

static int emb_color_count(float *out, int r, int c, int in_c, int out_c,
                             const int *inp, const int *out_grid, int H, int W) {
    int sz = H * W;
    float inv_sz = 1.0f / (sz > 0 ? sz : 1);
    one_hot(out, in_c);
    one_hot(out + 10, out_c);
    out[20] = (float)count_val(inp, sz, in_c) * inv_sz;
    out[21] = (float)count_val(out_grid, sz, out_c) * inv_sz;
    /* Mode: find most frequent color */
    int in_mode = 0, out_mode = 0;
    int in_max = 0, out_max = 0;
    for (int i = 0; i < N_COLORS; i++) {
        int ci = count_val(inp, sz, i);
        int co = count_val(out_grid, sz, i);
        if (ci > in_max) { in_max = ci; in_mode = i; }
        if (co > out_max) { out_max = co; out_mode = i; }
    }
    out[22] = (in_c == in_mode) ? 1.0f : 0.0f;
    out[23] = (out_c == out_mode) ? 1.0f : 0.0f;
    return 24;
}

static int emb_diagonal(float *out, int r, int c, int in_c, int out_c,
                          const int *inp, const int *out_grid, int H, int W) {
    one_hot(out, in_c);
    one_hot(out + 10, out_c);
    out[20] = (H > 1) ? (float)r / (H - 1) : 0.0f;
    out[21] = (W > 1) ? (float)c / (W - 1) : 0.0f;
    int maxHW = (H > W) ? H : W;
    int minHW = (H < W) ? H : W;
    out[22] = (float)(r - c + maxHW) / (2.0f * maxHW);
    float denom = (float)(H + W - 2) + 1e-6f;
    out[23] = (float)(r + c) / denom;
    out[24] = (r == c) ? 1.0f : 0.0f;
    out[25] = (r + c == minHW - 1) ? 1.0f : 0.0f;
    return 26;
}

/* Dispatch embedding by type ID */
typedef int (*emb_func_t)(float*, int, int, int, int, const int*, const int*, int, int);

static emb_func_t EMB_FUNCS[8] = {
    emb_hist_color,
    emb_color_only,
    emb_pos_color,
    emb_all,
    emb_row_features,
    emb_col_features,
    emb_color_count,
    emb_diagonal,
};

/* ── Public API ────────────────────────────────────────────────────────────── */

/*
 * precompute_cell_embeddings
 *
 * Inputs:
 *   emb_type    — embedding type ID (0-7)
 *   inp         — input grid, flat int array (H*W), row-major
 *   out_grid    — output grid proxy (usually == inp for test)
 *   used_colors — array of nc used color indices
 *   nc          — number of used colors
 *   H, W        — grid dimensions
 *   out_embs    — preallocated output array (H*W * nc * dim), row-major
 *                 layout: out_embs[(pos * nc + ci) * dim + d]
 *
 * Returns: embedding dimension (or -1 on invalid type)
 */
int precompute_cell_embeddings(int emb_type,
                                const int *inp,
                                const int *out_grid,
                                const int *used_colors,
                                int nc, int H, int W,
                                float *out_embs) {
    if (emb_type < 0 || emb_type > 7) return -1;
    emb_func_t fn = EMB_FUNCS[emb_type];
    int dim = EMB_DIMS[emb_type];

    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            int pos = r * W + c;
            int in_c = inp[pos];
            for (int ci = 0; ci < nc; ci++) {
                float *dst = out_embs + (pos * nc + ci) * dim;
                fn(dst, r, c, in_c, used_colors[ci], inp, out_grid, H, W);
            }
        }
    }
    return dim;
}

/*
 * build_score_tables
 *
 * Given precomputed embeddings, adjacency pairs, projection matrices W1/W2,
 * and transversal-derived JTm matrix, compute score_table[adj][ca][cb].
 *
 * Inputs:
 *   embs        — (H*W, nc, dim) embedding array, row-major
 *   dim         — embedding dimension
 *   adj_pairs   — (n_adj, 4) array of (r1,c1,r2,c2) adjacency pairs, flat
 *   n_adj       — number of adjacency pairs
 *   W1          — (4, 2*dim) projection matrix, row-major
 *   W2          — (4, 2*dim) projection matrix, row-major
 *   JTm         — (6, n_trans) = J6 @ transversals.T, row-major
 *   n_trans     — number of transversals
 *   nc          — number of candidate colors
 *   H, W        — grid dimensions
 *   out_scores  — preallocated (n_adj * nc * nc) output array
 */
void build_score_tables(const float *embs, int dim,
                         const int *adj_pairs, int n_adj,
                         const float *W1, const float *W2,
                         const float *JTm, int n_trans,
                         int nc, int H, int W_grid,
                         float *out_scores) {
    int combined_dim = 2 * dim;
    /* Plücker pairs: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) */
    static const int pi[6] = {0, 0, 0, 1, 1, 2};
    static const int pj[6] = {1, 2, 3, 2, 3, 3};

    /* Scratch buffers (per-call, could be stack allocated for small dims) */
    float combined[256]; /* max 2*44 = 88, plenty of room */
    float p1[4], p2[4];
    float L[6];
    float inner_buf[4096]; /* up to 4096 transversals */

    for (int ai = 0; ai < n_adj; ai++) {
        int r1 = adj_pairs[ai * 4 + 0];
        int c1 = adj_pairs[ai * 4 + 1];
        int r2 = adj_pairs[ai * 4 + 2];
        int c2 = adj_pairs[ai * 4 + 3];
        int idx_a = r1 * W_grid + c1;
        int idx_b = r2 * W_grid + c2;

        for (int ca = 0; ca < nc; ca++) {
            const float *ea = embs + (idx_a * nc + ca) * dim;

            for (int cb = 0; cb < nc; cb++) {
                const float *eb = embs + (idx_b * nc + cb) * dim;

                /* Concatenate [ea; eb] */
                memcpy(combined, ea, dim * sizeof(float));
                memcpy(combined + dim, eb, dim * sizeof(float));

                /* p1 = W1 @ combined, p2 = W2 @ combined */
                for (int i = 0; i < 4; i++) {
                    float s1 = 0.0f, s2 = 0.0f;
                    const float *w1_row = W1 + i * combined_dim;
                    const float *w2_row = W2 + i * combined_dim;
                    for (int j = 0; j < combined_dim; j++) {
                        s1 += w1_row[j] * combined[j];
                        s2 += w2_row[j] * combined[j];
                    }
                    p1[i] = s1;
                    p2[i] = s2;
                }

                /* Exterior product → Plücker 6-vector */
                for (int k = 0; k < 6; k++)
                    L[k] = p1[pi[k]] * p2[pj[k]] - p1[pj[k]] * p2[pi[k]];

                /* Normalize */
                float norm2 = 0.0f;
                for (int k = 0; k < 6; k++) norm2 += L[k] * L[k];
                float norm = sqrtf(norm2);

                float score;
                if (norm <= 1e-10f) {
                    score = 0.0f;
                } else {
                    float inv_norm = 1.0f / norm;
                    for (int k = 0; k < 6; k++) L[k] *= inv_norm;

                    /* inner = L @ JTm → (n_trans,) */
                    /* score = sum(log(|inner| + 1e-10)) */
                    score = 0.0f;
                    for (int t = 0; t < n_trans; t++) {
                        float dot = 0.0f;
                        for (int k = 0; k < 6; k++)
                            dot += L[k] * JTm[k * n_trans + t];
                        /* Clamp */
                        if (dot > 1e10f) dot = 1e10f;
                        if (dot < -1e10f) dot = -1e10f;
                        float v = logf(fabsf(dot) + 1e-10f);
                        /* nan/inf guard */
                        if (!isfinite(v)) v = (v != v) ? 0.0f : -100.0f;
                        score += v;
                    }
                }

                out_scores[ai * nc * nc + ca * nc + cb] = score;
            }
        }
    }
}


/*
 * build_hist_score_tables
 *
 * For a single histogram diff vector, build (n_adj, nc, nc) score tables
 * using the hist_color embedding structure.
 *
 * The hist_color embedding is [in_oh(10), out_oh(10), diff(10)] = 30 dims.
 * Here `diff` is fixed for the entire histogram, and in_oh depends on position,
 * out_oh depends on candidate color.
 *
 * Inputs:
 *   diff         — (10,) histogram difference vector
 *   inp          — flat input grid (H*W)
 *   used_colors  — (nc,) color array
 *   adj_pairs    — (n_adj, 4) flat
 *   n_adj, nc, H, W — dimensions
 *   W1, W2       — (4, 60) projection matrices (2*30=60)
 *   JTm          — (6, n_trans) matrix
 *   n_trans      — number of transversals
 *   out_scores   — (n_adj * nc * nc) output, ACCUMULATED (added to existing)
 */
void build_hist_score_tables(const float *diff,
                              const int *inp,
                              const int *used_colors,
                              const int *adj_pairs, int n_adj,
                              int nc, int H, int W_grid,
                              const float *W1, const float *W2,
                              const float *JTm, int n_trans,
                              float *out_scores) {
    static const int pi[6] = {0, 0, 0, 1, 1, 2};
    static const int pj[6] = {1, 2, 3, 2, 3, 3};
    const int dim = 30;
    const int combined_dim = 60;

    /* Precompute all possible embeddings: for each position × candidate color */
    /* emb[pos][ci] = [in_oh(10), out_oh(10), diff(10)] */
    int hw = H * W_grid;

    /* in_oh per position: depends on inp[pos] */
    /* out_oh per candidate: depends on used_colors[ci] */
    /* diff: constant */

    /* We can precompute the embeddings or compute inline. For small grids, inline is fine. */
    float ea[30], eb[30], combined[60];
    float p1[4], p2[4], L[6];

    for (int ai = 0; ai < n_adj; ai++) {
        int r1 = adj_pairs[ai * 4 + 0];
        int c1 = adj_pairs[ai * 4 + 1];
        int r2 = adj_pairs[ai * 4 + 2];
        int c2 = adj_pairs[ai * 4 + 3];
        int in_c_a = inp[r1 * W_grid + c1];
        int in_c_b = inp[r2 * W_grid + c2];

        for (int ca = 0; ca < nc; ca++) {
            /* Build ea = [in_oh_a, out_oh(ca), diff] */
            memset(ea, 0, 30 * sizeof(float));
            if (in_c_a >= 0 && in_c_a < N_COLORS) ea[in_c_a] = 1.0f;
            if (used_colors[ca] >= 0 && used_colors[ca] < N_COLORS)
                ea[10 + used_colors[ca]] = 1.0f;
            memcpy(ea + 20, diff, 10 * sizeof(float));

            for (int cb = 0; cb < nc; cb++) {
                /* Build eb = [in_oh_b, out_oh(cb), diff] */
                memset(eb, 0, 30 * sizeof(float));
                if (in_c_b >= 0 && in_c_b < N_COLORS) eb[in_c_b] = 1.0f;
                if (used_colors[cb] >= 0 && used_colors[cb] < N_COLORS)
                    eb[10 + used_colors[cb]] = 1.0f;
                memcpy(eb + 20, diff, 10 * sizeof(float));

                /* Concatenate */
                memcpy(combined, ea, 30 * sizeof(float));
                memcpy(combined + 30, eb, 30 * sizeof(float));

                /* p1 = W1 @ combined, p2 = W2 @ combined */
                for (int i = 0; i < 4; i++) {
                    float s1 = 0.0f, s2 = 0.0f;
                    const float *w1_row = W1 + i * combined_dim;
                    const float *w2_row = W2 + i * combined_dim;
                    for (int j = 0; j < combined_dim; j++) {
                        s1 += w1_row[j] * combined[j];
                        s2 += w2_row[j] * combined[j];
                    }
                    p1[i] = s1;
                    p2[i] = s2;
                }

                /* Exterior product */
                for (int k = 0; k < 6; k++)
                    L[k] = p1[pi[k]] * p2[pj[k]] - p1[pj[k]] * p2[pi[k]];

                float norm2 = 0.0f;
                for (int k = 0; k < 6; k++) norm2 += L[k] * L[k];
                float norm = sqrtf(norm2);

                float score;
                if (norm <= 1e-10f) {
                    score = 0.0f;
                } else {
                    float inv_norm = 1.0f / norm;
                    for (int k = 0; k < 6; k++) L[k] *= inv_norm;

                    score = 0.0f;
                    for (int t = 0; t < n_trans; t++) {
                        float dot = 0.0f;
                        for (int k = 0; k < 6; k++)
                            dot += L[k] * JTm[k * n_trans + t];
                        if (dot > 1e10f) dot = 1e10f;
                        if (dot < -1e10f) dot = -1e10f;
                        float v = logf(fabsf(dot) + 1e-10f);
                        if (!isfinite(v)) v = (v != v) ? 0.0f : -100.0f;
                        score += v;
                    }
                }

                out_scores[ai * nc * nc + ca * nc + cb] += score;
            }
        }
    }
}


/*
 * get_embedding_dim — return the dimension for a given embedding type ID.
 */
int get_embedding_dim(int emb_type) {
    if (emb_type < 0 || emb_type > 7) return -1;
    return EMB_DIMS[emb_type];
}
