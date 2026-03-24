"""
test_c_vs_python.py — Compare transversal computation between C and Python.

Gives the same 4 Plucker lines to both implementations and compares:
- Constraint matrix A (hodge duals)
- SVD singular values, null space vectors v1, v2
- Quadratic coefficients alpha, beta, gamma
- Discriminant
- Resulting transversals

Usage: uv run python test_c_vs_python.py
"""

import numpy as np
import subprocess
import json
import sys
import os
import tempfile

# ── Python computation ──────────────────────────────────────────────────────

from transversal_memory.plucker import (
    hodge_dual, plucker_relation,
    _PR_AB, _PR_CD, _PR_AC, _PR_BD, _PR_AD, _PR_BC,
)
from transversal_memory.solver import solve_p3

# Test lines — construct from random points so they satisfy Plucker relation
# and are in general position (guaranteeing 2 transversals)
from transversal_memory.plucker import line_from_points, random_line
rng = np.random.default_rng(42)
lines = np.array([random_line(rng) for _ in range(4)])

print("Test lines (from random points, satisfy Plucker relation):")
for i, L in enumerate(lines):
    print(f"  L{i}: [{', '.join(f'{v:+.12f}' for v in L)}]")
    print(f"       PR = {plucker_relation(L):.2e}")

print("=" * 70)
print("PYTHON IMPLEMENTATION")
print("=" * 70)

# 1. Constraint matrix
A = np.stack([hodge_dual(p) for p in lines])
print(f"\nConstraint matrix A (4x6):")
for i, row in enumerate(A):
    print(f"  row {i}: [{', '.join(f'{v:+.10f}' for v in row)}]")

# 2. SVD
_, S, Vt = np.linalg.svd(A, full_matrices=True)
print(f"\nSingular values: [{', '.join(f'{s:.10f}' for s in S)}]")

v1 = Vt[-1].copy()
v2 = Vt[-2].copy()
print(f"\nv1 (last row of Vt):  [{', '.join(f'{v:+.10f}' for v in v1)}]")
print(f"v2 (2nd last of Vt):  [{', '.join(f'{v:+.10f}' for v in v2)}]")

# 3. Quadratic coefficients
alpha = plucker_relation(v1)
gamma = plucker_relation(v2)
beta = ((v1[_PR_AB]*v2[_PR_CD] + v2[_PR_AB]*v1[_PR_CD])
      - (v1[_PR_AC]*v2[_PR_BD] + v2[_PR_AC]*v1[_PR_BD])
      + (v1[_PR_AD]*v2[_PR_BC] + v2[_PR_AD]*v1[_PR_BC]))

disc = beta**2 - 4*alpha*gamma

print(f"\nalpha = {alpha:+.15e}")
print(f"beta  = {beta:+.15e}")
print(f"gamma = {gamma:+.15e}")
print(f"disc  = {disc:+.15e}")

# 4. Solve
results = solve_p3(v1, v2)
print(f"\nTransversals found: {len(results)}")
for i, (T, resid) in enumerate(results):
    print(f"  T{i}: [{', '.join(f'{v:+.10f}' for v in T)}]  resid={resid:.2e}")

# Verify transversals meet all 4 lines
from transversal_memory.plucker import plucker_inner
for i, (T, resid) in enumerate(results):
    ips = [plucker_inner(T, L) for L in lines]
    print(f"  T{i} inner products with lines: [{', '.join(f'{v:+.2e}' for v in ips)}]")

# ── C computation ───────────────────────────────────────────────────────────

# Write a minimal C program that does the same computation and prints results
c_test_code = r"""
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *A,
                    int *lda, double *S, double *U, int *ldu, double *Vt,
                    int *ldvt, double *work, int *lwork, int *info);
typedef int __CLPK_integer;
#endif

/* Plucker relation indices */
#define PR_AB 0
#define PR_CD 5
#define PR_AC 1
#define PR_BD 4
#define PR_AD 2
#define PR_BC 3

static double vec_norm(const double *v, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += v[i] * v[i];
    return sqrt(s);
}

static double plucker_relation(const double *p) {
    return p[PR_AB] * p[PR_CD] - p[PR_AC] * p[PR_BD] + p[PR_AD] * p[PR_BC];
}

static void hodge_dual(const double *p, double *out) {
    out[0] =  p[5];
    out[1] = -p[4];
    out[2] =  p[3];
    out[3] =  p[2];
    out[4] = -p[1];
    out[5] =  p[0];
}

static double plucker_inner_fn(const double *p, const double *q) {
    return p[0]*q[5] - p[1]*q[4] + p[2]*q[3]
         + p[3]*q[2] - p[4]*q[1] + p[5]*q[0];
}

static void svd_right(const double *A, int m, int n, double *Vt, double *S) {
    double *At = (double *)malloc(m * n * sizeof(double));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            At[j * m + i] = A[i * n + j];

    double *Vt_lapack = (double *)malloc(n * n * sizeof(double));
    __CLPK_integer m_ = m, n_ = n, lda_ = m, ldu_ = 1, ldvt_ = n, info;

    __CLPK_integer lwork = -1;
    double work_query;
    char jobu = 'N', jobvt = 'A';
    dgesvd_(&jobu, &jobvt, &m_, &n_, At, &lda_, S, NULL, &ldu_,
            Vt_lapack, &ldvt_, &work_query, &lwork, &info);
    lwork = (__CLPK_integer)work_query;
    double *work = (double *)malloc(lwork * sizeof(double));

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            At[j * m + i] = A[i * n + j];

    dgesvd_(&jobu, &jobvt, &m_, &n_, At, &lda_, S, NULL, &ldu_,
            Vt_lapack, &ldvt_, work, &lwork, &info);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Vt[i * n + j] = Vt_lapack[j * n + i];

    free(At); free(Vt_lapack); free(work);
}

static int solve_p3(const double *v1, const double *v2,
                    double sol[2][6], double res[2]) {
    double tol = 1e-10;
    int nsol = 0;

    double alpha = plucker_relation(v1);
    double gamma = plucker_relation(v2);
    double beta = (v1[PR_AB]*v2[PR_CD] + v2[PR_AB]*v1[PR_CD])
                - (v1[PR_AC]*v2[PR_BD] + v2[PR_AC]*v1[PR_BD])
                + (v1[PR_AD]*v2[PR_BC] + v2[PR_AD]*v1[PR_BC]);

    printf("alpha = %+.15e\n", alpha);
    printf("beta  = %+.15e\n", beta);
    printf("gamma = %+.15e\n", gamma);
    printf("disc  = %+.15e\n", beta*beta - 4.0*alpha*gamma);

    #define ADD_SOL(is_v1_only, t_val) do { \
        double T[6]; \
        if (is_v1_only) { \
            for (int _k = 0; _k < 6; _k++) T[_k] = v1[_k]; \
        } else { \
            double _t = (t_val); \
            for (int _k = 0; _k < 6; _k++) T[_k] = _t * v1[_k] + v2[_k]; \
        } \
        double _n = vec_norm(T, 6); \
        if (_n > tol) { \
            for (int _k = 0; _k < 6; _k++) sol[nsol][_k] = T[_k] / _n; \
            res[nsol] = fabs(plucker_relation(sol[nsol])); \
            nsol++; \
        } \
    } while(0)

    if (fabs(alpha) < tol) {
        if (fabs(beta) > tol) {
            ADD_SOL(0, -gamma / beta);
        }
        ADD_SOL(1, 0.0);
    } else {
        double disc = beta * beta - 4.0 * alpha * gamma;
        if (disc >= 0.0) {
            double sq = sqrt(fabs(disc));
            ADD_SOL(0, (-beta + sq) / (2.0 * alpha));
            if (nsol < 2) {
                ADD_SOL(0, (-beta - sq) / (2.0 * alpha));
            }
        }
    }
    #undef ADD_SOL

    if (nsol == 2 && res[1] < res[0]) {
        double tmp[6]; double tr;
        memcpy(tmp, sol[0], 6 * sizeof(double));
        memcpy(sol[0], sol[1], 6 * sizeof(double));
        memcpy(sol[1], tmp, 6 * sizeof(double));
        tr = res[0]; res[0] = res[1]; res[1] = tr;
    }

    return nsol;
}

int main(void) {
    double lines[4][6] = {
        {0.0998825797219819, -0.2870649827690026, -0.1655380603417241, 0.7337495434307627, 0.2371851688442816, 0.5343866762089703},
        {-0.0892440959262567, 0.0482678961402157, 0.5563408448558046, -0.1311304431072646, -0.6781469154483000, -0.4506796354929750},
        {0.4008191121664234, 0.1366009339358746, 0.3735097147762055, -0.5058931539708322, 0.3061671355728111, 0.5757677648135123},
        {0.4535967463011593, -0.1141137565511841, 0.6801600989277983, 0.1019411112231264, -0.5550164398954293, -0.0132303977908497},
    };

    /* 1. Constraint matrix */
    double A[4][6];
    for (int i = 0; i < 4; i++)
        hodge_dual(lines[i], A[i]);

    printf("Constraint matrix A (4x6):\n");
    for (int i = 0; i < 4; i++) {
        printf("  row %d: [", i);
        for (int j = 0; j < 6; j++) {
            printf("%+.10f", A[i][j]);
            if (j < 5) printf(", ");
        }
        printf("]\n");
    }

    /* 2. SVD */
    double Vt[36], S[6];
    svd_right((const double *)A, 4, 6, Vt, S);

    printf("\nSingular values: [");
    for (int i = 0; i < 4; i++) {
        printf("%.10f", S[i]);
        if (i < 3) printf(", ");
    }
    printf("]\n");

    const double *v1 = &Vt[5 * 6];
    const double *v2 = &Vt[4 * 6];

    printf("\nv1 (last row of Vt):  [");
    for (int j = 0; j < 6; j++) {
        printf("%+.10f", v1[j]);
        if (j < 5) printf(", ");
    }
    printf("]\n");

    printf("v2 (2nd last of Vt):  [");
    for (int j = 0; j < 6; j++) {
        printf("%+.10f", v2[j]);
        if (j < 5) printf(", ");
    }
    printf("]\n");

    /* 3-4. Quadratic + solve */
    printf("\n");
    double sol[2][6], res[2];
    int nsol = solve_p3(v1, v2, sol, res);

    printf("\nTransversals found: %d\n", nsol);
    for (int i = 0; i < nsol; i++) {
        printf("  T%d: [", i);
        for (int j = 0; j < 6; j++) {
            printf("%+.10f", sol[i][j]);
            if (j < 5) printf(", ");
        }
        printf("]  resid=%.2e\n", res[i]);
    }

    /* Verify inner products */
    for (int i = 0; i < nsol; i++) {
        printf("  T%d inner products with lines: [", i);
        for (int j = 0; j < 4; j++) {
            printf("%+.2e", plucker_inner_fn(sol[i], lines[j]));
            if (j < 3) printf(", ");
        }
        printf("]\n");
    }

    return 0;
}
"""

# Write, compile, and run C test
c_file = "/Volumes/PRO-G40/Code/transversal-memory/test_c_transversal.c"
c_bin = "/Volumes/PRO-G40/Code/transversal-memory/test_c_transversal"

with open(c_file, "w") as f:
    f.write(c_test_code)

print("\n" + "=" * 70)
print("C IMPLEMENTATION")
print("=" * 70)

# Compile
import platform
if platform.system() == "Darwin":
    cc = "cc"
    flags = ["-O2", "-framework", "Accelerate"]
else:
    cc = "cc"
    flags = ["-O2", "-lm", "-llapack"]

compile_cmd = [cc] + flags + ["-o", c_bin, c_file]
result = subprocess.run(compile_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Compilation failed:\n{result.stderr}")
    sys.exit(1)

# Run
result = subprocess.run([c_bin], capture_output=True, text=True)
if result.returncode != 0:
    print(f"C program failed:\n{result.stderr}")
    sys.exit(1)

print(result.stdout)

# Clean up
os.unlink(c_file)
os.unlink(c_bin)

# ── Comparison summary ─────────────────────────────────────────────────────

print("=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

# Parse C output for automated comparison
c_lines_out = result.stdout.strip().split("\n")

# Extract C transversals by parsing output
c_transversals = []
c_resids = []
for line in c_lines_out:
    line = line.strip()
    if line.startswith("T") and "resid=" in line:
        # Parse transversal
        bracket_start = line.index("[")
        bracket_end = line.index("]")
        vals = [float(x) for x in line[bracket_start+1:bracket_end].split(",")]
        c_transversals.append(np.array(vals))
        resid_str = line.split("resid=")[1]
        c_resids.append(float(resid_str))

py_transversals = [T for T, _ in results]
py_resids = [r for _, r in results]

print(f"\nPython found {len(results)} transversals, C found {len(c_transversals)} transversals")

if len(py_transversals) == len(c_transversals):
    for i in range(len(py_transversals)):
        # Transversals may differ by sign (both T and -T represent the same line)
        diff_pos = np.linalg.norm(py_transversals[i] - c_transversals[i])
        diff_neg = np.linalg.norm(py_transversals[i] + c_transversals[i])
        diff = min(diff_pos, diff_neg)
        sign = "same" if diff_pos < diff_neg else "negated"
        print(f"  T{i}: ||py - c|| = {diff:.2e} (sign: {sign})")
        print(f"    py resid = {py_resids[i]:.2e}, c resid = {c_resids[i]:.2e}")
        if diff > 1e-6:
            print(f"    *** DIVERGENCE DETECTED ***")
        else:
            print(f"    MATCH")

# Also check: do both sets of transversals meet all 4 lines?
print(f"\nVerification (inner products should be ~0):")
for label, trans_list in [("Python", py_transversals), ("C", c_transversals)]:
    for i, T in enumerate(trans_list):
        ips = [abs(plucker_inner(T, lines[j])) for j in range(4)]
        max_ip = max(ips)
        print(f"  {label} T{i}: max |inner product| = {max_ip:.2e} {'OK' if max_ip < 1e-6 else 'FAIL'}")

# Check Plucker relation satisfaction
print(f"\nPlucker relation (should be ~0):")
for label, trans_list in [("Python", py_transversals), ("C", c_transversals)]:
    for i, T in enumerate(trans_list):
        pr = abs(plucker_relation(T))
        print(f"  {label} T{i}: |PR| = {pr:.2e} {'OK' if pr < 1e-6 else 'FAIL'}")

# Index ordering check
print(f"\nIndex ordering verification:")
print(f"  Python: _PR_AB={_PR_AB}, _PR_CD={_PR_CD}, _PR_AC={_PR_AC}, _PR_BD={_PR_BD}, _PR_AD={_PR_AD}, _PR_BC={_PR_BC}")
print(f"  C:      PR_AB=0,  PR_CD=5,  PR_AC=1,  PR_BD=4,  PR_AD=2,  PR_BC=3")
match = (_PR_AB==0 and _PR_CD==5 and _PR_AC==1 and _PR_BD==4 and _PR_AD==2 and _PR_BC==3)
print(f"  {'MATCH' if match else '*** MISMATCH ***'}")

# Hodge dual check
print(f"\nHodge dual check (first line):")
py_hd = hodge_dual(lines[0])
print(f"  Python: [{', '.join(f'{v:+.10f}' for v in py_hd)}]")
# C formula: [p5, -p4, p3, p2, -p1, p0]
c_hd = np.array([lines[0][5], -lines[0][4], lines[0][3], lines[0][2], -lines[0][1], lines[0][0]])
print(f"  C:      [{', '.join(f'{v:+.10f}' for v in c_hd)}]")
print(f"  {'MATCH' if np.allclose(py_hd, c_hd) else '*** MISMATCH ***'}")
