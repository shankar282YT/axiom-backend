# ============================================================
#  AXIOM · matrix.py
#  Pure Python matrix math — no numpy, no external libs
#  All matrices are list[list[float]]
# ============================================================

import math
import random


# ── Constructors ─────────────────────────────────────────────

def zeros(rows: int, cols: int) -> list:
    return [[0.0] * cols for _ in range(rows)]

def identity(n: int) -> list:
    m = zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m

def rand_matrix(rows: int, cols: int, scale: float = 0.01) -> list:
    """He/Xavier-style small random init."""
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


# ── Shape ────────────────────────────────────────────────────

def shape(m: list) -> tuple:
    if not m:
        return (0, 0)
    if isinstance(m[0], list):
        return (len(m), len(m[0]))
    return (len(m),)          # 1-D vector


# ── Element-wise ops ─────────────────────────────────────────

def add(a: list, b: list) -> list:
    """Add two matrices or two vectors."""
    if isinstance(a[0], list):
        return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    return [a[i] + b[i] for i in range(len(a))]

def sub(a: list, b: list) -> list:
    if isinstance(a[0], list):
        return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    return [a[i] - b[i] for i in range(len(a))]

def mul_scalar(m: list, s: float) -> list:
    if isinstance(m[0], list):
        return [[v * s for v in row] for row in m]
    return [v * s for v in m]

def mul_elem(a: list, b: list) -> list:
    """Element-wise multiply (Hadamard)."""
    if isinstance(a[0], list):
        return [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    return [a[i] * b[i] for i in range(len(a))]


# ── Matrix multiply ──────────────────────────────────────────

def matmul(a: list, b: list) -> list:
    """a: (m x k), b: (k x n) → (m x n)"""
    m, k  = len(a), len(a[0])
    k2, n = len(b), len(b[0])
    assert k == k2, f"matmul shape mismatch {k} vs {k2}"
    bt = [[b[r][c] for r in range(k)] for c in range(n)]
    result = zeros(m, n)
    for i in range(m):
        ai = a[i]
        for j in range(n):
            result[i][j] = sum(ai[p] * bt[j][p] for p in range(k))
    return result

def matvec(m: list, v: list) -> list:
    """Matrix (r x c) × vector (c,) → vector (r,)"""
    return [sum(row[j] * v[j] for j in range(len(v))) for row in m]

def vecmat(v: list, m: list) -> list:
    """Row-vector (1 x r) × matrix (r x c) → vector (c,)"""
    r, c = len(m), len(m[0])
    return [sum(v[i] * m[i][j] for i in range(r)) for j in range(c)]

def transpose(m: list) -> list:
    rows, cols = len(m), len(m[0])
    return [[m[r][c] for r in range(rows)] for c in range(cols)]

def outer(a: list, b: list) -> list:
    """Outer product: (n,) x (m,) → (n x m)"""
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]


# ── Activations ──────────────────────────────────────────────

def relu(v: list) -> list:
    return [max(0.0, x) for x in v]

def relu_deriv(v: list) -> list:
    return [1.0 if x > 0 else 0.0 for x in v]

def softmax(v: list) -> list:
    m = max(v)
    exps = [math.exp(x - m) for x in v]
    s = sum(exps)
    return [e / s for e in exps]

def leaky_relu(v: list, alpha: float = 0.01) -> list:
    return [x if x > 0 else alpha * x for x in v]

def leaky_relu_deriv(v: list, alpha: float = 0.01) -> list:
    return [1.0 if x > 0 else alpha for x in v]


# ── Loss ─────────────────────────────────────────────────────

def cross_entropy(probs: list, target_idx: int) -> float:
    return -math.log(max(probs[target_idx], 1e-12))

def cross_entropy_deriv(probs: list, target_idx: int) -> list:
    """Gradient of softmax + cross-entropy combined."""
    grad = probs[:]
    grad[target_idx] -= 1.0
    return grad


# ── Utilities ────────────────────────────────────────────────

def dot(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))

def norm(v: list) -> float:
    return math.sqrt(sum(x * x for x in v))

def clip(v: list, lo: float = -5.0, hi: float = 5.0) -> list:
    return [max(lo, min(hi, x)) for x in v]

def clip_matrix(m: list, lo: float = -5.0, hi: float = 5.0) -> list:
    return [[max(lo, min(hi, v)) for v in row] for row in m]
