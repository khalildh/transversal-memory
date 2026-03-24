# Benchmark: 5 uncertain cells × 10 colors = 100K candidates
# Parallel search via bend trees, compiled to C

def penalty(a, b):
  d = a - b
  return d * d

def score_3x3(c00, c01, c02, c10, c11, c12, c20, c21, c22):
  h0 = penalty(c00, c01) + penalty(c01, c02)
  h1 = penalty(c10, c11) + penalty(c11, c12)
  h2 = penalty(c20, c21) + penalty(c21, c22)
  v0 = penalty(c00, c10) + penalty(c10, c20)
  v1 = penalty(c01, c11) + penalty(c11, c21)
  v2 = penalty(c02, c12) + penalty(c12, c22)
  return h0 + h1 + h2 + v0 + v1 + v2

# Helper: min of two (score, args) tuples
def best(a, b):
  (sa, _) = a
  (sb, _) = b
  if sa < sb:
    return a
  else:
    return b

# Search: 5 uncertain cells in middle row + c20
# Fixed: c00=0, c01=2, c02=0, c21=2, c22=0
def search(nc):
  bend a = 0:
    when a < nc:
      r = search_b(a, nc)
      return best(r, fork(a + 1))
    else:
      return (99999, 0)

def search_b(a, nc):
  bend b = 0:
    when b < nc:
      r = search_c(a, b, nc)
      return best(r, fork(b + 1))
    else:
      return (99999, 0)

def search_c(a, b, nc):
  bend c = 0:
    when c < nc:
      r = search_d(a, b, c, nc)
      return best(r, fork(c + 1))
    else:
      return (99999, 0)

def search_d(a, b, c, nc):
  bend d = 0:
    when d < nc:
      r = search_e(a, b, c, d, nc)
      return best(r, fork(d + 1))
    else:
      return (99999, 0)

def search_e(a, b, c, d, nc):
  bend e = 0:
    when e < nc:
      s = score_3x3(0, 2, 0, a, b, c, d, 2, e)
      return best((s, (a, (b, (c, (d, e))))), fork(e + 1))
    else:
      return (99999, 0)

def main():
  return search(10)
