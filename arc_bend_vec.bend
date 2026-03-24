# Parallel ARC scorer using bend trees
# No SUPs — just parallel map + reduce

def penalty(a, b):
  d = a - b
  return d * d

def score_2x2(c00, c01, c10, c11):
  h0 = penalty(c00, c01)
  h1 = penalty(c10, c11)
  v0 = penalty(c00, c10)
  v1 = penalty(c01, c11)
  return h0 + h1 + v0 + v1

# Parallel search: try all values for c11, return (best_score, best_c11)
# Fixed: c00=0, c01=1, c10=1
def search_c11(n):
  bend x = 0:
    when x < n:
      s = score_2x2(0, 1, 1, x)
      rest = fork(x + 1)
      # Keep the min
      (rs, rx) = rest
      if s < rs:
        return (s, x)
      else:
        return rest
    else:
      return (9999, -1)

# Parallel search: two uncertain cells c10 and c11, each 0..nc
def search_2cells(nc):
  bend i = 0:
    when i < nc:
      # For each c10=i, search all c11 values
      inner = search_inner(i, nc)
      rest = fork(i + 1)
      (is_, ix, iy) = inner
      (rs, rx, ry) = rest
      if is_ < rs:
        return inner
      else:
        return rest
    else:
      return (9999, -1, -1)

def search_inner(c10, nc):
  bend j = 0:
    when j < nc:
      s = score_2x2(0, 1, c10, j)
      rest = fork(j + 1)
      (rs, rx, ry) = rest
      if s < rs:
        return (s, c10, j)
      else:
        return rest
    else:
      return (9999, -1, -1)

def main():
  # Test 1: search 1 cell among 10 colors
  r1 = search_c11(10)

  # Test 2: search 2 cells among 5 colors = 25 combos
  r2 = search_2cells(5)

  return (r1, r2)
