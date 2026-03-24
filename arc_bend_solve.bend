# Bend: grid scorer with superpositions
# Using functional syntax for SUPs

def penalty(a, b):
  d = a - b
  return d * d

def score_2x2(c00, c01, c10, c11):
  h0 = penalty(c00, c01)
  h1 = penalty(c10, c11)
  v0 = penalty(c00, c10)
  v1 = penalty(c01, c11)
  return h0 + h1 + v0 + v1

def score_3x3(c00, c01, c02, c10, c11, c12, c20, c21, c22):
  h0 = penalty(c00, c01) + penalty(c01, c02)
  h1 = penalty(c10, c11) + penalty(c11, c12)
  h2 = penalty(c20, c21) + penalty(c21, c22)
  v0 = penalty(c00, c10) + penalty(c10, c20)
  v1 = penalty(c01, c11) + penalty(c11, c21)
  v2 = penalty(c02, c12) + penalty(c12, c22)
  return h0 + h1 + h2 + v0 + v1 + v2

# Test 1: concrete
def test1():
  return score_2x2(0, 1, 1, 0)

# Test 2: superposition via compiled HVM
# Bend compiles to HVM — we can inject SUPs via the gen-hvm pipeline
# For now, test if Bend supports the `!` DUP syntax inline

# Actually, let's use Bend's native `let` with tuples as a workaround
# and test the HVM output

def main():
  return test1()
