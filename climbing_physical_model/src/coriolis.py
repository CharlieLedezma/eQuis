import sympy as sp

# Define generalized coordinates q and velocities dq
q1, q2 = sp.symbols('q1 q2')
dq1, dq2 = sp.symbols('dq1 dq2')
q = [q1, q2]
dq = [dq1, dq2]

# Example symmetric mass matrix H(q)
# Replace this with your actual robotic/mechanical system's mass matrix
H = sp.Matrix([
    [q2**2 + 1, q1],
    [q1, sp.Integer(2)]
])

n = len(q)

# 1. Compute Christoffel symbols of the first kind: Gamma[i, j, k] = Gamma_ijk
# Gamma_ijk = 0.5 * (dH_ij/dq_k + dH_ik/dq_j - dH_jk/dq_i)
Gamma = [[[sp.Integer(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]

for i in range(n):
  for j in range(n):
    for k in range(n):
      term1 = sp.diff(H[i, j], q[k])
      term2 = sp.diff(H[i, k], q[j])
      term3 = sp.diff(H[j, k], q[i])
      Gamma[i][j][k] = 0.5 * (term1 + term2 + term3)

# 2. Compute the Coriolis matrix C(q, dq) where C_ij = sum_k (Gamma_ijk * dq_k)
C = sp.zeros(n, n)
for i in range(n):
  for j in range(n):
    c_ij = 0
    for k in range(n):
      c_ij += Gamma[i][j][k] * dq[k]
    C[i, j] = sp.simplify(c_ij)

print('Coriolis Matrix C(q, dq):')
sp.pprint(C)

