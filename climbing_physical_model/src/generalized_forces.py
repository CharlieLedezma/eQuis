import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. Define Time and Generalized Coordinates
t = sp.symbols('t')
q1, q2, q3 = sp.symbols('q1 q2 q3', cls=sp.Function)
q = sp.Matrix([q1(t), q2(t), q3(t)])

# Time derivatives (Velocities)
q_dot = q.diff(t)

# 2. Define Parameters and System Energies (Example: 3-mass spring system)
m1, m2, m3, k1, k2, k3, k4 = sp.symbols('m1 m2 m3 k1 k2 k3 k4')
Q1, Q2, Q3 = sp.symbols('Q1 Q2 Q3')  # Generalized forces

# Kinetic Energy (T) and Potential Energy (V)
T = 0.5 * m1 * q_dot[0]**2 + 0.5 * m2 * q_dot[1]**2 + 0.5 * m3 * q_dot[2]**2
V = 0.5 * k1 * q[0]**2 + 0.5 * k2 * (q[1] - q[0])**2 + 0.5 * k3 * (q[2] - q[1])**2 + 0.5 * k4 * q[2]**2

# Lagrangian
L = T - V

# 3. Formulate Lagrange's Equations
# d/dt(dL/dq_dot) - dL/dq = Q
eqs = []
for i in range(3):
    expr = sp.diff(sp.diff(L, q_dot[i]), t) - sp.diff(L, q[i])
    eqs.append(expr)
eqs = sp.Matrix(eqs)

# 4. Extract Mass Matrix (M) and Forcing Vector (F)
# System equation form: M(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = Q_vec
# Rewritten as: M(q) * q_ddot = F_vec (where F_vec contains generalized forces, stiffness, damping)
q_ddot = q_dot.diff(t)
M = eqs.jacobian(q_ddot)

# Force vector without the inertia terms
Q_vec = sp.Matrix([Q1, Q2, Q3])
F = Q_vec - (eqs - M * q_ddot)

# 5. Prepare for Vectorized/Numerical Integration
# Replace functions with temporary symbols for lambdification
q_s = sp.symbols('q1:4')
sp_dot_s = sp.symbols('q_dot1:4')
q_ddot_s = sp.symbols('q_ddot1:4')

# Mapping dictionary to replace time-dependent functions
sub_dict = {
    q_ddot[i]: q_ddot_s[i] for i in range(3)
} | {
    q_dot[i]: sp_dot_s[i] for i in range(3)
} | {
    q[i]: q_s[i] for i in range(3)
}

M_numeric = M.subs(sub_dict)
F_numeric = F.subs(sub_dict)

# Lambdify matrices for fast execution
# Signature: (t, [q1, q2, q3], [q_dot1, q_dot2, q_dot3], [m1, m2, m3, k1, k2, k3, k4, Q1, Q2, Q3])
param_symbols = (m1, m2, m3, k1, k2, k3, k4, Q1, Q2, Q3)
fn_M = sp.lambdify((t, q_s, sp_dot_s, param_symbols), M_numeric, 'numpy')
fn_F = sp.lambdify((t, q_s, sp_dot_s, param_symbols), F_numeric, 'numpy')

# 6. Define State-Space ODE for solve_ivp
def ode_system(t, y, params):
    # y = [q1, q2, q3, q_dot1, q_dot2, q_dot3]
    q_val = y[:3]
    q_dot_val = y[3:]
    
    # Evaluate numerical Mass matrix and Force vector
    M_mat = fn_M(t, q_val, q_dot_val, params)
    F_vec = fn_F(t, q_val, q_dot_val, params).flatten()
    
    # Solve for acceleration: q_ddot = M^-1 * F
    q_ddot_val = np.linalg.solve(M_mat, F_vec)
    
    # Return [velocities, accelerations]
    return np.concatenate([q_dot_val, q_ddot_val])

# 7. Numerical Evaluation
# Numerical parameter values
numerical_params = (1.0, 1.5, 2.0, 10.0, 10.0, 10.0, 10.0, 1.0, 0.0, 0.0) # masses, stiffnesses, external forces
y0 = [0.1, 0.0, -0.1, 0.0, 0.0, 0.0]  # Initial positions and velocities
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)

# Run solver
sol = solve_ivp(ode_system, t_span, y0, args=(numerical_params,), t_eval=t_eval, method='RK45')

print("Integration complete. Solution shape:", sol.y.shape)

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Joint 1 ($dq_2$)')
plt.plot(sol.t, sol.y[1], label='Joint 2 ($dq_3$)')
plt.plot(sol.t, sol.y[2], label='Joint 3 ($dq_3$)')
# #plt.plot(solution.t, solution.y[2], label='Joint 3 ($q_3$)')
plt.title('Car Displacement')
plt.xlabel('Time (s)')
plt.ylabel('Displacement(m))')
plt.legend()
plt.grid(True)
plt.show()