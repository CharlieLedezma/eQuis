import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==========================================
# 1. SYMBOLIC DERIVATION (SymPy)
# ==========================================
print("Deriving equations of motion symbolically...")

# Define symbols
t = sp.Symbol('t')
m1, m2, l1, l2, g = sp.symbols('m1 m2 l1 l2 g')
q1 = sp.Function('q1')(t)
q2 = sp.Function('q2')(t)

# Generalized coordinates and velocities
q = sp.Matrix([q1, q2])
dq = q.diff(t)
ddq = dq.diff(t)

# Positions of centers of mass (assuming uniform rods, CoM at middle)
x1 = (l1 / 2) * sp.sin(q1)
y1 = -(l1 / 2) * sp.cos(q1)

x2 = l1 * sp.sin(q1) + (l2 / 2) * sp.sin(q1 + q2)
y2 = -l1 * sp.cos(q1) - (l2 / 2) * sp.cos(q1 + q2)

# Velocities (time derivatives of positions)
vx1, vy1 = x1.diff(t), y1.diff(t)
vx2, vy2 = x2.diff(t), y2.diff(t)

# Kinetic Energy (T) and Potential Energy (V)
# (Including simplified rotational inertia I = 1/12 * m * l^2)
I1 = (1/12) * m1 * l1**2
I2 = (1/12) * m2 * l2**2

T1 = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * I1 * dq[0]**2
T2 = 0.5 * m2 * (vx2**2 + vy2**2) + 0.5 * I2 * (dq[0] + dq[1])**2
T = sp.simplify(T1 + T2)

V = m1 * g * y1 + m2 * g * y2

# Mass Matrix M(q)
# Derived from Kinetic energy terms matching 0.5 * dq^T * M * dq
M = sp.Matrix([[T.diff(dq[0]).diff(dq[0]), T.diff(dq[0]).diff(dq[1])],
               [T.diff(dq[1]).diff(dq[0]), T.diff(dq[1]).diff(dq[1])]])
M = sp.simplify(M)

# Gravity Vector g(q)
G = sp.Matrix([V.diff(q1), V.diff(q2)])

# Coriolis and Centrifugal Matrix C(q, dq) using Christoffel Symbols
C = sp.zeros(2, 2)
n = len(q)
# Temporary velocity symbols for partial differentiation
dq_sym = sp.symbols('dq1 dq2')
M_sym = M.subs({dq[0]: dq_sym[0], dq[1]: dq_sym[1]})

for i in range(n):
    for j in range(n):
        c_ij = 0
        for k in range(n):
            # Christoffel symbol formulation
            g_ijk = 0.5 * (M_sym[i, j].diff(q[k]) + M_sym[i, k].diff(q[j]) - M_sym[j, k].diff(q[i]))
            c_ij += g_ijk * dq_sym[k]
        C[i, j] = c_ij

# Substitute real time derivatives back into C matrix
C = C.subs({dq_sym[0]: dq[0], dq_sym[1]: dq[1]})
C = sp.simplify(C)

print("Matrices derived successfully.")

# ==========================================
# 2. LAMBDIFY FOR NUMERICAL SIMULATION
# ==========================================
# Create concrete numerical parameters
robot_params = {m1: 1.0, m2: 1.0, l1: 1.0, l2: 1.0, g: 9.81}

# Create dummy symbols to replace functions for clean lambdification
q1_s, q2_s, dq1_s, dq2_s = sp.symbols('q1_s q2_s dq1_s dq2_s')
sub_dict = {q[0]: q1_s, q[1]: q2_s, dq[0]: dq1_s, dq[1]: dq2_s}

M_num = sp.lambdify((q1_s, q2_s), M.subs(robot_params).subs(sub_dict), 'numpy')
C_num = sp.lambdify((q1_s, q2_s, dq1_s, dq2_s), C.subs(robot_params).subs(sub_dict), 'numpy')
G_num = sp.lambdify((q1_s, q2_s), G.subs(robot_params).subs(sub_dict), 'numpy')

# ==========================================
# 3. ODE INITIAL VALUE PROBLEM (solve_ivp)
# ==========================================
def forward_dynamics(t, y):
    """
    y = [q1, q2, dq1, dq2]
    returns [dq1, dq2, ddq1, ddq2]
    """
    q_vec = y[0:2]
    dq_vec = y[2:4]
    
    # 1. Unpack matrices numerically
    M_val = np.array(M_num(q_vec[0], q_vec[1]))
    C_val = np.array(C_num(q_vec[0], q_vec[1], dq_vec[0], dq_vec[1]))
    G_val = np.array(G_num(q_vec[0], q_vec[1])).flatten()
    
    # 2. Define Control Torque (Let's assume zero input torque -> passive drop)
    tau = np.array([0.0, 0.0]) 
    
    # 3. Solve for acceleration: ddq = M^-1 * (tau - C*dq - G)
    # Using np.linalg.solve is faster and more stable than inverting
    rhs = tau - C_val @ dq_vec - G_val
    ddq_vec = np.linalg.solve(M_val, rhs)
    
    return [dq_vec[0], dq_vec[1], ddq_vec[0], ddq_vec[1]]

# Initial conditions: [q1, q2, dq1, dq2] -> Start horizontal at rest
y0 = [np.pi/2, 0.0, 0.0, 0.0]
t_span = (0, 5)  # 5 seconds simulation
t_eval = np.linspace(0, 5, 500)

print("Running solve_ivp integration...")
sol = solve_ivp(forward_dynamics, t_span, y0, t_eval=t_eval, method='RK45')

# ==========================================
# 4. PLOT RESULTS
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='Joint 1 Angle (rad)')
plt.plot(sol.t, sol.y[1], label='Joint 2 Angle (rad)')
plt.title('2-DOF Robotic Arm Passive Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Position (rad)')
plt.legend()
plt.grid(True)
plt.show()
