import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

# =============================================================================
# 1. SYMBOLIC DERIVATION (SymPy)
# =============================================================================
# Define symbolic variables for a simple 2-DOF planar robot arm
q1, q2, dq1, dq2, ddq1, ddq2 = sp.symbols('q1 q2 dq1 dq2 ddq1 ddq2')
m1, m2, l1, l2, g = sp.symbols('m1 m2 l1 l2 g')

q = sp.Matrix([q1, q2])
dq = sp.Matrix([dq1, dq2])
ddq = sp.Matrix([ddq1, ddq2])

# Define a simplified Mass Matrix M(q)
M11 = m1*l1**2 + m2*(l1**2 + l2**2 + 2*l1*l2*sp.cos(q2))
M12 = m2*(l2**2 + l1*l2*sp.cos(q2))
M21 = M12
M22 = m2*l2**2
M = sp.Matrix([[M11, M12], [M21, M22]])

# Derive Coriolis/Centrifugal Matrix C(q, dq) using Christoffel symbols
C = sp.zeros(2, 2)
for i in range(2):
    for j in range(2):
        for k in range(2):
            # Christoffel symbol of the first kind: 0.5 * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)
            christoffel = 0.5 * (sp.diff(M[i, j], q[k]) + sp.diff(M[i, k], q[j]) - sp.diff(M[j, k], q[i]))
            C[i, j] += christoffel * dq[k]

# Define Gravity Vector G(q)
G = sp.Matrix([
    (m1 + m2)*g*l1*sp.cos(q1) + m2*g*l2*sp.cos(q1 + q2),
    m2*g*l2*sp.cos(q1 + q2)
])

# Lambdify for blistering fast numerical evaluation
# Structure: f(q1, q2, dq1, dq2, m1, m2, l1, l2, g)
num_M = sp.lambdify((q1, q2, dq1, dq2, m1, m2, l1, l2, g), M, 'numpy')
num_C = sp.lambdify((q1, q2, dq1, dq2, m1, m2, l1, l2, g), C, 'numpy')
num_G = sp.lambdify((q1, q2, dq1, dq2, m1, m2, l1, l2, g), G, 'numpy')

# Physical parameters mapping
params = {'m1': 1.0, 'm2': 1.0, 'l1': 1.0, 'l2': 1.0, 'g': 9.81}

# Helper function to unpack states into the numpy dynamics
def get_dynamics(q_vec, dq_vec):
    args = (q_vec[0], q_vec[1], dq_vec[0], dq_vec[1], 
            params['m1'], params['m2'], params['l1'], params['l2'], params['g'])
    return num_M(*args), num_C(*args), num_G(*args)

# =============================================================================
# 2. CONTROL & SIMULATION INTERFACE (SciPy)
# =============================================================================
def desired_trajectory(t):
    """Defines target position, velocity, and acceleration profiles."""
    q_d = np.array([np.sin(t), np.cos(t)])
    dq_d = np.array([np.cos(t), -np.sin(t)])
    ddq_d = np.array([-np.sin(t), -np.cos(t)])
    return q_d, dq_d, ddq_d

def robot_dynamics_ode(t, x):
    """State-space equations for solve_ivp: x = [q1, q2, dq1, dq2]"""
    q_curr = x[0:2]
    dq_curr = x[2:4]
    
    # 1. Compute physical robot matrices
    M_mat, C_mat, G_mat = get_dynamics(q_curr, dq_curr)
    G_mat = G_mat.flatten() # Ensure 1D array consistency
    
    # 2. Compute Tracking Controller (Computed Torque Control / Feedback Linearization)
    q_d, dq_d, ddq_d = desired_trajectory(t)
    
    Kp = np.diag([100.0, 100.0]) # Proportional Gain
    Kd = np.diag([20.0, 20.0])   # Derivative Gain
    
    e = q_d - q_curr
    de = dq_d - dq_curr
    
    # Control law: Tau = M*(ddq_d + Kd*de + Kp*e) + C*dq + G
    aq = ddq_d + Kd @ de + Kp @ e
    tau = M_mat @ aq + C_mat @ dq_curr + G_mat
    
    # 3. Solve Forward Dynamics for Acceleration: ddq = M^-1 * (Tau - C*dq - G)
    ddq_curr = np.linalg.inv(M_mat) @ (tau - C_mat @ dq_curr - G_mat)
    
    # Return state derivatives [dq1, dq2, ddq1, ddq2]
    return np.concatenate([dq_curr, ddq_curr])

# =============================================================================
# 3. RUN SIMULATION
# =============================================================================
# Initial conditions: [q1, q2, dq1, dq2] -> slightly offset from desired
x0 = np.array([0.0, 0.5, 0.0, 0.0]) 
t_span = (0.0, 10.0)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(robot_dynamics_ode, t_span, x0, t_eval=t_eval, method='RK45')

print(print(f"Simulation completed successfully with {len(sol.t)} time steps!"))


plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Joint 1 ($dq_2$)')
plt.plot(sol.t, sol.y[1], label='Joint 2 ($dq_3$)')
#plt.plot(sol.t, desired_trajectory[0], label='Joint 3 ($dq_3$)')
# #plt.plot(solution.t, solution.y[2], label='Joint 3 ($q_3$)')
plt.title('Car Displacement')
plt.xlabel('Time (s)')
plt.ylabel('Displacement(m))')
plt.legend()
plt.grid(True)
plt.show()