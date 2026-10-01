import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def system_dynamics(t, state, params):
    """
    Defines the first-order ODEs for a 3DOF Lagrangian system.
    state = [q1, q2, q3, q1_dot, q2_dot, q3_dot]
    """
    # 1. Unpack generalized coordinates (q) and velocities (q_dot)
    q = state[:3]
    q_dot = state[3:]
    
    # 2. Extract physical parameters
    m1, m2, m3, l1, l2, l3, g = params
    
    # 3. Define the 3x3 Mass Matrix M(q)
    # (Replace these dummy equations with your system's actual derived matrices)
    M = np.zeros((3, 3))
    M[0, 0] = (m1 + m2 + m3) * l1**2
    M[1, 1] = (m2 + m3) * l2**2
    M[2, 2] = m3 * l3**2
    # Off-diagonal cross-coupling terms (example using cosines)
    M[0, 1] = M[1, 0] = (m2 + m3) * l1 * l2 * np.cos(q[0] - q[1])
    M[0, 2] = M[2, 0] = m3 * l1 * l3 * np.cos(q[0] - q[2])
    M[1, 2] = M[2, 1] = m3 * l2 * l3 * np.cos(q[1] - q[2])
    
    # 4. Define the Coriolis & Centrifugal forces C(q, q_dot) * q_dot
    # (Often simplified directly into a vector for computation)
    C_qdot = np.zeros(3)
    C_qdot[0] = (m2 + m3) * l1 * l2 * (q_dot[1]**2) * np.sin(q[0] - q[1])
    C_qdot[1] = - (m2 + m3) * l1 * l2 * (q_dot[0]**2) * np.sin(q[0] - q[1])
    # Note: complete equations include many more coupled sine variations
    
    # 5. Define the Gravity Vector G(q)
    G = np.array([
        (m1 + m2 + m3) * g * l1 * np.sin(q[0]),
        (m2 + m3) * g * l2 * np.sin(q[1]),
        m3 * g * l3 * np.sin(q[2])
    ])
    
    # 6. Generalized forces/torques applied (set to 0 for unforced/passive motion)
    tau = np.array([0.0, 0.0, 0.0])
    
    # 7. Solve for acceleration: q_ddot = M^-1 * (tau - C*q_dot - G)
    rhs = tau - C_qdot - G
    q_ddot = np.linalg.solve(M, rhs)  # More efficient than explicit inverse
    
    # 8. Return state derivatives [q_dot, q_ddot]
    return np.concatenate((q_dot, q_ddot))

# --- Simulation Setup ---

# Parameters: [m1, m2, m3, l1, l2, l3, g]
sys_params = (1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 9.81)

# Initial conditions: [q1, q2, q3, q1_dot, q2_dot, q3_dot]
# Let's release it from rest at a localized angle deviation
initial_state = [np.pi/4, 0.0, -np.pi/4, 0.0, 0.0, 0.0]

# Time span for the integration
t_span = (0.0, 10.0)
t_eval = np.linspace(t_span[0], t_span[1], 500) # Output timestamps

# Execute numerical integration
sol = solve_ivp(
    fun=system_dynamics,
    t_span=t_span,
    y0=initial_state,
    method='RK45',        # Excellent default solver for non-stiff dynamics
    t_eval=t_eval,
    args=(sys_params,),   # Pass structural parameters cleanly
    rtol=1e-6,            # Tighten tolerances for chaotic/multilink systems
    atol=1e-8
)

# --- Plotting Coordinates over Time ---
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Joint 1 ($q_1$)')
plt.plot(sol.t, sol.y[1], label='Joint 2 ($q_2$)')
plt.plot(sol.t, sol.y[2], label='Joint 3 ($q_3$)')
plt.title('3DOF Lagrangian System Trajectory via solve_ivp')
plt.xlabel('Time (s)')
plt.ylabel('Displacement / Angle (rad)')
plt.legend()
plt.grid(True)
plt.show()
