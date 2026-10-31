import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def manipulator(t, state, params):
    """
    Defines the first-order ODEs for a 2DOF Lagrangian system.'
    state = [q1, q2, q1_dot, q2_dot]
    """
        
    # 1. Unpack generalized coordinates (q) and velocities (q_dot)
    q = state[:2]
    q_dot = state[2:]
        
    # 2. Extract physical parameters
    m_1, m_2, I_1, I_2, l_g1,l_1,l_g2, g = params
        
    # 3. Define the 2x2 Mass Matrix M(q)
    # (Replace these dummy equations with your system's actual derived matrices)
    M = np.zeros((2, 2))
    M[0, 0] =I_1 + I_1 + m_1*(1/2)*l_g1**   2 + m_2*(l_1*l_1 + l_g2**2 + 2*l_1*l_g2*np.cos(q[1]))
    M[1, 1] = I_2 + m_2*l_g2**2
    # Off-diagonal cross-coupling terms (example using cosines)
    M[0, 1] = M[1, 0] = I_2 + m_2*(l_g2**2 + l_1*l_g2*np.cos(q[1]))
        
    # 4. Define the Coriolis & Centrifugal forces C(q, q_dot) * q_dot
    # (Often simplified directly into a vector for computation)
    C_qdot = np.zeros(2)
    C_qdot[0] = -m_2*l_1*l_g2*q_dot[1]*(2*q_dot[0] + q_dot[1])*np.sin(q[1])
    C_qdot[1] = m_2*l_1*l_g2*q_dot[0]*q_dot[0]*np.sin(q[1])
    # Note: complete equations include many more coupled sine variations
        
    # 5. Define the Gravity Vector G(q)
    G = np.array([
            m_1*g*l_g1*np.cos(q[0]) + m_2*g*(l_1*np.cos(q[0]) + l_g2*np.cos(q[0]+q[1])),
            m_2*g*l_g2*np.cos(q[0] + q[1])
        ])

    # 6. Generalized forces/torques applied (set to 0 for unforced/passive motion)
    tau = np.array([0.0, 0.0]) 
        
    # 7. Solve for acceleration: q_ddot = M^-1 * (tau - C*q_dot - G)
    rhs = tau - C_qdot - G
    q_ddot = np.linalg.solve(M, rhs)  # More efficient than explicit inverse
        
    # 8. Return state derivatives [q_dot, q_ddot]
    return np.concatenate((q_dot, q_ddot))
    

# Parameters: [m_1, m_2, I_1, I_2, l_g1,l_1,l_g2, g]
sys_params = (1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 9.81)

# Initial conditions: [q1, q2, q3, q1_dot, q2_dot, q3_dot]
# Let's release it from rest at a localized angle deviation
initial_state = [0.1*np.pi, 0.1*np.pi, 0.0, 0.0]

# Time span for the integration
t_span = (0.0, 1.0)
resolution = 20
t_eval = np.linspace(t_span[0], t_span[1], resolution) # Output timestamps


# Execute numerical integration
sol = solve_ivp(
    fun=manipulator,
    t_span=t_span,
    y0=initial_state,
    method='RK45',        # Excellent default solver for non-stiff dynamics
    t_eval=t_eval,
    args=(sys_params,),   # Pass structural parameters cleanly
    rtol=1e-6,            # Tighten tolerances for chaotic/multilink systems
    atol=1e-8
)

data = np.vstack((sol.t, sol.y)).T
np.savetxt('owaki_solution.csv', data, delimiter=',', header='t,y', comments='')



plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Joint 1 ($q_1$)')
plt.plot(sol.t, sol.y[1], label='Joint 2 ($q_2$)')
plt.title('2DOF Lagrangian System Trajectory via solve_ivp')
plt.xlabel('Time (s)')
plt.ylabel('Displacement / Angle (rad)')
plt.legend()
plt.grid(True)
plt.show()