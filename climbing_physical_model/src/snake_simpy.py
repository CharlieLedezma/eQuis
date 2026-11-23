import numpy as np
from scipy.integrate import solve_ivp

# 1. Robot & Simulation Parameters
N = 4                  # Number of links
m = 1.0                # Mass of each link (kg)
l = 0.1                # Half-length of each link (m)
J = (1/3) * m * l**2   # Moment of inertia of each link
c_t = 0.5              # Tangential friction coefficient (low)
c_n = 5.0              # Normal friction coefficient (high -> enables forward motion)

# 2. Structural Matrices for N-link Chain
A = np.zeros((N - 1, N))
D = np.zeros((N - 1, N))
for i in range(N - 1):
    A[i, i] = 1.0
    A[i, i + 1] = 1.0
    D[i, i] = 1.0
    D[i, i + 1] = -1.0

# Mapping from link angles to link center positions
V = A.T @ np.linalg.inv(D @ D.T) @ D
e = np.ones((N, 1))

def snake_dynamics(t, state):
    """Right-hand side dy/dt = f(t, y) for solve_ivp."""
    # Unpack state: positions (N+2) and velocities (N+2)
    q = state[:N + 2]
    dq = state[N + 2:]
    
    theta = q[:N]
    dtheta = dq[:N]
    dp_x, dp_y = dq[N], dq[N + 1]
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    S = np.diag(sin_t)
    C = np.diag(cos_t)
    
    # Link center velocities in global frame
    dx = -l * (V.T @ (sin_t * dtheta)) + dp_x
    dy =  l * (V.T @ (cos_t * dtheta)) + dp_y
    
    # Anisotropic viscous ground friction forces on each link
    fx = -(c_t * cos_t**2 + c_n * sin_t**2) * dx - (c_t - c_n) * sin_t * cos_t * dy
    fy = -(c_t - c_n) * sin_t * cos_t * dx - (c_t * sin_t**2 + c_n * cos_t**2) * dy
    
    # Lateral undulation control (PD tracking of a traveling sine wave)
    alpha, omega, delta = 0.6, 3.0, 1.2
    kp, kd = 15.0, 2.0
    u = np.zeros(N - 1)
    for i in range(N - 1):
        phi_ref = alpha * np.sin(omega * t + i * delta)
        dphi_ref = alpha * omega * np.cos(omega * t + i * delta)
        phi = theta[i] - theta[i + 1]
        dphi = dtheta[i] - dtheta[i + 1]
        u[i] = kp * (phi_ref - phi) + kd * (dphi_ref - dphi)
        
    # Mass matrix M_theta for rotational dynamics
    K = V.T @ V
    M_theta = J * np.eye(N) + m * (l**2) * (S @ K @ S + C @ K @ C)
    W = m * (l**2) * (S @ K @ C - C @ K @ S)
    
    # Accelerations: ddtheta, ddp_x, ddp_y
    rhs_theta = D.T @ u - W @ (dtheta**2) + l * (S @ V @ fx - C @ V @ fy)
    ddtheta = np.linalg.solve(M_theta, rhs_theta)
    ddp_x = np.sum(fx) / (N * m)
    ddp_y = np.sum(fy) / (N * m)
    
    ddq = np.concatenate([ddtheta, [ddp_x, ddp_y]])
    return np.concatenate([dq, ddq])

# 3. Initial Conditions & Integration
y0 = np.zeros(2 * (N + 2))
t_span = (0.0, 10.0)
t_eval = np.linspace(0.0, 10.0, 500)

sol = solve_ivp(
    fun=snake_dynamics,
    t_span=t_span,
    y0=y0,
    method='RK45',
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-8
)

# Extract Center-of-Mass (CoM) trajectory
com_x = sol.y[N]
com_y = sol.y[N + 1]
print(f"Final CoM Position: x = {com_x[-1]:.3f} m, y = {com_y[-1]:.3f} m")
