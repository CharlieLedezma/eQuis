import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==========================================
# 1. SYMBOLIC DERIVATION (SymPy)
# ==========================================
N = 9  # Number of links (keep small for fast symbolic derivation, e.g., 3 to 5)
t = sp.symbols('t')

# Generalized coordinates: head position (x, y) and link angles (theta_1 ... theta_N)
x = sp.Function('x')(t)
y = sp.Function('y')(t)
thetas = [sp.Function(f'theta_{i}')(t) for i in range(N)]
q = sp.Matrix([x, y] + thetas)
dq = q.diff(t)
ddq = dq.diff(t)

# Physical parameters (symbolic)
m, l, J, c_t, c_n = sp.symbols('m l J c_t c_n', positive=True)

# Kinematics: Center of mass (CoM) of each link
# Link 0 starts at (x, y); subsequent links attach end-to-end
px, py = [x], [y]
for i in range(1, N):
    px.append(px[i-1] + l * sp.cos(thetas[i-1]))
    py.append(py[i-1] + l * sp.sin(thetas[i-1]))

cx = [px[i] + (l / 2) * sp.cos(thetas[i]) for i in range(N)]
cy = [py[i] + (l / 2) * sp.sin(thetas[i]) for i in range(N)]

# Kinetic Energy (T)
T = 0
for i in range(N):
    vx_i = sp.diff(cx[i], t)
    vy_i = sp.diff(cy[i], t)
    omega_i = sp.diff(thetas[i], t)
    T += sp.Rational(1, 2) * m * (vx_i**2 + vy_i**2) + sp.Rational(1, 2) * J * omega_i**2

# Anisotropic Viscous Ground Friction (Rayleigh Dissipation / Generalized Forces)
# Snakes move forward because normal friction (c_n) >> tangential friction (c_t)
Q_fric = sp.zeros(N + 2, 1)
for i in range(N):
    vx_i = sp.diff(cx[i], t)
    vy_i = sp.diff(cy[i], t)
    
    # Unit tangent and normal vectors for link i
    tx, ty = sp.cos(thetas[i]), sp.sin(thetas[i])
    nx, ny = -sp.sin(thetas[i]), sp.cos(thetas[i])
    
    # Velocity components in link frame
    v_t = vx_i * tx + vy_i * ty
    v_n = vx_i * nx + vy_i * ny
    
    # Friction force in global frame
    fx_i = -c_t * v_t * tx - c_n * v_n * nx
    fy_i = -c_t * v_t * ty - c_n * v_n * ny
    
    # Project onto generalized coordinates via virtual work (Jacobian transpose)
    for j in range(N + 2):
        Q_fric[j] += fx_i * sp.diff(cx[i], q[j]) + fy_i * sp.diff(cy[i], q[j])

# Joint Actuator Torques (u_1 ... u_{N-1} acting between adjacent links)
u = sp.Matrix(sp.symbols(f'u_0:{N-1}'))
Q_act = sp.zeros(N + 2, 1)
for i in range(N - 1):
    # Torque u_i acts positively on link i+1 and negatively on link i
    Q_act[2 + i] -= u[i]
    Q_act[2 + i + 1] += u[i]

# Euler-Lagrange Equations: d/dt(dT/ddq) - dT/dq = Q_fric + Q_act
# Rearrange into Mass Matrix form: M(q) * ddq = rhs(q, dq, u)
dT_ddq = sp.Matrix([sp.diff(T, dqi) for dqi in dq])
dT_dq = sp.Matrix([sp.diff(T, qi) for qi in q])
EL_lhs = dT_ddq.diff(t) - dT_dq

M_mat = EL_lhs.jacobian(ddq)
# Extract remaining Coriolis, centrifugal, and generalized force terms
coriolis_grav = EL_lhs.subs({ddqi: 0 for ddqi in ddq})
rhs_vec = Q_fric + Q_act - coriolis_grav

# ==========================================
# 2. LAMBDIFY FOR FAST EVALUATION
# ==========================================
# Replace time-dependent functions with plain symbols before lambdifying
q_syms = sp.symbols(f'q_0:{N+2}')
dq_syms = sp.symbols(f'dq_0:{N+2}')
subs_dict = {q[i]: q_syms[i] for i in range(N + 2)}
subs_dict.update({dq[i]: dq_syms[i] for i in range(N + 2)})

M_num_expr = M_mat.subs(subs_dict)
rhs_num_expr = rhs_vec.subs(subs_dict)

param_syms = (m, l, J, c_t, c_n)
M_func = sp.lambdify((q_syms, param_syms), M_num_expr, 'numpy')
rhs_func = sp.lambdify((q_syms, dq_syms, u, param_syms), rhs_num_expr, 'numpy')

# ==========================================
# 3. CONTROLLER & ODE INTEGRATION (solve_ivp)
# ==========================================
# Numerical parameters: m=1kg, l=0.5m, J=m*l^2/12, c_t=0.5, c_n=10.0
params = (1.0, 0.5, (1.0 * 0.5**2) / 12.0, 0.5, 10.0)

def serpenoid_controller(t, q_val, dq_val):
    """PD controller tracking a traveling sine wave (serpenoid curve)."""
    A = 0.6       # Wave amplitude (rad)
    omega = 4.0   # Temporal frequency (rad/s)
    delta = 1.2   # Phase shift between joints (rad)
    kp, kd = 15.0, 2.0
    
    u_val = np.zeros(N - 1)
    for i in range(N - 1):
        phi_actual = q_val[2 + i + 1] - q_val[2 + i]
        dphi_actual = dq_val[2 + i + 1] - dq_val[2 + i]
        
        phi_ref = A * np.sin(omega * t + i * delta)
        dphi_ref = A * omega * np.cos(omega * t + i * delta)
        
        u_val[i] = kp * (phi_ref - phi_actual) + kd * (dphi_ref - dphi_actual)
    return u_val

def snake_ode(t, state):
    q_val = state[:N+2]
    dq_val = state[N+2:]
    
    u_val = serpenoid_controller(t, q_val, dq_val)
    
    M_val = np.array(M_func(q_val, params), dtype=float)
    rhs_val = np.array(rhs_func(q_val, dq_val, u_val, params), dtype=float).flatten()
    
    # Solve M(q) * ddq = rhs numerically (much faster than symbolic M^-1)
    ddq_val = np.linalg.solve(M_val, rhs_val)
    
    return np.concatenate([dq_val, ddq_val])

# Initial state: straight along x-axis, at rest
y0 = np.zeros(2 * (N + 2))
t_span = (0, 10)
t_eval = np.linspace(0, 10, 300)

sol = solve_ivp(snake_ode, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)

# ==========================================
# 4. PLOT HEAD TRAJECTORY
# ==========================================
print("\nSimulation complete!")
print(f"Time shape: {sol.t.shape}")
#fuerza = 
data = np.vstack((sol.t, sol.y)).T
np.savetxt('snake.csv', data, delimiter=',', header='t,y', comments='')


plt.figure(figsize=(8, 4))
plt.plot(sol.y[0], sol.y[1], label='Head Trajectory (x, y)')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title(f'{N}-Link Snake Robot Locomotion')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
