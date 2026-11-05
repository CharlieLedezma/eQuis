import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =============================================================================
# 1. SYMBOLIC DERIVATION (SymPy)
# =============================================================================
print("Deriving Equations of Motion symbolicaly...")

# Define time and generalized coordinates as functions of time
t = sp.symbols('t')
q = sp.Function('q')(t)         # Joint position (angle)
dq = sp.Function('dq')(t)       # Joint velocity (first derivative)
ddq = sp.Function('ddq')(t)     # Joint acceleration (second derivative)

# Define system physical parameters and control input (Torque)
m, l, g = sp.symbols('m l g')
tau = sp.symbols('tau')         # Control torque applied at the joint

# --- Kinematics & Energy ---
# Position of the center of mass (assuming a uniform rod)
x = (l / 2) * sp.sin(q)
y = -(l / 2) * sp.cos(q)

# Velocities (Time derivatives)
vx = sp.diff(x, t)
vy = sp.diff(y, t)

# Kinetic Energy (T) and Potential Energy (V)
# Including both translational kinetic energy and rotational kinetic energy (I = 1/12 * m * l^2)
I = (1 / 12) * m * l**2
T = (1 / 2) * m * (vx**2 + vy**2) + (1 / 2) * I * dq**2
V = m * g * y

# Lagrangian (L = T - V)
L = T - V

# --- Substitute velocity for differentiation convenience ---
# SymPy's diff handles explicit symbols easier than functions of time
q_s, dq_s, ddq_s = sp.symbols('q_s dq_s ddq_s')

# Substitute functions of time with static symbols for the Euler-Lagrange steps
L_sub = L.subs({sp.diff(q, t): dq_s, q: q_s})

# Euler-Lagrange Equation: d/dt(dL/ddq) - dL/dq = tau
# 1. Partial derivative with respect to velocity
dL_ddq = sp.diff(L_sub, dq_s)
# To take the total time derivative, change symbols back to time functions
dL_ddq_t = dL_ddq.subs({dq_s: sp.diff(q, t), q_s: q})
d_dt_dL_ddq = sp.diff(dL_ddq_t, t)

# 2. Partial derivative with respect to position
dL_dq = sp.diff(L_sub, q_s).subs({dq_s: sp.diff(q, t), q_s: q})

# 3. Combine into the full equation of motion
eq_motion = sp.Eq(d_dt_dL_ddq - dL_dq, tau)

# --- Solve for Acceleration (ddq) ---
# Substitute back static symbols to make it easily lambdifiable
eq_motion_static = eq_motion.subs({sp.diff(q, t, 2): ddq_s, sp.diff(q, t): dq_s, q: q_s})
acceleration_expr = sp.solve(eq_motion_static, ddq_s)[0]

print(f"Derived Joint Acceleration (ddq): {acceleration_expr}\n")

# Convert the symbolic expression into a fast numerical function
# Inputs: (q, dq, tau, m, l, g)
numeric_acceleration = sp.lambdify((q_s, dq_s, tau, m, l, g), acceleration_expr, 'numpy')


# =============================================================================
# 2. CONTROL LAW & NUMERICAL INTEGRATION (SciPy)
# =============================================================================
# Fixed physical parameters
M_VAL = 1.0   # kg
L_VAL = 1.0   # meters
G_VAL = 9.81  # m/s^2

# Control Gains (PD Controller to bring the arm to a target angle)
KP = 15.0
KD = 5.0
Q_DESIRED = 0.5  # Target angle in radians (~28.6 degrees)

def control_law(q_current, dq_current):
    """
    Proportional-Derivative (PD) Control Law.
    Calculates required torque to minimize error.
    """
    error = Q_DESIRED - q_current
    error_dot = 0.0 - dq_current  # Target velocity is 0
    
    # Torque = Kp*e + Kd*e_dot
    u_torque = KP * error + KD * error_dot
    return u_torque

def system_dynamics(t, state):
    """
    State space equations for solve_ivp.
    state = [q, dq]
    returns = [dq, ddq]
    """
    q_curr, dq_curr = state
    
    # 1. Compute control action (torque) based on current state
    u_torque = control_law(q_curr, dq_curr)
    
    # 2. Evaluate physical system acceleration using our SymPy derived equation
    ddq_curr = numeric_acceleration(q_curr, dq_curr, u_torque, M_VAL, L_VAL, G_VAL)
    
    return [dq_curr, ddq_curr]

# --- Solve Initial Value Problem (IVP) ---
t_span = (0.0, 5.0)       # Simulate for 5 seconds
t_eval = np.linspace(0, 5, 500)
initial_state = [0.0, 0.0] # Starts at rest at 0 radians

print("Simulating closed-loop system dynamics...")
sol = solve_ivp(system_dynamics, t_span, initial_state, t_eval=t_eval, method='RK45')

# =============================================================================
# 3. POST-PROCESSING & PLOTTING
# =============================================================================
# Calculate control inputs over time for visualization
torques = [control_law(sol.y[0, i], sol.y[1, i]) for i in range(len(sol.t))]

plt.figure(figsize=(10, 5))

# Plot Position Trajectory
plt.subplot(1, 2, 1)
plt.plot(sol.t, sol.y[0], label='Actual Angle ($q$)', color='blue', linewidth=2)
plt.axhline(y=Q_DESIRED, color='r', linestyle='--', label='Desired Angle ($q_d$)')
plt.title('Joint Position Tracking')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.grid(True)
plt.legend()

# Plot Control Effort (Torque)
plt.subplot(1, 2, 2)
plt.plot(sol.t, torques, label='Control Torque ($\\tau$)', color='purple', linewidth=2)
plt.title('Control Torque Applied')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
