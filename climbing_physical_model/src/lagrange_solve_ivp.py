import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

# =============================================================================
# 1. SYMBOLIC DERIVATION (SymPy)
# =============================================================================
# Define time and structural constants
t = sp.symbols("t")
g, l, m = sp.symbols("g l m", positive=True)

# Define generalized coordinates as functions of time
q = sp.Function("q")(t)
dq = q.diff(t)
d2q = q.diff(t, 2)

# Define Kinetic Energy (T) and Potential Energy (V)
T = sp.Rational(1, 2) * m * (l * dq) ** 2
V = -m * g * l * sp.cos(q)

# Lagrangian (L = T - V)
L = T - V

# Compute Euler-Lagrange Equation: d/dt(dL/ddq) - dL/dq = 0
dL_ddq = sp.diff(L, dq)
d_dt_dL_ddq = dL_ddq.diff(t)
dL_dq = sp.diff(L, q)

EOM = d_dt_dL_ddq - dL_dq

# Solve explicitly for the highest derivative (acceleration: d2q)
acceleration_expr = sp.solve(EOM, d2q)[0]
print(f"Derived Acceleration Equation: d2q/dt2 = {acceleration_expr}")

# =============================================================================
# 2. NUMERICAL CONVERSION (Lambdify)
# =============================================================================
# Substitute physical constants for numerical evaluation
constants = {g: 9.81, l: 1.0, m: 1.0}
acceleration_numeric_expr = acceleration_expr.subs(constants)

# Lambdify the expression into a fast NumPy function
# It takes state inputs: q and dq
calc_acceleration = sp.lambdify((q, dq), acceleration_numeric_expr, "numpy")

# Define the state-space system function for SciPy's solve_ivp
# State vector y = [position, velocity] -> [q, dq]
def system_dynamics(t, y):
    position, velocity = y
    dydt = [velocity, calc_acceleration(position, velocity)]
    return dydt


# =============================================================================
# 3. NUMERICAL INTEGRATION (SciPy)
# =============================================================================
# Initial conditions: [initial position (rad), initial velocity (rad/s)]
y0 = [np.pi / 4, 0.0]  # 45 degrees, released from rest
t_span = (0.0, 10.0)  # Integrate from t=0 to t=10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Output time steps

# Solve the Initial Value Problem
sol = solve_ivp(
    system_dynamics, t_span, y0, t_eval=t_eval, method="RK45", rtol=1e-8
)

# Access results via sol.t (time array) and sol.y (states array)
print("\nSimulation complete!")
print(f"Time shape: {sol.t.shape}")
print(f"Positions shape: {sol.y[0].shape}")
