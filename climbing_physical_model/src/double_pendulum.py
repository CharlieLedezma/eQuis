import numpy as np
import sympy as sym
from scipy.integrate import solve_ivp

# ==========================================
# 1. Symbolic Derivation (SymPy)
# ==========================================
# Define time and generalized coordinates / velocities
t = sym.Symbol('t')
q1 = sym.Function('q1')(t)
q2 = sym.Function('q2')(t)

# Velocities (derivatives with respect to time)
q1_dot = q1.diff(t)
q2_dot = q2.diff(t)

# Accelerations (second derivatives)
q1_ddot = q1.diff(t, 2)
q2_ddot = q2.diff(t, 2)

# System constants (masses, lengths, gravity)
m1, m2, l1, l2, g = sym.symbols('m1 m2 l1 l2 g')

# Positions of the centers of mass (assuming uniform rods)
x1 = (l1 / 2) * sym.sin(q1)
y1 = -(l1 / 2) * sym.cos(q1)

x2 = l1 * sym.sin(q1) + (l2 / 2) * sym.sin(q2)
y2 = -l1 * sym.cos(q1) - (l2 / 2) * sym.cos(q2)

# Velocities of the centers of mass
vx1, vy1 = x1.diff(t), y1.diff(t)
vx2, vy2 = x2.diff(t), y2.diff(t)

# Kinetic (T) and Potential (V) Energy
T = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * m2 * (vx2**2 + vy2**2)
V = m1 * g * y1 + m2 * g * y2

# Lagrangian (L = T - V)
L = T - V

# Generalized forces (torques applied at joints)
tau1, tau2 = sym.symbols('tau1 tau2')

# Lagrange's Equations: d/dt(dL/dq_dot) - dL/dq = tau
eq1 = L.diff(q1_dot).diff(t) - L.diff(q1) - tau1
eq2 = L.diff(q2_dot).diff(t) - L.diff(q2) - tau2

# Solve for the accelerations explicitly (q_ddot = M^-1 * (tau - C - G))
accelerations = sym.solve([eq1, eq2], (q1_ddot, q2_ddot))

# Simplify the resulting expressions
q1_ddot_expr = sym.simplify(accelerations[q1_ddot])
q2_ddot_expr = sym.simplify(accelerations[q2_ddot])

# ==========================================
# 2. Vectorization & Numerical Conversion
# ==========================================
# Replace state functions with pure symbols for lambdification
q1_s, q2_s, q1_dot_s, q2_dot_s = sym.symbols('q1_s q2_s q1_dot_s q2_dot_s')

q1_ddot_numerical = q1_ddot_expr.subs({
    q1: q1_s, q2: q2_s, q1_dot: q1_dot_s, q2_dot: q2_dot_s
})
q2_ddot_numerical = q2_ddot_expr.subs({
    q1: q1_s, q2: q2_s, q1_dot: q1_dot_s, q2_dot: q2_dot_s
})

# Lambdify the acceleration functions for speed
calc_q1_ddot = sym.lambdify(
    (q1_s, q2_s, q1_dot_s, q2_dot_s, tau1, tau2, m1, m2, l1, l2, g), 
    q1_ddot_numerical, 'numpy'
)
calc_q2_ddot = sym.lambdify(
    (q1_s, q2_s, q1_dot_s, q2_dot_s, tau1, tau2, m1, m2, l1, l2, g), 
    q2_ddot_numerical, 'numpy'
)

# ==========================================
# 3. Setting Up the ODE Function for solve_ivp
# ==========================================
# System parameter values
PARAMS = {'m1': 1.0, 'm2': 1.0, 'l1': 1.0, 'l2': 1.0, 'g': 9.81}

def ode_system(t, state, params):
    q1_val, q2_val, q1_dot_val, q2_dot_val = state
    
    # Define control inputs (e.g., passive system / zero torques)
    # You can change these to active controllers like PD loops: tau = P*(q_des - q)
    t1_val = 0.0
    t2_val = 0.0
    
    # Evaluate accelerations
    q1_ddot_val = calc_q1_ddot(
        q1_val, q2_val, q1_dot_val, q2_dot_val, t1_val, t2_val,
        params['m1'], params['m2'], params['l1'], params['l2'], params['g']
    )
    q2_ddot_val = calc_q2_ddot(
        q1_val, q2_val, q1_dot_val, q2_dot_val, t1_val, t2_val,
        params['m1'], params['m2'], params['l1'], params['l2'], params['g']
    )
    
    # Return state derivatives: [q1_dot, q2_dot, q1_ddot, q2_ddot]
    return [q1_dot_val, q2_dot_val, q1_ddot_val, q2_ddot_val]

# ==========================================
# 4. Numerical Integration (IVP)
# ==========================================
# Initial state: [q1, q2, q1_dot, q2_dot] (angles in radians, dropped from horizontal)
initial_state = [np.pi/2, 0.0, 0.0, 0.0]
time_span = (0, 10)  # Simulate for 10 seconds
t_eval = np.linspace(0, 10, 500)  # Dense output points

solution = solve_ivp(
    ode_system, 
    time_span, 
    initial_state, 
    args=(PARAMS,), 
    t_eval=t_eval, 
    method='RK45'
)

print("Simulation finished successfully!")
print(f"Shape of output trajectories: {solution.y.shape}")
