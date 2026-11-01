import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

# 1. Define Symbolic Variables
t = sp.symbols('t')
q1, q2 = sp.symbols('q1 q2')
dq1, dq2 = sp.symbols('dq1 dq2')

# Group states into symbolic vectors
q = sp.Matrix([q1, q2])
dq = sp.Matrix([dq1, dq2])

# 2. Define your Symbolic Matrices (Example: 2-DOF Robot/System)
# Placeholder Mass Matrix (must be symmetric and positive-definite)
M = sp.Matrix([
    [10 + 5*sp.cos(q2), 2 + sp.cos(q2)],
    [2 + sp.cos(q2),  5]
])

# Placeholder Coriolis/Centrifugal Matrix
C = sp.Matrix([
    [-5*sp.sin(q2)*dq2, -5*sp.sin(q2)*(dq1 + dq2)],
    [5*sp.sin(q2)*dq1,  0]
])

# Placeholder Gravity / External Forces Vector
G = sp.Matrix([
    [9.81 * sp.sin(q1)],
    [9.81 * sp.sin(q1 + q2)]
])

# Control input or external torque vector (can be symbolic or constant)
tau = sp.Matrix([0, 0]) 

# 3. Symbolically Isolate the Acceleration Matrix (ddq)
# ddq = M^-1 * (tau - C*dq - G)
ddq_expr = M.inv() * (tau - C*dq - G)

# 4. Convert Symbolic Expressions to Fast Numerical Functions
# We bundle all state variables: [q1, q2, dq1, dq2]
state_vars = [q1, q2, dq1, dq2]
ddq_numeric = sp.lambdify(state_vars, ddq_expr, modules='numpy')

# 5. Define the State-Space ODE Function for solve_ivp
def system_ode(t, y):
    # y contains [q1, q2, dq1, dq2]
    q1_val, q2_val, dq1_val, dq2_val = y
    
    # Compute numerical acceleration array from lambdified expression
    # .flatten() converts the matrix output to a 1D array
    ddq_val = ddq_numeric(q1_val, q2_val, dq1_val, dq2_val).flatten()
    
    # Return derivatives: [dq1, dq2, ddq1, ddq2]
    return [dq1_val, dq2_val, ddq_val[0], ddq_val[1]]

# 6. Set Up and Solve the IVP
t_span = (0, 10)          # Time horizon: 0 to 10 seconds
t_eval = np.linspace(0, 10, 500)
y0 = [0.1, 0.0, 0.0, 0.0]  # Initial conditions: [q1, q2, dq1, dq2]

sol = solve_ivp(system_ode, t_span, y0, t_eval=t_eval, method='RK45')

# Access results
# sol.t -> time steps
# sol.y -> state trajectories (shape: 4, 500)
print("Integration successful. Time steps evaluated:", len(sol.t))
