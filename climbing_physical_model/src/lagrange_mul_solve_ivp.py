import numpy as np
from scipy.integrate import solve_ivp

class GeneralizedLagrangeIVPSolver:
    def __init__(self, g=9.81, L=1.0):
        self.g = g
        self.L = L

    def constraint(self, q):
        """Holonomic constraint: C(q) = x^2 + y^2 - L^2 = 0"""
        x, y = q[0], q[1]
        return x**2 + y**2 - self.L**2

    def jacobian_C(self, q):
        """Gradient of the constraint: dC/dq = [2x, 2y]"""
        x, y = q[0], q[1]
        return np.array([2*x, 2*y])

    def mass_matrix(self):
        """Mass matrix M (assuming unit mass m=1 for simplicity)"""
        return np.eye(2)

    def external_forces(self, q, dq):
        """Unconstrained generalized forces (Gravity acting on y)"""
        return np.array([0.0, -self.g])

    def equations_of_motion(self, t, state):
        """
        Calculates the state derivatives [dq; ddq] using Generalized Lagrange Multipliers.
        State vector layout: [x, y, dx, dy]
        """
        q = state[0:2]
        dq = state[2:4]
        
        M = self.mass_matrix()
        F = self.external_forces(q, dq)
        J = self.jacobian_C(q)
        
        # To find acceleration (ddq) and the multiplier (lambda), we solve the DAE system:
        # [ M    J^T ] [ ddq ]   [      F      ]
        # [ J     0  ] [ lambda] = [ -dJ/dt * dq ]
        # Note: dJ/dt * dq for our constraint is: 2*(dx^2 + dy^2)
        
        rhs_constraint = -2.0 * np.sum(dq**2)
        
        # Construct the KKT matrix
        A = np.block([
            [M, J.reshape(2, 1)],
            [J.reshape(1, 2), 0.0]
        ])
        
        b = np.append(F, rhs_constraint)
        
        # Solve the linear system for [ddq, lambda]
        sol = np.linalg.solve(A, b)
        ddq = sol[0:2]
        # lam = sol[2] # The Lagrange Multiplier (tension force magnitude)
        
        # Return derivatives: [dx, dy, ddx, ddy]
        return np.concatenate([dq, ddq])

    def solve(self, t_span, y0, t_eval=None):
        """Wrapper around scipy's solve_ivp"""
        return solve_ivp(
            fun=self.equations_of_motion,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )

# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":
    # Initialize the generalized system
    L_rod = 1.5
    solver = GeneralizedLagrangeIVPSolver(g=9.81, L=L_rod)

    # Initial conditions: Pendulum released horizontally from the right
    # State format: [x, y, dx, dy]
    initial_state = [L_rod, 0.0, 0.0, 0.0]
    time_span = (0, 5)
    time_steps = np.linspace(0, 5, 500)

    # Solve the constrained IVP
    result = solver.solve(t_span=time_span, y0=initial_state, t_eval=time_steps)

    # Verify constraint satisfaction at the final step
    final_positions = result.y[0:2, -1]
    constraint_error = solver.constraint(final_positions)
    
    print(f"Simulation Successful: {result.success}")
    print(f"Final coordinates: x = {final_positions[0]:.4f}, y = {final_positions[1]:.4f}")
    print(f"Final constraint violation |x^2 + y^2 - L^2|: {abs(constraint_error):.2e}")
