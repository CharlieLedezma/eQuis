import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp


class InitialValueProblemSolver:

    def __init__(self):
        # 1. Define symbolic variables
        t = sp.Symbol("t")
        y = sp.Symbol("y")
        k = sp.Symbol("k")  # parameter

        # 2. Define the symbolic differential equation: dy/dt = -k * y^2 + sin(t)
        self.sympy_rhs = -k * y**2 + sp.sin(t)

        # 3. Lambdify the expression into a fast numpy function
        # Order of arguments in lambda matches what we will pass in self.rhs
        self._lambdifed_func = sp.lambdify((t, y, k), self.sympy_rhs, "numpy")

    def rhs(self, t, y, k):
        """Wrapper method that matches the signature required by solve_ivp.

        solve_ivp passes 'y' as a numpy array, so we unpack it if necessary.
        """
        # self._lambdifed_func evaluates the numerical result
        return self._lambdifed_func(t, y[0], k)

    def solve(self, y0, t_span, k_val):
        """Solves the IVP using scipy's solve_ivp."""
        # Pass the instance method self.rhs.
        # Use 'args' to pass the constant parameter 'k' to the RHS function.
        sol = solve_ivp(
            fun=self.rhs, t_span=t_span, y0=[y0], args=(k_val,), t_eval=np.linspace(t_span[0], t_span[1], 100)
        )
        return sol


# --- Execution ---
if __name__ == "__main__":
    # Initialize the solver class
    solver = InitialValueProblemSolver()

    # Define simulation parameters
    y0_initial = 1.0  # Initial condition y(0) = 1
    time_span = (0, 10)  # Time from t=0 to t=10
    k_parameter = 0.5  # Decay constant

    # Run the solver
    solution = solver.solve(y0=y0_initial, t_span=time_span, k_val=k_parameter)

    # Print results
    print("Status:", solution.message)
    print("First 5 time steps:\n", solution.t[:5])
    print("First 5 solution values:\n", solution.y[0][:5])

