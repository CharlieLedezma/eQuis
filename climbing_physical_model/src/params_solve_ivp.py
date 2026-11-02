import numpy as np
from scipy.integrate import solve_ivp

class GeneralizedLagrangeSystem:
    def __init__(self, num_coordinates):
        """
        Initializes the system for a given number of generalized coordinates.
        """
        self.n = num_coordinates

    def equations_of_motion(self, t, state, *params):
        """
        Defines the state-space equations: dy/dt = f(t, y, args)
        
        state vector: [q_1, q_2, ..., dq_1, dq_2, ...]
        params: Unpacked dynamically from the `args` tuple in solve_ivp
        """
        # Split the state into generalized coordinates (q) and velocities (dq)
        q = state[:self.n]
        dq = state[self.n:]
        
        # Unpack your system parameters (example: mass, length, gravity)
        m, l, g = params
        
        # --- Generalized Lagrange Framework ---
        # 1. Compute Mass Matrix M(q)
        # 2. Compute Generalized Forces vector C(q, dq) + G(q) + Q_ext
        # For demonstration, this represents a simple non-linear pendulum:
        M = np.array([[m * l**2]])
        C_G = np.array([m * g * l * np.sin(q[0])])
        
        # Solve for generalized accelerations: ddq = M^-1 * (-C_G)
        ddq = np.linalg.solve(M, -C_G)
        
        # Return the derivative of the state vector: [dq, ddq]
        return np.concatenate([dq, ddq])

    def simulate(self, t_span, t_eval, initial_conditions, system_args):
        """
        Solves the IVP using SciPy's solve_ivp.
        """
        # system_args must be passed as a tuple to the `args` parameter
        sol = solve_ivp(
            fun=self.equations_of_motion,
            t_span=t_span,
            y0=initial_conditions,
            t_eval=t_eval,
            args=system_args,  # <--- Passing the parameters here
            method='RK45'
        )
        return sol

# --- Execution ---
if __name__ == "__main__":
    # System: 1 Generalized coordinate (e.g., Pendulum Angle)
    sys = GeneralizedLagrangeSystem(num_coordinates=1)
    
    # Simulation Window
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 100)
    
    # Initial State: [Initial Angle (rad), Initial Angular Velocity (rad/s)]
    init_state = [np.radians(45), 0.0]
    
    # Arguments to pass: (Mass = 1.0kg, Length = 0.5m, Gravity = 9.81m/s^2)
    arguments = (1.0, 0.5, 9.81)
    
    # Run Simulation
    solution = sys.simulate(t_span, t_eval, init_state, arguments)
    
    print("Simulation status:", solution.message)
    print("Shape of results:", solution.y.shape)
