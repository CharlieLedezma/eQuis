import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# 1. Define Link Lengths (Ground, Crank, Coupler, Follower)
# Consistent with classic mechanical system problem formulations
r1 = 10.0  # Ground link (d)
r2 = 4.0   # Crank link (a)
r3 = 11.0  # Coupler link (b)
r4 = 8.0   # Follower link (c)

def loop_closure_equations(vars, theta2):
    """
    Defines the vector loop closure equations for a 4-bar mechanism.
    vars: vector containing unknown angles [theta3, theta4]
    theta2: input crank angle (radians)
    """
    theta3, theta4 = vars
    
    # Real (X) and Imaginary (Y) components of the vector loop
    eq1 = r2 * np.cos(theta2) + r3 * np.cos(theta3) - r4 * np.cos(theta4) - r1
    eq2 = r2 * np.sin(theta2) + r3 * np.sin(theta3) - r4 * np.sin(theta4)
    
    return [eq1, eq2]

# 2. Simulate over a full rotation of the crank (0 to 360 degrees)
theta2_range = np.linspace(0, 2 * np.pi, 100)
theta3_results = []
theta4_results = []

# Initial guess for [theta3, theta4] in radians
initial_guess = [np.radians(20), np.radians(60)]

for t2 in theta2_range:
    # Solve non-linear algebraic equations numerically
    solution = fsolve(loop_closure_equations, initial_guess, args=(t2,))
    theta3_results.append(np.degrees(solution[0]))
    theta4_results.append(np.degrees(solution[1]))
    # Update initial guess to the last solution for faster convergence
    initial_guess = solution

# 3. Plot the Kinematic Output
plt.figure(figsize=(10, 5))
plt.plot(np.degrees(theta2_range), theta3_results, label='Coupler Angle ($\\theta_3$)', color='blue')
plt.plot(np.degrees(theta2_range), theta4_results, label='Follower Angle ($\\theta_4$)', color='orange')
plt.title('Kinematic Analysis of a Four-Bar Linkage')
plt.xlabel('Crank Angle ($\\theta_2$) [degrees]')
plt.ylabel('Output Angles [degrees]')
plt.grid(True)
plt.legend()
plt.show()
