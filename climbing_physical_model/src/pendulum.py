import sympy as smp
from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols, mechanics_printing

# Enable readable physics printing notation
mechanics_printing(pretty_print=True)

# 1. Define time and physical constants
t = smp.Symbol('t')
m, l, g = smp.symbols('m l g', positive=True)

# 2. Define generalized coordinates (position) and velocities
# q = theta(t)
q = dynamicsymbols('q')
q_dot = dynamicsymbols('q', 1)  # First derivative w.r.t time

# 3. Formulate Kinetic (T) and Potential (V) Energy
# Example: Simple Pendulum link setup
T = smp.Rational(1, 2) * m * (l * q_dot)**2
V = -m * g * l * smp.cos(q)

# 4. Define the Lagrangian (L = T - V)
L = T - V

# 5. Apply the Euler-Lagrange Method
# Pass the Lagrangian and the list of generalized coordinates
LM = LagrangesMethod(L, [q])

# Derive the symbolic equations of motion
equations = LM.form_lagranges_equations()

print("Euler-Lagrange Equation of Motion:")
smp.pprint(equations)

print("\nDerived Mass Matrix:")
smp.pprint(LM.mass_matrix)

print("\nForcing Vector (Gravity / External Forces):")
smp.pprint(LM.forcing)
