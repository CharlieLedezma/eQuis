from sympy import symbols
from sympy.utilities.lambdify import lambdify
import sympy as sym

""" x, y = symbols('x y')
expr = x + y

# Incorrect: missing the second argument when calling
f = lambdify((x, y), expr)
# f(5)  -> TypeError: missing 1 required positional argument: 'y'

# Correct: pass all required positional arguments
result = f(5, 10)
print(result) """

m, L, g, l = sym.symbols('m, L, g, l')
x, y = sym.symbols('x y')
t = sym.symbols("t")
q = sym.Array([sym.Function("w")(t),sym.Function("q_1")(t),sym.Function("q_2")(t)])
dq = q.diff(t)
dq
P1 = sym.Matrix([q[0]*sym.cos(q[1]), q[0]*sym.sin(q[1]), q[1]])

p1 = sym.lambdify([L, l, q[0], q[1], q[2]],P1, 'numpy')

result = p1(5,10,2,3,-1)
print(result)




