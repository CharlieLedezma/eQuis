import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import solve_ivp


# https://medium.com/@bldevries/simply-solving-differential-equations-using-python-scipy-and-solve-ivp-f6185da2572d
# F = lambda t, s: np.dot(np.array([[0, t**2], [-t, 0]]), s)

# tl = np.arange(0, 10.0, 0.01)
# sol = solve_ivp(F, [0, 10], [1, 1], t_eval=tl)

# plt.figure(figsize = (12, 8))
# plt.plot(sol.y.T[:, 0], sol.y.T[:, 1])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

def fun(t, U):
  x, v = U
  return [v, -x]

U_0 = [0, 1]
t_pts = np.linspace(0, 15, 100)
#x = np.arange(0, 5*np.pi, 0.1)
result = solve_ivp(fun, (0, 20*math.pi), U_0, t_eval=t_pts)

plt.plot(result.y[0,:], label = "Numerical solution")
plt.plot([np.sin(t) for t in t_pts], "o", label="Analytical solution")


plt.xlabel("t")
plt.ylabel("x(t)")
plt.xlim(0,40)
plt.legend()

#print(result.y[0,:])
#print(len(t_pts))
print(result.y.shape)

#plt.show()