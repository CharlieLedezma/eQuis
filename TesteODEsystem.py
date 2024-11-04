import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import solve_ivp

#Solve de system
#
#	dA/dt = -k A B, A(0) = 2m
#	dB/dt = -k A B, B(0) = 1m
#	dC/dt = k A B, C(0) = 0m
#
#	where k = 0.1 for t=[0,50]
#
#	convert to standard form:
#
#	dy0/dt = -k y0 y1
#	dy1/dt = -k y0 y1
#	dy2/dt = -k y0 y1


def f(t,y):
	A = y[0]
	B = y[1]
	C = y[2]
	
	k = 0.1
	dA_dt = -k*A*B
	dB_dt = -k*A*B
	dC_dt = k*A*B
	return np.array([dA_dt,dB_dt,dC_dt])

t_span = np.array([0,50])
times = np.linspace(t_span[0],t_span[1],200)

y0 = np.array([2,1,0])
soln = solve_ivp(f, t_span, y0, t_eval=times)
t = soln.t
A = soln.y[0]

print(soln)