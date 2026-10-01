import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class climbing_robot_dynamic:
    def __init__(self,q1, q2, L, I, m):
        self.q1 = q1
        self.q2 = q2
        self.L = L
        self.I = I
        self.m = m

#############################################################################
#                           Direct Kinematic
#############################################################################
    def directKinematic(self):
        x1 = self.q1*np.cos(self.q2)
        y1 = self.q1*np.sin(self.q2)