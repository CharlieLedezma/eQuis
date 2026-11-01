import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sym
from sympy import diff
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
from sympy.utilities.lambdify import lambdastr

""" class climbing_robot_dynamic:
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
        y1 = self.q1*np.sin(self.q2) """

"""
Defines the first-order ODEs for a 2DOF Lagrangian system.'
state = [q1, q2, q3, q1_dot, q2_dot, q3_dot]
"""

class climbing_robot_dynamic:
    #Symbolic representations, try in jupyter
    def __init__(self):

        #q = state[:3] 
        #q_dot = state[3:]
         
        m, L, g, l = sym.symbols('m, L, g, l')        
        t = sym.symbols("t")
        q = sym.Array([sym.Function("w")(t),sym.Function("q_1")(t),sym.Function("q_2")(t)])
        dq = q.diff(t)


        
        P1 = sym.Matrix([q[0]*sym.cos(q[1]), q[0]*sym.sin(q[1]), q[1]])
        P2 = sym.Matrix([q[0]-(L+l)*sym.cos(q[1]-q[2])*sym.cos(q[1]), q[0]-(L+l)*sym.cos(q[1]-q[2])*sym.sin(q[1]), q[2]])
        P3 = sym.Matrix([q[0]+(L+l)*sym.cos(q[1]-q[2])*sym.cos(q[1]), q[0]+(L+l)*sym.cos(q[1]-q[2])*sym.sin(q[1]), 2*q[1]-q[2]])

        #Jacobian calc
        I1 = m * 2*L ** 3 / 12  #Check Inertia to link 1
        I2 = m * 2*L ** 2 / 12
        I3 = m * 2*L ** 2 / 12

        # inertia matrices
        G1 = np.diag(np.array([m, m, I1]))
        G2 = np.diag(np.array([m, m, I2]))
        G3 = np.diag(np.array([m, m, I3]))

        J1 = P1.jacobian(q)
        J2 = P2.jacobian(q)
        J3 = P3.jacobian(q)
        # gravity
        G = 9.8

        # mass matrix
        M = J1.transpose() * G1 * J1 + J2.transpose() * G2 * J2 + J3.transpose() * G3 * J3

        # Christoffel symbols and Coriolis matrix
        dMdq = M.diff(q)
        Γ = sym.permutedims(dMdq, (2, 1, 0)) - 0.5 * dMdq
        C = sym.tensorcontraction(sym.tensorproduct(dq, Γ), (0, 2)).tomatrix()

        # gravity vector
        V = G * (m * P1[1] + m * P2[1] + m * P3[1])
        g = V.diff(q)

        tau = np.array([0.0, 0.0, 0.0])

        #self.symbolic_rhs =  C - V

        self.mass_matrix = sym.lambdify((m,L,l,q[0],q[1],q[2]),M, 'numpy')
        #christoffel_matrix = sym.lambdify([q], Γ)
        self.coriolis_matrix = sym.lambdify((m,L,l,q[0],q[1],q[2],dq[0],dq[1],dq[2]),C, 'numpy')
        self.gravity_vector = sym.lambdify((m,L,l,q[0],q[1],q[2]),g, 'numpy')

        self.rhs = tau - self.coriolis_matrix  - self.gravity_vector



    def rhs(self, t, y):  
        q = y[:3]
        q_dot = y[3:]   
        q_ddot = np.linalg.solve(self.mass_matrix, self.rhs)

        #return[q1_dot, q2_dot, q3_dot]
        return np.concatenate((q_dot, q_ddot))
    

    
    def solve(self, initial_state, t_span, k_val):
         #initial_state = [q0_0 ,]
         sol = solve_ivp(
             fun=self.rhs, 
             t_span=t_span, 
             y0= initial_state, 
             method='RK45',        # Excellent default solver for non-stiff dynamics
             args=(k_val,), 
             t_eval=np.linspace(t_span[0], t_span[1], 100),
             rtol=1e-6,
             atol=1e-8
             )
        


if __name__ == "__main__":

    robot = climbing_robot_dynamic()
    # Parameters: [m, L, l, g]
    sys_params = (1.0, 2.0, 1.0, 9.81)
    y0_initial = [0.0, 0.1*np.pi, 0.4*np.pi, 1.0, 0.0, 0.0]
    time_span = (0.0, 5.0)
    # Run the solver
    solution = robot.solve(initial_state=y0_initial, t_span=time_span, k_val=sys_params)


""" # Execute numerical integration
data = np.vstack((sol.t, sol.y)).T
np.savetxt('climbing_sol.csv', data, delimiter=',', header='t,y', comments='') """