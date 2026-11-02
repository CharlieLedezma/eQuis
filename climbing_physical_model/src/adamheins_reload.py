import numpy as np
import math
#from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import sympy as sym
from sympy import diff
from sympy.matrices import Matrix
from sympy.physics.mechanics import dynamicsymbols, init_vprinting
import matplotlib.pyplot as plt
#from dataclasses import dataclass

class parameters:
    def __init__(self, m1, m2, m3, lx, ly, l2, l3, g ):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.lx = lx
        self.ly = ly
        self.l2 = l2
        self.l3 = l3
        self.g = g


class adamheis:

    def __init__(self, params:parameters):
    # Define generalized coordinates q and velocities dq
        q1, q2, q3 = sym.symbols('q1 q2 q3')
        dq1, dq2, dq3 = sym.symbols('dq1 dq2 dq3')
        self.q = [q1, q2, q3]
        self.dq = [dq1, dq2, dq3] 
        t = sym.symbols("t")
        #self.x, self.y = sp.symbols('x y')

        self.n = len(self.q)
        #self.params = params

        # momets of inertia (base link doesn't rotate so don't need it's inertia)
        self.I2 = params.m2 * params.l2 ** 2 / 12
        self.I3 = params.m2 * params.l3 ** 2 / 12



    def kinematic(self, params:parameters):
        P1 = sym.Matrix([self.q[0], 0, 0])
        P2 = P1 + sym.Matrix([params.lx + 0.5 * params.l2 * sym.cos(self.q[1]),params.ly + 0.5 * params.l2 * sym.sin(self.q[1]),self.q[1],])
        P3 = P2 + sym.Matrix([0.5 * params.l2 * sym.cos(self.q[1]) + 0.5 * params.l3 * sym.cos(self.q[1] + self.q[2]),0.5 * params.l2 * sym.sin(self.q[1]) + 0.5 * params.l3 * sym.sin(self.q[1] + self.q[2]),
        self.q[2]])
        return [P1, P2, P3]
        
        #Kin = kinematic()

    def jacobian(self, params:parameters):
        K= self.kinematic(params)        
        J1 = K[0].jacobian(self.q)
        J2 = K[1].jacobian(self.q)
        J3 = K[2].jacobian(self.q)   
        return [J1, J2, J3]   
    
    def mass_matrix(self, params:parameters):
        J = self.jacobian(params)
        G1 = np.diag(np.array([self.m1, self.m1, 0]))
        G2 = np.diag(np.array([self.m2, self.m2, self.I2]))
        G3 = np.diag(np.array([self.m3, self.m3, self.I3]))
        M = J[0].transpose() * G1 * J[0] + J[1].transpose() * G2 * J[1] + J[2].transpose() * G3 * J[2]
        return M
    
    def colioris(self, params:parameters):
        M  = self.mass_matrix(params)
        n= self.n
        #MM = np.zeros((n, n))
        Gamma = [[[sym.Integer(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    term1 = sym.diff(M[i, j], self.q[k])
                    term2 = sym.diff(M[i, k], self.q[j])
                    term3 = sym.diff(M[j, k], self.q[i])
                    Gamma[i][j][k] = 0.5 * (term1 + term2 + term3)
        C = sym.zeros(n, n)
        for i in range(n):
            for j in range(n):
                c_ij = 0
                for k in range(n):
                    c_ij += Gamma[i][j][k] * self.dq[k]
                C[i, j] = sym.simplify(c_ij)
        return C
    


    def gravity_vector(self,params:parameters):
        P  = self.kinematic(params)
        V = params.g * (params.m1 * P[0][1] + params.m2 * P[1][1] + params.m3 * P[2][1])
        g = V.diff(sym.Matrix(self.q))
    
    def physical_model(self, params:parameters):
        M  = self.mass_matrix(params)
        C = self.colioris(params)

    
    def system_ode(self, t, y):
    # y contains [q1, q2, dq1, dq2]
        q1_val, q2_val, q3_val, dq1_val, dq2_val, dq3_val = y
        
        # Compute numerical acceleration array from lambdified expression
        # .flatten() converts the matrix output to a 1D array
        ddq_val = self.ddq_numeric(q1_val, q2_val, q3_val, dq1_val, dq2_val, dq3_val).flatten()
        
        # Return derivatives: [dq1, dq2, ddq1, ddq2]
        return [dq1_val, dq2_val, dq2_val, ddq_val[0], ddq_val[1], ddq_val[2]]
    
    def solve(self, t_span, t_eval, y0):
        sol = solve_ivp(
            fun=self.system_ode, 
            t_span=t_span, 
            y0= y0, 
            t_eval=t_eval, 
            method='RK45'
            #rtol=1e-6,
            #atol=1e-8
            )
        return sol




        
        # gravity
        G = 9.8





        

        #dMdq = M.diff(q)
        #Γ = sym.permutedims(dMdq, (2, 1, 0)) - 0.5 * dMdq
        #C = sym.tensorcontraction(sym.tensorproduct(dq, Γ), (0, 2)).tomatrix()

        #V = G * (M1 * P1[1] + M2 * P2[1] + M3 * P3[1])
        #g = V.diff(q)



        tau = sym.Matrix([0, 0, 0]) 
        rhs = tau - self.colioris()*sym.Matrix(self.dq) - g
        ddq_expr = self.M.inv() * rhs
        state_vars = [q1, q2, q3, dq1, dq2, dq3]
        self.ddq_numeric = sym.lambdify(state_vars,ddq_expr,'numpy')
    




     
if __name__ == "__main__":
    

    config = parameters(3.0, 2.0, 1.0, 0.1, 0.1, 1.0, 0.5, 9.81)
    objeto = adamheis(params=config)
    fun = objeto.jacobian(params=config)
    print(f"Time shape: {fun[2]}")

    #fun = object.kinematic(params=config)
    
    #arguments = (3.0, 2.0, 1.0, 0.1, 0.1, 1.0, 0.5, 9.81)   
    #initial_state = [0.0, np.deg2rad(90), np.deg2rad(60), 20.0, 0, 0]
    #t_span = (0, 1)
    #t_eval=np.linspace(t_span[0], t_span[1], 100)


    #solution = objeto.solve(t_span,t_eval, initial_state, sys_params)

    ##solution = objeto.solve(t_span,t_eval, initial_state)



    ####### Execute numerical integration  #######
    # print("\nSimulation complete!")
    # print(f"Time shape: {solution.t.shape}")
    # data = np.vstack((solution.t, solution.y)).T
    # np.savetxt('adamheis.csv', data, delimiter=',', header='t,y', comments='')

    # plt.figure(figsize=(10, 6))
    # plt.plot(solution.t, np.rad2deg(solution.y[4]), label='Joint 1 ($dq_2$)')
    # plt.plot(solution.t, np.rad2deg(solution.y[5]), label='Joint 2 ($dq_3$)')
    # #plt.plot(solution.t, solution.y[2], label='Joint 3 ($q_3$)')
    # plt.title('Car Displacement')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Displacement(m))')
    # plt.legend()
    # plt.grid(True)
    # plt.show()