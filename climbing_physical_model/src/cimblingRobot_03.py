import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sym
from sympy import diff
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
from sympy.utilities.lambdify import lambdastr


class climbing_robot_dynamic:
    #Symbolic representations, try in jupyter
    def __init__(self):
    # Define generalized coordinates q and velocities dq
        q1, q2, q3, dq1, dq2, dq3 = sym.symbols('q1 q2 q3 dq1 dq2 dq3')
        m1, L, phi, b, g = sym.symbols('m1 L phi b g')
        #m1, L, phi, b, g =  args
        #dq1, dq2, dq3 = sym.symbols('dq1 dq2 dq3')
        self.q = [q1, q2, q3]
        self.dq = [dq1, dq2, dq3] 
        #t = sym.symbols("t")

        self.n = len(self.q)
        


        self.m1 = m1
        self.b =b
        self.L = L
        self.phi = phi
        self.g = g         

        
        self.I1 = m1 * 2*L ** 3 / 12  #Check Inertia to link 1
        self.I2 = m1 * 2*L ** 2 / 12
        self.I3 = m1 * 2*L ** 2 / 12

        state_vars = [*self.q, *self.dq, self.m1, self.L, self.phi, self.b, self.g]
        self.params = {'m1': 1.0, 'L': 0.5, 'phi': np.rad2deg(30), 'b': 0.2, 'g': 9.81}

        #self.ddq_numeric = sym.lambdify(state_vars,ddq_expr,'numpy')

        #q[0] = q1
        #q[1] = q2
        #q[2] = q3

        # Lambdify for blistering fast numerical evaluation
        # Structure: f(q1, q2, dq1, dq2, m1, m2, l1, l2, g)
        M = self.mass_matrix()
        C = self.coriolis()
        G = self.gravity_vector()
        self.num_M = sym.lambdify((*self.q, *self.dq, self.m1, self.L, self.phi, self.b, self.g), M, 'numpy')
        self.num_C = sym.lambdify((*self.q, *self.dq, self.m1, self.L, self.phi, self.b, self.g), C, 'numpy')
        self.num_G = sym.lambdify((*self.q, *self.dq, self.m1, self.L, self.phi, self.b, self.g), G, 'numpy')



    def kinematic(self):
        P1 = sym.Matrix([self.q[0]*sym.cos(self.phi), self.q[0]*sym.sin(self.phi), self.phi])

        P2 = sym.Matrix([((self.q[0] - 2*self.L*sym.cos(self.phi-self.q[1]))*sym.cos(self.phi) + self.b*sym.cos(np.pi-self.phi) + self.L*sym.cos(self.q[1])), 
                         ((self.q[0] - 2*self.L*sym.cos(self.phi-self.q[1]))*sym.sin(self.phi) - self.b*sym.sin(np.pi-self.phi) + self.L*sym.sin(self.q[1])), 
                         self.q[1]])


        P3 = P2 + sym.Matrix([self.L* (sym.cos(self.q[1]) +  sym.cos(self.q[2])), 
                         self.L* (sym.sin(self.q[1]) +  sym.sin(self.q[2])), 
                         self.q[2]])
        
        return [P1, P2, P3]

    #Jacobian calc    
    def jacobian(self):
        K= self.kinematic()              
        J1 = K[0].jacobian(self.q)
        J2 = K[1].jacobian(self.q)
        J3 = K[2].jacobian(self.q)   
        return [J1, J2, J3]   
    
    #Mass matrix calc
    def mass_matrix(self):
        J = self.jacobian()
        G1 = np.diag(np.array([self.m1, self.m1, 0]))
        G2 = np.diag(np.array([self.m1, self.m1, self.I2]))
        G3 = np.diag(np.array([self.m1, self.m1, self.I3]))
        M = J[0].transpose() * G1 * J[0] + J[1].transpose() * G2 * J[1] + J[2].transpose() * G3 * J[2]
        return M
    
    def coriolis(self):
        M  = self.mass_matrix()
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
    

    def gravity_vector(self):
        P  = self.kinematic()
        V = self.g * (self.m1* P[0][1] + self.m1 * P[1][1] + self.m1 * P[2][1])
        g = V.diff(sym.Matrix(self.q))
        return g

    def get_dynamics(self, q_vec, dq_vec):
        args = (q_vec[0], q_vec[1], q_vec[2], dq_vec[0], dq_vec[1], dq_vec[2], self.params['m1'], self.params['L'], self.params['phi'], self.params['b'], self.params['g'])
        return self.num_M(*args), self.num_C(*args), self.num_G(*args)
      
    
    def desired_trajectory(t):
        """Defines target position, velocity, and acceleration profiles."""
        #q_d = np.array([5*t,np.deg2rad(35), np.deg2rad(35)])
        #dq_d = np.array([0,0,0])
        #ddq_d = np.array([0,0,0])

        #q_d = np.array([2, np.sin(t), np.cos(t)])
        #dq_d = np.array([0, np.cos(t), -np.sin(t)])
        #ddq_d = np.array([0, -np.sin(t), -np.cos(t)])
        #return q_d, dq_d, ddq_d


    def system_ode(self, t, y):
        # y contains [q1, q2, dq1, dq2]         
        q_curr = y[0:3]
        dq_curr = y[3:6]  

  
        # 1. Compute physical robot matrices
        M_mat, C_mat, G_mat = self.get_dynamics(q_curr, dq_curr)
        G_mat = G_mat.flatten() # Ensure 1D array consistency

        #print ("TiPO de G_mat,: ",type(G_mat))
        #print ("Tamano G_mat,: ",G_mat.shape)
        #print ("TiPO de M_mat,: ",type(M_mat))
        #print ("Tamano M_mat,: ",M_mat.shape)

        # 2. Compute Tracking Controller (Computed Torque Control / Feedback Linearization)
        #q_d, dq_d, ddq_d = self.desired_trajectory(t)

        q_d = np.array([1*t ,np.deg2rad(35), np.deg2rad(65)])
        dq_d = np.array([1,0,0])
        ddq_d = np.array([0,0,0])

        #q_d = np.array([3, np.sin(t), np.cos(t)])
        #dq_d = np.array([0, np.cos(t), -np.sin(t)])
        #ddq_d = np.array([0, -np.sin(t), -np.cos(t)])



        Kp = np.diag([30.0, 30.0, 50.0]) # Proportional Gain
        Kd = np.diag([20.0, 20.0, 20.0])   # Derivative Gain

        e = q_d - q_curr
        de = dq_d - dq_curr

        # Control law: Tau = M*(ddq_d + Kd*de + Kp*e) + C*dq + G

        aq = ddq_d + Kd @ de + Kp @ e
        tau = M_mat @ aq + C_mat @ dq_curr + G_mat
        #print ("Dimension de tau: ",tau.shape)

        # 3. Solve Forward Dynamics for Acceleration: ddq = M^-1 * (Tau - C*dq - G)
        ddq_curr = np.linalg.inv(M_mat) @ (tau - C_mat @ dq_curr - G_mat)
        

        return np.concatenate([dq_curr, ddq_curr])

    
    
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

if __name__ == "__main__":


    #arguments = (1.0, 0.7, 0.25*np.pi, 0.3, 9.81)

    robot = climbing_robot_dynamic()
    #print(robot.num_G(1.0, np.deg2rad(30), np.deg2rad(60), 0.0, 0.0, 0.0, 1.0, 0.3, 0.7, 0.25*np.pi, 9.81))


    initial_state  = [0.1, np.deg2rad(30), np.deg2rad(60), 0.0, 0.0, 0.0]
    t_span = (0, 10)
    t_eval=np.linspace(t_span[0], t_span[1], 100)
    solution = robot.solve(t_span,t_eval, initial_state)

    ####### Execute numerical integration  #######
    print("\nSimulation complete!")
    print(f"Time shape: {solution.t.shape}")
    data = np.vstack((solution.t, solution.y)).T
    np.savetxt('adamheis_evolution.csv', data, delimiter=',', header='t,y', comments='')

    # Window 1
    fig1, ax1 = plt.subplots()
    #ax1.figure(figsize=(10, 6))
    ax1.plot(solution.t, np.rad2deg(solution.y[3]), label='Joint 1 ($q_1$)')
    #ax1.plot(solution.t, np.rad2deg(solution.y[3]), label='Joint 2 ($dq_1$)')
    #ax1.plot(solution.t, np.deg2rad(35)*(solution.t/solution.t), label='Ref:($5*t)', linestyle='--')
    ax1.set_title('Robot Displacement and Speed over Pole ')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Displacement(m)-Speed(m/s)')
    ax1.legend()
    ax1.grid(True)

    

 
    # Window 2
    fig2, ax2 = plt.subplots()
    #ax2.figure(figsize=(10, 6))
    ax2.plot(solution.t, np.rad2deg(solution.y[1]), label='Joint 1 ($q_1$)')
    ax2.plot(solution.t, np.rad2deg(solution.y[2]), label='Joint 1 ($q_2$)')
    ax2.set_title('Links Angle Variations')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle(o)')
    ax2.legend()
    ax2.grid(True)


    # Window 3
    fig3, ax3 = plt.subplots()
    #ax1.figure(figsize=(10, 6))
    ax3.plot(solution.t, np.rad2deg(solution.y[4]), label='Joint 1 ($dq_2$)')
    ax3.plot(solution.t, np.rad2deg(solution.y[5]), label='Joint 2 ($dq_3$)')
    ax3.set_title('Links Angular Velocity')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle(o/s)')
    ax3.legend()
    ax3.grid(True) 

    plt.show()

    