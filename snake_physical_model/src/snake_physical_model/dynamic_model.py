import numpy as np
from numpy import sin, cos

#from sympy import *

class DynamicModel:
    def __init__(self, params):
        self.params = params
        self.count = 0

    def compute_auxiliary_matrices(self, N):
        A = np.zeros((N - 1, N))
        D = np.zeros((N - 1, N))
        J = np.zeros((N, N + 1))
        K = np.zeros((N, N + 1))
        J2 = np.zeros((N, N - 1))
        Jf = np.zeros((N, N - 1))

        for i1 in range(0,N-1):
            for i2 in range(0,N):
                if i1==i2:
                    A[i1,i2] = 1
                if i2==i1+1:
                    A[i1,i2] = 1
        for i1 in range(0,N-1):
            for i2 in range(0,N):
                if i1==i2:
                    D[i1,i2] = 1
                if i2==i1+1:
                    D[i1,i2] = -1

        Va= np.transpose(A) @ np.linalg.inv(D @ np.transpose(D)) @ A
        Ka= np.transpose(A) @ np.linalg.inv(D @ np.transpose(D)) @ D

        for i1 in range(0,N):
            for i2 in range(0,N+1):
                if i1==i2:
                    J[i1,i2] = -1
                if i2==i1+1:
                    J[i1,i2] = 1

        for i1 in range(0,N):
            for i2 in range(0,N+1):
                if i1==i2:
                    K[i1,i2] = 1

        for i1 in range(0,N):
            for i2 in range(0,N-1):
                if i1==i2:
                    J2[i1,i2] = 1
                if i1==i2+1:
                    J2[i1,i2] = 1

        for i1 in range(0,N):
            for i2 in range(0,N-1):
                if i1==i2:
                    Jf[i1,i2] = 1
                if i1==i2+1:
                    Jf[i1,i2] = -1


        HH = np.triu(np.ones((N, N)))
        J3 = -J2
        J1 = -Jf
        NN = J @ np.linalg.pinv(K)
        T = np.abs(NN)

        e = np.ones((N+1,1))
        e[N,0] = 0
        k = np.ones((N,1))
        j = np.zeros((N,1))
        j[N-1][0] = -1

        return A, D, Va, Ka, J, K, J2, Jf, HH, J3, J1, NN, T, j, e, k

    def dynamic_model(self, t, x):
        print(x.reshape(32))
        p = self.params
        N = p.N
        l = p.l
        m = p.m
        g = p.g
        alphaA = p.alphaA
        omega = p.omega
        delta = p.delta
        offset = p.offset
        kd = p.kd
        kp = p.kp
        umax = p.umax
        I = p.I

        A, D, Va, Ka, J, K, J2, Jf, HH, J3, J1, NN, T, j, e, k = self.compute_auxiliary_matrices(N)

        phi = x[:N]
        p = x[N:N+2]
        phiDot = x[N+2:2*N+2]
        pDot = x[2*N+2:]

        theta = HH @ phi
        thetaDot = HH @ phiDot

        # s_vect = np.zeros((N,1))
        # c_vect = np.zeros((N,1))
        Cm = np.zeros((N,N))
        Sm = np.zeros((N,N))
        # sgn = np.zeros((N,1))
        # dThetaqSqared = np.zeros((N,1))

        # for o1 in range(0,N):
        #     s_vect[o1,0] = sin(Float(theta[o1]))
        #     c_vect[o1,0] = cos(Float(theta[o1]))
        #     sgn[o1,0] = sign(Float(theta[o1]))
        #     dThetaqSqared[o1,0] = Float(thetaDot[o1]**2)
        for o1 in range(0,N):
            for o2 in range(0,N):
                if (o1==o2):
                    Cm[o1,o2]=cos(float(theta[o2]))
                    Sm[o1,o2]=sin(float(theta[o2]))

        A = J@J.T
        B = (1/N)*j
        C = (1/N)*j.T
        D = (1/N)


        DD = float(np.linalg.inv(D-C@np.linalg.inv(A)@B))                   #Return double (escalar)
        AA = np.linalg.inv(A)+(np.linalg.inv(A)@B)@(DD*C@np.linalg.inv(A))  #Return matrix NxN
        BB =  -DD*(np.linalg.inv(A))@B
        CC = -DD*C@np.linalg.inv(A)

        # ==============================================================================================================
        # Dynamic model
        # ==============================================================================================================

        # Initialize u as a zero array of size N-1
        u = np.zeros((N-1, 1))

        # Loop for phi, phiDot, phiDotDot calculations and control input u calculation
        for i in range(1, N):  # Python uses 0-based indexing
            phi_required = alphaA * np.sin(omega * t + (i-1) * delta) + offset
            phiDot_required = alphaA * omega * np.cos(omega * t + (i-1) * delta)
            phiDotDot_required = -alphaA * omega**2 * np.sin(omega * t + (i-1) * delta)

            u[i-1, 0] = phiDotDot_required + kp * (phi_required - phi[i]) + kd * (phiDot_required - phiDot[i])

            u[i-1, 0] = np.clip(u[i-1, 0], -umax, umax)


        # # Define matrices M and W
        M = I * np.eye(N) + m * l**2 * Sm * Va * Sm + m * l**2 * Cm * Va * Cm
        W = m * l**2 * Sm * Va * Cm - m * l**2 * Cm * Va * Sm

        # # Define phi2 and phi2Dot
        # phi2 = phi.copy()
        # phi2[N-1, 0] = theta[N-1]
        phi2 = phi.reshape(-1, 1)
        phi2[N-1, 0] = theta[N-1]

        # phi2Dot = phiDot.copy()
        # phi2Dot[N-1, 0] = thetaDot[N-1]
        phi2Dot = phiDot.reshape(-1, 1)
        phi2Dot[N-1, 0] = thetaDot[N-1]

        # # Define i1M, WWW, GGG, and BBB
        i1M = np.block([
          [HH.T @ M @ HH, np.zeros((N, 2))],
          [np.zeros((2, N)), N * m * np.eye(2)]
        ])

        WWW = np.block([
          [HH.T @ W @ np.diagflat(HH @ phiDot.reshape(-1, 1)) @ HH @ phiDot.reshape(-1 ,1)],
          [np.zeros((2, 1))]
        ])

        GGG = np.block([
          [-l * HH.T @ Sm @ Ka, l * HH.T @ Cm @ Ka],
          [-k.T, np.zeros((1, N))],
          [np.zeros((1, N)), -k.T]
        ])

        BBB = np.block([
          [np.eye(N-1, N-1)],
          [np.zeros((3, N-1))]
        ])

        # i11 = i1M[0:N-1, 0:N-1]
        # i12 = i1M[0:N-1, N:N+2]
        M21 = i1M[N-1:N+2, 0:N-1]
        M22 = np.array(i1M[N-1:N+2, N-1:N+2], dtype=np.float64)

        W1  = WWW[0:N-1]
        W2  = WWW[N-1:N+2]
        # G1  = GGG[0:N-1, 0:2*N]
        G2  = GGG[N:N+2, 0:2*N]

        Aq = -np.linalg.inv(M22) @ (W2)
        Bq = -np.linalg.inv(M22) @ M21

        xDot = np.vstack([phiDot.reshape(-1 ,1), pDot.reshape(-1 ,1), u, Aq + Bq @ u])

        self.count += 1

        #return xDot
        #return self.count
        
