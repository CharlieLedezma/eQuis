import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys

#dynamic_model:

    #def __init__(self):
N = 9  # Número de elos do robô
m = 0.406  # Massa do elo do robô
l = 0.0525  # Raio do elo do robô
g = 9.81  # Aceleração gravitacional
diameter = 0.3  # Diâmetro do tubo
diameterInfluence = 0.10
pipeLength = 3
shotsNumber = 50
dt = 0.01
ct = 0.015
cn = 0.03
ut = 0.15
un = 0.3
ctPipe = 0.08
utPipe = 0.2
umax = 3
Erub = 400000
vrub = 0.49
friction = 1
contact = 1
minLinkVel = 0.001
dimensionPlot3D = 0
resultsShow = 0
alphaA = 0.3981
omega = 0.6936
delta = 0.4914
offset = 0
acumulador = 0

kp  = 25
kd  = 10
tmax = 2
ti = 0
tf = 20
# self.npt = 200
t = np.arange(ti, tf, dt)

# if yaml_file:
#     self.load_from_yaml(yaml_file)

d = diameter - 2*l
qmax = 400 * dt
I = (m*(2*l)**2)/3


def compute_auxiliary_matrices(N):

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

    #acc = 0
    #acc = acc + 1

    return A, D, Va, Ka, J, K, J2, Jf, HH, J3, J1, NN, T, j, e, k


Cm = np.zeros((N,N))
Sm = np.zeros((N,N))
A, D, Va, Ka, J, K, J2, Jf, HH, J3, J1, NN, T, j, e, k = compute_auxiliary_matrices(N)
u = np.zeros((N-1, 1))

def dynamic_model( t, x):

    #size of x = (8,1)
    #print("shape of x:", type(x))
    phi = x[:N]         #First
    p = x[N:N+2]
    phiDot = x[N+2:2*N+2]
    pDot = x[2*N+2:]

    theta = HH @ phi
    thetaDot = HH @ phiDot


    ##################VOY POR AQUI ################

    # s_vect = np.zeros((N,1))
    # c_vect = np.zeros((N,1))



    #Cm = np.zeros((N,N))
    #Sm = np.zeros((N,N))
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
                Cm[o1,o2]=np.cos(theta[o2].item())
                Sm[o1,o2]=np.sin(theta[o2].item())
                #Cm[o1,o2]=np.cos(float(theta[o2]))
                #Sm[o1,o2]=np.sin(float(theta[o2]))


    A = J@J.T
    B = (1/N)*j
    C = (1/N)*j.T
    D = (1/N)

    #print("el valor de la matriz A es:", Cm)



    #DD = float(np.linalg.inv(D-C@np.linalg.inv(A)@B))                   #Return double (escalar)
    DD = np.linalg.inv(D-C@np.linalg.inv(A)@B)
    AA = np.linalg.inv(A)+(np.linalg.inv(A)@B)@(DD*C@np.linalg.inv(A))  #Return matrix NxN
    BB =  -DD*(np.linalg.inv(A))@B
    CC = -DD*C@np.linalg.inv(A)


    #u = np.zeros((N-1, 1))

    # Loop for phi, phiDot, phiDotDot calculations and control input u calculation
    for i in range(1, N):  # Python uses 0-based indexing
        phi_required = alphaA * np.sin(omega * t + (i-1) * delta) + offset
        phiDot_required = alphaA * omega * np.cos(omega * t + (i-1) * delta)
        phiDotDot_required = -alphaA * omega**2 * np.sin(omega * t + (i-1) * delta)

        u[i-1, 0] = phiDotDot_required + kp * (phi_required - phi[i].item()) + kd * (phiDot_required - phiDot[i].item())

        u[i-1, 0] = np.clip(u[i-1, 0], -umax, umax)


    # # Define matrices M and W
    M = I * np.eye(N) + m * l**2 * Sm * Va * Sm + m * l**2 * Cm * Va * Cm    
    W = m * l**2 * Sm * Va * Cm - m * l**2 * Cm * Va * Sm

    # # Define phi2 and phi2Dot
    # phi2 = phi.copy()
    # phi2[N-1, 0] = theta[N-1]
    phi2 = phi.reshape(-1, 1)
    phi2[N-1, 0] = theta[N-1].item()

    # phi2Dot = phiDot.copy()
    # phi2Dot[N-1, 0] = thetaDot[N-1]
    phi2Dot = phiDot.reshape(-1, 1)
    phi2Dot[N-1, 0] = thetaDot[N-1].item()

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

    #Dot.shape = (22, 1)

    return xDot


theta = np.zeros((N, 1))
thetaDot = np.zeros((N, 1))
phi = np.zeros((1, N - 1))
phiDot = np.zeros((N - 1, 1))
p = np.zeros((2, 1))
pDot = np.zeros((2, 1))

qa = phi.T
qu = np.array([theta[-1, 0], p[0, 0], p[1, 0]])
qaDot = phiDot
quDot = np.array([thetaDot[-1, 0], pDot[0, 0], pDot[1, 0]])

t_span = (0.0, 10.0)
t_eval = np.linspace(t_span[0], t_span[1], 200)
x0 = np.concatenate((qa.flatten(), qu.flatten(), qaDot.flatten(), quDot.flatten()))
sol = solve_ivp(dynamic_model, t_span, x0, t_eval=t_eval, method='RK45', vectorized=True, dense_output=True)



print(print(f"Simulation completed successfully with {len(sol.t)} time steps!"))       
data = np.vstack((sol.t, sol.y)).T
np.savetxt('snake.csv', data, delimiter=',', header='t,y', comments='')        

print(u.shape)



# #win 1
# fig1, ax1 = plt.subplots()
# for i  in range(1, N):
#     phi_required_plot = alphaA * np.sin(omega * sol.t + (i-1) * delta) + offset
#     # ax1.plot(sol.t, phi_required_plot, label='($q_{i}$)')
#     ax1.plot(sol.t, phi_required_plot, label=f"q_{i}")
# ax1.set_title('Reference Serpenoide Plot')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Angular Displacement(m)-Speed(rad)')
# ax1.legend()
# ax1.grid(True)



#win 2
fig1, ax5 = plt.subplots()
#for i  in range(1, N):
#phi_required_plot = alphaA * np.sin(omega * sol.t + (2-1) * delta) + offset
ax5.plot(sol.t, alphaA * np.sin(omega * sol.t + (1-1) * delta) + offset, label='Joint 1 ($phi_required$)')
ax5.plot(sol.t, sol.y[0], label=' 1 ($y[0]$)')
#ax5.plot(sol.t, sol.y[1], label='Joint 1 ($y[1]$)')
ax5.set_title('Reference Plot')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Angular Displacement(m)-Speed(rad)')
ax5.legend()
ax5.grid(True)




# #Window 1
# fig4, ax4 = plt.subplots()
# #ax1.figure(figsize=(10, 6))
# ax4.plot(sol.t, sol.y[9], label=' ($q_19$)')
# ax4.plot(sol.t, sol.y[10], label=' ($q_19$)')
# ax4.set_title('Robot Displacement P Point ')
# ax4.set_xlabel('Time (s)')
# ax4.set_ylabel('Displacement(m)-Speed(m/s)')
# ax4.legend()
# ax4.grid(True)




# # Window 2
fig2, ax2 = plt.subplots()
#ax2.figure(figsize=(10, 6))
ax2.plot(sol.t, sol.y[0], label=' ($q_1$)')
ax2.plot(sol.t, sol.y[1], label=' ($q_2$)')
ax2.plot(sol.t, sol.y[2], label=' ($q_3$)')
ax2.plot(sol.t, sol.y[3], label=' ($q_4$)')
ax2.plot(sol.t, sol.y[4], label=' ($q_5$)')
ax2.plot(sol.t, sol.y[5], label=' ($q_6$)')
ax2.plot(sol.t, sol.y[6], label=' ($q_7$)')
ax2.plot(sol.t, sol.y[7], label=' ($q_8$)')
ax2.plot(sol.t, sol.y[8], label=' ($q_9$)')
ax2.set_title('Links Angle Variations')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angle(o)')
ax2.legend()
ax2.grid(True)


# # Window 3
fig3, ax3 = plt.subplots()
# #ax1.figure(figsize=(10, 6))
ax3.plot(sol.t, np.rad2deg(sol.y[11]), label=' ($q_1$)')
ax3.plot(sol.t, np.rad2deg(sol.y[12]), label=' ($q_2$)')
ax3.plot(sol.t, np.rad2deg(sol.y[13]), label=' ($q_3$)')
ax3.plot(sol.t, np.rad2deg(sol.y[14]), label=' ($q_4$)')
ax3.plot(sol.t, np.rad2deg(sol.y[15]), label=' ($q_5$)')
ax3.plot(sol.t, np.rad2deg(sol.y[16]), label=' ($q_6$)')
ax3.plot(sol.t, np.rad2deg(sol.y[17]), label=' ($q_7$)')
ax3.plot(sol.t, np.rad2deg(sol.y[18]),  label='($q_8$)')
ax3.plot(sol.t, np.rad2deg(sol.y[19]), label=' ($q_9$)')
ax3.set_title('Links Angular Velocity')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Angle(o/s)')
ax3.legend()
ax3.grid(True) 

plt.show()