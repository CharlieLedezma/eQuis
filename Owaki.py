#http://oscillex.org/wp-content/uploads/2019/01/RoboticsIIday9_material_c.pdf
from scipy.integrate import odeint
import numpy as np
import math
import sympy as sym
import matplotlib.pyplot as plt
#from sympy import Float

m_1 = 1.0
m_2 = 1.0
I_1 = 1.0
I_2 = 1.0
l_1 = 1.0
l_g1 = 0.5
l_g2 = 0.5
g = 9.8

params = [m_1, m_2, I_1, I_2, l_g1,l_1,l_g2, g]

#initial conditions
max_t = 20.0
dt = 0.1 

#theta_1 = sym.Symbol("theta_1")
#theta_2 = sym.Symbol("theta_2")

def Manipulator(p,t):
    
    theta_1, dtheta_1, theta_2, dtheta_2 = p
    M_11 = I_1 + I_1 + m_1*(1/2)*l_g1**2 + m_2*(l_1*l_1 + l_g2**2 + 2*l_1*l_g2*math.cos(theta_2))
    M_12 = I_2 + m_2*(l_g2**2 + l_1*l_g2*math.cos(theta_2))
    M_21 = I_2 + m_2*(l_g2**2 + l_1*l_g2*math.cos(theta_2))
    M_22 = I_2 + m_2*l_g2**2

    N_1 = -m_2*l_1*l_g2*dtheta_2*(2*dtheta_1 + dtheta_2)*math.sin(theta_2)
    N_2 = m_2*l_1*l_g2*dtheta_1*dtheta_1*math.sin(theta_2)

    G_1 = m_1*g*l_g1*math.cos(theta_1) + m_2*g*(l_1*math.cos(theta_1) + l_g2*math.cos(theta_1 +theta_2))
    G_2 = m_2*g*l_g2*math.cos(theta_1 +theta_2)

    #define matrix
    M = np.matrix([[M_11, M_12],[M_21, M_22]])
    N = np.matrix([[N_1],[N_2]])
    G = np.matrix([[G_1],[G_2]])

    #calc inverse matrix
    IM = np.linalg.inv(M)
    A = (-1)*IM.dot(N+G)


    ddtheta_1, ddtheta_2 = A
    return [dtheta_1, ddtheta_1, dtheta_2, ddtheta_2]

t = np.arange(0.0, max_t, dt)
x0 = [0.1*math.pi, 0.0, 0.1*math.pi, 0.0]
p = odeint(Manipulator, x0, t)

#Plot   
print("La dimension del vector solucions es:",p.shape)

#print(p[:,2])
#plt.plot(t,p[:,0],'b--')
#plt.plot(t,p[:,1],'b')
plt.plot(t,p[:,0],'r--')
plt.plot(t,p[:,1],'r')
plt.xlabel('Time')
plt.ylabel('Plot')
#plt.plot(t,theta_1,'b--',label= r'$\frac{d\theta_1}{dt} = \theta_2 $')
#plt.plot(t,theta_2,'r--',label= r'$\frac{d\theta_2}{dt} = -\frac{b}{m}\theta_2 -\frac{g}{l}sin\theta_1 $')
plt.xlim([0,10])
plt.legend(loc = 'best')
plt.show()