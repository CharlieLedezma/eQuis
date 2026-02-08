import numpy as np
from scipy.integrate import solve_ivp
from snake_physical_model import parameters, dynamic_model

np.set_printoptions(precision=4, suppress=True, linewidth=np.inf)


def main():

    objeto = parameters.Parameters()
    N = objeto.N
    t = objeto.t
    #N = params.N
    #t = params.t

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

    x0 = np.concatenate((qa.flatten(), qu.flatten(), qaDot.flatten(), quDot.flatten()))

    sol = solve_ivp(
         lambda t, y: dynamic_model.dynamic_model(t, y),
         [t[0], t[-1]],
         x0,
         method="RK45",
         vectorized=True,
         dense_output=True,
    )
    #print(dynamic_model.count)
    print("Hola")

if __name__ == "__main__":
    main()