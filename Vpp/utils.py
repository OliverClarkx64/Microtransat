from foils import foil 
from foils import spring
import numpy as np
import matplotlib.pyplot as plt





def sum_F_M(foils, spring, AWA, AOA, beta, V, rho = 1.225): # beta is flap angle
    Lift = 0
    Drag = 0
    mmnt = 0
    for foil in foils:
        cl, L = foil.Lift(AOA, beta, V, rho)
        cd, D = foil.Drag(AOA, beta, V, rho)
        M = foil.moment(AOA, beta, V, rho)
        
        Lift += L
        Drag += D
        mmnt += M
    SA = AWA - AOA
    mmnt -= spring.moment(SA) #subtract moment from spring
    return Lift, Drag, mmnt


def Solve_beta(beta, *args): #solves trim angle for given AOA and AWA
    _, __, mmnt = sum_F_M(foils =args[0] , spring=args[1], AWA=args[2], AOA=args[3], beta=beta, V=args[5], rho=args[6])
    return mmnt
    

def solve_AOA(AOA, *args):
    _, __, mmnt = sum_F_M(foils =args[0] , spring=args[1], AWA=args[2], AOA=AOA, beta=args[4], V=args[5], rho=args[6])
    return mmnt




def plot_foils(foils, AWA, AOA, beta):
    Ca, Sa = np.cos(AOA), np.sin(AOA)
    Cb, Sb = np.cos(beta), np.sin(beta)
    Cw, Sw = np.cos(AWA), np.sin(AWA)

    Rota_A = np.array([[Ca, -Sa], [Sa, Ca]])
    Rota_B = np.array([[Cb, -Sb], [Sb, Cb]])
    Rota_W = np.array([[Cw, -Sw], [Sw, Cw]])

    for foil in foils:
        LE = np.array([-0.25*foil.Chord, 0])
        TE = np.array([0.75*foil.Chord, 0])
        if foil.type == "flap":
            LE = Rota_B @ LE
            TE = Rota_B @ TE
            P2 = np.array([foil.X, 0])
        LE = LE + np.array([foil.X, 0])
        TE = TE + np.array([foil.X, 0])

        LE = Rota_A @ LE
        TE = Rota_A @ TE
        plt.plot([LE[0], TE[0]], [LE[1], TE[1]], '-', label = foil.name)
    P2  = Rota_A @ P2
    FP, AP = np.array([-0.5, 0]), np.array([1.5, 0])
    FP, AP = Rota_W @ FP, Rota_W @ AP
    plt.plot([FP[0], AP[0]], [FP[1], AP[1]], 'k:')
    plt.plot(FP[0], FP[1], 'go', label = "FP")
    plt.plot(AP[0], AP[1], 'go', label = "AP")
    plt.plot([0, P2[0]], [0, P2[1]], 'k--', zorder=0)
    plt.plot(P2[0], P2[1], 'ro')
    plt.plot(0, 0 , 'ro')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()
