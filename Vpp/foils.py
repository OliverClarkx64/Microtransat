import numpy as np
import warnings
from scipy.optimize import fsolve


class foil:
    def __init__(self, name, type = "wing", X = 0,
                Chord = None, 
                span  =  None,
                polars = None,
                P_type = "2D",
                tou = 0.1,
                delta = 0.04):


        self.X = X #location of foil COP with respect to mast step
        self.name = name
        self.type = type
        self.polars = polars
        self.Chord = Chord
        self.span = span
        self.P_type = P_type
        self.tou = tou
        self.delta = delta

        self.oef = 1/(1+self.delta)
        self.area = self.Chord * self.span
        self.aspect_ratio = self.span / self.Chord


#########################################

        
    def Lift(self, AOA, beta, V, rho = 1.225):
        if self.type == "flap":
            alpha = AOA + beta
        else:
            alpha = AOA
        alpha = alpha % (2*np.pi)

        Cl0 = np.interp(alpha, self.polars["alpha"], self.polars["Cl"])
        if self.P_type == "2D":
            Cl = Cl0 * 1 # add better correction maybe
        else:
            Cl = Cl0
        lift = 0.5*rho * Cl * self.area * V**2
        return Cl, lift

#########################################

    def Drag(self, AOA, beta, V, rho = 1.225):
        if self.type == "flap":
            alpha = AOA + beta
        else:
            alpha = AOA

        alpha = alpha % (2*np.pi)

        Cd0 = np.interp(alpha, self.polars["alpha"], self.polars["Cd"]) + 0.02 # approx skin friction


        if self.P_type == "2D":
            Cl = self.Lift(AOA, beta, V, rho)[0]
            Cdi = (Cl **2) / (np.pi * self.oef * self.aspect_ratio)
            Cd = Cd0 + Cdi
        else:
            Cd = Cd0
        drag = 0.5*rho * Cd * self.area * V**2
        return Cd, drag

#########################################

    def moment(self, AOA, beta, V, rho = 1.225):
        if self.type == "flap":
            alpha = AOA + beta
        else:
            alpha = AOA        
        _, L = self.Lift(AOA, beta, V, rho)
        _, D = self.Drag(AOA, beta, V, rho)
        M = L * self.X * np.cos(alpha) + D * self.X * np.sin(alpha)
        return M




    

class spring:
    def __init__(self, k, x0 = 0):
        self.k = k
        self.x0 = x0

    def moment(self, sheeting_angle):
        sign = 1 if sheeting_angle >= 0 else -1
        dL = self.x0 * np.sqrt(2 - 2 * np.cos(sheeting_angle)) # change in length of spring
        force = self.k * dL
        Moment = force * self.x0 * np.sin(90-0.5*abs(sheeting_angle)) * sign # moment about mast step
        return Moment
    

class rig:
    def __init__(self, foils, spring):
        self.foils = foils
        self.spring = spring

        AX = 0
        A = 0
        for foil in self.foils:
            if foil.type == "wing":
                ax = foil.area * foil.span * 0.5
                AX += ax
                A += foil.area
        self.Healing_arm = AX / A # distance from mast step to center of lift of wing foil(s)

    def sum_F_M(self, AWA, AOA, beta, V, rho = 1.225): # beta is flap angle
        Lift = 0
        Drag = 0
        mmnt = 0
        for foil in self.foils:
            cl, L = foil.Lift(AOA, beta, V, rho)
            cd, D = foil.Drag(AOA, beta, V, rho)
            M = foil.moment(AOA, beta, V, rho)
            
            Lift += L
            Drag += D
            mmnt += M
        SA = AWA - AOA
        mmnt -= self.spring.moment(SA) #subtract moment from spring
        return Lift, Drag, mmnt
   

    def AOA_func(self, AOA, *args):
        _, __, mmnt = self.sum_F_M(AWA=args[0], AOA=AOA, beta=args[1], V=args[2], rho=args[3])
        return mmnt

    def Solve_AOA(self, AOA0, AWA, beta, V, rho = 1.225):
        AOA = min(fsolve(self.AOA_func, x0 = [AOA0], args = (AWA, beta, V, rho)  )  )
        return AOA

    def solve_for_AWA(self, AWA,V, rho = 1.225):
        N = 500
        betas = np.linspace(-np.pi/2, np.pi/2, N)
        AOAs, Ls, Ds= np.zeros(N), np.zeros(N), np.zeros(N)
        for i in range(N):
            AOA = self.Solve_AOA(AOA0 = AWA, AWA = AWA, beta = betas[i], V = V, rho = rho)
            AOAs[i] = AOA
            L, D, M = self.sum_F_M(AWA = AWA, AOA = AOA, beta = betas[i], V = V, rho = rho)
            Ls[i] = L
            Ds[i] = D

        return betas, AOAs, Ls, Ds

class keel:
    def __init__(self, name, type = "keel", X = 0,
                Chord = None, 
                span  =  None,
                polars = None,
                gz = None,
                displacement = None):


        self.X = X #location of foil COP with respect to mast step
        self.name = name
        self.type = type
        self.polars = polars
        self.Chord = Chord
        self.span = span
        self.gz = gz
        self.displacement = displacement

        self.area = self.Chord * self.span

    def get_leeway_V_roll(self, FX, FY, Roll0, H_arm):
        L_D = FX/FY

        cd = self.polars["Cd"] * (1+Roll0) #add roll correction, this is a placeholder

        cl_cd = self.polars["Cl"]/cd

        leeway = np.interp(L_D, cl_cd, self.polars["alpha"])
        cl = np.interp(leeway, self.polars["alpha"], self.polars["Cl"])
        V = np.sqrt(FY/(0.5*1.225*self.area*cl))

        HM = FY * H_arm
        roll = np.interp(HM, self.gz["GM"], self.gz["roll"])

        return leeway, V, roll