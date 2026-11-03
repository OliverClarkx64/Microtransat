from foils import foil 
from foils import spring
from foils import rig
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve

import utils


NACA0012P = pd.read_excel("clean_polars.xlsx").to_dict(orient = "list")

Main_foil = foil(name = "Main",
                type = "wing",
                X = 0.05,
                Chord = 0.4,
                span = 1.5,
                polars = NACA0012P
                )

Trim_tab = foil(name = "Trim tab",
                type = "flap",
                X = 0.75,
                Chord = 0.1,
                span = 0.5,
                polars = NACA0012P
                )


Mainsheet = spring(k=150,
                x0=0.15
                )



Simple_rig = rig(foils = [Main_foil], spring = Mainsheet)

trim_tab_rig = rig(foils = [Main_foil, Trim_tab], spring = Mainsheet)


def calculate_AWA(Vs, Vt, TWA, leeway0):
    # AWA is defined relative to the boat heading NOT direction of travel
    tan_AWAh = np.sin(TWA) / (Vs/Vt + np.cos(TWA))
    AWAh = np.arctan(tan_AWAh)

    Va = Vt*np.sin(TWA) / np.sin(AWAh)
    AWA = AWAh + leeway0
    return AWA, Va


AWA = 30*np.pi/180
Vw = 5

betas, AOAs, Ls, Ds = trim_tab_rig.solve_for_AWA(AWA, V = Vw)

print("betas: ", betas)
print("AOAs: ", AOAs)
print("Ls: ", Ls)
print("Ds: ", Ds)


# TODO:
# from arrays of beta, find Fx, Fy, then leeway and V
# update AWA and repeat until convergence
