"""
4-DOF autonomous sailboat simulation (Tipsuwan et al. 2023, Sec. 3.2 / Xiao & Jouffroy 2014)

Frames : NED. x = North, y = East, z = Down.  psi = heading, clockwise from North.
State  : X = [x, y, phi, psi, u, v, p, r]   (eta, nu)
Inputs : delta_s (sail angle), delta_r (rudder angle), true wind (v_tw, alpha_tw)
         alpha_tw = direction the wind blows TOWARD, bearing from North (rad).
Plot   : plot-x = East, plot-y = North.

Controls: drag mouse = set wind (arrow points where wind blows TO)
          a / d = turn desired heading to port / starboard
          r     = reset boat
"""
import sys
import numpy as np

##Boat paramaters
m, Ixx, Izz = 100.0, 30.0, 60.0     # [kg], [kgm^2], [kgm^2]
X_ud, Y_vd, K_pd, N_rd = -10.0, -80.0, -10.0, -20.0      # added-mass derivatives

CL_fun = lambda a: 2 * np.pi * np.sin(a) * np.cos(a)     # crude flat-plate-ish placeholder

##main ramamater
##density of water and air
##equation constants
P = {
    "rho_w": 1025.0, "rho_a": 1.225,
    "keel":   {"A": 0.10, "s": 0.5,  "pos": (0.0, 0.0, 0.4),
               "CL": CL_fun, "CD": lambda a: 0.01 + 1.0 * np.sin(a) ** 2},
    "hull":   {"pos": (0.0, 0.0, 0.0),
               "F_rh": lambda v: 20.0 * v * abs(v)},     # quadratic hull resistance [N]
    "sail":   {"A": 3.0, "s": 3.0,  "pos": (0.0, 0.0, -1.5),   # z<0 = above CO
               "x_m": 0.1, "x_sm": 0.15,
               "CL": CL_fun, "CD": lambda a: 0.02 + 1.2 * np.sin(a) ** 2},
    "rudder": {"A": 0.03, "s": 0.25, "pos": (-0.6, 0.0, 0.2),
               "CL": CL_fun, "CD": lambda a: 0.01 + 1.0 * np.sin(a) ** 2},
    "a": 60.0,    # heel damping   a*phi_dot*|phi_dot|
    "b": 60.0,    # yaw damping    b*r*|r|*cos(phi) phi dot but not sure
    "c": 0.0,     # restoring      c*phi^2 + d*phi
    "d": 490.0,   # ~ m*9.81*GM with GM = 0.5 m
}

M_RB = np.diag([m, m, Ixx, Izz])
M_A = -np.diag([X_ud, Y_vd, K_pd, N_rd])    ##Equations  9
M_INV = np.linalg.inv(M_RB + M_A)           ##Equations  6 M constant and only used inverted so calc once

#######dynamic model

##translation to global CO from local
def J(eta): ##Equations  5
    phi, psi = eta[2], eta[3]
    return np.array([
        [np.cos(psi), -np.sin(psi) * np.cos(phi), 0, 0],
        [np.sin(psi),  np.cos(psi) * np.cos(phi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.cos(phi)],
    ])


def compute_C(nu):  ##Equations  7, 10, 11
    u, v, p, r = nu
    C_RB = np.array([[0, -m * r, 0, 0],
                     [m * r, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])
    C_A = np.array([[0, 0, 0, Y_vd * v],
                    [0, 0, 0, -X_ud * u],
                    [0, 0, 0, 0],
                    [-Y_vd * v, X_ud * u, 0, 0]])
    return C_RB + C_A


def compute_g(eta):  ##gravitational restoring force ##Equations  23
    phi = eta[2]
    return np.array([0.0, 0.0, P["c"] * phi ** 2 + P["d"] * phi, 0.0])


def foil_LD(alpha, v_a, foil, rho):    ##shorthand foil lift and drag calcs Equations  12-14
    q = 0.5 * rho * foil["A"] * v_a ** 2
    CL = foil["CL"](alpha)
    CDi = CL ** 2 * foil["A"] / (np.pi * foil["s"] ** 2)
    return q * CL, q * (foil["CD"](alpha) + CDi)


def compute_D(nu, eta, P):    ##D term Equations  15-22
    u, v, p, r = nu
    phi = eta[2]
    cphi = max(np.cos(phi), 0.1)                         # guard sec(phi)
    rho = P["rho_w"]

    k = P["keel"]
    xk, yk, zk = k["pos"]
    v_aku = -u + r * yk
    v_akv = -v + r * xk + p * zk
    v_ak = np.hypot(v_aku, v_akv)
    a_ak = np.arctan2(v_akv, -v_aku)
    L_k, D_k = foil_LD(a_ak, v_ak, k, rho)
    sa, ca = np.sin(a_ak), np.cos(a_ak)
    Dk = np.array([-L_k * sa + D_k * ca,
                   -L_k * ca - D_k * sa,
                   (-L_k * ca - D_k * sa) * abs(zk),
                   (L_k * ca + D_k * sa) * abs(xk)])

    h = P["hull"]
    xh, yh, zh = h["pos"]
    v_ahu = -u + r * yh
    v_ahv = (-v - r * xh + p * zh) / cphi
    v_ah = np.hypot(v_ahu, v_ahv)
    a_ah = np.arctan2(v_ahv, -v_ahu)
    F = h["F_rh"](v_ah)
    sh, ch = np.sin(a_ah), np.cos(a_ah)
    Dh = np.array([F * ch,
                   -F * sh * cphi,
                   -F * sh * cphi * abs(zh),
                   F * sh * cphi * abs(xh)])

    Dhy = np.array([0.0, 0.0,
                    P["a"] * p * abs(p),
                    P["b"] * r * abs(r) * np.cos(phi)])
    return Dk + Dh + Dhy


def apparent_wind(nu, eta, wind, S):    ## Equations 24-25
    u, v, p, r = nu
    phi, psi = eta[2], eta[3]
    v_tw, a_tw = wind
    xs, ys, zs = S["pos"]
    v_awu = v_tw * np.cos(a_tw - psi) - u + r * ys
    v_awv = v_tw * np.sin(a_tw - psi) * np.cos(phi) - v - r * xs + p * zs
    return np.hypot(v_awu, v_awv), np.arctan2(v_awv, -v_awu)


def compute_tau(nu, eta, delta_s, delta_r, wind, P):     ##Equations 26-32
    u, v, p, r = nu
    S = P["sail"]
    zs = S["pos"][2]
    v_aw, a_aw = apparent_wind(nu, eta, wind, S)
    L_s, D_s = foil_LD(a_aw - delta_s, v_aw, S, P["rho_a"])     ##Equations  29
    sa, ca = np.sin(a_aw), np.cos(a_aw)
    Fx, Fy = L_s * sa - D_s * ca, L_s * ca + D_s * sa
    tau_s = np.array([Fx, Fy, Fy * abs(zs),
                      -Fx * S["x_sm"] * np.sin(delta_s)
                      + Fy * (S["x_m"] - S["x_sm"] * np.cos(delta_s))])

    R = P["rudder"]
    xr, yr, zr = R["pos"]
    v_aru = -u + r * yr
    v_arv = -v - r * xr + p * zr
    v_ar = np.hypot(v_aru, v_arv)
    a_ar = np.arctan2(v_arv, -v_aru)
    L_r, D_r = foil_LD(a_ar - delta_r, v_ar, R, P["rho_w"])     ##Equations  31
    sr, cr = np.sin(a_ar), np.cos(a_ar)
    Fry = L_r * cr + D_r * sr
    tau_r = np.array([L_r * sr - D_r * cr, Fry, Fry * abs(zr), -Fry * abs(xr)])
    return tau_s + tau_r


####dull dynamic model together
## ds = sail pos, dr = rudder pos
##f calculates how fast each state is moving given the states so is the full system model calculation that outputs 
##so outputs nu_dot and eta_dot the rate of change of position and speed states
##rk4 basically dead reckons the rates of changes into states
def f(X, delta_s, delta_r, wind):
    ##separate whole state matrix into eta and nu
    eta, nu = X[:4].copy(), X[4:]
    eta[2] = np.clip(eta[2], -1.2, 1.2)                  # keep cos(phi) away from 0
    tau = compute_tau(nu, eta, delta_s, delta_r, wind, P)
    D = compute_D(nu, eta, P)
    nu_dot = M_INV @ (tau - compute_C(nu) @ nu - D - compute_g(eta))
    return np.concatenate([J(eta) @ nu, nu_dot]) ##recombine j(eta) * nu to make eta dot and nu dot which is a rate of change matrix for all states pos/rotation and velocity
##eta is position/orientation nu is linear / angular velocity
##used for integrating first order ODE
##operates rk4 over all states as in matrix notation
def rk4_step(X, dt, delta_s, delta_r, wind):
    k1 = f(X, delta_s, delta_r, wind)
    k2 = f(X + 0.5 * dt * k1, delta_s, delta_r, wind)
    k3 = f(X + 0.5 * dt * k2, delta_s, delta_r, wind)
    k4 = f(X + dt * k3, delta_s, delta_r, wind)
    X = X + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    X[3] = wrap(X[3])
    return X


def wrap(a): ##
    return (a + np.pi) % (2 * np.pi) - np.pi


######control

##LOS guidance calcs heading giveninfo about path
def LOS():
    return

def control_wrapper(controlFunc):
    def wrapper(*args, **kwargs):
        ##determine mode
        ##calc desired heading psi_d
        ##need set path
        ##if relative wind direction to path then keep, tack, jibe

        #parse wind info
        #calc difference to path
        #decide on keep tack or jibe
        #then LOS or other algorithm
        ds, dr = (controlFunc(*args, **kwargs)) ## calls/runs wrapped function
        return ds, dr
    return wrapper


##for PID bind the integral term to prevent accumulative issues
##maybe also decay the accumulated error value
##only let integral raise when certain error threholds??
##try these with the goal of keeping integral error memory short since the control is continuously changing and not holding steady state errors

##[min speed, p, i, d]
rudderGainMap = np.array([
        [0, 1.0, 0.02, 1.5], ##0-a [m/s]
        [0, 0, 0, 0], ##a-b [m/s]
        [0, 0, 0, 0]  ##b-c [m/s] expand more slots if necessary
    ])

class Autopilot:
    """Heading PID (Eq. 37-38) on rudder + linear sail angle (Eq. 40) on sail."""
    ##rudder gain map to replace individual gains
    def __init__(self, rudderGainMap, Kp=1.0, Ki=0.02, Kd=1.5, dr_max=0.5, ): ##init PID values - tune with nomoto model and introoduce gain schedule
        self.Kp, self.Ki, self.Kd, self.dr_max = Kp, Ki, Kd, dr_max
        self.integ = 0.0
        self.rudderGainMap = rudderGainMap

    @control_wrapper ##wraps heading logic onto rudder control
    def step(self, X, psi_d, wind, dt): ##states, desired heading, wind params, timestep
    ##add path following instead of just heading control
    ##actually wrap path following ontop of heading control

        eta, nu = X[:4], X[4:]
        ##get pid response
        self.Kp, self.Ki, self.Kd = self.rudderGainMap[0][1:] ## replace with speed dependant pid response


        ##############ADD WHEN THRESHOLDS KNOWN#################
##        ## speed in 2 components u and v which are so abs(sqrt(eta[0]**2 + eta[1]**2)
##        ##optimise this to not run every cycle and to switch when thresholds met
##        speed = abs(sqrt(nu[0]**2 + nu[1]**2)
##        for i in range(len(self.rudderGainMap): ##range len so that prev data can be selected
##            if speed < i[0]: ##works because map goes up in speed so when speed finally smaller then resposne then select prev response
##                self.Kp, self.Ki, self.Kd = self.rudderGainMap[i-1][1:] ##selects prev map data
##                break
##

        
        ##calc error
        e = wrap(psi_d - eta[3])   ##desired heading - current heading / wrap forces slow motions  --error                    ##Equations 37
        r = nu[3]   ##current rate of change of heading
        summedTerms = self.Kp * e + self.Ki * self.integ - self.Kd * r * np.cos(eta[2])  ##add P I D terms
        dr = np.clip(summedTerms, -self.dr_max, self.dr_max) ##limits rudder angle mechanically
        if dr == summedTerms:                                   # prevents integral windup only accumulates when the rudder isnt saturated - prevents large integral erm
            self.integ += e * dt ##add up integral term

        ##rough sail control
        _, a_aw = apparent_wind(nu, eta, wind, P["sail"])
        ds_max, ds_min = np.pi / 2, 0.0
        ds = (ds_max - ds_min) * a_aw / np.pi + ds_min    ##Equations 40, signed by tack
        return ds, dr ##returns desired sail and rudder angle


#####plot
def main():
    import matplotlib.pyplot as plt
    global ds, dr, rudderGainMap ##so adjustments can be made in different functions
    ds , dr = np.pi/2,0 ##sets initial sail and rudder angles

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(xlim=(-12, 12), ylim=(-12, 12))
    ax.set_aspect("equal")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.grid(alpha=0.3)

    ##plot ship parts
    hull_line, = ax.plot([], [], lw=3, color="k")
    sail_line, = ax.plot([], [], lw=3, color="tab:orange")
    rudder_line, = ax.plot([], [], lw=3, color="tab:green")
    ##plot previous motion, heading and info
    trail_line, = ax.plot([], [], lw=1, color="gray", alpha=0.6)
    hd_line, = ax.plot([], [], "--", lw=1, color="tab:red")
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace")
    ##plot wind control line
    wind_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                             arrowprops=dict(arrowstyle="->", lw=3, color="tab:blue"))
    wind_text = ax.text(0, 0, "", color="tab:blue")

    ##init environmental conditions, X is ship states
    ##psi_d is desired heading
    ##stores all the info for the model including which control to use
    state = {"X": np.zeros(8), "psi_d": 0.0, "wind_vec": np.array([5.0, 0.0]),
             "dragging": False, "trail": [], "running": True, "ap": Autopilot(rudderGainMap)} ##initialise autopilot object under name state["ap"]

    def reset():
        state["X"] = np.zeros(8)
        state["psi_d"] = 0.0
        state["ap"] = Autopilot()
        state["trail"] = []

    def calc_heading():
        ##determine state to be in then LOS or manouvre
        return

    def wind_tuple():
        wx, wy = state["wind_vec"]                        # plot-x = East, plot-y = North
        return np.hypot(wx, wy), np.arctan2(wx, wy)       # (speed, bearing toward)

    def body_to_plot(xb, yb, X):
        N, E, psi = X[0], X[1], X[3] ##x,y and heading draw line from COG 
        return (E + xb * np.sin(psi) + yb * np.cos(psi),
                N + xb * np.cos(psi) - yb * np.sin(psi))

    def wind_origin():
        X = state["X"]
        return X[1] - 7.0, X[0] + 5.0
    
    def on_key(event):
        global ds, dr
        if event.key == "a": state["psi_d"] = wrap(state["psi_d"] - 0.15) ## manual control rudder dr -= 0.15##
        elif event.key == "d": state["psi_d"] = wrap(state["psi_d"] + 0.15) ##change desired heading ## manual control rudder dr += 0.15##
        elif event.key == "r": reset()

    def set_wind(event):
        if state["dragging"] and event.inaxes == ax and event.xdata is not None:
            ox, oy = wind_origin()
            state["wind_vec"] = np.array([event.xdata - ox, event.ydata - oy])

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", lambda e: state.update(running=False))
    fig.canvas.mpl_connect("button_press_event",
                           lambda e: (state.update(dragging=True), set_wind(e)))
    fig.canvas.mpl_connect("motion_notify_event", set_wind)
    fig.canvas.mpl_connect("button_release_event", lambda e: state.update(dragging=False))

    dt, substeps = 0.01, 4
    plt.ion()

    while state["running"]:
        v_tw, a_tw = wind_tuple() ##true wind
        wind = (v_tw, a_tw)
        ##run calcs
        ## get sail and rudder pos from controllers
        ##push sail and rudder angles theough rk4 which handles first order ODE integration
        ##over time step
        ##this calls f which handles the model calculations
        for _ in range(substeps):
            ##.step()is the method in the controller class
            ##update rudder and sail positions from conroller
            ds, dr = state["ap"].step(state["X"], state["psi_d"], wind, dt) ##update control response taking in error etc
            state["X"] = rk4_step(state["X"], dt, ds, dr, wind)   # inputs held over the step


        ##unpack state dic for updating screen
        X = state["X"]
        ######draw everything
        # hull
        hx, hy = body_to_plot(np.array([-1.0, 1.4]), np.zeros(2), X)
        hull_line.set_data(hx, hy)
        # boom: from mast, aft and swung to leeward by delta_s
        mx, my = P["sail"]["x_m"], 0.0
        sx, sy = body_to_plot(np.array([mx, mx - 1.2 * np.cos(ds)]),
                              np.array([my, my + 1.2 * np.sin(ds)]), X)
        sail_line.set_data(sx, sy)
        # rudder at stern
        xr = P["rudder"]["pos"][0]
        rx, ry = body_to_plot(np.array([xr, xr - 0.4 * np.cos(dr)]),
                              np.array([0.0, 0.4 * np.sin(dr)]), X)
        rudder_line.set_data(rx, ry)
        # desired heading
        d0 = body_to_plot(np.array([0.0]), np.array([0.0]), X)
        hd_line.set_data([d0[0][0], d0[0][0] + 3 * np.sin(state["psi_d"])],
                         [d0[1][0], d0[1][0] + 3 * np.cos(state["psi_d"])])
        # trail
        state["trail"].append((X[1], X[0]))
        state["trail"] = state["trail"][-600:]
        t = np.array(state["trail"])
        trail_line.set_data(t[:, 0], t[:, 1])

        
        # view + wind arrow follow boat
        ax.set_xlim(X[1] - 12, X[1] + 12); ax.set_ylim(X[0] - 12, X[0] + 12)
        ox, oy = wind_origin()
        wind_arrow.set_position((ox, oy))
        wind_arrow.xy = (ox + state["wind_vec"][0], oy + state["wind_vec"][1])
        wind_text.set_position((ox, oy + 1)); wind_text.set_text(f"Wind {v_tw:.1f} m/s")
        _, a_aw = apparent_wind(X[4:], X[:4], wind, P["sail"])
        info.set_text(f"speed {np.hypot(X[4], X[5]):4.2f} m/s\n"
                      f"heading {np.degrees(X[3]):6.1f}  (des {np.degrees(state['psi_d']):6.1f})\n"
                      f"heel {np.degrees(X[2]):5.1f} deg\n"
                      f"app. wind {np.degrees(a_aw):6.1f} deg\n"
                      f"sail {np.degrees(ds):5.1f}  rudder {np.degrees(dr):5.1f}")
        plt.pause(0.001)


if __name__ == "__main__":
    main()
