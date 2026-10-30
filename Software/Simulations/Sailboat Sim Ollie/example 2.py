from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import sys #for manual input

# initializing a figure in 
# which the graph will be plotted
fig = plt.figure() 
 
# marking the x-axis and y-axis
axis = plt.axes(xlim =(-10, 10), 
                ylim =(-10, 10)) 

# initializing a line variable
line, = axis.plot([], [], lw = 3)
COG = axis.scatter([0], [0], s = 50 )

##draws the body and COG
##COG is at X-Y
##front and back lines defined by
body_a = 1 ##distance from COG to tail [m]
body_b = 1.4 ##distance from COG to head [m]
heading = 0 ##radians
x,y  = 0, 0 ##set global xy coords for COG

def draw_body(x, y, heading):
##    consider heading from horizontal upward
    x = [x - body_a*np.cos(heading), x + body_b*np.cos(heading)]
    y = [y - body_a*np.sin(heading), y + body_b*np.sin(heading)]
    line.set_data(x, y) ##needs a sequence because its a line

##get keypresses and directly move
def on_press(event):
    global heading, x, y
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'x':
        heading += 0.5
        print("true")
    elif event.key == 'w':
        y += 0.5
        print("true")
    elif event.key == 'd':
        x += 0.5
        print("true")
    elif event.key == 'a':
        x -= 0.5
        print("true")



##### Setup simulaton environment variables
wind_speed = 0.0       # m/s (length of arrow)
wind_dir = 0.0         # radians, direction wind blows TOWARD
wind_origin = (-7, 7)  # where the arrow is anchored on the plot
wind_scale = 1.0       # arrow length per 1 m/s

wind_arrow = axis.annotate(
    '', xy=wind_origin, xytext=wind_origin,
    arrowprops=dict(arrowstyle='->', lw=3, color='tab:blue'))
wind_text = axis.text(wind_origin[0], wind_origin[1] + 1,
                      'Wind: 0.0 m/s', color='tab:blue')

#### Vessel dynamic model
##solve ν̇ = M⁻¹(τ − Cν − D − g)
##computes each part needed for kinematic model
##variable list
vesselMass = 100 #[kg]
vesselIxx = 5 #[kgm^2]
vesselIyy = 5 #[kgm^2]

##state vectors
##stores position and orientation
eta = np.array([
    [0],
    [0],
    [0],
    [0]
]) 

##body frame velocities
##
nu = np.array([
    [0],
    [0],
    [0],
    [0]
])



M = 0 ##matrix variable for kinematics
x_u, y_v, k_p, n_r = 0,0,0,0 ##test for values
def init_M():
    global vesselMass, vesselIxx, vesselIyy
    global M
    global x_u, y_v, k_p, n_r
    M_RB = np.array([
        [vesselMass, 0, 0, 0],
        [0, vesselMass, 0, 0],
        [0, 0, vesselIxx, 0 ],
        [0, 0, 0, vesselIyy]
    ])
    
    M_A = np.array([
        [x_u,0,0,0],
        [0,y_v,0,0],
        [0,0,k_p,0],
        [0,0,0,n_r]
    ])
    M = np.add(M_RB, M_A)
    print(M)
    return
init_M() ##only needs to call once since unchanging

def compute_C(nu): ##r i imagine is yaw rate or something, get from global state matrix
    global vesselMass
    global x_u, y_v, k_p, n_r
    
    C_RB = np.array([
        [0, -vesselMass*r, 0, 0],
        [vesselMass*r, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    
    C_A = np.array([
        [0,0,0,y_v*v],
        [0,0,0,-x_u*u],
        [0,0,0,0],
        [y_v*v,x_u*u,0,0]
    ])
    return np.add(C_RB, C_A)

P = {
    "rho_w": 1025.0,
    "keel": {"A": 0.05, "s": 0.4, "pos": (0.0, 0.0, 0.3),
             "CL": lambda a: 2*np.pi*np.sin(a)*np.cos(a),      # crude placeholder
             "CD": lambda a: 0.01 + 1.0*np.sin(a)**2},
    "hull": {"pos": (0.0, 0.0, 0.0),
             "F_rh": lambda v: 5.0 * v * abs(v)},              # placeholder quadratic drag
    "a": 5.0, "b": 2.0,
}

def compute_D(nu, eta, P):
    """
    Damping force vector D(nu, eta) = D_k + D_h + D_heel + D_yaw  (paper Eqs. 15-22)

    nu  : [u, v, p, r]        body-frame velocities (m/s, m/s, rad/s, rad/s)
    eta : [x, y, phi, psi]    n-frame pose (m, m, rad, rad)
    P   : parameter dict with keys
          rho_w          water density (kg/m^3)
          keel           dict: A, s, pos=(x,y,z), CL(alpha), CD(alpha)  [callables]
          hull           dict: pos=(x,y,z), F_rh(v_ah) [callable, N]
          a, b           heel / yaw damping constants (inclining test)
    Returns: 4-element numpy array [Fx, Fy, K, N]
    """
    u, v, p, r = nu ##unpack array
    phi = eta[2]
    rho = P["rho_w"]

    # ---------- Keel: Eqs. 16, 19, 20, plus lift/drag Eqs. 12-14 ----------
    k = P["keel"]
    xk, yk, zk = k["pos"]
    v_aku = -u + r * yk
    v_akv = -v + r * xk + p * zk
    v_ak = np.hypot(v_aku, v_akv)
    alpha_ak = np.arctan2(v_akv, -v_aku)      # alpha_k = alpha_ak (Eq. 19)

    CL = k["CL"](alpha_ak)
    CDi = CL**2 * k["A"] / (np.pi * k["s"]**2)           # Eq. 14
    L_k = 0.5 * rho * k["A"] * v_ak**2 * CL              # Eq. 12
    D_k = 0.5 * rho * k["A"] * v_ak**2 * (k["CD"](alpha_ak) + CDi)  # Eq. 13

    s_a, c_a = np.sin(alpha_ak), np.cos(alpha_ak)
    Dk = np.array([
        -L_k * s_a + D_k * c_a,
        -L_k * c_a - D_k * s_a,
        (-L_k * c_a - D_k * s_a) * abs(zk),
        ( L_k * c_a + D_k * s_a) * abs(xk),
    ])                                                   # Eq. 16

    # ---------- Hull: Eqs. 17, 21, 22 ----------
    h = P["hull"]
    xh, yh, zh = h["pos"]
    v_ahu = -u + r * yh
    v_ahv = (-v - r * xh + p * zh) / np.cos(phi)         # sec(phi); keep |phi| << pi/2
    v_ah = np.hypot(v_ahu, v_ahv)
    alpha_ah = np.arctan2(v_ahv, -v_ahu)

    F = h["F_rh"](v_ah)
    s_h, c_h = np.sin(alpha_ah), np.cos(alpha_ah)
    Dh = np.array([
         F * c_h,
        -F * s_h * np.cos(phi),
        -F * s_h * np.cos(phi) * abs(zh),
         F * s_h * np.cos(phi) * abs(xh),
    ])                                                   # Eq. 17

    # ---------- Heel + yaw damping: Eq. 18 ----------
    phi_dot = p                                          # from J: row 3 is phi_dot = p
    D_heel_yaw = np.array([
        0.0,
        0.0,
        P["a"] * phi_dot * abs(phi_dot),
        P["b"] * phi_dot * abs(phi_dot) * np.cos(phi),
    ])

    return Dk + Dh + D_heel_yaw                          # Eq. 15

def compute_g(nu):
    ##constants
    c = 0
    d = 0
    g = np.array([
        [0],
        [0],
        [c*phi**2 + d*phi],
        [0]
    ])
    return g


def foil_LD(alpha, v_a, foil, rho):
    """Lift and drag on a foil (paper Eqs. 12-14).
    alpha : angle of attack (rad), used as the argument to CL/CD
    v_a   : apparent flow speed at the foil (m/s)
    foil  : dict with A (m^2), s (m), CL(alpha), CD(alpha)
    rho   : fluid density (air for the sail, water for the rudder)
    """
    q = 0.5 * rho * foil["A"] * v_a**2
    CL = foil["CL"](alpha)
    CDi = CL**2 * foil["A"] / (np.pi * foil["s"]**2)     # Eq. 14
    L = q * CL                                           # Eq. 12
    D = q * (foil["CD"](alpha) + CDi)                    # Eq. 13
    return L, D

##new parameters needed for tau

P.update({
    "rho_a": 1.225,
    "sail":   {"A": 0.5, "s": 1.0, "pos": (0.0, 0.0, -0.8),   # z negative = above CO (NED, z down)
               "x_m": 0.1, "x_sm": 0.15,
               "CL": lambda a: 2*np.pi*np.sin(a)*np.cos(a),    # placeholder
               "CD": lambda a: 0.02 + 1.2*np.sin(a)**2},
    "rudder": {"A": 0.02, "s": 0.2, "pos": (-0.6, 0.0, 0.2),
               "CL": lambda a: 2*np.pi*np.sin(a)*np.cos(a),
               "CD": lambda a: 0.01 + 1.0*np.sin(a)**2},
})
def compute_tau(nu, eta, delta_s, delta_r, wind, P):
    """
    Propulsive force/moment vector tau = tau_s + tau_r  (paper Eqs. 24-32)

    nu      : [u, v, p, r]        body-frame velocities
    eta     : [x, y, phi, psi]    n-frame pose (rad for angles)
    delta_s : sail angle (rad)
    delta_r : rudder angle (rad)
    wind    : (v_tw, alpha_tw)    true wind speed (m/s) and direction (rad, n-frame)
    P       : parameter dict with keys
              rho_a, rho_w   air / water density (kg/m^3)
              sail           dict: A, s, pos=(x,y,z), CL, CD, x_m, x_sm
              rudder         dict: A, s, pos=(x,y,z), CL, CD
    Returns: 4-element numpy array [Fx, Fy, K, N]
    """
    u, v, p, r = nu
    phi, psi = eta[2], eta[3]
    v_tw, alpha_tw = wind

    # ---------------- Sail: Eqs. 24, 25, 27, 29 ----------------
    S = P["sail"]
    xs, ys, zs = S["pos"]

    v_awu = v_tw * np.cos(alpha_tw - psi) - u + r * ys
    v_awv = v_tw * np.sin(alpha_tw - psi) * np.cos(phi) - v - r * xs + p * zs
    v_aw = np.hypot(v_awu, v_awv)                        # Eq. 25
    alpha_aw = np.arctan2(v_awv, -v_awu)                 # Eq. 24

    alpha_s = alpha_aw - delta_s                         # Eq. 29 (angle of attack)
    L_s, D_s = foil_LD(alpha_s, v_aw, S, P["rho_a"])

    sa, ca = np.sin(alpha_aw), np.cos(alpha_aw)          # note: apparent angle, not alpha_s
    F_x = L_s * sa - D_s * ca
    F_y = L_s * ca + D_s * sa
    tau_s = np.array([
        F_x,
        F_y,
        F_y * abs(zs),
        -F_x * S["x_sm"] * np.sin(delta_s)
        + F_y * (S["x_m"] - S["x_sm"] * np.cos(delta_s)),
    ])                                                   # Eq. 27

    # ---------------- Rudder: Eqs. 28, 30-32 ----------------
    R = P["rudder"]
    xr, yr, zr = R["pos"]

    v_aru = -u + r * yr
    v_arv = -v - r * xr + p * zr
    v_ar = np.hypot(v_aru, v_arv)                        # Eq. 32
    alpha_ar = np.arctan2(v_arv, -v_aru)                 # Eq. 30

    alpha_r = alpha_ar - delta_r                         # Eq. 31 (angle of attack)
    L_r, D_r = foil_LD(alpha_r, v_ar, R, P["rho_w"])

    sr, cr = np.sin(alpha_ar), np.cos(alpha_ar)
    Fr_y = L_r * cr + D_r * sr
    tau_r = np.array([
        L_r * sr - D_r * cr,
        Fr_y,
        Fr_y * abs(zr),
        -Fr_y * abs(xr),
    ])                                                   # Eq. 28

    return tau_s + tau_r                                 # Eq. 26

##handling wind control 
dragging = False

def update_wind(mx, my):
    global wind_speed, wind_dir
    dx = mx - wind_origin[0]
    dy = my - wind_origin[1]
    wind_speed = np.hypot(dx, dy) / wind_scale
    wind_dir = np.arctan2(dy, dx)
    wind_arrow.xy = (mx, my)   # arrow tip follows the mouse
    wind_text.set_text(f'Wind: {wind_speed:.1f} m/s, '
                       f'{np.degrees(wind_dir):.0f}°')

def on_mouse_press(event):
    global dragging
    if event.inaxes == axis and event.xdata is not None:
        dragging = True
        update_wind(event.xdata, event.ydata)

def on_mouse_move(event):
    if dragging and event.inaxes == axis and event.xdata is not None:
        update_wind(event.xdata, event.ydata)

def on_mouse_release(event):
    global dragging
    dragging = False

##### compute dynamics



####running the environment
plt.ion() ## interactive plot mode
running = True
##stopFunc = lambda e: running = False
time = 0
timeStep = 0.01
def endFunc(arg):
    global running
    running = False

##handlers for inputs
fig.canvas.mpl_connect('key_press_event', on_press)
fig.canvas.mpl_connect('close_event', endFunc)
fig.canvas.mpl_connect('button_press_event', on_mouse_press)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_release_event', on_mouse_release)



##running environment
while running:
    time += timeStep ##record time
    heading += timeStep

    x += wind_speed * np.cos(wind_dir) * timeStep
    y += wind_speed * np.sin(wind_dir) * timeStep


    ##draw the body
    draw_body(x,y,heading)
    COG.set_offsets([[x, y]])
    plt.draw()
    ##delay and check for closed app
    plt.pause(timeStep)
    
##  rk4 for stable intgration of final values
