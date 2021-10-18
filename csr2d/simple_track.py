import numpy as np
import scipy.constants
mec2 = scipy.constants.value('electron mass energy equivalent in MeV')*1e6
from numba import guvectorize, float64


def sinc(x):
    return np.sinc(x / np.pi)


@np.vectorize
def cosc(x):
    if x == 0:
        return -0.5
    else:
        return (np.cos(x) - 1) / x ** 2


def track_a_bend(b, p0c, L=0, theta=0, g_err=0):
    """
    Tracks a 6-D beam through a bending magnet.
    See chapter 23.6 of the Bmad manual.
    
    Input:
        b: initial 6D bmad beam coord
        p0c: reference momentum in eV/c
        L: length to track in m
        theta: bending angle 
        g_err: error in g = theta/L

    Ouptut:
        d: final 6D bmad beam coord
    """
    x = b[0]
    px = b[1]
    y = b[2]
    py = b[3]
    z = b[4]
    pz = b[5]

    px_norm = np.sqrt((1 + pz) ** 2 - py ** 2)  # For simplicity

    phi1 = np.arcsin(px / px_norm)

    g = theta / L
    g_tot = g + g_err
    gp = g_tot / px_norm

    alpha = (2 * (1 + g * x) * np.sin(theta + phi1) * L * sinc(theta)
        - gp * ((1 + g * x) * L * sinc(theta)) ** 2)

    x2_t1 = x * np.cos(theta) + L ** 2 * g * cosc(theta)
    x2_t2 = np.sqrt((np.cos(theta + phi1) ** 2) + gp * alpha)
    x2_t3 = np.cos(theta + phi1)

    x2 = np.where(np.abs(theta + phi1) < np.pi / 2,
        (x2_t1 + alpha / (x2_t2 + x2_t3)),
        (x2_t1 + (x2_t2 - x2_t3) / gp))
    
    Lcu = x2 - L ** 2 * g * cosc(theta) - x * np.cos(theta)
    Lcv = -L * sinc(theta) - x * np.sin(theta)

    theta_p = 2 * (theta + phi1 - np.pi / 2 - np.arctan2(Lcv, Lcu))

    Lc = np.sqrt(Lcu ** 2 + Lcv ** 2)
    Lp = Lc / sinc(theta_p / 2)

    P = p0c * (1 + pz)  # in eV
    E = np.sqrt(P**2 + mec2**2)  # in eV
    E0 = np.sqrt(p0c**2 + mec2**2)  # in eV
    beta = P / E
    beta0 = p0c / E0

    xf = x2
    pxf = px_norm * np.sin(theta + phi1 - theta_p)
    yf = y + py * Lp / px_norm
    pyf = py
    zf = z + (beta * L / beta0) - ((1 + pz) * Lp / px_norm)
    pzf = pz

    return np.array([xf, pxf, yf, pyf, zf, pzf])


def track_entrance(b, L=0, theta=0, g_err=0, e1=0, f_int=0, h_gap=0):
    """
    Tracks a 6-D beam through the entrance fringe of a bending magnet.
    See chapter 16.2 of the Bmad manual.
    
    Input:
        b: initial 6D bmad beam coord
        L: length to track in m
        theta: bending angle in rad
        g_err: error in g = theta/L
        e1: entrance edge angle in rad

    Ouptut:
        d: final 6D bmad beam coord
    """
    x = b[0]
    px = b[1]
    y = b[2]
    py = b[3]
    z = b[4]
    pz = b[5]

    g = theta / L
    g_tot = g + g_err

    sin = np.sin(e1)
    tan = np.tan(e1)
    sec = 1 / np.cos(e1)

    Sigma_M1 = (
        (x ** 2 - y ** 2) * g_tot * tan / 2
        + y ** 2 * g_tot ** 2 * sec ** 3 * (1 + sin ** 2) * f_int * h_gap / 2 / (1 + pz)
        - x ** 3 * g_tot ** 2 * tan ** 3 / 12 / (1 + pz)
        + x * y ** 2 * g_tot ** 2 * tan * sec ** 2 / 4 / (1 + pz)
        + (x ** 2 * px - 2 * x * y * py) * g_tot * tan ** 2 / 2 / (1 + pz)
        - y ** 2 * px * g_tot * (1 + tan ** 2) / 2 / (1 + pz)
    )

    xf = x + g_tot / 2 / (1 + pz) * (-x ** 2 * tan ** 2 + y ** 2 * sec ** 2)

    pxf = (
        px
        + x * g_tot * tan
        + y ** 2 * g_tot ** 2 * (tan + 2 * tan ** 3) / 2 / (1 + pz)
        + (x * px - y * py) * g_tot * tan ** 2 / (1 + pz)
    )

    yf = y + x * y * g_tot * tan ** 2 / (1 + pz)

    pyf = (
        py
        + y
        * (
            -g_tot * tan
            + g_tot ** 2 * (1 + sin ** 2) * sec ** 3 * f_int * h_gap / (1 + pz)
        )
        + x * y * g_tot ** 2 * sec ** 2 * tan / (1 + pz)
        - x * py * g_tot * tan ** 2 / (1 + pz)
        - y * px * g_tot * (1 + tan ** 2) / (1 + pz)
    )

    zf = z + (Sigma_M1 - (x ** 2 - y ** 2) * g_tot * tan / 2) / (1 + pz)

    pzf = pz

    return np.array([xf, pxf, yf, pyf, zf, pzf])


def track_exit(b, L=0, theta=0, g_err=0, e2=0, f_int=0, h_gap=0):
    """
    Tracks a 6-D beam through the exit fringe of a bending magnet.
    See chapter 16.2 of the Bmad manual.
    
    Input:
        b: initial 6D bmad beam coord
        L: length to track in m
        theta: bending angle 
        g_err: error in g = theta/L
        e2: exit edge angle

    Ouptut:
        d: final 6D bmad beam coord
    """
    x = b[0]
    px = b[1]
    y = b[2]
    py = b[3]
    z = b[4]
    pz = b[5]

    g = theta / L
    g_tot = g + g_err
    print("g_tot:", g_tot)

    sin = np.sin(e2)
    tan = np.tan(e2)
    sec = 1 / np.cos(e2)

    # The 5th and 6th term have the opposite sign from Sigma_M1
    Sigma_M2 = (
        (x ** 2 - y ** 2) * g_tot * tan / 2
        + y ** 2 * g_tot ** 2 * sec ** 3 * (1 + sin ** 2) * f_int * h_gap / 2 / (1 + pz)
        - x ** 3 * g_tot ** 2 * tan ** 3 / 12 / (1 + pz)
        + x * y ** 2 * g_tot ** 2 * tan * sec ** 2 / 4 / (1 + pz)
        + (-x ** 2 * px + 2 * x * y * py) * g_tot * tan ** 2 / 2 / (1 + pz)
        + y ** 2 * px * g_tot * (1 + tan ** 2) / 2 / (1 + pz)
    )

    xf = x + g_tot / 2 / (1 + pz) * (x ** 2 * tan ** 2 - y ** 2 * sec ** 2)

    pxf = (
        px
        + x * g_tot * tan
        - (x ** 2 + y ** 2) * g_tot ** 2 * tan ** 3 / 2 / (1 + pz)
        + (-x * px + y * py) * g_tot * tan ** 2 / (1 + pz)
    )

    yf = y - x * y * g_tot * tan ** 2 / (1 + pz)

    pyf = (
        py
        + y
        * (
            -g_tot * tan
            + g_tot ** 2 * (1 + sin ** 2) * sec ** 3 * f_int * h_gap / (1 + pz)
        )
        + x * y * g_tot ** 2 * sec ** 2 * tan / (1 + pz)
        + x * py * g_tot * tan ** 2 / (1 + pz)
        + y * px * g_tot * (1 + tan ** 2) / (1 + pz)
    )

    zf = z + (Sigma_M2 - (x ** 2 - y ** 2) * g_tot * tan / 2) / (1 + pz)

    pzf = pz

    return np.array([xf, pxf, yf, pyf, zf, pzf])


def track_a_drift(b, p0c, L=0):
    """
    Tracks a 6-D beam through a drift.
    See chapter 23.7 of the Bmad manual.
    
    Input:
        b: initial 6D bmad beam coord
        p0c: reference momentum in eV/c
        L: length to track in m

    Ouptut:
        d: final 6D bmad beam coord
    """
    x = b[0]
    px = b[1]
    y = b[2]
    py = b[3]
    z = b[4]
    pz = b[5]

    pl = np.sqrt(1 - (px ** 2 + py ** 2) / (1 + pz ** 2))  # unitless

    P = p0c * (1 + pz)  # in eV
    E = np.sqrt(P ** 2 + mec2 ** 2)  # in eV
    E0 = np.sqrt(p0c ** 2 + mec2 ** 2)  # in eV
    beta = P / E
    beta0 = p0c / E0

    xf = x + L * px / (1 + pz) / pl
    pxf = px
    yf = y + L * py / (1 + pz) / pl
    pyf = py
    zf = z + (beta / beta0 - 1 / pl) * L
    pzf = pz

    return np.array([xf, pxf, yf, pyf, zf, pzf])


# Functions below require numba

def track_a_bend_parallel(b, p0c, L=0, theta=0, g_err=0):
    """
    The shape of the beam has to be (N, 6) for numba vectorization!
    """
    return track_a_bend_numba(b.T, p0c, L, theta, g_err).T

def track_a_drift_parallel(b, p0c, L=0):
    """
    The shape of the beam has to be (N, 6) for numba vectorization!
    """
    return track_a_drift_numba(b.T, p0c, L).T


@guvectorize([(float64[:], float64, float64, float64, float64, float64[:])], '(n),(),(),(),()->(n)', target='parallel')
def track_a_bend_numba(b, p0c, L, theta, g_err, res):
    """
    Tracks a 6-D beam through a bending magnet via numba guvectorization.
    See chapter 23.6 of the Bmad manual.
    
    Input:
        b: initial 6D bmad beam coord
        p0c: reference momentum in eV/c
        L: length to track in m
        theta: bending angle 
        g_err: error in g = theta/L

    Ouptut:
        d: final 6D bmad beam coord    
    """
    assert len(b)==6, "Bad beam dimension!!"
    
    x = b[0]
    px = b[1]
    y = b[2]
    py = b[3]
    z = b[4]
    pz = b[5]

    px_norm = np.sqrt((1 + pz)**2 - py**2)  # For simplicity

    phi1 = np.arcsin(px / px_norm)

    g = theta / L
    g_tot = g + g_err
    gp = g_tot / px_norm

    #sinc_th = np.sin(theta)/theta
    sinc_th = np.sinc(theta/np.pi)
    if theta ==0:
        cosc_th = -0.5
    else:
        cosc_th = (np.cos(theta) - 1) / theta**2
    
    alpha = ( 2 * (1 + g * x) * np.sin(theta + phi1) * L * sinc_th
        - gp * ((1 + g * x) * L * sinc_th )**2 )

    x2_t1 = x * np.cos(theta) + L**2 * g * cosc_th
    x2_t2 = np.sqrt((np.cos(theta + phi1)**2) + gp * alpha)
    x2_t3 = np.cos(theta + phi1)

    if ((np.abs(theta + phi1)) < (np.pi / 2)):
        x2 = (x2_t1 + alpha / (x2_t2 + x2_t3))
    else:
        x2 = (x2_t1 + (x2_t2 - x2_t3) / gp)

    Lcu = x2 - L**2 * g * cosc_th - x * np.cos(theta)
    Lcv = -L * sinc_th - x * np.sin(theta)

    theta_p = 2 * (theta + phi1 - np.pi/2 - np.arctan2(Lcv, Lcu))

    Lc = np.sqrt(Lcu**2 + Lcv**2)
    
    sinc_half_th_p = np.sinc(theta_p / 2 / np.pi)
    Lp = Lc / sinc_half_th_p

    P = p0c * (1 + pz)  # in eV
    E = np.sqrt(P**2 + mec2**2)  # in eV
    E0 = np.sqrt(p0c**2 + mec2**2)  # in eV
    beta = P / E
    beta0 = p0c / E0

    xf = x2
    pxf = px_norm * np.sin(theta + phi1 - theta_p)
    yf = y + py * Lp / px_norm
    pyf = py
    zf = z + (beta * L / beta0) - ((1 + pz) * Lp / px_norm)
    pzf = pz
    
    res[0] = xf 
    res[1] = pxf 
    res[2] = yf 
    res[3] = pyf 
    res[4] = zf 
    res[5] = pzf 
    
    
@guvectorize([(float64[:], float64, float64, float64[:])], '(n),(),()->(n)', target='parallel')
def track_a_drift_numba(b, p0c, L, res):
    """
    Tracks a 6-D beam through a drift.
    See chapter 23.7 of the Bmad manual.
    
    Input:
        b: initial 6D bmad beam coord
        p0c: reference momentum in eV/c
        L: length to track in m

    Ouptut:
        d: final 6D bmad beam coord
    """
    x = b[0]
    px = b[1]
    y = b[2]
    py = b[3]
    z = b[4]
    pz = b[5]

    pl = np.sqrt(1 - (px**2 + py**2) / (1 + pz**2))  # unitless

    P = p0c * (1 + pz)  # in eV
    E = np.sqrt(P**2 + mec2** 2)  # in eV
    E0 = np.sqrt(p0c**2 + mec2** 2)  # in eV
    beta = P / E
    beta0 = p0c / E0

    xf = x + L * px / (1 + pz) / pl
    pxf = px
    yf = y + L * py / (1 + pz) / pl
    pyf = py
    zf = z + (beta / beta0 - 1 / pl) * L
    pzf = pz

    res[0] = xf 
    res[1] = pxf 
    res[2] = yf 
    res[3] = pyf 
    res[4] = zf 
    res[5] = pzf 