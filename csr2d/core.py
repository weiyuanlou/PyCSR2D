import numpy as np
import scipy.special as ss
import scipy.signal as ss2

from numpy import abs, sin, cos, real, exp, pi


def psi_s(z, x, beta):
    """
    2D longitudinal potential
    Eq. (23) from Ref[1] with no constant factor (e*beta**2/2/rho**2).
    Ref[1]: Y. Cai and Yuantao. Ding, PRAB 23, 014402 (2020).
    Note that 'x' here corresponds to 'chi = x / rho' in the paper.
    """
    #try:
    out = (cos(2 * alpha(z, x, beta)) - 1 / (1+x)) / (
            kappa(z, x, beta) - beta * (1+x) * sin(2*alpha(z, x, beta)))
    #except ZeroDivisionError:
        # out = 0
        # print(f"Oops!  ZeroDivisionError at (z,x)= ({z:5.2f},{x:5.2f}). Returning 0.")
    return np.nan_to_num(out)

def psi_x_where_x_equals_zero(z, dx, beta):
    """
    Evaluate psi_x close to x = 0
    This is a rough approximation of the singularity across x = 0
    """
    return (psi_x(z, -dx/2, beta) + psi_x(z, dx/2, beta))/2


def alpha_where_z_equals_zero(x, beta):
    """
    Evaluate alpha(z,x) when z is zero.
    Eq. (24) from Ref[1] simplifies to a quadratic equation for alpha^2.
    """
    b = nu(x,beta)
    c = -3*(beta**2 * x**2)/4/beta**2/(1+x)
    root1 = (-b + np.sqrt(b**2 - 4*c))/2
    # root2 = (-b - np.sqrt(b**2 - 4*c))/2   
    # since b>0, root2 is always negative and discarded
    
    return np.sqrt(root1)
    
    
@np.vectorize
def ss_ellipf(phi, m):
    y = ss.ellipkinc(phi, m)
    # y = np.float(y)
    return y


@np.vectorize
def ss_ellipe(phi, m):
    y = ss.ellipeinc(phi, m)
    # y = np.float(y)
    return y


def psi_x(z, x, beta):
    """
    Eq.(24) from Ref[1] with argument zeta=0 and no constant factor e*beta**2/2/rho**2.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    # z = np.float(z)
    # x = np.float(x)
    kap = kappa(z, x, beta)
    alp = alpha(z, x, beta)
    arg2 = -4 * (1+x) / x**2
    try:
        T1 = (1/abs(x)/(1 + x) * ((2 + 2*x + x**2) * ss.ellipkinc(alp, arg2)- x**2 * ss.ellipeinc(alp, arg2)))
        D = kap**2 - beta**2 * (1 + x)**2 * sin(2*alp)**2
        T2 = ((kap**2 - 2*beta** 2 * (1+x)**2 + beta**2 * (1+x) * (2 + 2*x + x**2) * cos(2*alp))/ beta/ (1+x)/ D)
        T3 = -kap * sin(2 * alp) / D
        T4 = kap * beta ** 2 * (1 + x) * sin(2 * alp) * cos(2 * alp) / D
        T5 = 1 / abs(x) * ss.ellipkinc(alp, arg2)  # psi_phi without e/rho**2 factor
        out = real((T1 + T2 + T3 + T4) - 2 / beta ** 2 * T5)
    except ZeroDivisionError:
        out = 0
        # print(f"Oops!  ZeroDivisionError at (z,x)= ({z:5.2f},{x:5.2f}). Returning 0.")
    return np.nan_to_num(out)


def nu(x, beta):
    """
    Eq. (6) from Ref[1] (coeffient of alpha**2)
    Note that 'x' here corresponds to 'chi = x/rho' in the paper.
    """
    return 3 * (1 - beta**2 - beta**2*x) / beta**2 / (1+x)


def eta(z, x, beta):
    """
    Eq. (6) from Ref[1] (coeffient of alpha)
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    return -6 * z / beta**2 / (1+x)


def zeta(z, x, beta):
    """
    Eq. (6) from Ref[1] (constant term)
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    return 3 * (4* z**2 - beta**2 * x**2) / 4 / beta**2 / (1+x)


def Omega(z, x, beta):
    """
    Eq. (A3) from Ref[1]
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    temp = (eta(z, x, beta)**2/16
        - zeta(z, x, beta) * nu(x, beta)/6
        + nu(x, beta)**3/216)
    return temp + (temp**2 - (zeta(z, x, beta)/3 + nu(x, beta)**2/36)**3)**(1/2)


def m(z, x, beta):
    """
    Eq. (A2) from Ref[1]
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    return (-nu(x, beta)/3
        + (zeta(z, x, beta)/3 + nu(x, beta)**2/36) * Omega(z, x, beta)**(-1/3)
        + Omega(z, x, beta)**(1/3))



def alpha(z, x, beta):
    """
    Eq. (A4) from Ref[1]
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    arg1 = np.sqrt(2 * abs(m(z, x, beta)))
    arg2 = -2 * (m(z, x, beta) + nu(x, beta))
    arg3 = 2 * eta(z, x, beta) / arg1
    # out1 = real(1/2*(-arg1 + np.sqrt( abs( arg2 + arg3) )))
    # out2 = real(1/2*( arg1 + np.sqrt( abs( arg2 - arg3) )))
    # return np.nan_to_num(np.where(z<0, out1, out2))
    return np.nan_to_num(
        np.where(z < 0,
            real(1 / 2 * (-arg1 + np.sqrt(abs(arg2 + arg3)))),
            real(1 / 2 * (arg1 + np.sqrt(abs(arg2 - arg3)))) ) )


def kappa(z, x, beta):
    """
    Eq. (13) from Ref[1] with argumaent zeta = 0.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    return (x**2 + 4*(1+x) * sin(alpha(z, x, beta))**2)**(1/2)


### Functions below are obsolete


def lambda_p_Gauss(z, x):
    """
    The z derivative of a 2D Gaussian G(z,x)
    """
    sigmaz = 10e-6
    sigmax = 10e-6
    return (
        1/(2*pi*sigmaz*sigmax)
        * exp(-x**2 / 2 / sigmax**2)
        * exp(-z**2 / 2 / sigmaz**2)
        * (-z / sigmaz**2))


def make_2dgrid(func, zmin, zmax, dz, xmin, xmax, dx):
    """
    Make a 2D grid of a function
    """
    zvec = np.arange(zmin, zmax, dz)
    xvec = np.arange(xmin, xmax, dx)
    list2d = [[func(i, j) for j in xvec] for i in zvec]
    return np.array(list2d, dtype=float)


def WsOld(gamma, rho, sigmaz, sigmax, dz, dx):
    """
    Apply 2D convolution to compute the longitudinal wake Ws on a grid 
    Also returns the zvec and xvec which define the grid
    
    Still needs to improve the convolution step
    """
    beta = (1 - 1 / gamma ** 2) ** (1 / 2)

    zvec = np.arange(-5 * sigmaz, 5 * sigmaz, dz)
    xvec = np.arange(-5 * sigmax, 5 * sigmax, dx)
    lambdap_list = [[lambda_p_Gauss(i, j) for j in xvec] for i in zvec]
    lambdap_grid = np.array(lambdap_list, dtype=float)

    zvec2 = np.arange(-10 * sigmaz, 10 * sigmaz, dz)
    xvec2 = np.arange(-10 * sigmax, 10 * sigmax, dx)
    psi_s_list = [[psi_s(i / 2 / rho, j, beta) for j in xvec2] for i in zvec2]
    psi_s_grid = np.array(psi_s_list, dtype=float)

    conv_s = ss2.convolve2d(
        lambdap_grid, psi_s_grid, mode="same", boundary="fill", fillvalue=0
    )
    WsConv = beta ** 2 / rho * conv_s * (dz) * (dx)
    return zvec, xvec, WsConv


def WxOld(gamma, rho, sigmaz, sigmax, dz, dx):
    """
    Apply 2D convolution to compute the transverse wake Wx on a grid 
    Also returns the zvec and xvec which define the grid
    
    Still needs to improve the convolution step
    """
    beta = (1 - 1 / gamma ** 2) ** (1 / 2)

    zvec = np.arange(-5 * sigmaz, 5 * sigmaz, dz)
    xvec = np.arange(-5 * sigmax, 5 * sigmax, dx)
    lambdap_list = [[lambda_p_Gauss(i, j) for j in xvec] for i in zvec]
    lambdap_grid = np.array(lambdap_list, dtype=float)

    zvec2 = np.arange(-10 * sigmaz, 10 * sigmaz, dz)
    xvec2 = np.arange(-10 * sigmax, 10 * sigmax, dx)
    psi_x_list = [[psi_x(i / 2 / rho, j, beta) for j in xvec2] for i in zvec2]
    psi_x_grid = np.array(psi_x_list, dtype=float)

    conv_x = ss2.convolve2d(
        lambdap_grid, psi_x_grid, mode="same", boundary="fill", fillvalue=0
    )
    WxConv = beta ** 2 / rho * conv_x * (dz) * (dx)
    return zvec, xvec, WxConv
