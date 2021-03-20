import numpy as np
import scipy.special as ss
import scipy.signal as ss2
import scipy

#from numba import jit

from numpy import abs, sin, cos, real, exp, pi, cbrt, sqrt

#@jit(nopython=True)
def psi_s(z, x, beta):
    """
    2D longitudinal potential
    Eq. (23) from Ref[1] with no constant factor (e*beta**2/2/rho**2).
    Ref[1]: Y. Cai and Yuantao. Ding, PRAB 23, 014402 (2020).
    Note that 'x' here corresponds to 'chi = x / rho' in the paper.
    """
    #try:
    
    beta2 = beta**2
    
    out = (cos(2 * alpha(z, x, beta2)) - 1 / (1+x)) / (
            kappa(z, x, beta2) - beta * (1+x) * sin(2*alpha(z, x, beta2)))
    #except ZeroDivisionError:
        # out = 0
        # print(f"Oops!  ZeroDivisionError at (z,x)= ({z:5.2f},{x:5.2f}). Returning 0.")
    return np.nan_to_num(out)

    
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
    
    beta2 = beta**2
        
    alp = alpha(z, x, beta2)
    kap = sqrt(x**2 + 4*(1+x) * sin(alp)**2) # kappa(z, x, beta2) inline
    
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)    
    
    arg2 = -4 * (1+x) / x**2
    
    ellipeinc = ss.ellipeinc(alp, arg2)
    ellipkinc = ss.ellipkinc(alp, arg2) 

    T1 = (1/abs(x)/(1 + x) * ((2 + 2*x + x**2) * ellipkinc - x**2 * ellipeinc))
    D = kap**2 - beta2 * (1 + x)**2 * sin2a**2
    T2 = ((kap**2 - 2*beta2 * (1+x)**2 + beta2 * (1+x) * (2 + 2*x + x**2) * cos2a)/ beta/ (1+x)/ D)
    T3 = -kap * sin2a / D
    T4 = kap * beta2 * (1 + x) * sin2a * cos2a / D
    T5 = 1 / abs(x) * ellipkinc # psi_phi without e/rho**2 factor
    out = (T1 + T2 + T3 + T4) - 2 / beta2 * T5

    return out

def psi_x_where_x_equals_zero(z, dx, beta):
    """
    Evaluate psi_x close to x = 0
    This is a rough approximation of the singularity across x = 0
    """
    return (psi_x(z, -dx/2, beta) + psi_x(z, dx/2, beta))/2


def psi_sx(z, x, beta):
    """
    2D longitudinal and transverse potential
    Eq. (23) from Ref[1] with no constant factor (e*beta**2/2/rho**2).
    
    Eq.(24) from Ref[1] with argument zeta=0 and no constant factor e*beta**2/2/rho**2.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    
    This is the most efficient routine.
    
    Parameters
    ----------
    z : array-like
        z/(2*rho)
        
    x : array-like
        x/rho
        
    beta : float
        Relativistic beta
        
        
    Returns
    -------
    
    psi_s, psi_x : tuple(ndarray, ndarray)
    
    
    """
    
    # beta**2 appears far more than beta. Use this in internal functions
    beta2 = beta**2
    
    alp = alpha(z, x, beta2)
    kap = sqrt(x**2 + 4*(1+x) * sin(alp)**2) # kappa(z, x, beta2) inline
    
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)    
    
    # psi_s calc
    out_psi_s = (cos2a - 1 / (1+x)) / (
            kap - beta * (1+x) * sin2a)    
      
    # psi_x calc    
    arg2 = -4 * (1+x) / x**2
    
    ellipeinc = ss.ellipeinc(alp, arg2)
    ellipkinc = ss.ellipkinc(alp, arg2) 

    T1 = (1/abs(x)/(1 + x) * ((2 + 2*x + x**2) * ellipkinc - x**2 * ellipeinc))
    D = kap**2 - beta2 * (1 + x)**2 * sin2a**2
    T2 = ((kap**2 - 2*beta2 * (1+x)**2 + beta2 * (1+x) * (2 + 2*x + x**2) * cos2a)/ beta/ (1+x)/ D)
    T3 = -kap * sin2a / D
    T4 = kap * beta2 * (1 + x) * sin2a * cos2a / D
    T5 = 1 / abs(x) * ellipkinc # psi_phi without e/rho**2 factor
    out_psi_x = (T1 + T2 + T3 + T4) - 2 / beta2 * T5

    return out_psi_s, out_psi_x



def nu(x, beta2):
    """
    Eq. (6) from Ref[1] (coeffient of alpha**2)
    Note that 'x' here corresponds to 'chi = x/rho' in the paper.
    """
    return 3 * (1 - beta2 - beta2*x) / beta2 / (1+x)


def eta(z, x, beta2):
    """
    Eq. (6) from Ref[1] (coeffient of alpha)
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    return -6 * z / beta2 / (1+x)


def zeta(z, x, beta2):
    """
    Eq. (6) from Ref[1] (constant term)
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    return 3 * (4* z**2 - beta2 * x**2) / 4 / beta2 / (1+x)


def Omega(z, x, beta2):
    """
    Eq. (A3) from Ref[1]
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    
    nu0 = nu(x, beta2)
    zeta0 = zeta(z, x, beta2)
    
    temp = (eta(z, x, beta2)**2/16
        - zeta0 * nu0/6
        + nu0**3/216)
    return temp + sqrt(temp**2 - (zeta0/3 + nu0**2/36)**3)


def m(z, x, beta2):
    """
    Eq. (A2) from Ref[1]
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    
    omega3 = cbrt(Omega(z, x, beta2))
    return (-nu(x, beta2)/3
        + (zeta(z, x, beta2)/3 + nu(x, beta2)**2/36) /omega3
        + omega3)    
    
    # Old, slightly different results
    #return (-nu(x, beta2)/3
    #    + (zeta(z, x, beta2)/3 + nu(x, beta2)**2/36) * Omega(z, x, beta2)**(-1/3)
    #    + Omega(z, x, beta2)**(1/3))


def alpha_where_z_equals_zero(x, beta2):
    """
    Evaluate alpha(z,x) when z is zero.
    Eq. (24) from Ref[1] simplifies to a quadratic equation for alpha^2.
    """
    b = nu(x,beta2)
    c = -3*(beta2 * x**2)/4/beta2/(1+x)
    root1 = (-b + np.sqrt(b**2 - 4*c))/2
    # root2 = (-b - np.sqrt(b**2 - 4*c))/2   
    # since b>0, root2 is always negative and discarded
    
    return sqrt(root1)


def alpha_where_z_not_zero(z, x, beta2):
    """
    Eq. (A4) from Ref[1]
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """

    arg1 = np.sqrt(2 * np.abs(m(z, x, beta2)))
    arg2 = -2 * (m(z, x, beta2) + nu(x, beta2))
    arg3 = 2 * eta(z, x, beta2) / arg1
    
    zsign=np.sign(z)
    
    return np.real(1 / 2 * (zsign*arg1 + np.sqrt(abs(arg2 -zsign*arg3))))


def alpha(z, x, beta2):
    on_x_axis = z == 0
    # Check for scalar, then return the normal functions
    if not isinstance(z, np.ndarray):
        if on_x_axis:
            return alpha_where_z_equals_zero(x, beta2)
        else:
            return alpha_where_z_not_zero(z, x, beta2)
    # Array z
    out = np.empty(z.shape)
    ix1 = np.where(on_x_axis)
    ix2 = np.where(~on_x_axis)
    
    if len(ix1)==0:
        print('ix1:', ix1)
        print(z)
    # Check for arrays
    if isinstance(x, np.ndarray):
        x1 = x[ix1]
        x2 = x[ix2]
    else:
        x1 = x
        x2 = x

    out[ix1] = alpha_where_z_equals_zero(x1, beta2)
    out[ix2] = alpha_where_z_not_zero(z[ix2], x2, beta2)
    return out




@np.vectorize
def alpha_exact(z, x, beta2):
    """
    Exact alpha calculation using numerical root finding.
    
    For testing only!
    
    Eq. (23) from Ref[1]
    """
    beta = sqrt(beta2)
    f = lambda a: a - beta/2*np.sqrt(x**2 + 4*(1+x)*np.sin(a)**2 ) - z
    
    res = scipy.optimize.root_scalar(f, bracket=(-1,1))
    
    return res.root


def kappa(z, x, beta2):
    """
    Eq. (13) from Ref[1] with argumaent zeta = 0.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    return sqrt(x**2 + 4*(1+x) * sin(alpha(z, x, beta2))**2)


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
