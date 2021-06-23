
from numba import vectorize, float64, njit
# For special functions
from numba.extending import get_cython_function_address
import ctypes

import numpy as np
import scipy.special as ss
import scipy.signal as ss2
import scipy

from numpy import abs, sin, cos, real, exp, pi, cbrt, sqrt

from quantecon.optimize.root_finding import brentq


def old_psi_s(z, x, beta):
    """
    2D longitudinal potential
    Eq. (23) from Ref[1] with no constant factor (e*beta**2/2/rho**2).
    Ref[1]: Y. Cai and Yuantao. Ding, PRAB 23, 014402 (2020).
    Note that 'x' here corresponds to 'chi = x / rho' in the paper.
    """

    beta2 = beta**2
    
    out = (cos(2 * alpha(z, x, beta2)) - 1 / (1+x)) / (
            kappa(z, x, beta2) - beta * (1+x) * sin(2*alpha(z, x, beta2)))

    return out

def old_psi_x(z, x, beta):
    """
    Eq.(24) from Ref[1] with argument zeta=0 and no constant factor e*beta**2/2/rho**2.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    
    beta2 = beta**2
        
    alp = old_alpha(z, x, beta2)
    kap = sqrt(x**2 + 4*(1+x) * sin(alp)**2) # kappa(z, x, beta2) inline
    
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)    
    
    arg2 = -4 * (1+x) / x**2
    
    ellipkinc = ss.ellipkinc(alp, arg2) 
    ellipeinc = ss.ellipeinc(alp, arg2)
    
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


def old_alpha(z, x, beta2):
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



@vectorize([float64(float64, float64, float64)])
def alpha(z, x, beta2):
    """
    Numba vectorized form of alpha.
    See: https://numba.pydata.org/numba-doc/dev/user/vectorize.html
    
    
    Eq. (6) from Ref[X] using the solution in 
    Eq. (A4) from Ref[1] 
    
    
    """
    if z == 0:
        # Quadratic solution
        
        b = 3 * (1 - beta2 - beta2*x) / beta2 / (1+x)    
        c = -3*(x**2)/(4*(1+x))
    
        root1 = (-b + np.sqrt(b**2 - 4*c))/2
        
        return np.sqrt(root1)
        
    # Quartic solution 
        
    # Terms of the depressed quartic equation
    eta = -6 * z / (beta2 * (1+x))
    nu = 3 * (1/beta2 - 1 - x) / (1+x)
    zeta = (3/4) * (4* z**2 /beta2 - x**2) / (1+x)
    
    # Omega calc and cube root
    temp = (eta**2/16 - zeta * nu/6 + nu**3/216)  
    Omega =  temp + np.sqrt(temp**2 - (zeta/3 + nu**2/36)**3)  
    #omega3 = np.cbrt(Omega) # Not supported in Numba! See: https://github.com/numba/numba/issues/5385
    omega3= Omega**(1/3)
    
    # Eq. (A2) from Ref[1]
    m = -nu/3 + (zeta/3 + nu**2/36) /omega3 + omega3
     
    arg1 = np.sqrt(2 * abs(m))
    arg2 = -2 * (m + nu)
    arg3 = 2 * eta / arg1
    
    zsign= np.sign(z)
    
    return (zsign*arg1 + np.sqrt(abs(arg2 -zsign*arg3)))/2


@vectorize([float64(float64, float64, float64)], target='parallel')
def psi_s(z, x, beta):
    """
    2D longitudinal potential
    
    Numba vectorized
    
    Eq. (23) from Ref[1] with no constant factor (e*beta**2/2/rho**2).
    Ref[1]: Y. Cai and Yuantao. Ding, PRAB 23, 014402 (2020).
    Note that 'x' here corresponds to 'chi = x / rho' in the paper.
    """
    
    if z == 0 and x == 0:
        return 0
    
    beta2 = beta**2
    
    alp = alpha(z, x, beta2)
    kap = sqrt(x**2 + 4*(1+x) * sin(alp)**2)
    
    out = (cos(2*alp)- 1/(1+x)) / (
            kap - beta * (1+x) * sin(2*alp))   

    return out

# Include special functions for Numba
#
# Tip from: https://github.com/numba/numba/issues/3086
# and http://numba.pydata.org/numba-doc/latest/extending/high-level.html
#
addr1 = get_cython_function_address('scipy.special.cython_special', 'ellipkinc')
addr2 = get_cython_function_address('scipy.special.cython_special', 'ellipeinc')
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
my_ellipkinc = functype(addr1)
my_ellipeinc = functype(addr2)


@vectorize([float64(float64, float64, float64)])
def psi_x(z, x, beta):
    """
    Eq.(24) from Ref[1] with argument zeta=0 and no constant factor e*beta**2/2/rho**2.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    
    
    #if x == 0.0:
    ## Can't do this   
    #dx = 1e-15
    #    return (new_psi_x(z, -dx, beta) + new_psi_x(z, dx, beta))/2
    
    beta2 = beta**2
        
    alp = alpha(z, x, beta2)
    kap = sqrt(x**2 + 4*(1+x) * sin(alp)**2) # kappa(z, x, beta2) inline
    
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)    
    
    arg2 = -4 * (1+x) / x**2
    
    F = my_ellipkinc(alp, arg2) 
    E = my_ellipeinc(alp, arg2)
    
    
    T1 = (1/abs(x)/(1 + x) * ((2 + 2*x + x**2)*F - x**2*E))
    D = kap**2 - beta2 * (1 + x)**2 * sin2a**2
    T2 = ((kap**2 - 2*beta2*(1+x)**2 + beta2*(1+x)*(2 + 2*x + x**2)*cos2a)/ beta/ (1+x)/ D)
    T3 = -kap * sin2a / D
    T4 = kap * beta2 * (1 + x) * sin2a * cos2a / D
    T5 = 1 / abs(x) * F # psi_phi without e/rho**2 factor
    out = (T1 + T2 + T3 + T4) - 2 / beta2 * T5
    
    return out

@vectorize([float64(float64, float64, float64, float64)], target='parallel')
def psi_x0(z, x, beta, dx):
    """
    Same as psi_x, but checks for x==0, and averages over +/- dx/2
    
    """
    
    if x == 0:
        return (psi_x(z, -dx/2, beta) +  psi_x(z, dx/2, beta))/2
    else:
        return  psi_x(z, x, beta)


##################################################
### Transient fields and potentials ##############
##################################################


############ Case A ##############################

@vectorize([float64(float64, float64, float64, float64)])
def eta_case_A(z, x, beta2, alp):
    """
    Eq.(?) from Ref[1] slide 11
    "eta" here is H/rho, not to be confused with the eta function.
    "alp" here is half of the bending angle, not the alpha function.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    sin2a = sin(2*alp)
    
    a = (1-beta2)/4
    b = alp - z - beta2*(1+x)*sin2a/2
    c = alp**2 - 2*alp*z + z**2 - beta2*x**2/4 - beta2*(1+x)*sin(alp)**2
    
    return (-b + sqrt(b**2 - 4*a*c)) / (2*a)


@vectorize([float64(float64, float64, float64, float64)])
def Es_case_A(z, x, beta, alp):
    """
    Eq.(?) from Ref[2] with no constant factor e/gamma**2/rho**2.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    'alp' is half the observational angle here.
    """

    if z == 0 and x == 0 and alp==0:
        return 0
    
    beta2 = beta**2
    sin2a = sin(2*alp)
    cos2a = cos(2*alp) 
    eta = eta_case_A(z, x, beta2, alp)
    kap = sqrt( eta**2 + x**2 + 4*(1+x)*sin(alp)**2 + 2*eta*(1+x)*sin2a) # kappa for case A
    
    N = sin2a + (eta - beta*kap)*cos2a
    D = kap - beta*(eta + (1+x)*sin2a)
    
    return N/D**3
    
    
@vectorize([float64(float64, float64, float64, float64)])
def Fx_case_A(z, x, beta, alp):
    """
    Eq.(?) from Ref[2] with no constant factor e**2/gamma**2/rho**2.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    'alp' is half the observational angle here.
    """
        
    beta2 = beta**2
    sin2a = sin(2*alp)
    cos2a = cos(2*alp) 
    eta = eta_case_A(z, x, beta2, alp)
    kap = sqrt( eta**2 + x**2 + 4*(1+x)*sin(alp)**2 + 2*eta*(1+x)*sin2a) # kappa for case A
    
    N1 = (1 + beta2)*(1+x)
    N2 = -(1 + beta2*(1+x)**2)*cos2a
    N3 = (eta - beta*kap)*sin2a
    D = kap - beta*(eta + (1+x)*sin2a)
    
    return (N1+N2+N3)/D**3


########### Case B #################################
########## Note that psi_s and psi_x above are also for case_B

@vectorize([float64(float64, float64, float64)])
def Es_case_B(z, x, beta):
    """
    Eq.(?) from Ref[2] slide #21 with no constant factor e*beta**2/rho**2.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
  
    if z == 0 and x == 0:
        return 0
    
    beta2 = beta**2
    alp = alpha(z, x, beta2)
    sin2a = sin(2*alp)
    cos2a = cos(2*alp) 

    kap = sqrt(x**2 + 4*(1+x)*sin(alp)**2) # kappa for case B
    
    N1 = cos2a - (1+x)
    N2 = (1+x)*sin2a - beta*kap
    D = kap - beta*(1+x)*sin2a
    
    return N1*N2/D**3


@vectorize([float64(float64, float64, float64, float64)], target='parallel')
def Es_case_B0(z, x, beta, dx):
    """
    Same as Es_case_B, but checks for x==0, and averages over +/- dx/2
    
    """
    
   # if x == 0:
   #     return (Es_case_B(z, -dx/2, beta) +  Es_case_B(z, dx/2, beta))/2
    
    if z == 0:
        return 0
    else:
        return Es_case_B(z, x, beta)
    

############# Case C ####################################
@vectorize([float64(float64, float64, float64, float64, float64)])
def eta_case_C(z, x, beta2, alp, lamb):
    """
    Eq.(?) from Ref[1] slide 11
    "eta" here is H/rho, not to be confused with the eta function.
    "alp" here is half of the bending angle, not the alpha function.
    "lamb" is L/rho, where L is the bunch center location down the bending exit.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)
    
    a = (1-beta2)/4
    b = alp - z + lamb/2 - lamb*beta2*cos2a/2 - beta2*(1+x)*sin2a/2
    c = alp**2 + alp*lamb + (1-beta2)*lamb**2/4 - 2*alp*z - lamb*z + z**2 - beta2*x**2/4 - beta2*(1+x)*sin(alp)**2 - lamb*beta2*sin2a/2
    
    return (-b + sqrt(b**2 - 4*a*c)) / (2*a)


@vectorize([float64(float64, float64, float64, float64, float64)])
def Es_case_C(z, x, beta, alp, lamb):
    """
    Eq.(?) from Ref[2] with no constant factor e/gamma**2/rho**2.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    'alp' is half the observational angle here.
    """

    if z == 0 and x == 0 and alp == 0:
        return 0
    
    beta2 = beta**2
    sin2a = sin(2*alp)
    cos2a = cos(2*alp) 
    eta = eta_case_C(z, x, beta2, alp, lamb)
    kap = sqrt( lamb**2 + eta**2 + x**2 + 4*(1+x)*sin(alp)**2 + 2*(lamb + eta*(1+x))*sin2a + 2*lamb*eta*cos2a) # kappa for case C
    
    N = lamb + sin2a + (eta - beta*kap)*cos2a
    D = kap - beta*(eta + lamb*cos2a + (1+x)*sin2a)
    
    return N/D**3


@vectorize([float64(float64, float64, float64, float64, float64)])
def Fx_case_C(z, x, beta, alp, lamb):
    """
    Eq.(?) from Ref[2] with no constant factor e**2/gamma**2/rho**2.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    'alp' is half the observational angle here.
    """
        
    beta2 = beta**2
    sin2a = sin(2*alp)
    cos2a = cos(2*alp) 
    eta = eta_case_C(z, x, beta2, alp, lamb)
    kap = sqrt( lamb**2 + eta**2 + x**2 + 4*(1+x)*sin(alp)**2 + 2*(lamb + eta*(1+x))*sin2a + 2*lamb*eta*cos2a) # kappa for case C
    
    N1 = (1 + beta2)*(1+x)
    N2 = -(1 + beta2*(1+x)**2)*cos2a
    N3 = (eta - beta*kap + beta2*lamb*(1+x))*sin2a
    D = kap - beta*(eta + lamb*cos2a + (1+x)*sin2a)
    
    return (N1+N2+N3)/D**3


############################### Case D #################################

@np.vectorize
def alpha_exact_case_D(z, x, beta, lamb):
    """
    Exact alpha calculation using numerical root finding.

    """
    #beta = np.sqrt(beta2)
    f = lambda a: a + 1/2 * (lamb - beta*sqrt(lamb**2 + x**2 + 4*(1+x)*sin(a)**2 + 2*lamb*sin(2*a)) - z)
    
    res = scipy.optimize.root_scalar(f, bracket=(-1,1))
    
    return res.root

@njit
def f_root_case_D(a, z, x, beta, lamb):
    return a + 1/2 * (lamb - beta* sqrt(lamb**2 + x**2 + 4*(1+x)*sin(a)**2 + 2*lamb*sin(2*a)) - z)


#@vectorize([float64(float64, float64, float64, float64)], target='parallel')
@vectorize([float64(float64, float64, float64, float64)])
def alpha_exact_case_D_brentq(z, x, beta, lamb):
    """
    Exact alpha calculation for case D using numerical Brent's method of root finding.

    """
    #return brentq(ff, -0.01, 0.1, args=(z, x, beta, lamb))[0]
    return brentq(f_root_case_D, -1, 1, args=(z, x, beta, lamb))[0]


#@vectorize([float64(float64, float64, float64, float64)])
#@np.vectorize
@vectorize([float64(float64, float64, float64, float64)])
def Es_case_D(z, x, beta, lamb):
    """
    Eq.(?) from Ref[2] slide #21 with no constant factor e*beta**2/rho**2.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
  
    if z == 0 and x == 0:
        return 0
    
    #alp = alpha_exact_case_D(z, x, beta, lamb)  # old method
    alp = alpha_exact_case_D_brentq(z, x, beta, lamb)
    sin2a = sin(2*alp)
    cos2a = cos(2*alp) 

    kap = sqrt(lamb**2 + x**2 + 4*(1+x)*sin(alp)**2 + 2*lamb*sin(2*alp)) # kappa for case D
    
    N1 = cos2a - (1+x)
    N2 = lamb*cos2a + (1+x)*sin2a - beta*kap
    D = kap - beta*(lamb*cos2a + (1+x)*sin2a)
    
    return N1*N2/D**3