import mpmath as mp
import numpy as np
import scipy.signal as ss
from mpmath import mpf, sin, cos, exp, re, pi, ellipf, ellipe

def psi_s(z,x,beta):
    """
    2D Longitudinal potential of Eq.(23) with no constant factor (e*beta**2/2/rho**2).
    """
    z = float(z)
    x = float(x)
    try:
        out = (cos(2*alpha(z,x,beta))-1/(1+x))/(kappa(z,x,beta)-beta*(1+x)*sin(2*alpha(z,x,beta)))
    except ZeroDivisionError:
        out = 0
        #print(f"Oops!  ZeroDivisionError at (z,x)= ({z:5.2f},{x:5.2f}). Returning 0.")
    return out

def psi_x(z,x,beta):
    """
    2D transverse potential of Eq.(24) with no constant factor e*beta**2/2/rho**2.
    """
    z = float(z)
    x = float(x)
    try:     
        T1 = 1/abs(x)/(1+x)*((2+2*x+x**2)*ellipf(alpha(z,x,beta),-4*(1+x)/x**2)  -  x**2*ellipe(alpha(z,x,beta),-4*(1+x)/x**2))
        D = kappa(z,x,beta)**2-beta**2*(1+x)**2*(sin(2*alpha(z,x,beta)))**2
        T2 = (kappa(z,x,beta)**2 - 2*beta**2*(1+x)**2 + beta**2*(1+x)*(2+2*x+x**2)*cos(2*alpha(z,x,beta)))/beta/(1+x)/D
        T3 = -kappa(z,x,beta)*sin(2*alpha(z,x,beta))/D
        T4 = kappa(z,x,beta)*beta**2*(1+x)*sin(2*alpha(z,x,beta))*cos(2*alpha(z,x,beta))/D
        T5 = 1/abs(x)*ellipf(alpha(z,x,beta),-4*(1+x)/x**2)   # psi_phi without e/rho**2 factor
        out = re((T1+T2+T3+T4)-2/beta**2*T5)
    except ZeroDivisionError:
        out = 0
        #print(f"Oops!  ZeroDivisionError at (z,x)= ({z:5.2f},{x:5.2f}). Returning 0.")
    return out


def nu(x,beta):
    return 3*(1-beta**2-beta**2*x)/beta**2/(1+x)
def eta(z,x,beta):
    return -6*z/beta**2/(1+x)
def zeta(z,x,beta):
    return 3*(4*z**2-beta**2*x**2)/4/beta**2/(1+x)
def Omega(z,x,beta):
    temp = eta(z,x,beta)**2/16 - zeta(z,x,beta)*nu(x,beta)/6 + nu(x,beta)**3/216
    return temp + (temp**2-(zeta(z,x,beta)/3+nu(x,beta)**2/36)**3)**(1/2)
def m(z,x,beta):
    return -nu(x,beta)/3 + (zeta(z,x,beta)/3+nu(x,beta)**2/36)*Omega(z,x,beta)**(-1/3) + Omega(z,x,beta)**(1/3)
def alpha(z,x,beta):
    if z<0: 
        out=  1/2*(-(2*m(z,x,beta))**(1/2) + (  -2*(m(z,x,beta)+nu(x,beta))+2*eta(z,x,beta)*(2*m(z,x,beta))**(-1/2)   )**(1/2))
    else: 
        out=  1/2*( (2*m(z,x,beta))**(1/2) + (  -2*(m(z,x,beta)+nu(x,beta))-2*eta(z,x,beta)*(2*m(z,x,beta))**(-1/2)   )**(1/2))
    return re(out)
def kappa(z,x,beta):
    return (x**2+4*(1+x)*sin(alpha(z,x,beta))**2)**(1/2)


def Lambda_p_Gauss(z, x): 
    """
    The z derivative of a 2D Gaussian G(z,x)
    """
    sigmaz = 10E-6 
    sigmax = 10E-6
    return 1/(2*pi*sigmaz*sigmax)*exp(-x**2/2/sigmax**2)*exp(-z**2/2/sigmaz**2)*(-z/sigmaz**2)

def make_2dgrid(func,zmin,zmax,dz, xmin, xmax, dx):
    """
    Make a 2D grid of a function
    """
    zvec = np.arange(zmin, zmax, dz)
    xvec = np.arange(xmin, xmax, dx)
    list2d= [[func(i,j) for j in xvec] for i in zvec] 
    return np.array(list2d,dtype=float)
    
def Ws(gamma,rho,sigmaz,sigmax,dz,dx):
    """
    Apply 2D convolution to compute the longitudinal wake Ws on a grid 
    Also returns the zvec and xvec which define the grid
    """
    #dz = 0.09*sigmaz
    #dx = 0.09*sigmax
    beta = (1-1/gamma**2)**(1/2)
    zvec = np.arange(-5*sigmaz, 5*sigmaz, dz)
    xvec = np.arange(-5*sigmax, 5*sigmax, dx)
    lambdap_list = [[Lambda_p_Gauss(i,j) for j in xvec] for i in zvec] 
    lambdap_grid = np.array(lambdap_list,dtype=float)
    zvec2 = np.arange(-10*sigmaz, 10*sigmaz, dz)
    xvec2 = np.arange(-10*sigmax, 10*sigmax, dx)
    psi_s_list = [[psi_s(i/2/rho,j,beta) for j in xvec2] for i in zvec2]   
    psi_s_grid = np.array(psi_s_list,dtype=float)
    conv_s=ss.convolve2d(lambdap_grid, psi_s_grid, mode='same', boundary='fill', fillvalue=0)
    WsConv=beta**2/rho*conv_s*(dz)*(dx)
    return zvec, xvec, WsConv