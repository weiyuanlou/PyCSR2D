from csr2d.deposit import histogram_cic_2d
from csr2d.central_difference import central_difference_z
from csr2d.core2 import psi_sx, psi_s, psi_x0,  psi_x0_hat, Es_case_B0, Es_case_A, Fx_case_A, Es_case_C, Fx_case_C, Es_case_D
from csr2d.core2 import psi_s_SC, psi_x0_SC
from csr2d.convolution import fftconvolve2

import numpy as np

from scipy.signal import savgol_filter
from scipy.interpolate import RectBivariateSpline
#from scipy.signal import convolve2d, fftconvolve, oaconvolve
from scipy.ndimage import map_coordinates

from numba import njit

import scipy.constants

mec2 = scipy.constants.value("electron mass energy equivalent in MeV") * 1e6
c_light = scipy.constants.c
e_charge = scipy.constants.e
r_e = scipy.constants.value("classical electron radius")

import time


def csr2d_kick_calc(
    z_b,
    x_b,
    weight,
    *,
    gamma=None,
    rho=None,
    nz=100,
    nx=100,
    xlim=None,
    zlim=None,
    reuse_psi_grids=False,
    psi_s_grid_old=None,
    psi_x_grid_old=None,
    map_f=map,
    species="electron",
    imethod='map_coordinates',
    debug=False,
):
    """
    
    Calculates the 2D CSR kick on a set of particles with positions `z_b`, `x_b` and charges `charges`.
    
    
    Parameters
    ----------
    z_b : np.array
        Bunch z coordinates in [m]

    x_b : np.array
        Bunch x coordinates in [m]
  
    weight : np.array
        weight array (positive only) in [C]
        This should sum to the total charge in the bunch
        
    gamma : float
        Relativistic gamma
        
    rho : float
        bending radius in [m]
        if neagtive, particles with a positive x coordinate is on the inner side of the magnet
        
    nz : int
        number of z grid points
        
    nx : int
        number of x grid points        
    
    zlim : floats (min, max) or None
        z grid limits in [m]
        
    xlim : floats (min, max) or None  
        x grid limits in [m]
        
    map_f : map function for creating potential grids.
            Examples:
                map (default)
                executor.map
    
    species : str
        Particle species. Currently required to be 'electron'
        
    imethod : str
        Interpolation method for kicks. Must be one of:
            'map_coordinates' (default): uses  scipy.ndimage.map_coordinates 
            'spline': uses: scipy.interpolate.RectBivariateSpline
    
    debug: bool
        If True, returns the computational grids. 
        Default: False
        
              
    Returns
    -------
    dict with:
    
        ddelta_ds : np.array
            relative z momentum kick [1/m]
            
        dxp_ds : np.array
            relative x momentum kick [1/m]
    """
    assert species == "electron", "TODO: support species {species}"
    # assert np.sign(rho) == 1, 'TODO: negative rho'

    # Grid setup
    if zlim:
        zmin = zlim[0]
        zmax = zlim[1]
    else:
        zmin = z_b.min()
        zmax = z_b.max()

    if xlim:
        xmin = xlim[0]
        xmax = xlim[1]
    else:
        xmin = x_b.min()
        xmax = x_b.max()

    dz = (zmax - zmin) / (nz - 1)
    dx = (xmax - xmin) / (nx - 1)

    # Charge deposition
    t1 = time.time()
    charge_grid = histogram_cic_2d(z_b, x_b, weight, nz, zmin, zmax, nx, xmin, xmax)

    if debug:
        t2 = time.time()
        print("Depositing particles takes:", t2 - t1, "s")

    # Normalize the grid so its integral is unity
    norm = np.sum(charge_grid) * dz * dx
    lambda_grid = charge_grid / norm

    # Apply savgol filter
    lambda_grid_filtered = np.array([savgol_filter(lambda_grid[:, i], 13, 2) for i in np.arange(nx)]).T

    # Differentiation in z
    lambda_grid_filtered_prime = central_difference_z(lambda_grid_filtered, nz, nx, dz, order=1)

    # Grid axis vectors
    zvec = np.linspace(zmin, zmax, nz)
    xvec = np.linspace(xmin, xmax, nx)

    beta = np.sqrt(1 - 1 / gamma ** 2)

    t3 = time.time()

    if reuse_psi_grids == True:
        psi_s_grid = psi_s_grid_old
        psi_x_grid = psi_x_grid_old

    else:
        # Creating the potential grids        
        psi_s_grid, psi_x_grid, zvec2, xvec2 = green_meshes(nz, nx, dz, dx, rho=rho, beta=beta)  
    
    if debug:
        t4 = time.time()
        print("Computing potential grids take:", t4 - t3, "s")

    # Compute the wake via 2d convolution
    conv_s, conv_x = fftconvolve2(lambda_grid_filtered_prime, psi_s_grid, psi_x_grid)

    if debug:
        t5 = time.time()
        print("Convolution takes:", t5 - t4, "s")

    Ws_grid = (beta ** 2 / abs(rho)) * (conv_s) * (dz * dx)
    Wx_grid = (beta ** 2 / abs(rho)) * (conv_x) * (dz * dx)

    # Calculate the kicks at the particle locations
    
    # Overall factor
    Nb = np.sum(weight) / e_charge
    kick_factor = r_e * Nb / gamma  # m
        
    # Interpolate Ws and Wx everywhere within the grid
    if imethod == 'spline':
        # RectBivariateSpline method
        Ws_interp = RectBivariateSpline(zvec, xvec, Ws_grid)
        Wx_interp = RectBivariateSpline(zvec, xvec, Wx_grid)
        delta_kick = kick_factor * Ws_interp.ev(z_b, x_b)
        xp_kick = kick_factor * Wx_interp.ev(z_b, x_b)
    elif imethod == 'map_coordinates':
        # map_coordinates method. Should match above fairly well. order=1 is even faster.
        zcoord = (z_b-zmin)/dz
        xcoord = (x_b-xmin)/dx
        delta_kick = kick_factor * map_coordinates(Ws_grid, np.array([zcoord, xcoord]), order=2)
        xp_kick    = kick_factor * map_coordinates(Wx_grid, np.array([zcoord, xcoord]), order=2)    
    else:
        raise ValueError(f'Unknown interpolation method: {imethod}')
    
    if debug:
        t6 = time.time()
        print(f'Interpolation with {imethod} takes:', t6 - t5, "s")        


    result = {"ddelta_ds": delta_kick, "dxp_ds": xp_kick}

    if debug:
        timing = np.array([t2-t1, t4-t3, t5-t4, t6-t5])
        result.update(
            {
                "zvec": zvec,
                "xvec": xvec,
                "zvec2": zvec2,
                "xvec2": xvec2,
                "Ws_grid": Ws_grid,
                "Wx_grid": Wx_grid,
                "psi_s_grid": psi_s_grid,
                "psi_x_grid": psi_x_grid,
                "charge_grid": charge_grid,
                "lambda_grid_filtered_prime": lambda_grid_filtered_prime,
                "timing": timing
            }
        )

    return result


def green_meshes(nz, nx, dz, dx, rho=None, beta=None):
    """
    Computes Green funcion meshes for psi_s and psi_x simultaneously.
    These meshes are in real space (not scaled space).
    
    Parameters
    ----------
    nz, nx : int
        Size of the density mesh in z and x

    dz, dx : float
        Grid spacing of the density mesh in z and x [m]
        
    rho : float
        bending radius (must be positve)
        
    beta : float
        relativistic beta
    
    Returns:
    tuple of:
        psi_s_grid : np.array
            Double-sized array for the psi_s Green function
        
        psi_x_grid : 
            Double-sized array for the psi_x Green function
        
        zvec2 : array-like
            Coordinate vector in z (real space) [m]

        xvec2 : array-like
            Coordinate vector in x (real space) [m]
    
    """
    rho_sign = 1 if rho>=0 else -1
    
    # Change to internal coordinates
    dx = dx/rho
    dz = dz/(2*abs(rho))
    
    # Double-sized array for convolution with the density
    zvec2 = np.arange(-nz+1,nz+1,1)*dz # center = 0 is at [nz-1]
    xvec2 = np.arange(-nx+1,nx+1,1)*dx # center = 0 is at [nx-1]
    
    # Corrections to avoid the singularity at x=0
    # This will calculate just off axis. Note that we don't need the last item, 
    # because the density mesh does not span that far
    #xvec2[nx-1] = -dx/2
    #xvec2[-1] = dx/2 
    
    zm2, xm2 = np.meshgrid(zvec2, xvec2, indexing="ij")
    
    # Evaluate
    #psi_s_grid, psi_x_grid = psi_sx(zm2, xm2, beta)
    psi_s_grid = psi_s(zm2, xm2, beta) # Numba routines!
    psi_x_grid = rho_sign*psi_x0(zm2, xm2, beta, abs(dx)) # Will average around 0
    
    # Average out the values around x=0
    #psi_s_grid[:,nx-1] = (psi_s_grid[:,nx-1] + psi_s_grid[:,-1])/2
    #psi_x_grid[:,nx-1] = (psi_x_grid[:,nx-1] + psi_x_grsid[:,-1])/2    
    
    # Remake this 
    #xvec2 = np.arange(-nx+1,nx+1,1)*dx*rho
    
    return psi_s_grid, psi_x_grid, zvec2*2*rho, xvec2*rho

def green_meshes_hat(nz, nx, dz, dx, rho=None, beta=None):
    """
    Computes Green funcion meshes for psi_s and psi_x simultaneously.
    These meshes are in real space (not scaled space).
    
    Parameters
    ----------
    nz, nx : int
        Size of the density mesh in z and x

    dz, dx : float
        Grid spacing of the density mesh in z and x [m]
        
    rho : float
        bending radius (must be positve)
        
    beta : float
        relativistic beta
    
    Returns:
    tuple of:
        psi_s_grid : np.array
            Double-sized array for the psi_s Green function
        
        psi_x_grid : 
            Double-sized array for the psi_x Green function
        
        zvec2 : array-like
            Coordinate vector in z (real space) [m]

        xvec2 : array-like
            Coordinate vector in x (real space) [m]
    
    """
    rho_sign = 1 if rho>=0 else -1
    
    # Change to internal coordinates
    dx = dx/rho
    dz = dz/(2*abs(rho))
    
    # Double-sized array for convolution with the density
    zvec2 = np.arange(-nz+1,nz+1,1)*dz # center = 0 is at [nz-1]
    xvec2 = np.arange(-nx+1,nx+1,1)*dx # center = 0 is at [nx-1]
    
    
    zm2, xm2 = np.meshgrid(zvec2, xvec2, indexing="ij")
    
    # Evaluate
    psi_s_grid = psi_s(zm2, xm2, beta) # Numba routines!
    psi_x_grid = rho_sign*psi_x0_hat(zm2, xm2, beta, abs(dx)) # Will average around 0
    
    
    return psi_s_grid, psi_x_grid, zvec2*2*rho, xvec2*rho


def green_meshes_SC(nz, nx, dz, dx, rho=None, beta=None):
    """
    Computes Green funcion meshes for psi_s and psi_x simultaneously.
    These meshes are in real space (not scaled space).
    
    Parameters
    ----------
    nz, nx : int
        Size of the density mesh in z and x

    dz, dx : float
        Grid spacing of the density mesh in z and x [m]
        
    rho : float
        bending radius (must be positve)
        
    beta : float
        relativistic beta
    
    Returns:
    tuple of:
        psi_s_grid : np.array
            Double-sized array for the psi_s Green function
        
        psi_x_grid : 
            Double-sized array for the psi_x Green function
        
        zvec2 : array-like
            Coordinate vector in z (real space) [m]

        xvec2 : array-like
            Coordinate vector in x (real space) [m]
    
    """
    rho_sign = 1 if rho>=0 else -1
    
    # Change to internal coordinates
    dx = dx/rho
    dz = dz/(2*abs(rho))
    
    # Double-sized array for convolution with the density
    zvec2 = np.arange(-nz+1,nz+1,1)*dz # center = 0 is at [nz-1]
    xvec2 = np.arange(-nx+1,nx+1,1)*dx # center = 0 is at [nx-1]
    
    
    zm2, xm2 = np.meshgrid(zvec2, xvec2, indexing="ij")
    
    # Evaluate
    psi_s_grid = psi_s_SC(zm2, xm2, beta) # Numba routines!
    psi_x_grid = psi_x0_SC(zm2, xm2, beta, abs(dx)) # Will average around 0
    
    
    return psi_s_grid, psi_x_grid, zvec2*2*rho, xvec2*rho


def green_meshes_case_B(nz, nx, dz, dx, rho=None, beta=None):
    """
    Computes Green funcion meshes for psi_s and psi_x simultaneously.
    These meshes are in real space (not scaled space).
    
    Parameters
    ----------
    nz, nx : int
        Size of the density mesh in z and x

    dz, dx : float
        Grid spacing of the density mesh in z and x [m]
        
    rho : float
        bending radius (must be positve)
        
    beta : float
        relativistic beta
    
    Returns:
    tuple of:
        Es_case_B_grid : np.array
            Double-sized array for the Es field function of case B
        
        zvec2 : array-like
            Coordinate vector in z (real space) [m]

        xvec2 : array-like
            Coordinate vector in x (real space) [m]
    
    """
    rho_sign = 1 if rho>=0 else -1
    
    # Change to internal coordinates
    dx = dx/rho
    dz = dz/(2*abs(rho))
    
    # Double-sized array for convolution with the density
    zvec2 = np.arange(-nz+1,nz+1,1)*dz # center = 0 is at [nz-1]
    xvec2 = np.arange(-nx+1,nx+1,1)*dx # center = 0 is at [nx-1]
    
    
    zm2, xm2 = np.meshgrid(zvec2, xvec2, indexing="ij")
    
    Es_case_B_grid = Es_case_B0(zm2, xm2, beta, dx) # Numba routines!
    
    return Es_case_B_grid, zvec2*2*rho, xvec2*rho


def green_meshes_case_A(nz, nx, dz, dx, rho=None, beta=None, alp=None):
    """
    Computes Green funcion meshes for psi_s and psi_x simultaneously.
    These meshes are in real space (not scaled space).
    
    Parameters
    ----------
    nz, nx : int
        Size of the density mesh in z and x

    dz, dx : float
        Grid spacing of the density mesh in z and x [m]
        
    rho : float
        bending radius (must be positve)
        
    beta : float
        relativistic beta
        
    alp : half of the bending angle = L/2/abs(rho)
    
    Returns:
    tuple of:
        Es_case_A_grid : np.array
            Double-sized array for the Es function of case A

        Fx_case_A_grid : np.array
            Double-sized array for the Fx function of case A
            
        zvec2 : array-like
            Coordinate vector in z (real space) [m]

        xvec2 : array-like
            Coordinate vector in x (real space) [m]
    
    """
    rho_sign = 1 if rho>=0 else -1
    
    # Change to internal coordinates
    dx = dx/rho
    dz = dz/(2*abs(rho))
    
    # Double-sized array for convolution with the density
    zvec2 = np.arange(-nz+1,nz+1,1)*dz # center = 0 is at [nz-1]
    xvec2 = np.arange(-nx+1,nx+1,1)*dx # center = 0 is at [nx-1]
    
    
    zm2, xm2 = np.meshgrid(zvec2, xvec2, indexing="ij")
    
    Es_case_A_grid = Es_case_A(zm2, xm2, beta, alp) # Numba routines!
    Fx_case_A_grid = Fx_case_A(zm2, xm2, beta, alp) # Numba routines!
    
    return Es_case_A_grid, Fx_case_A_grid, zvec2*2*rho, xvec2*rho

    
def green_meshes_case_C(nz, nx, dz, dx, rho=None, beta=None, alp=None, lamb=None):
    """
    Computes Green funcion meshes for psi_s and psi_x simultaneously.
    These meshes are in real space (not scaled space).
    
    Parameters
    ----------
    nz, nx : int
        Size of the density mesh in z and x

    dz, dx : float
        Grid spacing of the density mesh in z and x [m]
        
    rho : float
        bending radius (must be positve)
        
    beta : float
        relativistic beta
        
    alp : half of the bending angle = Lm/2/abs(rho)
    
    lamb : L/rho, L is the bunch center location down the bending exit
    
    Returns:
    tuple of:
        Es_case_C_grid : np.array
            Double-sized array for the Es function of case C

        Fx_case_C_grid : np.array
            Double-sized array for the Fx function of case C
            
        zvec2 : array-like
            Coordinate vector in z (real space) [m]

        xvec2 : array-like
            Coordinate vector in x (real space) [m]
    
    """
    rho_sign = 1 if rho>=0 else -1
    
    # Change to internal coordinates
    dx = dx/rho
    dz = dz/(2*abs(rho))
    
    # Double-sized array for convolution with the density
    zvec2 = np.arange(-nz+1,nz+1,1)*dz # center = 0 is at [nz-1]
    xvec2 = np.arange(-nx+1,nx+1,1)*dx # center = 0 is at [nx-1]
    
    
    zm2, xm2 = np.meshgrid(zvec2, xvec2, indexing="ij")
    
    Es_case_C_grid = Es_case_C(zm2, xm2, beta, alp, lamb) # Numba routines!
    Fx_case_C_grid = Fx_case_C(zm2, xm2, beta, alp, lamb) # Numba routines!
    
    return Es_case_C_grid, Fx_case_C_grid, zvec2*2*rho, xvec2*rho


def green_meshes_case_D(nz, nx, dz, dx, rho=None, beta=None, lamb=None):
    """
    Computes Green funcion meshes for psi_s and psi_x simultaneously.
    These meshes are in real space (not scaled space).
    
    Parameters
    ----------
    nz, nx : int
        Size of the density mesh in z and x

    dz, dx : float
        Grid spacing of the density mesh in z and x [m]
        
    rho : float
        bending radius (must be positve)
        
    beta : float
        relativistic beta
    
    Returns:
    tuple of:
        Es_case_B_grid : np.array
            Double-sized array for the Es field function of case B
        
        zvec2 : array-like
            Coordinate vector in z (real space) [m]

        xvec2 : array-like
            Coordinate vector in x (real space) [m]
    
    """
    rho_sign = 1 if rho>=0 else -1
    
    # Change to internal coordinates
    dx = dx/rho
    dz = dz/(2*abs(rho))
    
    # Double-sized array for convolution with the density
    zvec2 = np.arange(-nz+1,nz+1,1)*dz # center = 0 is at [nz-1]
    xvec2 = np.arange(-nx+1,nx+1,1)*dx # center = 0 is at [nx-1]
    
    
    zm2, xm2 = np.meshgrid(zvec2, xvec2, indexing="ij")
    
    Es_case_D_grid = Es_case_D(zm2, xm2, beta, lamb)
    
    return Es_case_D_grid, zvec2*2*rho, xvec2*rho



def csr1d_steady_state_kick_calc(z, weights, nz=100, rho=1, species="electron", normalized_units=False):

    """

    Steady State CSR 1D model kick calc

    
    Parameters
    ----------
    z : np.array
        Bunch z coordinates in [m]    
        
    weights : np.array
        weight array (positive only) in [C]
        This should sum to the total charge in the bunch        
        
    nz : int
        number of z grid points        
        
    rho : float
        bending radius in [m]        
        
    species : str
        Particle species. Currently required to be 'electron'   
        
    normalized_units : bool
        If True, will return in normalized units [1/m^2]
            This multiplied by Qtot / e_charge * r_e * mec2 gives:
        Otherwise, units of [eV/m] are returned (default).
        Default: False
        
    Returns
    -------
    dict with:
    
        denergy_ds : np.array
            energy kick for each particle in [eV/m], or [1/m^2] if normalized_units=True
            
        wake : np.array
            wake array that kicks were interpolated on
            
        zvec : np.array
            z coordinates for wake array
    
    """

    assert species == "electron", f"TODO: support species {species}"

    # Density
    H, edges = np.histogram(z, weights=weights, bins=nz)
    zmin, zmax = edges[0], edges[-1]
    dz = (zmax - zmin) / (nz - 1)

    zvec = np.linspace(zmin, zmax, nz)  # Sloppy with bin centers

    Qtot = np.sum(weights)
    density = H / dz / Qtot

    # Density derivative
    densityp = np.gradient(density) / dz
    densityp_filtered = savgol_filter(densityp, 13, 2)

    # Green function
    zi = np.arange(0, zmax - zmin, dz)
    #factor =
    # factor = -3**(2/3) * Qtot/e_charge * r_e * rho**(-2/3) / gamma  # factor for ddelta/ds [1/m]
    if normalized_units:
        factor =  -3**(2/3) * rho**(-2/3) # factor for normalized uinits [1/m^2]
    else:
        factor =  ( -3**(2/3) * Qtot / e_charge * r_e * mec2 * rho**(-2/3)  )  # factor for denergy/dz [eV/m]
    green = factor * np.diff(zi ** (2 / 3))

    # Convolve to get wake
    wake = np.convolve(densityp_filtered, green, mode="full")[0 : len(zvec)]

    # Interpolate to get the kicks
    delta_kick = np.interp(z, zvec, wake)

    return {"denergy_ds": delta_kick, "zvec": zvec, "wake": wake}
