import numpy as np
import time

from scipy.ndimage import convolve as conv
from scipy.signal import convolve2d, fftconvolve, oaconvolve
from cupyx.scipy.ndimage import convolve as cupy_conv


from csr2d.deposit import split_particles, deposit_particles, histogram_cic_2d
from csr2d.central_difference import central_difference_z
from csr2d.core import psi_s, psi_x
#from csr2d.beam_conversion import particle_group_to_bmad, bmad_to_particle_group
from csr2d.simple_track import track_a_bend, track_a_drift

import concurrent.futures as cf

#from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.interpolate import RectBivariateSpline

def calc_csr_kick(beam, charges, Np, gamma, rho, Nz=100, sigma_z=1E-3, Nx=100, sigma_x=1E-3, reuse_psi_grids=False, psi_s_grid_old=None, psi_x_grid_old=None, verbose=True):
    """
    """
    (x_b, xp_b, y_b, yp_b, z_b, zp_b) = beam
    zx_positions = np.stack((z_b, x_b)).T
    
    # Fix the grid here
    # The grid needs to enclose most particles in z-x space  
    mins = np.array([-6*sigma_z,-6*sigma_x])  # Lower bounds of the grid
    maxs = np.array([ 6*sigma_z, 6*sigma_x])     # Upper bounds of the grid
    sizes = np.array([Nz, Nx])

    (zmin,xmin) = mins
    (zmax,xmax) = maxs
    (Nz,Nx) = sizes
    (dz,dx) = (maxs - mins)/(sizes-1)  # grid steps
    
    
    #indexes, contrib = split_particles(zx_positions, charges, mins, maxs, sizes)
    #t1 = time.time();
    #charge_grid = deposit_particles(Np, sizes, indexes, contrib)
    #t2 = time.time();
    
    t1 = time.time();
    charge_grid = histogram_cic_2d( z_b, x_b, charges, Nz, zmin, zmax, Nx, xmin, xmax )
    t2 = time.time();
    
    
        
    # Normalize the grid so its integral is unity
    norm = np.sum(charge_grid)*dz*dx
    lambda_grid = charge_grid/norm
    
    # Apply savgol filter
    lambda_grid_filtered = np.array([savgol_filter( lambda_grid[:,i], 13, 2 ) for i in np.arange(Nx)]).T
    
    # Differentiation in z
    lambda_grid_filtered_prime = central_difference_z(lambda_grid_filtered, Nz, Nx, dz, order=1)
    
    
    zvec = np.linspace(zmin, zmax, Nz)
    xvec = np.linspace(xmin, xmax, Nx)
    
    beta = (1-1/gamma**2)**(1/2)
    
    
    t3 = time.time()
    
    if (reuse_psi_grids == True):
        psi_s_grid = psi_s_grid_old
        psi_x_grid = psi_x_grid_old
        
    else:
        # Creating the potential grids 
        zvec2 = np.linspace(2*zmin,2*zmax,2*Nz)
        xvec2 = np.linspace(2*xmin,2*xmax,2*Nx)
        zm2, xm2 = np.meshgrid(zvec2, xvec2, indexing='ij')
        #psi_s_grid = psi_s(zm2,xm2,beta)
    
        beta_grid = beta*np.ones(zm2.shape)
        with cf.ProcessPoolExecutor(max_workers = 12) as executor:
            temp = executor.map(psi_s, zm2/2/rho, xm2, beta_grid)
            psi_s_grid = np.array(list(temp))
            temp2 = executor.map(psi_x, zm2/2/rho, xm2, beta_grid)
            psi_x_grid = np.array(list(temp2))
    
    t4 = time.time();

    if verbose:
        print('Depositting particles takes:', t2 - t1, 's')
        print('Computing potential grids take:', t4 - t3, 's')
        
        
    # Compute the wake via 2d convolution
    conv_s = oaconvolve(lambda_grid_filtered_prime, psi_s_grid, mode='same')
    conv_x = oaconvolve(lambda_grid_filtered_prime, psi_x_grid, mode='same')
        
    Ws_grid = (beta**2/rho)*(conv_s)*(dz*dx)
    Wx_grid = (beta**2/rho)*(conv_x)*(dz*dx)
    
    # Interpolate Ws and Wx everywhere within the grid
    Ws_interp = RectBivariateSpline(zvec, xvec, Ws_grid)
    Wx_interp = RectBivariateSpline(zvec, xvec, Wx_grid)


    r_e = 2.8179403227E-15 
    q_e = 1.602176634E-19
    Nb =  np.sum(charge_grid)/q_e
    kick_factor = r_e*Nb/gamma
    
    # Calculate the kicks at the paritcle locations
    delta_kick = kick_factor * Ws_interp.ev(z_b, x_b)
    xp_kick    = kick_factor * Wx_interp.ev(z_b, x_b)
    
    
    return {'zvec':zvec, 'xvec':xvec, 'delta_kick':delta_kick, 'xp_kick':xp_kick, 'Ws_grid':Ws_grid, 'Wx_grid':Wx_grid, 'psi_s_grid':psi_s_grid, 'psi_x_grid':psi_x_grid, 'charge_grid':charge_grid}
    
    #return delta_kick, xp_kick