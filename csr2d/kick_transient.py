from csr2d.deposit import histogram_cic_2d
from csr2d.central_difference import central_difference_z
from csr2d.core2 import psi_sx, psi_s, psi_x0, Es_case_A, Fx_case_A, Es_case_C, Fx_case_C
from csr2d.convolution import fftconvolve2

import numpy as np

from scipy.signal import savgol_filter
from scipy.interpolate import RectBivariateSpline
#from scipy.signal import convolve2d, fftconvolve, oaconvolve
from scipy.ndimage import map_coordinates


import scipy.constants
mec2 = scipy.constants.value("electron mass energy equivalent in MeV") * 1e6
c_light = scipy.constants.c
e_charge = scipy.constants.e
r_e = scipy.constants.value("classical electron radius")

import time


# @njit (doesn't like savgol filter...)
def compute_dist_grid(z_b, x_b, weight, *, nz=100, nx=100, xlim=None, zlim=None, debug=False):

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
    
    return zvec, xvec, dz, dx, lambda_grid_filtered, lambda_grid_filtered_prime




#@njit # (jit doesn't like np.meshgrid...)
def compute_potential_grids(case, nz=100, nx=100, dz=None, dx=None, rho=None, gamma=None, phi=None, phi_m=None, lamb=None):
    """
    The output of the 4 cases are:
        Case A, C, D: Es_grid, Fx_grid, zvec2, xvec2
        Case B:       psi_s_grid, psi_x_grid, zvec2, xvec2    
    """
    
    assert rho>0 , "rho (bending angle) must be positive!!!"
    
    #rho_sign = 1 if rho>=0 else -1
    
    ## Change to internal coordinates
    dx = dx/rho
    dz = dz/(2*abs(rho))
    
    ## Double-sized array for convolution with the density
    zvec2 = np.arange(-nz+1,nz+1,1)*dz # center = 0 is at [nz-1]
    xvec2 = np.arange(-nx+1,nx+1,1)*dx # center = 0 is at [nx-1]
    
    zm2, xm2 = np.meshgrid(zvec2, xvec2, indexing="ij")

    if   case=='A': 
        assert phi>0 , "phi (entrance angle) must be positive!!!"
        Es_case_A_grid = Es_case_A(zm2, xm2, gamma, phi/2) # Numba routines!
        Fx_case_A_grid = Fx_case_A(zm2, xm2, gamma, phi/2) # Numba routines!
        
        return Es_case_A_grid, Fx_case_A_grid, zvec2*2*rho, xvec2*rho
        
        #return green_meshes_case_A(nz, nx, dz, dx, rho=rho, beta=beta, alp=phi/2) 
    
    elif case=='B':
        
        psi_s_grid = psi_s(zm2, xm2, gamma) # Numba routines!
        psi_x_grid = psi_x0(zm2, xm2, gamma, abs(dx)) # Will average around 0
    
        return psi_s_grid, psi_x_grid, zvec2*2*rho, xvec2*rho
    
    #    return green_meshes(nz, nx, dz, dx, rho=rho, beta=beta)  
    elif case=='C':
        assert phi_m>0 , "phi_m must be positive!!!"
        assert lamb>0 , "lamb (exit distance over rho) must be positive!!!"
        
        Es_case_C_grid = Es_case_C(zm2, xm2, gamma, phi_m/2, lamb) # Numba routines!
        Fx_case_C_grid = Fx_case_C(zm2, xm2, gamma, phi_m/2, lamb) # Numba routines!
        
        return Es_case_C_grid, Fx_case_C_grid, zvec2*2*rho, xvec2*rho
    #    return green_meshes_case_C(nz, nx, dz, dx, rho=rho, beta=beta, alp=phi_m/2, lamb=lamb) 
    
    
    else:
        print('INVALID CASE GIVEN. MUST be one of ABCDE!!')



#@njit #(doesn't like fftconvole2 and condition_grid )
def boundary_convolve(
    case, x_observe, zvec=None, xvec=None,  zvec2=None, xvec2=None, 
    G_lamb = None, G_lamb_p = None, Gs=None, Gx=None, 
    beta=None, rho=None, phi=None, phi_m=None, lamb=None, dx=None):
    """
    The grids required for the 4 cases are:
        Case A, C, D: 
            G_lamb = lambda_grid, Gs = Es_grid, Gx = Fx_grid
        Case B: 
            G_lamb = lambda_grid, G_lamb_p = lambda_grid_prime, Gs = psi_s_grid, Gx = psi_x_grid

    The "optional" parameters required by the 4 cases are:
        Case A: phi 
        Case B: phi, zvec, dx (for on-axis calculation and boundary terms ) 
        Case C: phi_m, lamb
        Case D: phi_m, lamb (, and dx for boundary terms if formula available)
    """

    # Strictly speaking x_observe should be a value from xvec
    x_observe_index = np.argmin(np.abs(xvec - x_observe))
    #print('x_observe_index :', x_observe_index )
    
    if case == 'A':
        # Boundary condition 
        temp = (x_observe - xvec2)/abs(rho)
        zi_vec = abs(rho)*(phi-beta*np.sqrt(temp**2 + 4*(1+temp)*np.sin(phi/2)**2))
        #zo_vec = -beta*np.abs(x_observe - xvec2)  

        # Want "True" if (z < zi), where the potential grid values are set to ZERO
        condition_grid = np.array([(zvec2 < zi_vec[i]) for i in range(len(xvec2))])
        Es_grid_bounded = np.where(condition_grid.T, 0, Gs)
        Fx_grid_bounded = np.where(condition_grid.T, 0, Gx)

        
        #### Method one: FFT #######
        conv_s, conv_x = fftconvolve2(G_lamb, Es_grid_bounded, Fx_grid_bounded)

        return conv_s[:,x_observe_index], conv_x[:,x_observe_index]

        
        ####### Method 2: William writing new Convolve to compute values only along specified x_oberseve_index
        #Ws_along_x_observe = \
        #my_super_convolve2(lambda_grid_filtered, Es_grid_bounded, np.arange(len(zvec)), x_observe_index)
        #Wx_along_x_observe = \
        #my_super_convolve2(lambda_grid_filtered, Fx_grid_bounded, np.arange(len(zvec)), x_observe_index)
        
        
        # convolve4 is slower than convolve2....
        #Ws_along_x_observe, Wx_along_x_observe= \
        #my_super_convolve4(lambda_grid_filtered, Es_grid_bounded, Fx_grid_bounded, np.arange(len(zvec)), x_observe_index)
        
        #return Ws_along_x_observe, Wx_along_x_observe
        ###############################################
        
    elif case=='C':
        temp = (x_observe - xvec2)/rho
        zid_vec = rho*(phi_m + lamb - beta*np.sqrt(lamb**2 + temp**2 + 4*(1+temp)*np.sin(phi_m/2)**2 + 2*lamb*np.sin(phi_m)))

        condition_grid = np.array([(zvec2 < zid_vec[i]) for i in range(len(xvec2))])
        Es_grid_bounded = np.where(condition_grid.T, 0, Gs)
        Fx_grid_bounded = np.where(condition_grid.T, 0, Gx)

        conv_s, conv_x = fftconvolve2(G_lamb, Es_grid_bounded, Fx_grid_bounded)
    
        return conv_s[:,x_observe_index], conv_x[:,x_observe_index]
    
    elif case=='B':
        temp = (x_observe - xvec2)/rho
        zi_vec = rho*( phi - beta*np.sqrt(temp**2 + 4*(1+temp)*np.sin(phi/2)**2))
        zo_vec = -beta*np.abs(x_observe - xvec2)


        # Computing the integral term of case B
        # Want "True" if (z > zi) OR (z < zo), where the potential values are set to ZERO
        condition_grid = np.array([(zvec2 > zi_vec[i]) | (zvec2 < zo_vec[i]) for i in range(len(xvec2))])
        psi_s_grid_bounded = np.where(condition_grid.T, 0, Gs)
        psi_x_grid_bounded = np.where(condition_grid.T, 0, Gx)
        conv_s, conv_x = fftconvolve2(G_lamb_p, psi_s_grid_bounded, psi_x_grid_bounded)
        #Ws_grid = (beta**2 / abs(rho)) * (conv_s) * (dz * dx)
        #Wx_grid = (beta**2 / abs(rho)) * (conv_x) * (dz * dx)

        # Computing the two boundary terms of case B 
        # ==========================================
        lambda_interp = RectBivariateSpline(zvec, xvec, G_lamb)  # lambda lives in the observation grid

        lambda_zi_vec = lambda z: lambda_interp.ev( z - zi_vec, xvec2 )
        psi_s_zi_vec = psi_s(zi_vec/2/np.abs(rho), temp, beta)
        Ws_zi = lambda z:  np.dot(psi_s_zi_vec, lambda_zi_vec(z))
        Ws_zi_vec = np.array(list(map(Ws_zi, zvec)))
        psi_x_zi_vec = psi_x0(zi_vec/2/np.abs(rho), temp, beta, dx)
        Wx_zi = lambda z:  np.dot(psi_x_zi_vec, lambda_zi_vec(z))
        Wx_zi_vec = np.array(list(map(Wx_zi, zvec)))


        lambda_zo_vec = lambda z: lambda_interp.ev( z - zo_vec, xvec2 )
        psi_s_zo_vec = psi_s(zo_vec/2/np.abs(rho), temp, beta)
        Ws_zo = lambda z: (-1.0) * np.dot(psi_s_zo_vec, lambda_zo_vec(z))
        Ws_zo_vec = np.array(list(map(Ws_zo, zvec)))
        psi_x_zo_vec = psi_x0(zo_vec/2/np.abs(rho), temp, beta, dx)
        Wx_zo = lambda z: (-1.0) * np.dot(psi_x_zo_vec, lambda_zo_vec(z))
        Wx_zo_vec = np.array(list(map(Wx_zo, zvec)))

        return conv_s[:,x_observe_index], Ws_zi_vec, Ws_zo_vec, conv_x[:,x_observe_index], Wx_zi_vec, Wx_zo_vec
    
    else:
        print('INVALID CASE GIVEN. MUST be one of ABCDE!!')