import numpy as np
import scipy.special as ss

from scipy.optimize import root_scalar
from scipy import integrate

    
from csr2d.core2 import psi_x0, psi_s, Es_case_B, Fx_case_B_Chris, Es_case_A, Fx_case_A, Es_case_C, Fx_case_C, Es_case_D, Fx_case_D, psi_s_case_E, Es_case_E

from csr2d.core2 import alpha_exact_case_B_brentq, alpha_exact_case_D_brentq

from numba import njit

from quantecon.optimize.root_finding import newton 

from scipy import optimize

from scipy.signal import find_peaks 


def symmetric_vec(n, d):
    """
    Returns a symmetric vector about 0 of length 2*n with spacing d.
    The center = 0 is at [n-1]
    """
    return np.arange(-n+1,n+1,1)*d







def green_mesh(density_shape, deltas, rho=None, gamma=None, offset=(0,0,0), 
               component='psi_s', map_f=map, phi=None, phi_m=None, lamb=None, 
               include_break_points=True, debug=False):
    """
    Computes Green funcion meshes for a particular component
    These meshes are in real space (not scaled space).
    
    Parameters
    ----------
    shape : tuple(int, int)
        Shape of the charge mesh (nz, nx)
    
    deltas : tuple(float, float)
        mesh spacing corresonding to dz, dx
        
    gamma : float
        relativistic gamma

    map_f : map function for creating potential grids.
            Examples:
                map (default)
                executor.map
    Returns:
        Double-sized array for the Green function with the speficied component

    """
    
    nz, nx = tuple(density_shape)
    dz, dx = tuple(deltas) # Convenience
    
    if debug:
        print('component:', component)
    # Change to internal coordinates
    if (component != 'psi_s_case_E') & (component != 'Es_case_E_IGF') :
        if debug:
            print('Change to internal coordinates...')
        # handle negative rho
        #rho_sign = np.sign(rho)
        #rho = abs(rho)
        dx = dx/rho
        dz = dz/(2*rho)
    
    # Make an offset grid
    vecs = [symmetric_vec(n, delta) for n, delta, o in zip(density_shape, [dz,dx], offset)] 
    #vecs[0] = rho_sign*vecs[0] # Flip sign of x
    meshes = np.meshgrid(*vecs, indexing='ij')  # this gives zm2 and xm2

    
    # Only case B has a potential form of psi_s            
    if component == 'psi_s':
        green = psi_s(*meshes, gamma)

    # psi_x is incorrect
    #elif component == 'psi_x':
    #    green = rho_sign*psi_x0(*meshes, gamma, dz, dx)  

    elif component == 'psi_s_case_E':
        green = psi_s_case_E(*meshes, gamma)
    #elif component == 'Es_case_E':
    #    green = Es_case_E(*meshes, gamma)
    
    elif component == 'Es_case_D':
        assert lamb>=0 , "lamb (exit distance over rho) must be positive for case D !"
        green = Es_case_D(*meshes, gamma, lamb)        

    # Case A fields
    elif component =='Es_case_A': 
        assert phi>=0 , "phi (entrance angle) must be positive for case A !"
        green = Es_case_A(*meshes, gamma, phi/2)
    elif component =='Fx_case_A': 
        assert phi>=0 , "phi (entrance angle) must be positive for case A !"
        green = Fx_case_A(*meshes, gamma, phi/2)
             
    # Case C fields
    elif component =='Es_case_C': 
        assert phi_m>=0 , "phi_m must be positive for case C !"
        assert lamb>=0 , "lamb (exit distance over rho) must be positive for case C !"
        green = Es_case_C(zm2, xm2, gamma, phi_m/2, lamb)
    elif component =='Fx_case_C': 
        assert phi_m>=0 , "phi_m must be positive for case C !"
        assert lamb>=0 , "lamb (exit distance over rho) must be positive for case C !"
        green = Fx_case_C(zm2, xm2, gamma, phi_m/2, lamb)

    # ===================================================    
    # Case B fields IGF
    elif component in ['Fx_case_B_IGF', 'Es_case_B_IGF','Es_case_E_IGF']:
        if component == 'Es_case_B_IGF':
            F = Es_case_B 
        elif component == 'Fx_case_B_IGF':
            F = Fx_case_B_Chris
        else:
            F = Es_case_E
            
        # Flat meshes
        Z = meshes[0].flatten()
        X = meshes[1].flatten()
        
        # Select special points for IGF
        ix_for_IGF = np.where(abs(Z) < dz*3.5)
        # ix_for_IGF = np.where(np.logical_and( abs(Z)<dz*2, abs(X)<dx*2 ))        
        
        if debug:
            print(f'Finding IGF for {len(ix_for_IGF[0])} points...')
        
        Z_special = Z[ix_for_IGF]
        X_special = X[ix_for_IGF]
        
        if include_break_points == True:
            xvec2 = vecs[1]
        
            # The spike_list can not be an numpy array since its elements have potentially different sizes
            def find_Es_case_B_spike_x(x):
                return find_Es_case_B_spike(x, gamma)
            
            spike_list = list(map(find_Es_case_B_spike_x, xvec2))           
            
            fzx = lambda z, x: IGF_z_case_B(F, z, x, dz, dx, gamma, xvec2=xvec2, spike_list=spike_list)/dz  # evaluate special
        
        else:
            fzx = lambda z, x: IGF_z_case_B(F, z, x, dz, dx, gamma)/dz  # evaluate special
        
        res = map(fzx, Z_special, X_special)
        G_short = np.array(list(res))
        
        if debug:
            print(f'Done. Starting midpoint method...')
        
        G = F(Z, X, gamma)    # Simple midpoint evaluation
        G[ix_for_IGF] = G_short   # Replace at special points with calculated IGF
        green = G.reshape(meshes[0].shape) # reshape

    # ===================================================
    # Case D fields IGF
    elif component in ['Fx_case_D_IGF', 'Es_case_D_IGF']:
        assert lamb>=0 , "lamb (exit distance over rho) must be positive for case D !"
        if component == 'Es_case_D_IGF':
            F = Es_case_D
        else:
            F = Fx_case_D        

        # Flat meshes
        Z = meshes[0].flatten()
        X = meshes[1].flatten()

        # Select special points for IGF
        ix_for_IGF = np.where(abs(Z) < dz*3.5)
        # ix_for_IGF = np.where(np.logical_and( abs(Z)<dz*2, abs(X)<dx*2 ))        
        
        print(f'Finding IGF for {len(ix_for_IGF[0])} points...')
        
        Z_special = Z[ix_for_IGF]
        X_special = X[ix_for_IGF]
        
        fzx = lambda z, x: IGF_z_case_D(F, z, x, dz, dx, gamma, lamb)/dz  # evaluate special
        res = map(fzx, Z_special, X_special)
        G_short = np.array(list(res))
        
        print(f'Done. Starting midpoint method...')
        
        G = F(Z, X, gamma, lamb)    # Simple midpoint evaluation
        G[ix_for_IGF] = G_short   # Replace at special points with calculated IGF
        green = G.reshape(meshes[0].shape) # reshape
        
    else:
        raise ValueError(f'Unknown component: {component}')
    
    return green 
    

    
def IGF_z_case_B(func, z, x, dz, dx, gamma, xvec2=None, spike_list=None):
    """
    Special Integrated Green Function (IGF) in the z direction only
    """
    
    #func_x = lambda x: func(z, x, gamma)
    func_z = lambda z: func(z, x, gamma)

    #if abs(z) < 1e-14:
    #    if (abs(x) < 1e-14):
    #        return 0

    points = [z]
    
    if spike_list != None:
        x_index = np.argmin(np.abs(xvec2 - x))
        spikes = spike_list[x_index]   # a list of z_poisition of the spikes at xvecs[x_index]
        spikes_in_dz = [zp for zp in spikes if zp < z+dz/2 and zp > z-dz/2] 
        
        # A rare situation in which too many break points are found (oscillatory curve)
        # Only use the first 20 points ( the integrator can't take more than 100? )
        if len(spikes_in_dz) > 20:
            points = [z] + spikes_in_dz[0:19]       
        
        else:
            points = [z] + spikes_in_dz 
        
    return integrate.quad(func_z, z-dz/2, z+dz/2, points = points, epsrel=1e-6, limit=100)[0]        



def IGF_z_case_D(func, z, x, dz, dx, gamma, lamb):
    """
    Special Integrated Green Function (IGF) in the z direction only
    """
    
    #func_x = lambda x: func(z, x, gamma)
    func_z = lambda z: func(z, x, gamma, lamb)

    if abs(z) < 1e-14:
        if (abs(x) < 1e-14):
            return 0

    return integrate.quad(func_z, z-dz/2, z+dz/2, 
                          points = [z], 
                          epsrel=1e-6, # Coarse
                          limit=100)[0]   

    
def IGF_z_case_E(func, z, x, dz, dx, gamma):
    """
    Special Integrated Green Function (IGF) in the z direction only
    """
    
    #func_x = lambda x: func(z, x, gamma)
    func_z = lambda z: func(z, x, gamma)

    if abs(z) < 1e-14:
        if (abs(x) < 1e-14):
            return 0

    return integrate.quad(func_z, z-dz/2, z+dz/2, 
                          points = [z], 
                          epsrel=1e-6, # Coarse
                          limit=100)[0]   


def Es_case_B_N2(z,x,gamma):
    
    beta2 = 1-1/gamma**2
    beta = np.sqrt(beta2)

    alp = alpha_exact_case_B_brentq(z, x, beta)
    sin2a = np.sin(2*alp)

    kap = (2*(alp - z))/beta # kappa for case B
    
    return (1+x)*sin2a - beta*kap


def find_Es_case_B_spike(xval,gamma):
    """
    Return a list of z values at which Es_case_B(z,xval) has spikes
    """
    def Es_case_B_N2_z(z):
        return Es_case_B_N2(z,xval,gamma)
    
    # First find where N2 ~ 0, a good reference point close to spike(s)
    op = optimize.root(Es_case_B_N2_z, 0, tol=1E-6)
    if op.success == False:
        #print('no N2 root found!!')
        return [0]
    
    root = op.x[0]

    def Es_case_B_z(z):
        return Es_case_B(z, xval, gamma)
    
    zv = np.linspace( root - 2E-11, root + 2E-11, 2001 ) # The range and resolution are subjected to changes...
    peak_ix = np.union1d(find_peaks( Es_case_B_z(zv))[0], find_peaks( -Es_case_B_z(zv))[0])
    
    return list(zv[peak_ix])


## ============== Below are higher level functions ===================================

@njit
def my_2d_convolve2(g1, g2, ix1, ix2):
    """
    Convolution for a specific observation point only, at (ix1, ix2)
    Assumption: g2 is a double-sized grid of g1. 
    
    Parameters
    ----------
    g1 : 2D array of size (nz, nx)
    
    g2 : 2D array of size (2*nz, 2*nx)
        
    ix1, ix2 : int
    
    Returns:
        A single value, the convolution result at (ix1, ix2)
    """
    d1, d2 = g1.shape
    g2_flip = np.flip(g2)
    g2_cut = g2_flip[d1-ix1:2*d1-ix1, d2-ix2:2*d2-ix2]
    
    sums = 0
    for i in range(d1):
        for j in range(d2):
            sums+= g1[i,j]*g2_cut[i,j]
    return sums


@njit
def boundary_convolve(case, z_observe, x_observe, zvec, xvec, dz, dx, lambda_grid_filtered, Green, gamma=None, rho=None, phi=None):

    beta2 = 1-1/gamma**2
    beta = np.sqrt(beta2)
    
    x_observe_index = np.argmin(np.abs(xvec - x_observe))
    z_observe_index = np.argmin(np.abs(zvec - z_observe))

    nz = len(zvec)
    nx = len(xvec)
    cond = np.zeros( (nz,nx) ) # To be filled with True and Flase
    
    # Boundary condition 
    temp = (x_observe - xvec)/rho
    
    if case == 1:
        zi_vec = rho*( phi - beta*np.sqrt(temp**2 + 4*(1 + temp)*np.sin(phi/2)**2))
        for i in range(nx):
            cond[:,i]  = (zvec > z_observe - zi_vec[i])   
            
    elif case == 2:
        zi_vec = rho*( phi - beta*np.sqrt(temp**2 + 4*(1 + temp)*np.sin(phi/2)**2))
        zo_vec = -beta*np.abs(x_observe - xvec)
        for i in range(nx):
            cond[:,i]  = (zvec > z_observe - zo_vec[i]) | (zvec < z_observe - zi_vec[i])
        
    else:
        print('Unknown case !!!')
        #raise ValueError(f'Unknown case: {case} !!!')
        
    lambda_grid_filtered_bounded = np.where(cond, 0, lambda_grid_filtered)
    
    conv = my_2d_convolve2(lambda_grid_filtered_bounded, Green, z_observe_index, x_observe_index) 
    
    return conv