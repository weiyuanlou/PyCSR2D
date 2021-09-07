import numpy as np
import scipy.special as ss

from scipy.optimize import root_scalar
from scipy import integrate

    
from csr2d.core2 import psi_x0, psi_s, Fx_case_B_Chris, psi_x0_hat, psi_x0_SC


def symmetric_vec(n, d):
    """
    Returns a symmetric vector about 0 of length 2*n with spacing d.
    The center = 0 is at [n-1]
    """
    return np.arange(-n+1,n+1,1)*d

def green_mesh(density_shape, deltas, rho=None, gamma=None, offset=(0,0,0), component='s', map_f=map):
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
    
    # handle negative rho
    rho_sign = np.sign(rho)
    rho = abs(rho)
    
    nz, nx = tuple(density_shape)
    dz, dx = tuple(deltas) # Convenience
    
    # Change to internal coordinates
    dx = dx/rho
    dz = dz/(2*rho)
    
    # Make an offset grid
    vecs = [symmetric_vec(n, delta) for n, delta, o in zip(density_shape, [dz,dx], offset)] 
    vecs[0] = rho_sign*vecs[0] # Flip sign of x
    meshes = np.meshgrid(*vecs, indexing='ij')

    beta = np.sqrt(1 - 1 / gamma ** 2)
    beta2 = 1-1/gamma**2
    
    if component == 'x':
        green = rho_sign*psi_x0(*meshes, beta, dz, dx)      
    elif component == 'xhat':
        green = rho_sign*psi_x0_hat(*meshes, beta, dz, dx)                   
    elif component == 's':
        green = psi_s(*meshes, gamma)
        
    elif component in ['Fx_IGF']:
        
        F = Fx_case_B_Chris

        # Flat meshes
        Z = meshes[0].flatten()
        X = meshes[1].flatten()

        # Select special points for IGF
        ix_for_IGF = np.where(abs(Z)<dz*1.5)
        # Select special points for IGF
       # ix_for_IGF = np.where(np.logical_and( abs(Z)<dz*2, abs(X)<dx*2 ))        
        

        print(f'IGF for {len(ix_for_IGF[0])} points...')
        
        Z_special = Z[ix_for_IGF]
        X_special = X[ix_for_IGF]
        

        # evaluate special
        f3 = lambda z, x: IGF_z(F, z, x, dz, dx, beta)/dz
        
        res = map(f3, Z_special, X_special)
        G_short = np.array(list(res))
        
        print(f'Done. Starting midpoint method...')
        
        # Simple midpoint evaluation
        G = F(Z, X, beta)
        # Fill with IGF
        G[ix_for_IGF] = G_short
        
        # reshape
        green = G.reshape(meshes[0].shape)
        
    else:
        raise ValueError(f'Unknown component: {component}')
    
    return green
    

    
def IGF_z(func, z, x, dz, dx, beta):
    """
    Special Integrated Green Function (IGF) in the z direction only
    
    """
    
    #func_x = lambda x: func(z, x, gamma)
    func_z = lambda z: func(z, x, beta)

    if abs(z) < 1e-14:
        if (abs(x) < 1e-14):
            return 0

    return integrate.quad(func_z, z-dz/2, z+dz/2, 
                          points = [z], 
                          epsrel=1e-4, # Coarse
                          limit=50)[0]        
