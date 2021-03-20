import numpy as np

import scipy.fft as sp_fft


def fftconvolve2(rho, *greens):
    """
    Efficiently perform a 2D convolution of a charge density rho and multiple Green functions. 
    
    Parameters
    ----------
    
    rho : np.array (2D)
        Charge mesh
        
    *greens : np.arrays (2D)
        Charge meshes for the Green functions, which should be twice the size of rho    
        
        
    Returns
    -------
    
    fields : tuple of np.arrays with the same shape as rho. 
    
    """

    # FFT Configuration
    fft  = lambda x: sp_fft.fft2(x,  overwrite_x=True)
    ifft = lambda x: sp_fft.ifft2(x, overwrite_x=True)    
    
    # Place rho in double-sized array. Should match the shape of green
    n0, n1 = rho.shape
    crho = np.zeros( (2*n0, 2*n1))
    crho[0:n0,0:n1] = rho[0:n0,0:n1]
    
    # FFT
    crho = fft(crho)    
   
    results = []
    for green in greens:
        assert crho.shape == green.shape, f'Green array shape {green.shape} should be twice rho shape {rho.shape}'
        result = ifft(crho*fft(green))
        # Extract the result
        result = np.real(result[n0-1:2*n0-1,n1-1:2*n1-1])
        results.append(result)
        
        
        
    return tuple(results)