import numpy as np
from numba import jit
import math

def split_particles(position, charge, mins, maxs, sizes):
    '''
    mins  = (xmin, ymin, zmin)
    maxs  = (xmax, ymax, zmax)
    sizes = (Nx, Ny, Nz)
    '''

    if ((mins.shape != maxs.shape) | (mins.shape != sizes.shape)):
        print('Dimension mismatch between mins, maxs, and sizes!!')
        return None
    
    if (np.any( position - mins < 0 ) | np.any( position - maxs >= 0 )):
        print(np.any( position - mins < 0 ) )
        print(np.any( position - maxs >= 0 ) )
        print('Particle position falls outside the boundary specified!')
        return None
    
    deltas = (maxs - mins) / (sizes - 1)
        #print('deltas: ',deltas)
    floors = np.floor( (position - mins)/deltas + 1 )  # index of the nearest "floor point"
    floors = floors.astype(int)
        #print('floors: ', floors)
    weights = ((mins - position) + floors*deltas) / deltas  # werights towards the floor point
        #print('weights: ', weights)
    
    dim = sizes.shape[0]
    if (dim > 3) :
        print('Dimension > 3 detected!!')
        return None

    elif (dim == 3):

        ip = floors[:,0]
        jp = floors[:,1]
        kp = floors[:,2]
        
        w1 = weights[:,0]
        w2 = weights[:,1]
        w3 = weights[:,2]        
 
        t1 = w1*w2*w3*charge
        t2 = w1*(1-w2)*w3*charge
        t3 = w1*(1-w2)*(1-w3)*charge
        t4 = w1*w2*(1-w3)*charge

        t5 = (1-w1)*w2*w3*charge
        t6 = (1-w1)*(1-w2)*w3*charge
        t7 = (1-w1)*(1-w2)*(1-w3)*charge
        t8 = (1-w1)*w2*(1-w3)*charge

        indexes = np.array([ip,jp,kp])
        contrib = np.array([t1,t2,t3,t4,t5,t6,t7,t8])
        
    elif (dim == 2):
        ip = floors[:,0]
        jp = floors[:,1]
        
        w1 = weights[:,0]
        w2 = weights[:,1]     
 
        t1 = w1*w2*charge
        t2 = w1*(1-w2)*charge
        t3 = (1-w1)*(1-w2)*charge
        t4 = (1-w1)*w2*charge

        indexes = np.array([ip,jp])
        contrib = np.array([t1,t2,t3,t4])
    
    elif (dim == 1):

        ip = floors[:,0]    
        w1 = weights[:,0]   
 
        t1 = w1*charge
        t2 = (1-w1)*charge

        indexes = np.array([ip])
        contrib = np.array([t1,t2])
        
    return indexes, contrib


def deposit_particles(Np, sizes, indexes, contrib):
    """
    Deposit the "splitted particles" on an empty grid.
    Use this function after the "split_particles" function.
    For large Np this function can be slow.
    """
    
    charge_grid = np.zeros(sizes)
    # Populate charge_grid
    for n in range(Np):
        (ip, jp) = indexes[:,n] # depositting index of the nth particle
        (t1, t2, t3, t4) = contrib[:,n] # contribtuion of the nth particle
    
        charge_grid[ip-1][jp-1] = charge_grid[ip-1][jp-1] + t1
        charge_grid[ip-1][jp]   = charge_grid[ip-1][jp]   + t2

        charge_grid[ip][jp]     = charge_grid[ip][jp]     + t3
        charge_grid[ip][jp-1]   = charge_grid[ip][jp-1]   + t4
        
    return charge_grid


@jit
def histogram_cic_2d( q1, q2, w,
    nbins_1, bins_start_1, bins_end_1,
    nbins_2, bins_start_2, bins_end_2 ):
    """
    Return an 2D histogram of the values in `q1` and `q2` weighted by `w`,
    consisting of `nbins_1` bins in the first dimension and `nbins_2` bins
    in the second dimension.
    Contribution to each bins is determined by the
    Cloud-in-Cell weighting scheme.

    Source: 
    ----------
        https://github.com/openPMD/openPMD-viewer/blob/dev/openpmd_viewer/openpmd_timeseries/utilities.py
    ----------
    
    Parameters:
    ----------
    q1/q2 : float, array
            q1/q2 position of the particles
    w: float, array
            weights (charges) of the particles
    nbins_1/nbins_2 : int
            number of bins (vertices) in the q1/q2 direction
    bins_start_1, bins_end_1, bins_start_2, bins_end_2: float
            start/end value in the q1/q2 direction
    ----------
    
    Returns:
    ----------
    A 2D array of size (nbins_1, nbins_2)
    ----------
    """
    # Define various scalars
    bin_spacing_1 = (bins_end_1-bins_start_1)/(nbins_1-1)
    inv_spacing_1 = 1./bin_spacing_1
    bin_spacing_2 = (bins_end_2-bins_start_2)/(nbins_2-1)
    inv_spacing_2 = 1./bin_spacing_2
    n_ptcl = len(w)

    # Allocate array for histogrammed data
    hist_data = np.zeros( (nbins_1, nbins_2), dtype=np.float64 )

    # Go through particle array and bin the data
    for i in range(n_ptcl):

        # Calculate the index of lower bin to which this particle contributes
        q1_cell = (q1[i] - bins_start_1) * inv_spacing_1
        q2_cell = (q2[i] - bins_start_2) * inv_spacing_2
        i1_low_bin = int( math.floor( q1_cell ) )
        i2_low_bin = int( math.floor( q2_cell ) )

        # Calculate corresponding CIC shape and deposit the weight
        S1_low = 1. - (q1_cell - i1_low_bin)
        S2_low = 1. - (q2_cell - i2_low_bin)
        if (i1_low_bin >= 0) and (i1_low_bin < nbins_1):
            if (i2_low_bin >= 0) and (i2_low_bin < nbins_2):
                hist_data[ i1_low_bin, i2_low_bin ] += w[i]*S1_low*S2_low
            if (i2_low_bin+1 >= 0) and (i2_low_bin+1 < nbins_2):
                hist_data[ i1_low_bin, i2_low_bin+1 ] += w[i]*S1_low*(1.-S2_low)
        if (i1_low_bin+1 >= 0) and (i1_low_bin+1 < nbins_1):
            if (i2_low_bin >= 0) and (i2_low_bin < nbins_2):
                hist_data[ i1_low_bin+1, i2_low_bin ] += w[i]*(1.-S1_low)*S2_low
            if (i2_low_bin+1 >= 0) and (i2_low_bin+1 < nbins_2):
                hist_data[ i1_low_bin+1, i2_low_bin+1 ] += w[i]*(1.-S1_low)*(1.-S2_low)

    return( hist_data )