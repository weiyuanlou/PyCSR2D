import numpy as np

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

        charge_grid[ip][jp-1]   = charge_grid[ip][jp-1]   + t3
        charge_grid[ip][jp]     = charge_grid[ip][jp]     + t4
        
    return charge_grid