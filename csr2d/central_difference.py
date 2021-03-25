import numpy as np

def central_difference_z(grid, Nz, Nx, dz, order=1):
    """
    Take the differentiation in z of a "2D z-x grid" via central difference. 
    """
    if (order==1):
        grid_temp = np.vstack((np.zeros(Nx), grid, np.zeros(Nx)))

        prime_grid = (grid_temp[2:Nz+2] - grid_temp[0:Nz]) / (2*dz)
        
        # Endpoints with 2nd order accuracy
        prime_grid[0]    = ((-3/2)*grid[0]   + 2*grid[1]    - (1/2)*grid[2]) / dz
        prime_grid[nz-1] = ((1/2)*grid[nz-3] - 2*grid[nz-2] + (3/2)*grid[nz-1]) / dz
        
    elif (order==2):
        grid_temp = np.vstack((np.zeros(Nx), np.zeros(Nx), grid, np.zeros(Nx), np.zeros(Nx)))

        prime_grid = (-   grid_temp[4:Nz+4] \
                      + 8*grid_temp[3:Nz+3] \
                      - 8*grid_temp[1:Nz+1] \
                      +   grid_temp[0:Nz]) / (12*dz)
        
    elif (order==3):
        grid_temp = np.vstack((np.zeros(Nx),np.zeros(Nx),np.zeros(Nx), grid, np.zeros(Nx),np.zeros(Nx),np.zeros(Nx)))

        prime_grid = (    grid_temp[6:Nz+6] \
                     -  9*grid_temp[5:Nz+5] \
                     + 45*grid_temp[4:Nz+4] \
                     - 45*grid_temp[2:Nz+2] \
                     +  9*grid_temp[1:Nz+1] \
                     -    grid_temp[0:Nz] ) / (60*dz)
    else:
        raise ValueError(' order value has to be 1 or 2 or 3!! ')
        
    return prime_grid