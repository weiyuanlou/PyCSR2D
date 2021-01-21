import numpy as np

def central_difference_z(lambda_grid, Nz, Nx, dz, order=1):
    """
    Take the differentiation in z of a "2D z-x grid" via central difference. 
    """
    if (order==1):
        lambda_grid_temp = np.vstack((np.zeros(Nx),lambda_grid,np.zeros(Nx)))
        ##print(lambda_grid_temp.shape)

        lambda_prime_grid = (lambda_grid_temp[2:Nz+2] - lambda_grid_temp[0:Nz]) / (2*dz)
        
    elif (order==2):
        lambda_grid_temp = np.vstack((np.zeros(Nx),np.zeros(Nx),lambda_grid,np.zeros(Nx),np.zeros(Nx)))
        ##print(lambda_grid_temp.shape)

        lambda_prime_grid = (-   lambda_grid_temp[4:Nz+4] \
                             + 8*lambda_grid_temp[3:Nz+3] \
                             - 8*lambda_grid_temp[1:Nz+1] \
                             +   lambda_grid_temp[0:Nz]) / (12*dz)
        
    elif (order==3):
        lambda_grid_temp = np.vstack((np.zeros(Nx),np.zeros(Nx),np.zeros(Nx),lambda_grid,np.zeros(Nx),np.zeros(Nx),np.zeros(Nx)))
        ##print(lambda_grid_temp.shape)

        lambda_prime_grid = (     lambda_grid_temp[6:Nz+6] \
                     -  9*lambda_grid_temp[5:Nz+5] \
                     + 45*lambda_grid_temp[4:Nz+4] \
                     - 45*lambda_grid_temp[3:Nz+3] \
                     +  9*lambda_grid_temp[5:Nz+5] \
                     -    lambda_grid_temp[6:Nz+6] ) / (60*dz)
    else:
        raise ValueError(' order value has to be 1 or 2 or 3!! ')
        
    return lambda_prime_grid