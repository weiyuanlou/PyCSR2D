

from csr2d.wake import green_mesh,  boundary_convolve
from csr2d.deposit import histogram_cic_2d
#from csr2d.central_difference import central_difference_z
from csr2d.convolution import fftconvolve2
from csr2d.simple_track import track_a_bend, track_a_drift, track_a_bend_parallel, track_a_drift_parallel
from csr2d.beam_conversion import particle_group_to_bmad, bmad_to_particle_group

import numpy as np
import time 

from scipy.signal import savgol_filter
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates

from numba import njit, vectorize, float64



import scipy.constants

mec2 = scipy.constants.value("electron mass energy equivalent in MeV") * 1e6
c_light = scipy.constants.c
e_charge = scipy.constants.e
r_e = scipy.constants.value("classical electron radius")


def csr2d_kick_calc_transient(
    z_b,
    x_b,
    weight,
    *,
    gamma=None,
    rho=None,
    phi=None,
    steady_state=False,
    nz=100,
    nx=100,
    xlim=None,
    zlim=None,
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
        
    phi : float
        entrance angle in radian
        
    nz : int
        number of z grid points
        
    nx : int
        number of x grid points     
        
    steady_state : boolean
        If True, the transient terms in case A and B are ignored
    
    zlim : floats (min, max) or None
        z grid limits in [m]
        
    xlim : floats (min, max) or None  
        x grid limits in [m]
    
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

    # Apply savgol filter to the distribution grid
    lambda_grid_filtered = np.array([savgol_filter(lambda_grid[:, i], 13, 2) for i in np.arange(nx)]).T
    
    # Differentiation in z 
    #lambda_grid_filtered_prime = central_difference_z(lambda_grid_filtered, nz, nx, dz, order=1)

    # Distribution grid axis vectors
    zvec = np.linspace(zmin, zmax, nz)
    xvec = np.linspace(xmin, xmax, nx)
    Z, X = np.meshgrid(zvec, xvec, indexing='ij')

    beta = np.sqrt(1 - 1/gamma**2)

    t3 = time.time()

    Es_case_B_grid_IGF = green_mesh((nz, nx), (dz, dx), rho=rho, gamma=gamma, component= 'Es_case_B_IGF')
    Fx_case_B_grid_IGF = green_mesh((nz, nx), (dz, dx), rho=rho, gamma=gamma, component= 'Fx_case_B_IGF')

    if debug:
        t4 = time.time()
        print("Computing case B field grids takes:", t4 - t3, "s")
    
    if steady_state==True:
        
        ### Compute the wake via 2d convolution (no boundary condition)
        #conv_s, conv_x = fftconvolve2(lambda_grid_filtered_prime, psi_s_grid, psi_x_grid)
        conv_s, conv_x = fftconvolve2(lambda_grid_filtered, Es_case_B_grid_IGF, Fx_case_B_grid_IGF)
        
        Ws_grid = (beta ** 2 / abs(rho)) * (conv_s) * (dz * dx)
        Wx_grid = (beta ** 2 / abs(rho)) * (conv_x) * (dz * dx)
        
    else: 
        Es_case_A_grid = green_mesh((nz, nx), (dz, dx), rho=rho, gamma=gamma, component= 'Es_case_A', phi = phi, debug=False)
        Fx_case_A_grid = green_mesh((nz, nx), (dz, dx), rho=rho, gamma=gamma, component= 'Fx_case_A', phi = phi, debug=False)
        if debug:
            print("Case A field grids computed!")
        
        @vectorize([float64(float64, float64)], target='parallel')
        def boundary_convolve_Ws_A_super(z_observe, x_observe):
            return boundary_convolve(1, z_observe, x_observe, zvec, xvec, dz, dx, lambda_grid_filtered, Es_case_A_grid, gamma=gamma, rho=rho, phi=phi)
        @vectorize([float64(float64, float64)], target='parallel')
        def boundary_convolve_Ws_B_super(z_observe, x_observe):
            return boundary_convolve(2, z_observe, x_observe, zvec, xvec, dz, dx, lambda_grid_filtered, Es_case_B_grid_IGF, gamma=gamma, rho=rho, phi=phi)

        @vectorize([float64(float64, float64)], target='parallel')
        def boundary_convolve_Wx_A_super(z_observe, x_observe):
            return boundary_convolve(1, z_observe, x_observe, zvec, xvec, dz, dx, lambda_grid_filtered, Fx_case_A_grid, gamma=gamma, rho=rho, phi=phi)
        @vectorize([float64(float64, float64)], target='parallel')
        def boundary_convolve_Wx_B_super(z_observe, x_observe):
            return boundary_convolve(2, z_observe, x_observe, zvec, xvec, dz, dx, lambda_grid_filtered, Fx_case_B_grid_IGF, gamma=gamma, rho=rho, phi=phi)
     
        if debug:
            print("mappable functions for field grids defined")

        # use Numba vectorization 
        factor_case_A = (1/gamma**2/rho**2)* (dz*dx)
        Ws_grid_case_A = boundary_convolve_Ws_A_super(Z, X) * factor_case_A
        Wx_grid_case_A = boundary_convolve_Wx_A_super(Z, X) * factor_case_A

        factor_case_B = (beta**2 / rho**2)* (dz*dx)
        Ws_grid_case_B = boundary_convolve_Ws_B_super(Z, X) * factor_case_B
        Wx_grid_case_B = boundary_convolve_Wx_B_super(Z, X) * factor_case_B

        Ws_grid = Ws_grid_case_B + Ws_grid_case_A
        Wx_grid = Wx_grid_case_B + Wx_grid_case_A

    if debug:
        t5 = time.time()
        print("Convolution takes:", t5 - t4, "s")

    # Calculate the kicks at the particle locations
    
    # Overall factor
    Nb = np.sum(weight) / e_charge
    kick_factor = r_e * Nb / gamma  # in m
        
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

    #result = {"ddelta_ds": delta_kick}
    result = {"ddelta_ds": delta_kick, "dxp_ds": xp_kick}
    
    result.update({"zvec": zvec,
                "xvec": xvec,
                "Ws_grid": Ws_grid,
                "Wx_grid": Wx_grid})
    
    if debug:
        timing = np.array([t2-t1, t4-t3, t5-t4, t6-t5])
        result.update(
            {   "charge_grid": charge_grid,
                "lambda_grid_filtered": lambda_grid_filtered,
                "timing": timing
            })        
        if steady_state == False:
            result.update(
                {   "Ws_grid_case_A": Ws_grid_case_A,
                    "Ws_grid_case_B": Ws_grid_case_B,
                    "Wx_grid_case_A": Wx_grid_case_A,
                    "Wx_grid_case_B": Wx_grid_case_B,
                    "Es_case_A_grid": Es_case_A_grid,
                    "Es_case_B_grid_IGF": Es_case_B_grid_IGF,
                    "charge_grid": charge_grid,
                })
    return result


def track_bend_with_2d_csr_transient(Pin, p0c=None, gamma=None, L=0, g=0, g_err=0, N_step=20, s0=0, nz=200, nx=200, zlim=None, xlim=None,
                           CSR_on=True, steady_state=False, CSR_1D_only=False, energy_kick_on=True, xp_kick_on=True, bend_name='the bend', 
                           debug=True, keep_Pin=True, save_all_P_h5=None, save_all_P=False, save_all_wake=False, bend_track_parallel=True):
    """
    Calculates the 2D CSR kick on a set of particles with positions `z_b`, `x_b` and charges `charges`.
    Tracks a bunch thorugh a bending magnet
    
    Parameters
    ----------
    z_b : np.array
        Bunch z coordinates in [m]

    p0c : float
        reference momentum in [eV]
        
    gamma : float
        Relativistic gamma

    L : float
        total length of magnet
        
    g : float
        bending curvature in [1/m] = 1/rho
        
    g_err : float
        error in bending curvature in [1/m] (see Bmad )

    s0 : float
        bending curvature in [1/m] = 1/rho
        
    N_step : int
        number of tracking steps
        
    nz : int
        number of z grid points
        
    nx : int
        number of x grid points     
        
    steady_state : boolean
        If True, 2D steady-state wake is applied, and no transient wake is calculated
        Default: True
    
    zlim : floats (min, max) or None
        z grid limits in [m]
        
    xlim : floats (min, max) or None  
        x grid limits in [m]
    
    debug : boolean
        If True, returns the computational grids. 
        Default : False

    save_all_P_h5 : h5py file signature
        If given, returns the bunch at every step in a h5 file
        Example: h5 = h5py.File('test.h5','w') ... h5.close()

    save_all_P : boolean
        If True: returns the bunch at every step
        Default : False

    save_all_wake : boolean
        If True: returns the wake grids and the defining coordinate vectors at every step
        Default : False
              
    Returns
    -------
    dict with:
    
        Pout : ParticleGroup at the end of tracking
        Wslist : A list of Ws wake grids at every tracking step
        Wxlist : A list of Wx wake grids at every tracking step
        zvec_list : A list of zvec defining the wake grids at every tracking step
        xvec_list : A list of xvec defining the wake grids at every tracking step
        s_list : a list of s-position of the bunch center at every tracking step
        
    """
    
    P_list = []       # To save the beam at each step
    s_list = []       # To save the s position at each step
    Ws_list = []
    Wx_list = []
    zvec_list = []
    xvec_list = []
    s = s0
    
    if keep_Pin:
        P_list.append(Pin)
        s_list.append(s0)
        if save_all_P_h5:
            print('Saving the initial beam into the h5 file...')
            group = h5.create_group(bend_name + str(0))
            group.attrs['s_position'] = s0
            Pin.write(group)
        
    if bend_track_parallel:
        track = track_a_bend_parallel
    else:
        track = track_a_bend
    
    rho = 1/g   

    beam, charges = particle_group_to_bmad(Pin, p0c = p0c)
    
    #N_steps = int(np.floor(L/ds_step))
    ds_step = L/N_step

    for i in range(N_step):
        print('Tracking through', bend_name, 'in the', i+1 , "th step starting at s=" , s,'m' ) 

        ## track through a bend of length ds/2
        beam = track(beam, p0c = p0c, L=ds_step/2, theta = ds_step/2/rho, g_err=g_err)
    
        ## Calculate CSR kicks to xp and delta
        ####===================================
    
        if (CSR_on):
            
            if (CSR_1D_only):
                print('Applying 1D s-s kick...')
                csr_data = csr1d_steady_state_kick_calc(beam[4,:], charges, nz=nz, rho=rho, normalized_units=False)
                delta_kick = csr_data['denergy_ds']/(gamma*mec2)
                beam[5] = beam[5] + delta_kick * ds_step
                
            else:
                phi = (i+0.5) * ds_step / rho
                if debug:
                    print(f'Begin computing transient wakes at entrance angle phi = {phi}')
                csr_data = csr2d_kick_calc_transient(beam[4], beam[0], charges, gamma=gamma, rho=rho,
                                           phi=phi, nz=nz, nx=nx, steady_state=steady_state, debug=debug)
                              
                if (energy_kick_on):
                    print('Applying energy kick...')
                    delta_kick = csr_data['ddelta_ds'] 
                    beam[5] = beam[5] + delta_kick * ds_step
            
                if (xp_kick_on):
                    print('Applying xp_kick...')
                    xp_kick = csr_data['dxp_ds'] 
                    beam[1] = beam[1] + xp_kick * ds_step
    
        ####====================================

        ## track through a bend of length ds/2
        beam = track(beam, p0c = p0c, L=ds_step/2, theta = ds_step/2/rho, g_err=g_err)
    
        s += ds_step
    
        # save the beam and s at every step into a h5 file
        if save_all_P_h5:
            print('Saving beam into the h5 file...')
            P = bmad_to_particle_group(beam, p0c = p0c, t_ref = 0, charges = charges, verbose=False)
            group = h5.create_group(bend_name + str(i+1))
            group.attrs['s_position'] = s
            P.write(group)
    
        # save the beam and s at every step
        if save_all_P:
            P = bmad_to_particle_group(beam, p0c = p0c, t_ref = 0, charges = charges, verbose=False)
            P_list.append(P)
            s_list.append(s)

        # save the wake grids at every step
        if save_all_wake:
            Ws_list.append(csr_data['Ws_grid'])
            Wx_list.append(csr_data['Wx_grid'])
            zvec_list.append(csr_data['zvec'])
            xvec_list.append(csr_data['xvec'])
            s_list.append(s)
    if debug:
        print('Tracking completed!! Saving output...')    
        
    output = {}
    output['Pout'] = bmad_to_particle_group(beam, p0c = p0c, t_ref = 0, charges = charges, verbose=False)

    if save_all_P:
        output['P_list'] = P_list
        output['s_list'] = s_list
        
    if save_all_wake:
        output['Ws_list'] = Ws_list
        output['Wx_list'] = Wx_list
        output['zvec_list'] = zvec_list
        output['xvec_list'] = xvec_list
        output['s_list'] = s_list
        
    return output