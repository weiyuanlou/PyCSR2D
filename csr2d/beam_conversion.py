import numpy as np
from pmd_beamphysics import ParticleGroup
   
def particle_group_to_bmad(particle_group,           
                           p0c=None,
                           t_ref=0,
                           verbose=False): 

    """
    Converts a particle group to a bmad beam.
    
    Parameters:
    ----------
    particle_group : float, array
                a 2D array of size (6, number_of_particles)
    p0c: float, 
         reference momentum in eV/c
    t_ref : float, optional
            reference time of the beam in seconds. Default: 0.
    ----------
    
    Returns:
    ----------
    (1) Bmad beam: a 2D numpy array of size (6,number_of_particles)
    (2) Charges: a 1D numpy array of size (number_of_particles))
    
    ----------
    
    Bmad's ASCII format is described in:
    
        https://www.classe.cornell.edu/bmad/manual.html
    
    Bmad normally uses s-based coordinates, with momenta:
        bmad px = px/p0
        bmad py = py/p0
        bmad pz = p/p0 - 1
    and longitudinal coordinate
        bmad z = -beta*c(t - t_ref)
    """

    n = particle_group.n_particle            
    x = particle_group.x
    y = particle_group.y

    px = particle_group.px/p0c
    py = particle_group.py/p0c

    z = -particle_group.beta*299792458*(particle_group.t - t_ref)
    pz = particle_group.p/p0c -1.0
    
    status  = particle_group.status 
    weight  = particle_group.weight
        
    beam = np.vstack([x,px,y,py,z,pz])
    return beam, weight


def bmad_to_particle_group(bmad_beam,
                       p0c=None,
                       charges=None,
                       t_ref=0,
                       verbose=False):

    """
    Converts a bmad beam to a particle group.
    Assumes electrons.
    
    Parameters:
    ----------
    bmad_beam : float, array
                a 2D array of size (6, number_of_particles)
    p0c: float, 
         reference momentum in eV/c
    t_ref : float, optional
            reference time of the beam in seconds. Default: 0.
    charges : float, array
              an 1D array of size (number_of_particles)
    ----------
    
    Returns:
    ----------
    A ParticleGroup object
    ----------
    
    
    Bmad's ASCII format is described in:
    
        https://www.classe.cornell.edu/bmad/manual.html
    
    Bmad normally uses s-based coordinates, with momenta:
        bmad px = px/p0
        bmad py = py/p0
        bmad pz = p/p0 - 1
    and longitudinal coordinate
        bmad z = -beta*c(t - t_ref)

    
    """
    if ( p0c <=0 ):
        raise ValueError(' invalid p0c value given!! ')
        
    if ( np.any(charges <= 0) ):
        raise ValueError(' invalid charges value(s) given!! ')

    Np = bmad_beam.shape[1]      # Number of macro particles

    delta = bmad_beam[5]
    p = p0c*(1 + delta) # in eV/c

    E = np.sqrt(p**2 + 510998.950**2) # in eV, assuming electrons
    beta = p/E

    x  = bmad_beam[0]        # in m
    px = bmad_beam[1]*p0c    # in eV/c
    y  = bmad_beam[2]        # in m  
    py = bmad_beam[3]*p0c    # in eV/c
    t  = bmad_beam[4]/(-1.*beta*299792458) + t_ref
    pz = np.sqrt(p**2 - px**2 - py**2)


    data = {'x':x,
            'px':px,
            'y':y,
            'py':py,
            'z':np.zeros(Np),
            'pz':pz, 
            't':t, 
            'status':np.ones(Np), 
            'weight':charges, 
            'species':'electron'}
    return ParticleGroup(data=data)