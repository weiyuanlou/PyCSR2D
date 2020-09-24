from mpmath import exp

def lambda_p_Gauss(z, x): 
    """
    The z derivative of a 2D Gaussian G(z,x)
    """
    sigmaz = 10E-6 
    sigmax = 10E-6
    return 1/(2*pi*sigmaz*sigmax)*exp(-x**2/2/sigmax**2)*exp(-z**2/2/sigmaz**2)*(-z/sigmaz**2)gg