import mpmath
from mpmath import fabs, sin, cos, re, sqrt, exp, pi, ellipf, ellipe


def psi_s(z, x, beta):
    """
    2D longitudinal potential
    Eq. (23) from Ref[1] with no constant factor (e*beta**2/2/rho**2).
    Ref[1]: Y. Cai and Yuantao. Ding, PRAB 23, 014402 (2020)
    Note that 'x' here corresponds to 'chi = x / rho' in the paper.
    """
    out = (cos(2 * alpha(z, x, beta)) - 1 / (1 + x)) / (
        kappa(z, x, beta) - beta * (1 + x) * sin(2 * alpha(z, x, beta))
    )
    return out


def psi_x(z, x, beta):
    """
    Eq.(24) from Ref[1] with argument zeta=0 and no constant factor e*beta**2/2/rho**2.
    Note that 'x' here corresponds to 'chi = x / rho' in the paper.
    """
    kap = kappa(z, x, beta)
    alp = alpha(z, x, beta)
    arg2 = -4 * (1 + x) / x ** 2

    T1 = (
        1
        / fabs(x)
        / (1 + x)
        * ((2 + 2 * x + x ** 2) * ellipf(alp, arg2) - x ** 2 * ellipe(alp, arg2))
    )
    D = kap ** 2 - beta ** 2 * (1 + x) ** 2 * sin(2 * alp) ** 2
    T2 = (
        (
            kap ** 2
            - 2 * beta ** 2 * (1 + x) ** 2
            + beta ** 2 * (1 + x) * (2 + 2 * x + x ** 2) * cos(2 * alp)
        )
        / beta
        / (1 + x)
        / D
    )
    T3 = -kap * sin(2 * alp) / D
    T4 = kap * beta ** 2 * (1 + x) * sin(2 * alp) * cos(2 * alp) / D
    T5 = 1 / fabs(x) * ellipf(alp, arg2)  # psi_phi without e/rho**2 factor
    out = re((T1 + T2 + T3 + T4) - 2 / beta ** 2 * T5)

    return out


def nu(x, beta):
    """
    Eq. (6) from Ref[1] (coeffient of alpha**2)
    Note that 'x' here corresponds to 'chi = x / rho' in the paper.
    """
    return 3 * (1 - beta ** 2 - beta ** 2 * x) / beta ** 2 / (1 + x)


def eta(z, x, beta):
    """
    Eq. (6) from Ref[1] (coeffient of alpha)
    """
    return -6 * z / beta ** 2 / (1 + x)


def zeta(z, x, beta):
    """
    Eq. (6) from Ref[1] (constant term)
    Note that 'x' here corresponds to 'chi = x / rho' in the paper.
    """
    return 3 * (4 * z ** 2 - beta ** 2 * x ** 2) / 4 / beta ** 2 / (1 + x)


def Omega(z, x, beta):
    """
    Eq. (A3) from Ref[1]
    Note that 'x' here corresponds to 'chi = x / rho' in the paper.
    """
    temp = (
        eta(z, x, beta) ** 2 / 16
        - zeta(z, x, beta) * nu(x, beta) / 6
        + nu(x, beta) ** 3 / 216
    )
    return temp + (temp ** 2 - (zeta(z, x, beta) / 3 + nu(x, beta) ** 2 / 36) ** 3) ** (
        1 / 2
    )


def m(z, x, beta):
    """
    Eq. (A2) from Ref[1]
    Note that 'x' here corresponds to 'chi = x / rho' in the paper.
    """
    return (
        -nu(x, beta) / 3
        + (zeta(z, x, beta) / 3 + nu(x, beta) ** 2 / 36) * Omega(z, x, beta) ** (-1 / 3)
        + Omega(z, x, beta) ** (1 / 3)
    )


def alpha(z, x, beta):
    """
    Eq. (A4) from Ref[1]
    Note that 'x' here corresponds to 'chi = x / rho' in the paper.
    """
    arg1 = sqrt(2 * fabs(m(z, x, beta)))
    arg2 = -2 * (m(z, x, beta) + nu(x, beta))
    arg3 = 2 * eta(z, x, beta) / arg1

    if z < 0:
        return re(1 / 2 * (-arg1 + sqrt(fabs(arg2 + arg3))))
    else:
        return re(1 / 2 * (arg1 + sqrt(fabs(arg2 - arg3))))


def kappa(z, x, beta):
    """
    Eq. (13) from Ref[1] with argumaent zeta = 0
    Note that 'x' here corresponds to 'chi = x / rho' in the paper.
    """
    return (x ** 2 + 4 * (1 + x) * sin(alpha(z, x, beta)) ** 2) ** (1 / 2)


def lambda_p_gauss(z, x, sigmaz, sigmax):
    """
    The z derivative of a 2D Gaussian G(z,x)
    """
    return (
        1
        / (2 * pi * sigmaz * sigmax)
        * exp(-x ** 2 / 2 / sigmax ** 2)
        * exp(-z ** 2 / 2 / sigmaz ** 2)
        * (-z / sigmaz ** 2)
    )


def lambda_gauss(z, x, sigmaz, sigmax):
    """
    The 2D Gaussian G(z,x)
    """
    return (
        1
        / (2 * pi * sigmaz * sigmax)
        * exp(-x ** 2 / 2 / sigmax ** 2)
        * exp(-z ** 2 / 2 / sigmaz ** 2)
    )
