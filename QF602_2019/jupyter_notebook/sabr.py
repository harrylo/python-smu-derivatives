import numpy as np
import matplotlib.pyplot as plt

def sabr_lognormal_vol(k, f, t, alpha, beta, rho, volvol):
    """
    Hagan's 2002 SABR lognormal vol expansion.
    The strike k can be a scalar or an array, the function will return an array
    of lognormal vols.
    """
    # Negative strikes or forwards
    if k <= 0 or f <= 0:
        return 0.
    eps = 1e-07
    logfk = np.log(f / k)
    fkbeta = (f*k)**(1 - beta)
    a = (1 - beta)**2 * alpha**2 / (24 * fkbeta)
    b = 0.25 * rho * beta * volvol * alpha / fkbeta**0.5
    c = (2 - 3*rho**2) * volvol**2 / 24
    d = fkbeta**0.5
    v = (1 - beta)**2 * logfk**2 / 24
    w = (1 - beta)**4 * logfk**4 / 1920
    z = volvol * fkbeta**0.5 * logfk / alpha
    # if |z| > eps
    if abs(z) > eps:
        vz = alpha * z * (1 + (a + b + c) * t) / (d * (1 + v + w) * _x(rho, z))
        return vz
    # if |z| <= eps
    else:
        v0 = alpha * (1 + (a + b + c) * t) / (d * (1 + v + w))
        return v0


def _x(rho, z):
    """Return function x used in Hagan's 2002 SABR lognormal vol expansion."""
    a = (1 - 2*rho*z + z**2)**.5 + z - rho
    b = 1 - rho
    return np.log(a / b)

def find_alpha(v_atm_ln, f, t, beta, rho, volvol):
    """
    Compute SABR parameter alpha to an ATM lognormal volatility.
    Alpha is determined as the root of a 3rd degree polynomial. Return a single
    scalar alpha.
    """
    f_ = f ** (beta - 1)
    p = [
        t * f_**3 * (1 - beta)**2 / 24,
        t * f_**2 * rho * beta * volvol / 4,
        (1 + t * volvol**2 * (2 - 3*rho**2) / 24) * f_,
        -v_atm_ln
    ]
    roots = np.roots(p)
    roots_real = np.extract(np.isreal(roots), np.real(roots))
    # Note: the double real roots case is not tested
    alpha_first_guess = v_atm_ln * f**(1-beta)
    i_min = np.argmin(np.abs(roots_real - alpha_first_guess))
    return roots_real[i_min]

# this part of code won't get called by ipynb
if __name__ == "__main__":
    f = 10
    t = 1
    lower_bound = 4
    upper_bound = 18
    k = np.arange(lower_bound, upper_bound, 0.5)


    alpha = 0.4
    rho = -0.5
    volvol = 0.4

    sabr_ivol_0 = []
    sabr_ivol_1 = []
    sabr_ivol_2 = []

    atm_ln_vol = 0.2

    for i in range(len(k)):

        beta = 0.1
        alpha = find_alpha(atm_ln_vol, f, t, beta, rho, volvol)
        sabr_ivol_0.append(sabr_lognormal_vol(k[i], f, t, alpha, beta, rho, volvol))

        beta = 0.3
        alpha = find_alpha(atm_ln_vol, f, t, beta, rho, volvol)
        sabr_ivol_1.append(sabr_lognormal_vol(k[i], f, t, alpha, beta, rho, volvol))

        beta = 0.5
        alpha = find_alpha(atm_ln_vol, f, t, beta, rho, volvol)
        sabr_ivol_2.append(sabr_lognormal_vol(k[i], f, t, alpha, beta, rho, volvol))

    plt1, = plt.plot(k, sabr_ivol_0, label="beta = 0.1")
    plt2, = plt.plot(k, sabr_ivol_1, label="beta = 0.3")
    plt3, = plt.plot(k, sabr_ivol_2, label="beta = 0.5")

    plt.legend(handles=[plt1, plt2, plt3])
    plt.grid(True)
    plt.show()