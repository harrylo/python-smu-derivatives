import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
import cmath

# characteristic function of black model
def black_cf(forward,   # forward
             vol,       # vol of the lognormal process
             t,         # time to maturity
             u          # dummy variable for the cf
             ):

    vvt_2 = 0.5*vol*vol*t
    z = complex(-vvt_2*u*u, u*(np.log(forward)-vvt_2))
    return np.exp(z)

# this is an undiscounted modified characteristic function
# the input cf is assumed to be a one parameter function, cf(u), where u is the dummy variable
def carr_madan_integrand(cf,    # characteristic function of the model
                         logk,  # log strike
                         u,     # dummy variable
                         alpha  # damping factor
                        ):
    temp = complex(u, -(alpha+1.0))
    top = cf(temp)
    top = top * np.exp(complex(0.0, -u*logk))
    bottom = complex(alpha*alpha + alpha - u*u, (2.0*alpha + 1.0)*u)
    result = top/bottom
    return result.real

# pricing a call option using carr madan fourier method
# the input cf is assumed to be a one parameter function, cf(u), where u is the dummy variable
def carr_madan_black_call_option(forward,   #  forward price
                                 vol,       # vol of the lognormal process
                                 t,         # time to maturity
                                 k,         # strike
                                 alpha = 0.75  # damping factor
                                 ):
    logk = np.log(k)
    cf = lambda u: black_cf(forward, vol, t, u)
    integrand = lambda u: carr_madan_integrand(cf, logk, u, alpha)

    # lower limit is 0.0, upper limit is infinity
    numerical_integration_result, error = integrate.quad(integrand, 0.0, np.inf)

    temp = np.exp(-alpha*logk) / np.pi
    result = temp * numerical_integration_result

    return result


# characteristic function of heston model. The notation follows The Little Heston Trap 2005.
# dF_t/F_t = sqrt(v_t)dW_t
# dv_t = kappa*(nu-v_t) dt + lambda * sqrt(v_t)dZ_t
# E[dWdZ] = rho dt
def heston_cf(forward, # forward
              vol_0, # the volatility at time 0, NOT variance
              kappa, # mean reversion rate
              nu,    # mean reversion level
              lam,   # vol of variance
              rho,   # correlation between the variance and the underlying
              t,     # time
              u      # dummy variable for the cf
              ):

    # pre computed values
    rliu_m_k = complex(-kappa, rho * lam * u)

    d = np.sqrt( rliu_m_k * rliu_m_k + complex(lam*lam*u*u, lam*lam*u) )

    k_m_rliu_m_d = complex(kappa, -rho * lam * u) - d

    g2 = complex(kappa-d, -rho*lam*u) / complex(kappa+d, -rho*lam*u)

    emdt = np.exp(-d*t)

    one_m_g2 = 1.0 - g2 * emdt

    one_m_emdt = 1.0 - emdt

    v0v0ll = vol_0*vol_0/(lam*lam)

    temp1 = np.exp(complex(0.0, u * np.log(forward)))

    temp2 = k_m_rliu_m_d*t - 2.0 * np.log (one_m_g2/(1.0-g2))
    temp2 = np.exp( temp2 * nu*kappa/(lam*lam) )

    temp3 = np.exp( v0v0ll * k_m_rliu_m_d * (1.0 - emdt) / one_m_g2 )

    result = temp1 * temp2 * temp3
    return result


# pricing a call option using carr madan fourier method
# the input cf is assume to be a one parameter function, cf(u), where u is the dummy variable
def carr_madan_heston_call_option(forward,  # forward
                                  vol_0,    # the volatility at time 0, NOT variance
                                  kappa,    # mean reversion rate
                                  nu,       # mean reversion level
                                  lam,      # vol of variance
                                  rho,      # correlation between the variance and the underlying
                                  t,        # time to maturity
                                  k,        #  strike
                                  alpha = 0.75  # damping factor
                                  ):
    logk = np.log(k)
    cf = lambda u: heston_cf(forward, vol_0, kappa, nu, lam, rho, t, u)
    integrand = lambda u: carr_madan_integrand(cf, logk, u, alpha)

    # lower limit is 0.0, upper limit is infinity
    numerical_integration_result, error = integrate.quad(integrand, 0.0, np.inf)

    temp = np.exp(-alpha*logk) / np.pi
    result = temp * numerical_integration_result

    return result