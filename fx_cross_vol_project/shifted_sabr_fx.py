import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from black_analytics import black_option_price
from black_analytics import black_implied_vol
from sobol_lib import *

T = 1

moneyness = np.arange(0.8, 1.2, 0.05)

EURUSD = 1.15
USDJPY = 100
EURJPY = EURUSD*USDJPY
JPYUSD = 1.0/USDJPY

r_EUR = -0.01
r_USD = 0.02
r_JPY = 0.01

DF_EUR = np.exp(-r_EUR*T)
DF_USD = np.exp(-r_USD*T)
DF_JPY = np.exp(-r_JPY*T)

Forward_EURUSD = EURUSD * np.exp((r_USD - r_EUR)*T)
Forward_JPYUSD = JPYUSD * np.exp((r_USD - r_JPY)*T)
Forward_EURJPY = EURJPY * np.exp((r_JPY - r_EUR)*T)

#implied vol for ATM and correlation
vol_EURUSD = 0.20
vol_JPYUSD = 0.20
rho_EURUSD_JPYUSD = 0.2

#shifted lognormal parameters to control the skew of the implied vol
shift_EURUSD = 0
shift_JPYUSD = 0

EURUSD_volofvol = 0
JPYUSD_volofvol = 0

# this is to choose the shifted lognormal vol such that the ATM price doesn't change
ATM_EURUSD_option_price = black_option_price(Forward_EURUSD, Forward_EURUSD, T, vol_EURUSD, 1)
sln_vol_EURUSD = black_implied_vol(ATM_EURUSD_option_price, Forward_EURUSD+shift_EURUSD, Forward_EURUSD+shift_EURUSD, T, 1)

ATM_USDJPY_option_price = black_option_price(Forward_JPYUSD, Forward_JPYUSD, T, vol_JPYUSD, 1)
sln_vol_JPYUSD = black_implied_vol(ATM_USDJPY_option_price, Forward_JPYUSD+shift_JPYUSD, Forward_JPYUSD+shift_JPYUSD, T, 1)

# ---------------------------------------------------------------------
# plot the implied vol smile for EURUSD and USDJPY
EURUSD_strike = Forward_EURUSD * moneyness
JPYUSD_strike = Forward_JPYUSD * moneyness

# ---------------------------------------------------------------------

sqrt_T = np.sqrt(T)

# generate sobol number
dimension = 4
skip = 2
power = 15
sample_size = 2 ** power - 1
sobol_uniform = i4_sobol_generate(dimension, sample_size, skip)
sobol_normal = scipy.stats.norm.ppf(sobol_uniform, 0, 1.0)


# --------------------------------------------
# The processes that we are simulating is the following
# d(F1+shift1)/(F1+shift1) = sln_vol1 dW, F1 = EURUSD forward price, dW is a BM in USD forward measure
# d(F2+shift2)/(F2+shift2) = sln_vol2 dZ, F2 = JPYUSD forward price, dZ is a BM in USD forward measure
# E[dWdZ] = rho dt
# d(sln_vol1)/sln_vol1 = nu1 dU
# d(sln_vol2)/sln_vol2 = nu2 dV
# where dU and dV are not correlated to each other, dW and dZ
# --------------------------------------------

#Cholesky matrix
M11 = 1
M21 = rho_EURUSD_JPYUSD
M22 = np.sqrt(1.0 - rho_EURUSD_JPYUSD**2)

#okay, this line is dumb but I want to make it clear that I am doing Cholesky factorization
correlated_sobol_normal_0 = M11 * sobol_normal[0]
correlated_sobol_normal_1 = M21 * sobol_normal[0] + M22 * sobol_normal[1]

# this approx is exact only if EURUSD and USDJPY are both lognormal
approx_vol_EURJPY = np.sqrt(vol_EURUSD**2.0 + vol_JPYUSD**2.0 - 2.0*vol_EURUSD*vol_JPYUSD*rho_EURUSD_JPYUSD)

# generate the integrated variance, \int_0^t \sigma^2 for both EURUSD and USDJPY
# it works only if the vol process is NOT correlated with the FX rate, or otherwise there will be a drift adjustment for the FX process
# The skewness is controlled by the shift, the kurtosis is controlled by the vol of vol parameter
EURUSD_sigma_0 = sln_vol_EURUSD
JPYUSD_sigma_0 = sln_vol_JPYUSD

# ------------------------ EURUSD ------------------------------
# simulate sigma_T conditions on sigma_0 and using trapezoidal rule to approx the integrated variance
EURUSD_sigma_T = EURUSD_sigma_0 * np.exp(-0.5 * EURUSD_volofvol**2 * T + EURUSD_volofvol * sqrt_T * sobol_normal[2])
EURUSD_int_var = 0.5 * T * (EURUSD_sigma_0**2 + EURUSD_sigma_T**2)

# generate shifted lognormal sample paths for EURUSD in USD measure with path-wise sqrt of integrated variance
EURUSD_martingale = np.exp(- 0.5 * EURUSD_int_var + np.sqrt(EURUSD_int_var) * correlated_sobol_normal_0)
EURUSD_T = (Forward_EURUSD+shift_EURUSD) * EURUSD_martingale - shift_EURUSD

# ------------------------ JPYUSD ------------------------------
# simulate sigma_T conditions on sigma_0 and using trapezoidal rule to approx the integrated variance
JPYUSD_sigma_T = JPYUSD_sigma_0 * np.exp(-0.5 * JPYUSD_volofvol**2 * T + JPYUSD_volofvol * sqrt_T * sobol_normal[3])
JPYUSD_int_var = 0.5 * T * (JPYUSD_sigma_0**2 + JPYUSD_sigma_T**2)

# generate shifted lognormal sample paths for JPY in USD measure with path-wise sqrt of integrated variance
JPYUSD_martingale = np.exp(- 0.5 * JPYUSD_int_var + np.sqrt(JPYUSD_int_var) * correlated_sobol_normal_1)
JPYUSD_T = (Forward_JPYUSD+shift_JPYUSD) * JPYUSD_martingale - shift_JPYUSD

# ---------------------------------------------------------------------
# plot the implied vol smile for EURUSD and JPYUSD using Monte Carlo

EURUSD_ivol = []
for i in range(len(EURUSD_strike)):
    K = EURUSD_strike[i]
    mc_price_in_USD = np.maximum(EURUSD_T - K, 0.0)
    mc_EUR_call_USD_put_in_USD = np.mean(mc_price_in_USD)
    EURUSD_ivol.append(black_implied_vol(mc_EUR_call_USD_put_in_USD, Forward_EURUSD, K, T, 1))

JPYUSD_ivol = []
for i in range(len(JPYUSD_strike)):
    K = JPYUSD_strike[i]
    mc_price_in_USD = np.maximum(JPYUSD_T - K, 0.0)
    mc_JPY_call_USD_put_in_USD = np.mean(mc_price_in_USD)
    JPYUSD_ivol.append( black_implied_vol(mc_JPY_call_USD_put_in_USD, Forward_JPYUSD, K, T, 1) )

plt1 ,= plt.plot(moneyness, EURUSD_ivol, label="MC EURUSD")
plt2 ,= plt.plot(moneyness, JPYUSD_ivol, label="MC JPYUSD")
plt.legend(handles=[plt1, plt2])
f = plt.figure(1)
plt.draw()

#----------------------------------------------------------------------

EURJPY_strike = Forward_EURJPY * moneyness

EURJPY_ivol = []
EURJPY_analytic_approx_ivol = []
for i in range(len(EURJPY_strike)):
    K = EURJPY_strike[i]
    mc_EUR_call_JPY_put_in_USD = np.maximum(EURUSD_T/JPYUSD_T-K, 0.0)*JPYUSD_T
    mc_EUR_call_JPY_put_in_USD = np.mean(mc_EUR_call_JPY_put_in_USD) * DF_USD
    mc_EUR_call_JPY_put_in_JPY = mc_EUR_call_JPY_put_in_USD * USDJPY

    EURJPY_ivol.append( black_implied_vol(mc_EUR_call_JPY_put_in_JPY/DF_JPY , Forward_EURJPY, K, T, 1 ) )
    EURJPY_analytic_approx_ivol.append(approx_vol_EURJPY)

f = plt.figure(2)
plt1 ,= plt.plot(moneyness, EURJPY_ivol, label="EURJPY MC")
plt2 ,= plt.plot(moneyness, EURJPY_analytic_approx_ivol, label="EURJPY approx.")
plt.legend(handles=[plt1, plt2])
plt.draw()

plt.show()



