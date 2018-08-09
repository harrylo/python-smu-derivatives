import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from black_analytics import black_option_price
from black_analytics import black_implied_vol
from sobol_lib import *

T = 1

moneyness = np.arange(0.6, 1.4, 0.05)

EURUSD = 1.15
USDJPY = 120
EURJPY = EURUSD*USDJPY

r_EUR = -0.02
r_USD = 0.03
r_JPY = 0.001

DF_EUR = np.exp(-r_EUR*T)
DF_USD = np.exp(-r_USD*T)
DF_JPY = np.exp(-r_JPY*T)

Forward_EURUSD = EURUSD * np.exp((r_USD - r_EUR)*T)
Forward_USDJPY = USDJPY * np.exp((r_JPY - r_USD)*T)
Forward_EURJPY = EURJPY * np.exp((r_JPY - r_EUR)*T)

#implied vol for ATM
vol_EURUSD = 0.15
vol_USDJPY = 0.25
rho_EURUSD_USDJPY = 0.2

#shifted lognormal parameters to control the skew of the implied vol
shift_EURUSD = 0.2
shift_USDJPY = 10

EURUSD_volofvol = 0.5
USDJPY_volofvol = 0.2

# this is to choose the shifted lognormal vol such that the ATM price doesn't change
ATM_EURUSD_option_price = black_option_price(Forward_EURUSD, Forward_EURUSD, T, vol_EURUSD, 1)
sln_vol_EURUSD = black_implied_vol(ATM_EURUSD_option_price, Forward_EURUSD+shift_EURUSD, Forward_EURUSD+shift_EURUSD, T, 1)

ATM_USDJPY_option_price = black_option_price(Forward_USDJPY, Forward_USDJPY, T, vol_USDJPY, 1)
sln_vol_USDJPY = black_implied_vol(ATM_USDJPY_option_price, Forward_USDJPY+shift_USDJPY, Forward_USDJPY+shift_USDJPY, T, 1)

# # ---------------------------------------------------------------------
# # plot the implied vol smile for EURUSD and USDJPY
# EURUSD_strike = Forward_EURUSD * moneyness
# USDJPY_strike = Forward_USDJPY * moneyness
#
# EURUSD_ivol = []
# for i in range(len(EURUSD_strike)):
#     K = EURUSD_strike[i]
#     price = black_option_price(Forward_EURUSD+shift_EURUSD, K+shift_EURUSD, T, sln_vol_EURUSD, 1)
#     EURUSD_ivol.append( black_implied_vol(price, Forward_EURUSD, K, T, 1) )
#
# USDJPY_ivol = []
# for i in range(len(USDJPY_strike)):
#     K = USDJPY_strike[i]
#     price = black_option_price(Forward_USDJPY+shift_USDJPY, K+shift_USDJPY, T, sln_vol_USDJPY, 1)
#     USDJPY_ivol.append( black_implied_vol(price, Forward_USDJPY, K, T, 1) )
#
# plt1 ,= plt.plot(moneyness, EURUSD_ivol, label="EURUSD")
# plt2 ,= plt.plot(moneyness, USDJPY_ivol, label="USDJPY")
# plt.legend(handles=[plt1, plt2])
# f = plt.figure(1)
# plt.draw()
#
#
# # ---------------------------------------------------------------------

# this approx is exact only if EURUSD and USDJPY are both lognormal
approx_vol_EURJPY = np.sqrt(vol_EURUSD**2.0 + vol_USDJPY**2.0 + 2.0*vol_EURUSD*vol_USDJPY*rho_EURUSD_USDJPY)

drift_EURUSD_in_USD_measure = r_USD - r_EUR - 0.5 * sln_vol_EURUSD ** 2

drift_USDJPY_in_JPY_measure = r_JPY - r_USD - 0.5 * sln_vol_USDJPY ** 2
drift_USDJPY_in_USD_measure = r_JPY - r_USD + sln_vol_USDJPY ** 2 - 0.5 * sln_vol_USDJPY ** 2

sqrt_T = np.sqrt(T)

# generate sobol number
dimension = 4
skip = 2
power = 18
sample_size = 2 ** power - 1
sobol_uniform = i4_sobol_generate(dimension, sample_size, skip)
sobol_normal = scipy.stats.norm.ppf(sobol_uniform, 0, 1.0)


# --------------------------------------------
# The processes that we are simulating is the following
# d(X1+shift1)/(X1+shift1) = drift1 dt + sln_vol1 dW, X1 = EURUSD
# d(X2+shift2)/(X2+shift2) = drift2 dt + sln_vol2 dZ, X2 = USDJPY
# E[dWdZ] = rho dt
# d(sln_vol1)/sln_vol1 = nu1 dU
# d(sln_vol2)/sln_vol2 = nu2 dV
# where dU and dV are not correlated to each other, dW and dZ
# --------------------------------------------

#Cholesky matrix
M11 = 1
M21 = rho_EURUSD_USDJPY
M22 = np.sqrt(1.0 - rho_EURUSD_USDJPY**2)

#okay, this line is dumb but I want to make it clear that I am doing Cholesky factorization
correlated_sobol_normal_0 = M11 * sobol_normal[0]
correlated_sobol_normal_1 = M21 * sobol_normal[0] + M22 * sobol_normal[1]

# generate the integrated variance, \int_0^t \sigma^2 for both EURUSD and USDJPY
# it works only if the vol process is NOT correlated with the FX rate, or otherwise there will be a drift adjustment for the FX process
# The skewness is controlled by the shift, the kurtosis is controlled by the vol of vol parameter
EURUSD_sigma_0 = sln_vol_EURUSD
USDJPY_sigma_0 = sln_vol_USDJPY

# simulate sigma_T conditions on sigma_0
EURUSD_sigma_T = EURUSD_sigma_0 * np.exp(-0.5 * EURUSD_volofvol**2 * T + EURUSD_volofvol * sqrt_T * sobol_normal[2])
USDJPY_sigma_T = USDJPY_sigma_0 * np.exp(-0.5 * USDJPY_volofvol**2 * T + USDJPY_volofvol * sqrt_T * sobol_normal[3])

# using trapezoidal rule to approx the integrated variance
EURUSD_sqrt_int_var = np.sqrt(0.5 * T * (EURUSD_sigma_0**2 + EURUSD_sigma_T**2))

# generate shifted lognormal sample paths for EURUSD in USD measure with pathwise sqrt of integrated variance
EURUSD_T = (EURUSD+shift_EURUSD) * np.exp(drift_EURUSD_in_USD_measure * T + EURUSD_sqrt_int_var *correlated_sobol_normal_0) - shift_EURUSD

# using trapezoidal rule to approx the integrated variance
USDJPY_sqrt_int_var = np.sqrt(0.5 * T * (USDJPY_sigma_0**2 + USDJPY_sigma_T**2))

# genearte shifted lognormal sample paths for USDJPY in USD measure with pathwise sqrt of integrated variance
USDJPY_T = (USDJPY+shift_USDJPY) * np.exp(drift_USDJPY_in_USD_measure * T + USDJPY_sqrt_int_var * correlated_sobol_normal_1) - shift_USDJPY

# ---------------------------------------------------------------------

EURJPY_strike = Forward_EURJPY * moneyness

EURJPY_ivol = []
EURJPY_analytic_approx_ivol = []
for i in range(len(EURJPY_strike)):
    K = EURJPY_strike[i]
    mc_EUR_call_JPY_put_in_USD = np.maximum(EURUSD_T*USDJPY_T-K, 0.0)/USDJPY_T
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




