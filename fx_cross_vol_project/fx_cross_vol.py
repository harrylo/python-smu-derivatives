import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from black_analytics import black_option_price
from black_analytics import black_implied_vol
from sobol_lib import *

T = 2

moneyness = np.arange(0.7, 1.3, 0.1)

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
vol_EURUSD = 0.20
vol_USDJPY = 0.20
rho_EURUSD_USDJPY = -0.3

#shifted lognormal parameters to control the skew of the implied vol
shift_EURUSD = 0.3
shift_USDJPY = -10

# this is to choose the shifted lognormal vol such that the ATM price doesn't change
ATM_EURUSD_option_price = black_option_price(Forward_EURUSD, Forward_EURUSD, T, vol_EURUSD, 1)
sln_vol_EURUSD = black_implied_vol(ATM_EURUSD_option_price, Forward_EURUSD+shift_EURUSD, Forward_EURUSD+shift_EURUSD, T, 1)

ATM_USDJPY_option_price = black_option_price(Forward_USDJPY, Forward_USDJPY, T, vol_USDJPY, 1)
sln_vol_USDJPY = black_implied_vol(ATM_USDJPY_option_price, Forward_USDJPY+shift_USDJPY, Forward_USDJPY+shift_USDJPY, T, 1)

# ---------------------------------------------------------------------
# plot the implied vol smile for EURUSD and USDJPY
EURUSD_strike = Forward_EURUSD * moneyness
USDJPY_strike = Forward_USDJPY * moneyness

EURUSD_ivol = []
for i in range(len(EURUSD_strike)):
    K = EURUSD_strike[i]
    price = black_option_price(Forward_EURUSD+shift_EURUSD, K+shift_EURUSD, T, sln_vol_EURUSD, 1)
    EURUSD_ivol.append( black_implied_vol(price, Forward_EURUSD, K, T, 1) )

USDJPY_ivol = []
for i in range(len(USDJPY_strike)):
    K = USDJPY_strike[i]
    price = black_option_price(Forward_USDJPY+shift_USDJPY, K+shift_USDJPY, T, sln_vol_USDJPY, 1)
    USDJPY_ivol.append( black_implied_vol(price, Forward_USDJPY, K, T, 1) )

plt1 ,= plt.plot(moneyness, EURUSD_ivol, label="EURUSD")
plt2 ,= plt.plot(moneyness, USDJPY_ivol, label="USDJPY")
plt.legend(handles=[plt1, plt2])
f = plt.figure(1)
plt.draw()


# ---------------------------------------------------------------------

# this approx is exact only if EURUSD and USDJPY are both lognormal
approx_vol_EURJPY = np.sqrt(vol_EURUSD**2.0 + vol_USDJPY**2.0 + 2.0*vol_EURUSD*vol_USDJPY*rho_EURUSD_USDJPY)

drift_EURUSD_in_USD_measure = r_USD - r_EUR - 0.5 * vol_EURUSD ** 2

drift_USDJPY_in_JPY_measure = r_JPY - r_USD - 0.5 * vol_USDJPY ** 2
drift_USDJPY_in_USD_measure = r_JPY - r_USD + vol_USDJPY ** 2 - 0.5 * vol_USDJPY ** 2

EURJPY_strike = Forward_EURJPY * moneyness

sqrt_T = np.sqrt(T)

# generate sobol number
dimension = 2
skip = 2
power = 16
sample_size = 2 ** power - 1
sobol_uniform = i4_sobol_generate(dimension, sample_size, skip)
sobol_normal = scipy.stats.norm.ppf(sobol_uniform, 0, 1.0)

#Cholesky matrix
M11 = 1
M21 = rho_EURUSD_USDJPY
M22 = np.sqrt(1.0 - rho_EURUSD_USDJPY**2)

#okay, this line is dumb but I want to make it clear that I am doing Cholesky factorization
correlated_sobol_normal_0 = M11 * sobol_normal[0]
correlated_sobol_normal_1 = M21 * sobol_normal[0] + M22 * sobol_normal[1]

#generate shifted lognormal sample paths for EURUSD in USD measure
EURUSD_T = (EURUSD+shift_EURUSD) * np.exp(drift_EURUSD_in_USD_measure * T + sln_vol_EURUSD * sqrt_T * correlated_sobol_normal_0) - shift_EURUSD

#genearte lognormal sample paths for USDJPY in USD measure
USDJPY_T = (USDJPY+shift_USDJPY) * np.exp(drift_USDJPY_in_USD_measure * T + sln_vol_USDJPY * sqrt_T * correlated_sobol_normal_1) - shift_USDJPY


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
plt1 ,= plt.plot(EURJPY_strike, EURJPY_ivol, label="EURJPY MC")
plt2 ,= plt.plot(EURJPY_strike, EURJPY_analytic_approx_ivol, label="EURJPY approx.")
plt.legend(handles=[plt1, plt2])
plt.draw()

plt.show()




