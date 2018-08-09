import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from black_analytics import black_option_price
from black_analytics import black_implied_vol
from sobol_lib import *

# input:
# sigma_0, volatility at the current time
# volofvol, volatility of volatility
# t, simulation time step
# normal, an array of standard normal random variables
# we assume the vol process is a drift-less log-normal process.
# output:
# path-wise integrated variance, \int_0^T sigma^2(u) du
def generate_integrated_variance_sample_paths(sigma_0, volofvol, t, normal):
    sqrt_t = np.sqrt(t)
    sigma_t = sigma_0 * np.exp(-0.5 * volofvol**2 * t + volofvol * sqrt_t * normal)
    # simulate sigma_t conditions on sigma_0 and using trapezoidal rule to approx the integrated variance
    int_var = 0.5 * t * (sigma_0**2 + sigma_t**2)
    return int_var

# --------------------------------------------
# The processes that we are simulating is the following
# d(FX1+shift_X1)/(F1+shift_X1) = sigma1_0 dW, FX1 = EURUSD forward price, dW is a BM in USD forward measure
# d(FX2+shift_X2)/(F2+shift_X2) = sigma2_0 dZ, FX2 = JPYUSD forward price, dZ is a BM in USD forward measure
# E[dWdZ] = rho dt
# d(sigma1_t)/sigma1_t = volofvol_1 dU
# d(sigma2_t)/sigma2_t = volofvol_2 dV
# where dU and dV are not correlated to each other, dW and dZ
# --------------------------------------------
# input:
# Forward_X1, forward for FX1
# Forward_X2, forward for FX2
# shift_X1, shift for FX1
# shift_X2, shift for FX2
# sigma1_0, shifted lognormal vol for FX1 at the current time
# sigma2_0, shifted lognormal vol for FX2 at the current time
# volofvol_1, volatility of volatility for FX1
# volofvol_2, volatility of volatility for FX2
# t, simulation time step
# rho, correlation between log FX1 and log FX2
# numsims, number of simulations
# output:
# sample paths for FX1 and FX2, that both process follows shifted log-normal with stochastic volatility
def generate_fx_spot_processes_sample_paths(Forward_X1, Forward_X2,
                                            shift_X1, shift_X2,
                                            sigma1_0, sigma2_0,
                                            volofvol_1, volofvol_2,
                                            t,
                                            rho,
                                            numsims):
    # generate sobol number
    dimension = 4
    skip = 2
    sobol_uniform = i4_sobol_generate(dimension, numsims, skip)
    sobol_normal = scipy.stats.norm.ppf(sobol_uniform, 0, 1.0)

    # Cholesky matrix
    M11 = 1
    M21 = rho
    M22 = np.sqrt(1.0 - rho ** 2)

    # okay, this line is dumb but I want to make it clear that I am doing Cholesky factorization
    correlated_sobol_normal_0 = M11 * sobol_normal[0]
    correlated_sobol_normal_1 = M21 * sobol_normal[0] + M22 * sobol_normal[1]

    # generate shifted lognormal sample paths for FX1 with path-wise integrated variance
    int_var1 = generate_integrated_variance_sample_paths(sigma1_0, volofvol_1, t, sobol_normal[2])
    FX1_martingale = np.exp(- 0.5 * int_var1 + np.sqrt(int_var1) * correlated_sobol_normal_0)
    FX1_T = (Forward_X1+shift_X1) * FX1_martingale - shift_X1

    # generate shifted lognormal sample paths for FX2 with path-wise integrated variance
    int_var2 = generate_integrated_variance_sample_paths(sigma2_0, volofvol_2, t, sobol_normal[3])
    FX2_martingale = np.exp(- 0.5 * int_var2 + np.sqrt(int_var2) * correlated_sobol_normal_1)
    FX2_T = (Forward_X2+shift_X2) * FX2_martingale - shift_X2

    return FX1_T, FX2_T

# compute implied vol smile given forward, moneyness, sampth paths, maturity
def compute_ivols(Forward, moneyness, sample_paths_T, T):
    strikes = Forward * moneyness
    ivols = []
    for i in range(len(strikes)):
        K = strikes[i]
        mc_price = np.maximum(sample_paths_T - K, 0.0)
        ivols.append(black_implied_vol(np.mean(mc_price), Forward, K, T, 1))

    return ivols


# ------------------------- program starts here -------------------------------------
# There are total 3 steps in this program
# You will see the FX single rate smiles for both FX1 and FX2
# You will see the FX cross smile verse the lognormal approx.
# If you set the shifts and the vol of vols to be 0 then FX cross smile and the lognormal approx should be very close.

# Step 1 - setting up the market data
T = 1

moneyness = np.arange(0.6, 1.4, 0.05)

EURUSD = 1.15
USDJPY = 100
EURJPY = EURUSD*USDJPY
JPYUSD = 1.0/USDJPY

r_EUR = -0.01
r_USD = 0.02
r_JPY = 0.01

# implied vol for ATM and correlation
vol_EURUSD = 0.20
vol_JPYUSD = 0.20
rho_EURUSD_JPYUSD = 0.5

# shifted lognormal parameters to control the skew of the implied vol
shift_EURUSD = 0.1
shift_JPYUSD = 0.00001

EURUSD_volofvol = 0.3
JPYUSD_volofvol = 0.4

DF_EUR = np.exp(-r_EUR*T)
DF_USD = np.exp(-r_USD*T)
DF_JPY = np.exp(-r_JPY*T)

Forward_EURUSD = EURUSD * np.exp((r_USD - r_EUR)*T)
Forward_JPYUSD = JPYUSD * np.exp((r_USD - r_JPY)*T)
Forward_EURJPY = EURJPY * np.exp((r_JPY - r_EUR)*T)

# this is to choose the shifted lognormal vol such that the ATM price doesn't change when we change the shift
ATM_EURUSD_option_price = black_option_price(Forward_EURUSD, Forward_EURUSD, T, vol_EURUSD, 1)
sln_vol_EURUSD = black_implied_vol(ATM_EURUSD_option_price, Forward_EURUSD+shift_EURUSD, Forward_EURUSD+shift_EURUSD, T, 1)

ATM_USDJPY_option_price = black_option_price(Forward_JPYUSD, Forward_JPYUSD, T, vol_JPYUSD, 1)
sln_vol_JPYUSD = black_implied_vol(ATM_USDJPY_option_price, Forward_JPYUSD+shift_JPYUSD, Forward_JPYUSD+shift_JPYUSD, T, 1)

# Step 2 - simulate the sample paths for FX1 and FX2

power = 18
sample_size = 2 ** power - 1

EURUSD_T, JPYUSD_T = generate_fx_spot_processes_sample_paths(Forward_EURUSD, Forward_JPYUSD,
                                                             shift_EURUSD, shift_JPYUSD,
                                                             sln_vol_EURUSD, sln_vol_JPYUSD,
                                                             EURUSD_volofvol, JPYUSD_volofvol,
                                                             T,
                                                             rho_EURUSD_JPYUSD,
                                                             sample_size)

# plot the implied vol smile for EURUSD and JPYUSD using Monte Carlo
EURUSD_ivols = compute_ivols(Forward_EURUSD, moneyness, EURUSD_T, T)
JPYUSD_ivols = compute_ivols(Forward_JPYUSD, moneyness, JPYUSD_T, T)
plt1, = plt.plot(moneyness, EURUSD_ivols, label="MC EURUSD")
plt2, = plt.plot(moneyness, JPYUSD_ivols, label="MC JPYUSD")
plt.legend(handles=[plt1, plt2])
f = plt.figure(1)
plt.draw()

# Step 3, compute the FX cross smile

# this approx is exact only if EURUSD and USDJPY are both pure lognormal, this is the GMDS method
approx_vol_EURJPY = np.sqrt(vol_EURUSD**2.0 + vol_JPYUSD**2.0 - 2.0*vol_EURUSD*vol_JPYUSD*rho_EURUSD_JPYUSD)

EURJPY_strike = Forward_EURJPY * moneyness

EURJPY_ivol = []
EURJPY_analytic_approx_ivol = []
for i in range(len(EURJPY_strike)):
    K = EURJPY_strike[i]
    # compute the EUR call JPY put in USD forward measure. The multiplication by JPYUSD_T is to convert the payoff to USD
    mc_EUR_call_JPY_put_in_USD = np.maximum(EURUSD_T/JPYUSD_T-K, 0.0)*JPYUSD_T
    # since we are in USD forward measure so we must discount using USD discount factor.
    mc_EUR_call_JPY_put_in_USD = np.mean(mc_EUR_call_JPY_put_in_USD) * DF_USD
    # convert the option price to JPY using today's spot
    mc_EUR_call_JPY_put_in_JPY = mc_EUR_call_JPY_put_in_USD * USDJPY
    # we divided by JPY discount factor to get the undiscount option price in JPY
    EURJPY_ivol.append( black_implied_vol(mc_EUR_call_JPY_put_in_JPY/DF_JPY , Forward_EURJPY, K, T, 1 ) )
    EURJPY_analytic_approx_ivol.append(approx_vol_EURJPY)

f = plt.figure(2)
plt1 ,= plt.plot(moneyness, EURJPY_ivol, label="EURJPY MC")
plt2 ,= plt.plot(moneyness, EURJPY_analytic_approx_ivol, label="EURJPY approx.")
plt.legend(handles=[plt1, plt2])
plt.draw()

plt.show()



