import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from black_analytics import black_option_price
from black_analytics import black_implied_vol
from sobol_lib import *

S = 100
T = 10
dt = 100
riskfree = 0.05
div = 0.02
vol = 0.15
drift = riskfree - div - 0.5 * vol * vol

# x = logS
dx = np.log(100) - np.log(99)
edx = np.exp(dx)

pu = -0.5 * dt * ( (vol/dx)**2 + drift/dx )
pm =  1.0 + dt * ( (vol/dx))**2 + r*dt
pd = -0.5 * dt * ( (vol/dx)**2 - drift/dx )

St = []

for i in range(len(EURUSD_strike)):

moneyness = np.arange(0.7, 1.3, 0.1)





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




