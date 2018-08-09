import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt

# from black_analytics import black_option_price
# from black_analytics import black_implied_vol

S = 100
riskfree = 0.0
div = 0.0
vol = 0.2

strike = 100

drift = riskfree - div - 0.5 * vol * vol

T = 1.0
num_dt = 13
dt = T/num_dt

num_std_coverage = 10
minS = 1
maxS = S * np.exp(drift * T + vol * np.sqrt(T) * num_std_coverage)

minx = np.log(minS)
maxx = np.log(maxS)

# x = logS
num_dx = 12
dx = (maxx-minx)/(num_dx-1)
x_grid = np.arange(0, num_dx, 1) * dx

# for future convenience use
S_grid = np.exp(x_grid)

edx = np.exp(dx)

pu = -0.5 * dt * ( (vol/dx)**2 + drift/dx )
pm =  1.0 + dt * ( (vol/dx))**2 + riskfree*dt
pd = -0.5 * dt * ( (vol/dx)**2 - drift/dx )

payoff_T = np.maximum( S_grid - strike, 0)

lambda_U = S_grid[num_dx-1] - S_grid[num_dx-2]
lambda_L = 0

from_time = T
#index running from 0 to num_dx-1
for i in range(num_dt, 0, -1):


    # debug information
    to_time = from_time-dt
    print("from = ", from_time, ", to = ", to_time)
    from_time = to_time








# EURJPY_analytic_approx_ivol = []
# for i in range(len(EURJPY_strike)):
#     K = EURJPY_strike[i]
#     mc_EUR_call_JPY_put_in_USD = np.maximum(EURUSD_T*USDJPY_T-K, 0.0)/USDJPY_T
#     mc_EUR_call_JPY_put_in_USD = np.mean(mc_EUR_call_JPY_put_in_USD) * DF_USD
#     mc_EUR_call_JPY_put_in_JPY = mc_EUR_call_JPY_put_in_USD * USDJPY
#
#     EURJPY_ivol.append( black_implied_vol(mc_EUR_call_JPY_put_in_JPY/DF_JPY , Forward_EURJPY, K, T, 1 ) )
#     EURJPY_analytic_approx_ivol.append(approx_vol_EURJPY)
#
# f = plt.figure(2)
# plt1 ,= plt.plot(EURJPY_strike, EURJPY_ivol, label="EURJPY MC")
# plt2 ,= plt.plot(EURJPY_strike, EURJPY_analytic_approx_ivol, label="EURJPY approx.")
# plt.legend(handles=[plt1, plt2])
# plt.draw()
#
# plt.show()




