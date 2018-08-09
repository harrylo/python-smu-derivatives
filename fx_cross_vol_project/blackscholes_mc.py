import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from black_analytics import black_option_price
from black_analytics import black_implied_vol
from sobol_lib import *

#TODO, I cannot get shifted lognormal to work for single asset implied vol? strange, 6 Aug 2018

# generate sobol number
dimension = 1
skip = 2
power = 15
sample_size = 2 ** power - 1
sobol_uniform = i4_sobol_generate(dimension, sample_size, skip)
sobol_normal = scipy.stats.norm.ppf(sobol_uniform, 0, 1.0)

spot = 100
r = 0.05
d = 0.02
T = 3
DF = np.exp(-r*T)
mu = r - d
Forward = spot * np.exp(mu*T)
vol = 0.20
shift = 50

sqrt_T = np.sqrt(T)

# this is to choose the shifted lognormal vol such that the ATM price doesn't change
undisc_ATM_option_price = black_option_price(Forward, Forward, T, vol, 1)
sln_vol = black_implied_vol(undisc_ATM_option_price, Forward+shift, Forward+shift, T, 1)

print("undisc_ATM_option_price =", undisc_ATM_option_price)
print("sln vol =", sln_vol)
print("lognormal vol =", vol)

moneyness = np.arange(0.9, 1.1, 0.1)
strike = Forward * moneyness

closed_form_ivol = []
print("from closed form")
for i in range(len(strike)):
    K = strike[i]
    undiscounted_price = black_option_price(Forward+shift, K+shift, T, sln_vol, 1)
    ivol = black_implied_vol(undiscounted_price, Forward, K, T, 1)
    closed_form_ivol.append( ivol )
    print("K = ", K, "Fwd = ", Forward, " price = ", undiscounted_price, " ivol = ", ivol)

martingale = np.exp((- 0.5 * sln_vol ** 2) * T + sln_vol * np.sqrt(T) * sobol_normal[0])
S_T = (Forward+shift) * martingale - shift

# price a forward contract
mc_fwd = np.mean(S_T)

print("from MC")
mc_ivol = []
mc_price_in_USD = []
call_or_put = 1
for i in range(len(strike)):
    K = strike[i]
    mc_price_in_USD = np.maximum(S_T - K, 0.0)
    undiscounted_price = np.mean(mc_price_in_USD)
    ivol = black_implied_vol(undiscounted_price, Forward, K, T, call_or_put)
    mc_ivol.append(ivol)
    print("K = ", K, "Fwd = ", mc_fwd, " price = ", undiscounted_price, " ivol = ", ivol)

plt1 ,= plt.plot(moneyness, closed_form_ivol, label="closed form ivol")
plt2 ,= plt.plot(moneyness, mc_ivol, label="MC ivol")
plt.legend(handles=[plt1, plt2])
f = plt.figure(1)
plt.draw()

plt.show()




