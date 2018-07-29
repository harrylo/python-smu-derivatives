import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt
from black_analytics import black_option_price
from black_analytics import black_option_vega
from black_analytics import black_implied_vol
from fourier import carr_madan_black_call_option
from fourier import carr_madan_heston_call_option

#call functions in black_analytics
forward = 100
vol = 0.3
t = 1
k = 100
call_price = black_option_price(forward, k, t, vol, 1)
ivol = black_implied_vol(call_price, forward, k, t, 1)

print("black call price = ", call_price)
print("black ivol = ", ivol)

#-------------------------------------------------------------------------------------------
alpha = 0.75 #carr-madan damping factor

test0 = carr_madan_black_call_option(forward, vol, t, 80, alpha)
ivol0 = black_implied_vol(test0, forward, 80, t, 1)

num_stdev = 1.0
lower_bound = forward * np.exp(-0.5*vol*vol*t - num_stdev * vol * np.sqrt(t))
upper_bound = forward * np.exp(-0.5*vol*vol*t + num_stdev * vol * np.sqrt(t))

#generate a range of strikes for options, strike step is 2.0
Ks = np.arange(lower_bound, upper_bound, 2.0)

vol_0 = vol    #the volatility at time 0, NOT variance
kappa = 0.02    #mean reversion rate
nu = vol       #mean reversion level
lam = 0.5      #vol of variance
rho = -0.5     #correlation between the variance and the underlying

heston_ivol_0 = []
heston_ivol_1 = []
heston_ivol_2 = []
heston_ivol_3 = []
heston_ivol_4 = []

for i in range(len(Ks)):
    temp_call = carr_madan_heston_call_option(forward, vol_0, kappa, nu, lam, -0.9, t, Ks[i], alpha)
    heston_ivol_0.append(black_implied_vol(temp_call, forward, Ks[i], t, 1))

    temp_call = carr_madan_heston_call_option(forward, vol_0, kappa, nu, lam, -0.7, t, Ks[i], alpha)
    heston_ivol_1.append(black_implied_vol(temp_call, forward, Ks[i], t, 1))

    temp_call = carr_madan_heston_call_option(forward, vol_0, kappa, nu, lam, -0.5, t, Ks[i], alpha)
    heston_ivol_2.append(black_implied_vol(temp_call, forward, Ks[i], t, 1))

    temp_call = carr_madan_heston_call_option(forward, vol_0, kappa, nu, lam, -0.3, t, Ks[i], alpha)
    heston_ivol_3.append(black_implied_vol(temp_call, forward, Ks[i], t, 1))

    temp_call = carr_madan_heston_call_option(forward, vol_0, kappa, nu, lam, -0.1, t, Ks[i], alpha)
    heston_ivol_4.append(black_implied_vol(temp_call, forward, Ks[i], t, 1))


plt1 ,= plt.plot(Ks, heston_ivol_0, label="rho = -0.9")
plt2 ,= plt.plot(Ks, heston_ivol_1, label="rho = -0.7")
plt3 ,= plt.plot(Ks, heston_ivol_2, label="rho = -0.5")
plt4 ,= plt.plot(Ks, heston_ivol_3, label="rho = -0.3")
plt5 ,= plt.plot(Ks, heston_ivol_4, label="rho = -0.1")

plt.legend(handles=[plt1, plt2, plt3, plt4, plt5])

plt.show()



