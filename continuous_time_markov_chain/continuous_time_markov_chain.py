import numpy as np
from scipy.stats import norm
from scipy import linalg

# generate non-uniform grid
def generate_non_uniform_grid(nx,           #number of grid points
                              min_x,        #lower bound
                              max_x,        #upper bound
                              mid_x,        #mid point value
                              mid_x_index,  #index corresponds to the mid point value
                              density_lower,#parameter to control the density of the grid point lower than the mid point value
                              density_upper #parameter to control the density of the grid point higher than the mid point value
                              ):
    c1 = np.arcsinh((min_x - mid_x) / density_lower);
    c2 = np.arcsinh((max_x - mid_x) / density_upper);

    nx_lower = mid_x_index;
    nx_upper = nx - nx_lower + 1;

    #generate uniform grid between 0 and 1
    x_lower = np.linspace(0.0, 1.0, nx_lower)
    result_lower = mid_x + density_lower * np.sinh(c1 * (1 - x_lower));

    # generate uniform grid between 0 and 1
    x_upper = np.linspace(0.0, 1.0, nx_upper)
    result_upper = mid_x + density_upper * np.sinh(c2 * x_upper);

    #concatenate 2 arrays. Note that the last element of result_lower is the same as the first element of
    #result_upper, that's why we truncate the first element of result_upper by using result_upper[1:]
    result = np.concatenate([result_lower, result_upper[1:]])
    return result

#generate the Markov Generator for the process
#dS = drift dt + vol dW
def generate_diffusion_MG(S,     #the grid for S
                          drift, #drift
                          vol    #vol
                          ):
    N = len(S)

    L = np.zeros((N, N))

    L[0,:] = 0;   # absorption lower boundary
    L[N-1,:] = 0; # absorption upper boundary

    #running from 1 to N-2
    for x in range(1,N-1):

        dSp = S[x+1]-S[x]
        dSm = S[x-1]-S[x]

        A = np.zeros((2, 2))
        b = np.zeros(2)
        A[0,0] = dSp
        A[0,1] = dSm
        A[1,0] = dSp*dSp
        A[1,1] = dSm*dSm

        b[0] = drift
        b[1] = vol*vol

        ret = np.linalg.solve(A, b)

        if (ret[0]<0 or ret[1] <0):
            raise ValueError('negative probabilities.')

        L[x,x+1] = ret[0];
        L[x,x-1] = ret[1];
        L[x,x  ] = -(ret[0]+ret[1])

    return L





N               = 70;      #num grid pts for the underlying
spotS_index     = 36;      #index for the spot price
minS            = 5;
maxS            = 700;
spotS           = 100;
density_lower   = 30;
density_upper   = 30;

s_grid = generate_non_uniform_grid(N, minS, maxS, spotS, spotS_index, density_lower, density_upper);
#print(s_grid)

r = 0
vol = 0.2

L = generate_diffusion_MG(s_grid, r, vol)
print(L)
