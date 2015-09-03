# -*- coding: utf-8 -*-

# g1 = exp(-.5*(y-yhat).^2/h1)./sqrt(2*pi*h1);
# g1 = [np.exp(-.5*(z)**2/h1) / sqrt(2*np.pi*h1) for z in y-yhat]

import numpy as np 
from scipy import stats

def dnorm(x, mu, sigma):
    return np.exp ( -.5*(x-mu)**2/(sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    
sigma = 0.3 ** .5
all_z = [0.5, .8, -2, 0]
  
# dnorm(z, 0, sigma)
print ([dnorm(z, 0, sigma) for z in all_z])
print ([stats.norm.pdf(z, 0, sigma) for z in all_z])

def test_dnorm():
    some_z = [0.5, .8, -2, 0]
    sigma = 2
    for z in some_z:
        assert abs(dnorm(z, 0, sigma) - stats.norm.pdf(z, 0, sigma)) < 0.00001

test_dnorm()

from scipy import stats
dnorm = stats.norm.pdf
errors = y-yhat
errors = [0.5, .8, -2, 0]
deviation = 0.3 ** .5
likelihood = sum (dnorm (z, 0, deviation) for z in errors)