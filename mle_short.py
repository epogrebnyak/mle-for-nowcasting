# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from mle import get_csv_rows_as_matrix, t, dnorm

def ncol(x):
    return np.shape(x)[1]

def nrow(x):
    return np.shape(x)[0]

#COMMENT: we use x and y from csv files for cleaner comparison
#         x and y can be derived from gdp time series
x = get_csv_rows_as_matrix("x_matlab.txt")
y = get_csv_rows_as_matrix("y_matlab.txt")
assert nrow(x) == nrow(y)
assert ncol(y) == 1
 
T = nrow(x)
k = ncol(x)

phi   = np.ones(k) 
sigma = np.ones(1)
theta = np.concatenate([phi, sigma])

def unpack(theta):
    "Returns phi as column-vector and sigma as scalar"
    return t(np.matrix(theta[0:-1])), theta[-1]

def negative_likelihood(theta, x, y):    
    # phi, sigma_squared = np.reshape(b[0:2], (2,1)), b[2] 
    # yhat = x*b(1:2);     

    phi, sigma = unpack(theta)    
    yhat = x * phi 
    residual = y - yhat
    log_likelihood_by_element = [np.log(dnorm(r, 0, sigma)) for r in residual]
    return -sum(log_likelihood_by_element)   
     
    # h1= b(3);    
    # g1 = exp(-.5*(y-yhat).^2/h1)./sqrt(2*pi*h1);
    # ll = (log(g1));
    # log_likelihood_by_element = [np.log(dnorm(z, 0, sigma_squared**.5)) for z in y-yhat]
    # fun=-(sum(ll));                     % Sum of the likelihood
    # return -np.asscalar(sum(log_likelihood_by_element))    
    
from scipy.optimize import minimize   
b = minimize(negative_likelihood, theta, args = (x,y), method = 'Nelder-Mead')
print (b)
