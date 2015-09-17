# -*- coding: utf-8 -*-
"""
Maximum likelihood estimate (MLE) example - python code 
========================================================

Data:   
   x_matlab.txt   
   y_matlab.txt

Model:

    y(t) is GDP growth rate
    x(t) = y(t-1) 
    y(t) = a + x(t)* b + e
    
    In matrix form:
    y = x*beta +  e 
    
Estimation result:
    beta 
   
Comment:
1. For data import and manipulation can use pandas functions pandas.read_csv(). 
Keeping existing fucntions now for transparency.

2. We use x and y from csv files for cleaner comparison
          x and y can be derived from gdp time series

3. may want to check output against some package specialised in MLE, eg 
https://github.com/ibab/python-mle or 
R's https://stat.ethz.ch/R-manual/R-devel/library/stats4/html/mle.html
    Things to look for:
       - accuracy
       - convergence of minimisation function
       - speed comparison   
   
4. Next thing is programming Kalman filter and comparing it with other implementations in R and python. 
"""


import numpy as np
from scipy.optimize import minimize  
from csv_io import get_csv_rows_as_matrix

##################################################################
# Row and column count
##################################################################

def ncol(x):
    return np.shape(x)[1]

def nrow(x):
    return np.shape(x)[0]
    
##################################################################
# Matrix manipulation and math functions 
##################################################################
    
def t(x):
    '''Transpose matrix'''
    return np.transpose(x)
    
def inv(x):
    '''Invert matrix'''
    return np.linalg.inv(x)

M_1_SQRT_2PI = 1 / np.sqrt(2 * np.pi)

def dnorm(x, mu, sigma):
    """ Density function for normal distribution """ 
    # Same as:
    #     return np.exp ( -.5*(x-mu)**2/(sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    # Similar to:
    #     scipy.norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi) 
    x = (x - mu) / sigma    
    return M_1_SQRT_2PI * np.exp(-0.5 * x * x) / sigma

##################################################################
# Example 
##################################################################
    
x = get_csv_rows_as_matrix("x_matlab.txt")
y = get_csv_rows_as_matrix("y_matlab.txt")
assert nrow(x) == nrow(y)
assert ncol(y) == 1
 
T = nrow(x)
k = ncol(x)

beta  = np.ones(k) 
sigma = np.ones(1)
theta = np.concatenate([beta, sigma])

def unpack(theta):
    """Returns beta as column-vector and sigma as scalar"""
    return t(np.matrix(theta[0:-1])), theta[-1]

def negative_likelihood(theta, x, y):    
    beta, sigma = unpack(theta)    
    yhat = x * beta 
    residual = y - yhat
    log_likelihood_by_element = [np.log(dnorm(r, 0, sigma)) for r in residual]
    return -sum(log_likelihood_by_element)  
 
opt = minimize(negative_likelihood, theta, args = (x,y), method = 'Nelder-Mead')
print (opt.x)


