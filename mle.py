# -*- coding: utf-8 -*-
"""
Maximum likelihood estimation - replication of repaso.m and ofn2.m MATLAB files.
For discussion and update. 

Comment:
1. For data import and manipulation can use pandas functions pandas.read_csv(). 
Keeping existing fucntions now for transparency.
2. Functions like get_gdp_values() can be entry point for data import by user. 
3. WARNING: deviation of x and y between python and matlab a bit high (see below)
4. inv(t(x) * x ) * t(x) * y - what's the algebra?
5. may want to check output against some package specialised in MLE, eg 

"""

import csv
import numpy as np
 

##################################################################
#  Basic CSV input-output functions
##################################################################

csv_flavor = {'delimiter': ',' , 'lineterminator' : '\n'}

# Output to CSV file

def dump_stream_to_csv(iterable, csv_filename):
    """ Write *iterable* stream into file *csv_filename*. """    
    with open(csv_filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile,  **csv_flavor)
        for row in iterable:        
             spamwriter.writerow(row)
    
def dump_list_to_csv(_list, csv_filename):
    dump_stream_to_csv(iter(_list), csv_filename)

# Input from CSV file

def yield_csv_rows(csv_filename):
    """ Open *csv_filename* and return rows as iterable."""
    with open(csv_filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, **csv_flavor)
        for row in spamreader:
            yield row
            
def get_csv_column_as_list(csv_filename, column = 0):
    return [float(x[column]) for x in yield_csv_rows(csv_filename)]

def get_csv_column_as_np_array(csv_filename, column = 0):
    return np.array(get_csv_column_as_list(csv_filename, column))

def get_csv_column_as_column_vector(csv_filename, column = 0):
    return t(np.matrix(get_csv_column_as_np_array(csv_filename, column)))
        
##################################################################
# Data import 
##################################################################
GDP_FILE = "gdp.txt"

def get_gdp_values():
    '''Returns first column from file *GDP_FILE* as column vector'''
    f = GDP_FILE 
    return get_csv_column_as_column_vector(f)

##################################################################
# Matrix manipulation  
##################################################################
    
def t(x):
    '''Transpose matrix'''
    return np.transpose(x)
    
def inv(x):
    '''Invert matrix'''
    return np.linalg.inv(x)

##################################################################
# Likelihood function
##################################################################

def dnorm(x, mu, sigma):
    """ Density function for normal distribution. """ 
    return np.exp ( -.5*(x-mu)**2/(sigma**2)) / (sigma * np.sqrt(2 * np.pi))

# function [fun] = ofn2(b)
def ofn2(b, x, y):
    # global x y
    # x, y = args    
    
    phi, sigma_squared = np.reshape(b[0:2], (2,1)), b[2] 
    # yhat = x*b(1:2); 
    yhat = x * phi 
     
    # h1= b(3);
    
    # g1 = exp(-.5*(y-yhat).^2/h1)./sqrt(2*pi*h1);
    # ll = (log(g1));
    log_likelihood_by_element = [np.log(dnorm(z, 0, sigma_squared**.5)) for z in y-yhat]
    # fun=-(sum(ll));                     % Sum of the likelihood
    return -np.asscalar(sum(log_likelihood_by_element))	
    
if __name__ == "__main__":
    
    ##################################################################
    # Import and transform data
    ##################################################################

    # Recreate the following matlab code:
    # yL=yL(:,2)
    # [T,n]=size(yL);                         
    # yv=100*log(yL(2:T,:)./yL(1:(T-1),:));   
    # ya=yv;
    # na=size(ya,1);
    # y=ya(2:na,:);
    # n=na-1
    # x =[ones(n,1) ya(1:n,:)];

    # load time series of gdp at constant prices    
    gdp = get_gdp_values()
    growth_rates = 100 * np.log(gdp[1:] / gdp[:-1])  

    # *y*  is regressed agianst its lagged version, *x*    
    # *y* includes *growth_rates* starting the second observation
    # y = growth_rates(t), t = 1, 2, ... n  
    y = growth_rates[1:]
    # *x* includes *growth_rates* starting first observation, but without last one
    # x(t) = growth_rates(t-1), t = 0, 1, ... (n-1)      
    B = growth_rates[:-1]
    A = np.ones_like(B)
    x = np.concatenate([A, B], axis=1)
    
    # later will also need x and y length (number of rows) 
    n = y.shape[0]    

    ##################################################################
    # Test we have same x and y in python and Matlab
    ##################################################################
    TOLERANCE = 0.01
 
    # WARNING: max deviation of x and y in python and matlab a bit high: 
    # >>> max(x[:,1] - x_matlab)
    # matrix([[ 0.00548916]])
    # >> max(y - matlab_y)
    # matrix([[ 0.00548916]])
    
    # test *x*
    x_matlab = get_csv_column_as_column_vector("x_matlab.txt", 1)
    assert np.allclose(x[:,1], x_matlab, atol = TOLERANCE)
    
    # test *y*
    y_matlab = get_csv_column_as_column_vector("y_matlab.txt")   
    assert np.allclose(y, y_matlab, atol = TOLERANCE)
    
    
    """
    # dump x and y to file:
    dump_list_to_csv(x, "x_python.txt")
    dump_list_to_csv(y, "y_python.txt")
    """

    ##################################################################
    # Prepare variables
    ##################################################################

    # b=inv(x'*x)*x'*y;
    # Where are we getting this from? y = x*b? what is matirx algebra for this?
    b = inv(t(x) * x ) * t(x) * y

    #e=y-x*b;
    e=y-x*b

    #sig=(e'*e)/n;
    sig=(t(e)*e)/n
    sig = np.asscalar(sig)

    #vari=(sig)*inv(x'*x);
    vari = sig*inv(t(x)*x)

    #B = 1;                                 % Matrix (n x 1)
    # we use B as a scalar, it can be used wherever (n x 1) matrix is used
    B = 1
     
    #phi=0.3*ones(2,1);                     % Matrix 0.3(2x1)
    phi = [0.3, 0.3]

    ##################################################################
    # Calculate maximum likelihood
    ##################################################################

    # startval=[phi;B];
    startval = phi + [B]
    
    # test this is the same function 
    assert abs(ofn2(startval, x, y) - 321.3745) < 0.1
    
    from scipy.optimize import minimize   
    b = minimize(ofn2, startval, args = (x,y), method = 'Nelder-Mead')
    print (b)    
    
    # check if MLE is similar to Matlab
    b_new = np.array([0.4843, 0.3641, 0.7277])
    ofn2_at_b_new = 308.7076     
    assert sum ((b.x - b_new) ** 2) < 0.01
    assert abs (ofn2(b.x, x, y) - ofn2_at_b_new) < 0.1   
    
    # % Row Concatanation
    # %% Maximizing the likelihood function
    # options = optimset('PlotFcns',@optimplotx,'DISPLAY','iter','TolFun',1e-8, 'MaxFunEvals',2000);
    # [x,lf]=fminunc(@ofn2, startval, options);
    ## lf = ofn2(b, args = (x,y))
    
    # optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)[source]
    # scipy.optimize.fmin_bfgs(f, x0, fprime=None, args=(), gtol=1e-05, norm=inf, epsilon=1.4901161193847656e-08, maxiter=None, full_output=0, disp=1, retall=0, callback=None)[source]