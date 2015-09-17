#
# Maximum likelihood estimate (MLE) example - R code 
#
# Model:
#    y(t) is GDP growth rate
#    x(t) = y(t-1) 
#    y(t) = a + x(t)* b  + e
# Model in matrix form:
#    y = x*beta +  e 
# Estimation result:
#    beta 
#

# may need to set working directory to where dare files are
setwd('D:/mle-for-nowcasting-master')

######################################################################
########## 1. Load and transform data
######################################################################

# загрузка данных из файлов
x <- read.table("x_matlab.txt", sep = ",")
y <- read.table("y_matlab.txt")

# преобразование данных в матрично-векторные типы
x <- as.matrix(x)
y <- as.matrix(y)

if (nrow(x) != nrow(y)) stop ("x and y must have equal row count")
if (ncol(y) != 1) stop ("y must be column-vector")

k <- ncol(x)
T <- nrow(x)

# initial guess for theta - all ones
theta <- rep(1, times = k+1) 

# theta consists of beta (vector) and sigma (scalar)
# functions below return beta as column-vector and sigma as scalar
get_beta   <- function(theta) return (as.matrix(theta[1:length(theta)-1]))
get_sigma <- function(theta) return (tail(theta, 1))
	
######################################################################
########## 2. Likelihood function 
######################################################################

negative_likelihood <- function(theta, x, y) {
    beta   = get_beta(theta)
	sigma = get_sigma(theta)  
	yhat = x %*% beta 
    residual = y - yhat
	log_likelihood_by_element = log(dnorm(residual, 0, sigma))
	return (-sum(log_likelihood_by_element))
}

# wrapper function to pass only one argument to optimisation procedure
ml_func = ml_func <- function(theta) negative_likelihood(theta, x, y) 

######################################################################
########## 3. Parameter estimation by ML
######################################################################

opt <- optim(fn = ml_func, par = theta, method = "Nelder-Mead")

# вектор оптимальных значений параметров
print(opt$par)


######################################################################
########## 4. Display estimation results
######################################################################

# график
#beta = get_beta(opt$par)
#yhat = x %*% beta
par(cex = 1.5)
par(mar = c(2.1,2.1,0.1,0.1))
#plot(x[,2], as.vector(y), pch = 16)
#lines(x[,2], y_hat, col = "red", lwd = 3)



