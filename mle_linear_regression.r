#
# MLE Example - R code by Yaroslav Blogov 
#
# Analytic representation:
#
# http://www.le.ac.uk/users/dsgp1/COURSES/MATHSTAT/13mlreg.pdf
#
# 16:08 09.09.2015
#
#

# загрузка данных из файлов
x <- read.table("x_matlab.txt", sep = ",")
y <- read.table("y_matlab.txt")

# преобразование данных в матрично-векторные типы
x <- as.matrix(x)
y <- as.matrix(y)

# запись функции правдоподобия для линейной регрессии y ~ x
# logL(beta, sigma) = -T/2*log(2*pi) -T/2*log(sigma^2) -1/(2*sigma^2)*(y-x*beta)'*(y-x*beta)
neg_llh <- function(x, y, par) { 
  
  # par[1] = beta_1, ..., par[n] = beta_n, par[n+1] = sigma^2
  if (nrow(x) != nrow(y)) stop ("x and y must have equal length")
  
  n <- ncol(x) # количество регрессоров (вкл. константу) 
  T <- nrow(x) # количество наблюдений

  # логарифм функции правдоподобия
  # cbind - предобразование списка значений в вектор-столбец
  # %*% - операция матрично-векторного умножения (как в линейной алгебре)
  # * - операция поэлементного умножения
  logL <- -T/2*log(2*pi) -T/2*log(par[n+1]) -1/(2*par[n+1])*t(y-x%*%cbind(par[1:n]))%*%(y-x%*%cbind(par[1:n]))

  return(-logL)

}

# подбор параметров регрессии численными методами оптимизации
n <- ncol(x); T <- nrow(x)
theta <- rep(1, times = n+1) # вектор параметров
# минимизируемая функция:
# отрицательное логарифмическое правдоподобие с подставленными значениями x и y
ml_func <- function(theta) neg_llh(x, y, theta)

opt <- optim(fn = ml_func, par = theta, method = "Nelder-Mead")

# вектор оптимальных значений параметров
opt$par

# график
y_hat <- apply(matrix(rep(opt$par[1:n], T), nrow = T, ncol = n, byrow = TRUE) * x, 1, sum)

par(cex = 1.5)
par(mar = c(2.1,2.1,0.1,0.1))
plot(x[,2], as.vector(y), pch = 16)
lines(x[,2], y_hat, col = "red", lwd = 3)


