#----------------------------------------------------------------
# Homework 4

# Boyao Zhu
#----------------------------------------------------------------


#===============================================================#
  # Problem 2
#===============================================================#
# (a)

import scipy.integrate as integrate
import numpy as np 

# define gaussian function with mean and std, mu and sigma respectively
def gaussian(x,mu,sigma):
	return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
# parameters for poin
mu    = 2
sigma = 1
# integrate from negative infinity to 1 for pion distribution 
alpha = integrate.quad(gaussian, -np.inf, 1, args=(mu, sigma))[0]

print ("the significance level of test is ", alpha)

# the significance level of test is  0.15865525393145707

# (b)
# parameter for electron
mu2   = 0
sigma2 = 1
# integrate from 1 to infinity for electron distribution
beta = integrate.quad(gaussian, 1, np.inf, args=(mu2, sigma2))[0]
# power = 1 - beta
power = 1 - beta

print ("the power of the test is ", power)
print ("the probability that an electron will be accepted as a pion is beta ", beta)
# the power of the test is  0.8413447460685429
# the probability that an electron will be accepted as a pion is beta  0.15865525393145707

# (c)
# purity = # of electron / (# of electron * percentage + # of pion * percentage)

N_electron = integrate.quad(gaussian, -np.inf, 1, args=(mu2,sigma2))[0]
N_pion     = integrate.quad(gaussian, -np.inf, 1, args=(mu,sigma))[0]
e_pct      = 0.01
pi_pct     = 0.99

purity = (N_electron * e_pct)/(N_electron*e_pct+N_pion*pi_pct)

print ("the purity is ", purity)
# the purity is  0.05084202446614325

# (d)

# To check which integers such that purity is satisfied
i = 0
while True:
	N_electron = integrate.quad(gaussian, -np.inf, i, args=(mu2,sigma2))[0]
	N_pion     = integrate.quad(gaussian, -np.inf, i, args=(mu,sigma))[0]

	purity = (N_electron * e_pct)/(N_electron*e_pct+N_pion*pi_pct)

	if purity > 0.95:
		break
	else:
		i = i - 1

j = i + 1 - 0.1

# TO check which value such that purity is satisfied in tenth order
while True:
	N_electron = integrate.quad(gaussian, -np.inf, j, args=(mu2,sigma2))[0]
	N_pion     = integrate.quad(gaussian, -np.inf, j, args=(mu,sigma))[0]

	purity = (N_electron * e_pct)/(N_electron*e_pct+N_pion*pi_pct)

	if purity > 0.95:
		break
	else:
		j = j - 0.1

k = j + 0.1 - 0.01

# To check in hundredth order
while True:
	N_electron = integrate.quad(gaussian, -np.inf, k, args=(mu2,sigma2))[0]
	N_pion     = integrate.quad(gaussian, -np.inf, k, args=(mu,sigma))[0]

	purity = (N_electron * e_pct)/(N_electron*e_pct+N_pion*pi_pct)

	if purity > 0.95:
		break
	else:
		k = k - 0.01

l = k + 0.01 - 0.001

# To check in thousandth order
while True:
	N_electron = integrate.quad(gaussian, -np.inf, l, args=(mu2,sigma2))[0]
	N_pion     = integrate.quad(gaussian, -np.inf, l, args=(mu,sigma))[0]

	purity = (N_electron * e_pct)/(N_electron*e_pct+N_pion*pi_pct)

	if purity > 0.95:
		break
	else:
		l = l - 0.001


print ("The cut value on T is ", l)

# The cut value on T is  -2.516

alp = integrate.quad(gaussian, l, np.inf, args=(mu2, sigma2))[0]

print ("The values of the significance is ", alp)



