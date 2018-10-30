#----------------------------------------------------------------
# Homework 4

# Boyao Zhu
#----------------------------------------------------------------


#===============================================================#
  # Problem 2
#===============================================================#
# (a)

import scipy.integrate as integrate
import matplotlib.pyplot as plt
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


#################################################################################
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


#################################################################################
# (c)
# purity = # of electron / (# of electron * percentage + # of pion * percentage)

N_electron = integrate.quad(gaussian, -np.inf, 1, args=(mu2,sigma2))[0]
N_pion     = integrate.quad(gaussian, -np.inf, 1, args=(mu,sigma))[0]
e_pct      = 0.01
pi_pct     = 0.99

purity = (N_electron * e_pct)/(N_electron*e_pct+N_pion*pi_pct)

print ("the purity is ", purity)
# the purity is  0.05084202446614325

#################################################################################
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



#===============================================================#
# Problem 3
#===============================================================#
# (a)

# read the data from text file
with open('hw4_chi2_histogram.txt', 'r') as f:
    data = f.readlines()

i = 0
b = []

# split data and save numerics into b
for line in data:
    a = line.split(":")
    i += 1
    if i >= 4:
        b.append(a[1])

thm1 = []
thm2 = []
data = []

# split b and save data into corresponding lists
for line in b:
    a = line.split(",")
    thm1.append(a[0])
    thm2.append(a[1])
    data.append(a[2])

# convert string characters into numerical values
for i in range(len(thm1)):
    
    thm1[i] = float(thm1[i])
    thm2[i] = float(thm2[i])
    data[i] = float(data[i])

# scale data
sum_data = np.sum(data)                # sum data
sum_thm1 = np.sum(thm1)                # sum h1 data
sum_thm2 = np.sum(thm2)                # sum h2 data
const1 = sum_data/sum_thm1             # calculate h1 scale coefficient
const2 = sum_data/sum_thm2             # calculate h2 scale coefficient

# scale back to original data
for i in range(len(thm1)):
    thm1[i] = const1*thm1[i]
    thm2[i] = const2*thm2[i]

chi1 = 0
chi2 = 0
# calculate chi-square for h1 and h2
for i in range(len(thm1)):
    chi1 += (data[i]-thm1[i])**2 / thm1[i]
    chi2 += (data[i]-thm2[i])**2 / thm2[i]
print ("The Chi-square for h1 is ", chi1)
print ("The Chi-square for h2 is ", chi2)

# The Chi-square for h1 is  82.35070545536655
# The Chi-square for h2 is  68.80411482783917

#################################################################################
# (b)

# calculate possible bins
rebins = []
for i in range(1,101):
    if 100%i == 0:
        rebins.append(i)


chi2_h1 = []
chi2_h2 = []

# calculate chi-square with different bins
for i in rebins:
    k = 100/i
    temp1h = np.zeros(i)
    temp2h = np.zeros(i)
    tempdt = np.zeros(i)
    
    for l in range(i):
        for j in range(int(k)):
            temp1h[l] += thm1[l*int(k)+j]
            temp2h[l] += thm2[l*int(k)+j]
            tempdt[l] += data[l*int(k)+j]
    chi1 = 0
    chi2 = 0
    for m in range(len(temp1h)):
        chi1 += (tempdt[m]-temp1h[m])**2 / temp1h[m]
        chi2 += (tempdt[m]-temp2h[m])**2 / temp2h[m]
    chi2_h1.append(chi1)
    chi2_h2.append(chi2)

# plot chi-square v.s. bins
plt.plot(rebins, chi2_h1, label="chi_h1")
plt.plot(rebins, chi2_h2, label="chi_h2")
plt.legend()
plt.title("3(b)")
plt.xlabel("bins")
plt.ylabel("chi_square")
plt.show()

#################################################################################
# (c)







