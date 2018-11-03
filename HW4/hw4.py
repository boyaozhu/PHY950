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
sum_data = np.ones(len(data))*np.sum(data)                # sum data

# scale back to original data

thm1 = sum_data*thm1
thm2 = sum_data*thm2

from scipy.stats import chisquare

chi1h = 0
chi2h = 0
# calculate chi-square for h1 and h2

chi1h = chisquare(data, f_exp=thm1)
chi2h = chisquare(data, f_exp=thm2)
print ("The Chi-square for h1 is ", chi1h[0])
print ("The Chi-square for h2 is ", chi2h[0])

# The Chi-square for h1 is  82.35070545536655
# The Chi-square for h2 is  68.80411482783917
# That is for 100 bins
# I also gave chi2 for 25 bin in (b)
#################################################################################
# (b)

# calculate possible bins
'''
rebins = []
for i in range(1,101):
    if 100%i == 0:
        rebins.append(i)
'''
chi2_h1 = []
chi2_h2 = []

# calculate chi-square with different bins
for i in range(1,101):
    k = 100//i                                 # count # of data per bin
    temp1h = np.zeros(i)
    temp2h = np.zeros(i)
    tempdt = np.zeros(i)
    
    for l in range(i):
        for j in range(int(k)):
            temp1h[l] += thm1[l*int(k)+j]
            temp2h[l] += thm2[l*int(k)+j]
            tempdt[l] += data[l*int(k)+j]
    
    chi1 = chisquare(tempdt, f_exp=temp1h)
    chi2 = chisquare(tempdt, f_exp=temp2h)
    if i == 25:
        a = temp1h          # save value for (d)
        b = temp2h
        c = tempdt
    chi2_h1.append(chi1[0])
    chi2_h2.append(chi2[0])
rebins = [i for i in range(1,101)]
# plot chi-square v.s. bins

plt.plot(rebins, chi2_h1, label="chi_h1")
plt.plot(rebins, chi2_h2, label="chi_h2")
plt.legend()
plt.title("3(b)")
plt.xlabel("bins")
plt.ylabel("chi_square")
plt.show()

print ("For bin of 25, as shown in the graph given, the chi2 for h1 is ", chi2_h1[24])
print ("the chi2 for h2 is", chi2_h2[24])

# if we rebin the data as 25, for example, then
# the chi2 for h1 is  28.995462324183322
# the chi2 for h2 is 16.19912802098431

#################################################################################
# (c)

# the number of degrees of freedom is exactly the same as the number of bins

#################################################################################
# (d)
chi21h = chisquare(c, f_exp=a, ddof = -1)
print (chi21h[1])
chi22h = chisquare(c, f_exp=b, ddof = -1)
print (chi22h[1])
# for bin = 25,
# The chi2 for h1 is 0.26410573986601965
# The chi2 for h2 is 0.9087695160208741


#################################################################################
# (e)

h1 = a
h2 = b
chi1 = []
chi2 = []
ex_h1 = np.zeros(len(h1))
ex_h2 = np.zeros(len(h2))

# Monte Carlo Similation for each bins with h1 and h2 values as being mean
for i in range(100000):           # simulate 100,000 times of experiments
    for j in range(len(h1)):
        ex_h1[j] = np.random.poisson(h1[j], 1)
        ex_h2[j] = np.random.poisson(h2[j], 1)
    chi_h1 = chisquare(ex_h1, f_exp = h1)[0]
    chi_h2 = chisquare(ex_h2, f_exp = h2)[0]
    chi1.append(chi_h1)
    chi2.append(chi_h2)

# plot h1 similation
count, bins, ignored = plt.hist(chi1, 1000, normed=True)  # plot using 1,000 bins
plt.title("chi2 of h1 3(e)")
plt.legend()
plt.xlabel("chi2")
plt.ylabel("Count")
plt.show()
# plot h2 similation
count2, bins2, ignored2 = plt.hist(chi2, 1000, normed=True)
plt.title("chi2of h2 3(e)")
plt.legend()
plt.xlabel("chi2")
plt.ylabel("Count")
plt.show()

newset = []
newset2 = []

# collect values for chi2 equal to or larger than those in (d)
for i in range(1000):
    if bins[i] >= chi2_h1[24]:
        newset.append(count[i])
    if bins2[i] >= chi2_h2[24]:
        newset2.append(count2[i])

# calculate the integration of p_value evaluated from chi2 obtained in (d)
# where (bins[1]-bins[0]) specify the width of bins
print (np.sum(newset)*(bins[1]-bins[0]))
print (np.sum(newset2)*(bins2[1]-bins2[0]))

# 0.2619699999999991
# 0.9080200000000022
# compared to (d)
# 0.26410573986601965
# 0.9087695160208741
