# Boyao Zhu
# HW 2 + one problem from HW3

import math
import numpy as np
import matplotlib.pyplot as plt

# the first function to print "hello world"
def HelloWorld():
    print ("Hello World")

# plot gaussian distribution
def Gaussian(mu, sigma):
    # sample standard normal distribution
    s = np.random.normal(mu,sigma,10000)
    # calculate mean and std
    mean = np.mean(s)
    std  = np.std(s)
    # plot histgram
    bins = np.linspace(-10,50,601)
    count, bins, ignored = plt.hist(s, bins, normed=True)
    # plot the corresponding function
    g = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins-mu)**2/(2*sigma**2))
    plt.plot(bins, g, linewidth = 2, color = "r")
    plt.show()
    
    return mean, std

# last problem corresponding to next homework
# sample sin() distribution
def random():
    # first sample standard uniform distribution
    s = np.random.uniform(0,1,10000)
    # calculate mean and std of uniform
    mean = np.mean(s)
    std  = np.std(s)
    # transform and get the sin() distribution
    x = np.arccos(1-2*s)/np.pi
    # plot histgram
    bins = np.linspace(0,1,101)
    count, bins, ignored = plt.hist(x, bins, normed=True)
    # plot sin() function
    g = np.pi/2*np.sin(np.pi*bins)
    plt.plot(bins, g, linewidth = 2, color = "r")
    plt.show()

    return mean, std

# Plot function x+Mx**2 taking the mean from gaussian as an argument
def function(mean):
    # get value M
    M = (mean*1e6)%10
    # get the range of x
    x = np.linspace(0,10,201)
    # calculate the function in the domain
    f = x + M*x**2
    # plot
    plt.plot(x,f,"r",label="f(x)=x+Mx^2")
    plt.xlabel("x")
    plt.ylabel("f")
    
    plt.legend()
    plt.show()


def main():
    # call the first function "hello world"
    HelloWorld()
    # assign arguments for gaussian
    mu, sigma = 16, 4
    # call gaussian distribution and return values assigned to mean and std
    # and print them out
    mean, std = Gaussian(mu, sigma)
    print ("the mean and standard deviation of gaussian distribution is", mean, std)
    # call sin() distribution and return values assigned to mean and std
    # and print them out
    mean, std = random()
    print ("the mean and standard deviation of Sin distribution is ", mean, std)
    # call function f= x+Mx**2 with argument taken from the mean of gaussian
    function(mean)

if __name__ == "__main__":
    main()
