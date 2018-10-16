
# coding: utf-8

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#% matplotlib inline
import random
import math



def main():
    
    ###########################################################################
    # Problem 1
    ##########################################################################

    random.seed(56575128)  # seed with my PID
    a = 8                  # element of probability to be calculated
    prob  = []             # create a list to store probability for each trial
    trial = []             # create a list to store trials in log2 base
    var   = []             # create a list of variance
    
    # create trials
    #t = [10,100,1000,10000,100000,1000000]
    t = []
    for i in range(3,21):
      t.append(2**i)
    
    for i in t:
        ran1 = np.random.randint(1,7, size=i)  # roll the first dice "i" times
        ran2 = np.random.randint(1,7, size=i)  # second dice "i" times
        sum  = ran1 + ran2                     # sum two dice
        
        var.append(np.var(sum))                # variance, typically the build-in function  \
                                               # of variance has bias when calculating.
        
        # count number of "8" occuring in trials
        num = 0
        for j in sum:
            if j == 8:
                num += 1

        p  = float(num)/i                  # calculate probability
        prob.append(p)
        trial.append(math.log(i,2))

    # calculate expected variance
    dice = [2,3,4,5,6,7,8,9,10,11,12]            # possible values of sum of two dice
    x_pr = np.array([1,2,3,4,5,6,5,4,3,2,1]) * (1/36)      # probability corresponding to each outcome
    mean = np.sum(dice * x_pr)


    s = 0
    for i in range(len(dice)):
        s += (dice[i]-mean)**2 * x_pr[i]              # expected variance
    
    s = np.zeros_like(trial)+s                        # list of variance
    


    # according to the diagram, the plot seems to be stable and converged as trials reach 10000
    f1 = plt.figure()
    plt.plot(trial, prob, '-*', label='Probability of 8')
    plt.xlabel('Log-trial')
    plt.ylabel('Probability')
    plt.xscale("log")
    plt.legend()
    plt.show()

    f2 = plt.figure()
    plt.plot(trial, var, '-o', label='Sample Variance')
    plt.plot(trial, s,'-',label='Expected Variance')
    plt.xlabel('Log-trial')
    plt.ylabel('Variance')
    plt.xscale("log")
    plt.legend()
    plt.show()
    

    ###########################################################################
    # Problem 2 has already been done in HW2
    ##########################################################################
    

    ###########################################################################
    # Problem 3
    ##########################################################################
    # (a)
    p = 500          # Mev/c
    m = 135          # Mev/c
    #c = 3.0e8
    E0 = np.sqrt(p**2 + m**2)
    beta = p/E0
    gamma = 1/np.sqrt(1-beta**2)    # define parameters
    #print (gamma)
    cos_theta = np.random.uniform(-1,1,100000)   # create random number from (-1,1)
    E = gamma * m/2 * (1+beta*cos_theta)         # calculate E
    bins = np.linspace(0,550,100)                # create bins
    f1 = plt.figure()
    count, bins, ingored = plt.hist(E, bins)    # plot histogram
    plt.show()
    
    # (b)
    mu = 500
    std = 50
    p_prime = np.random.normal(mu, std, 100000)       # create random momentum
    # print (np.mean(p_prime), np.std(p_prime))
    E0_prime = np.sqrt(p_prime**2 + m**2)               # doing the same calculation as in (a)
    beta_prime = p_prime/E0_prime
    gamma_prime = 1/np.sqrt(1-beta_prime**2)
    cos_prime = np.random.uniform(-1,1,100000)
    E_prime = gamma_prime * m/2 * (1+beta_prime*cos_prime)
    bins = np.linspace(0,660, 100)
    f2 = plt.figure()
    count, bins, ignored = plt.hist(E_prime, bins)
    plt.show()


    # (c)
    C = 200
    epsilon = 1/np.sqrt(1+np.exp(-E_prime/C))
    # E_pprime = [E_prime * epsilon for E_prime, epsilon in zip(E_prime, epsilon)]
    f3 = plt.figure()
    count, bins, ignored = plt.hist(E_prime, bins, weights=epsilon)
    plt.show()




if __name__ == "__main__":
    main()
