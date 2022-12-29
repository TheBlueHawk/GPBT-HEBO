import pandas as pd
import numpy as np
from math import exp

from numpy import genfromtxt

### Kolmogorov Smirnov test on array-like data

def empirical_distribution_twinned(data1,data2):
    """
    data1 and data2 are numpy array of shape (n,2)
    data1[i] gives the array [[t,y]] where t is the time of the observation and y is the value of the observation
    We suppose data1 and data2 are sorted according to the lexicographic order
    """
    n = data1.shape[0]
    m = data2.shape[0]
    tmax = max(data1[-1][0], data2[-1][0])
    
    distr1 = []
    distr2 = []
    curr1 = 0
    curr2 = 0
    i,j = 0,0
    while i < n and j < m:
        t1 = data1[i][0]
        t2 = data2[j][0]
        if t1 < t2:
            curr1 += data1[i][1]/tmax
            distr1.append([curr1,t1])
            distr2.append([curr2,t1])
            i += 1
        elif t1 > t2:
            curr2 += data2[j][1]/tmax
            distr1.append([curr1,t2])
            distr2.append([curr2,t2])
            j += 1
        else:
            curr1 += data1[i][1]/tmax
            curr2 += data2[j][1]/tmax
            distr1.append([curr1,t1])
            distr2.append([curr2,t1])
            i += 1
            j += 1
    while i < n:
        curr1 += data1[i][1]/tmax
        t1 = data1[i][0]
        distr1.append([curr1,t1])
        distr2.append([curr2,t1])
        i += 1
    while j < m:
        curr2 += data2[j][1]/tmax
        t2 = data2[j][0]
        distr1.append([curr1,t2])
        distr2.append([curr2,t2])
        j += 1
    return(np.array(distr1),np.array(distr2))

def kolmogorov_smirnov_distr(distr1,distr2):
    maxdist = 0
    n = distr1.shape[0]
    print(n)
    for i in range(n):
        d = abs(distr1[i][0] - distr2[i][0])
        maxdist = max(maxdist, d)
    alpha = 2*exp(-n*maxdist**2)
    return(maxdist,alpha)

def kolmogorov_smirnov(data1,data2):
    tmp = empirical_distribution_twinned(data1,data2)
    print(tmp)
    return(kolmogorov_smirnov_distr(tmp[0],tmp[1]))
    
### Data loading


"""hyperopt = genfromtxt('hyper.csv',
                     delimiter=',',
                     usecols=(0,3,4),
                     )

fsvn = genfromtxt('pb2.csv',
                     delimiter=',',
                     usecols=(0,3,4),
                     )
"""
"""
TODO LIST

- Vérifier KS (parce ça fait de la merde là)
- Implémenter un calcul de p-value pour dire que une mesure est différente des autres en supposant un modèle gaussien
- Analyse de l'overfit


"""

# p-value to test if our means are statistially different
def pval(mu1,sigma1,n1,mu2,sigma2,n2):
    from scipy.stats import t
    from math import sqrt
    # we perform a t-test
    tval = (mu1-mu2)/(sqrt((sigma1**2/n1) + (sigma2**2/n2)))
    nu = (sigma1**2/n1 + sigma2**2/n2)**2/(sigma1**4/(n1**2*(n1-1)) + sigma2**4/(n2**2*(n2-1)))
    p_val = 2*t.cdf(-abs(tval),nu)
    return(p_val)

print(pval(0.98324,0.00164365446490435,10,0.98324,0.001643654464904,10))


def convergence(p,mu,sigma,time,N):
    """
    (bad wording)
    Compares the final measured value with the previous ones, and find the first one with p-value > p
    """
    # Sanity check
    if not mu.shape == sigma.shape or not mu.shape == time.shape:
        raise "Invalid argument"
    n = mu.shape[0]
    mu_final = mu[-1]
    sigma_final = sigma[-1]
    idx = n-2
    while idx >= 0 and pval(mu_final,sigma_final, N, mu[idx], sigma[idx], N) > p:
        print(pval(mu_final,sigma_final, N, mu[idx], sigma[idx], N))
        idx -= 1
    return(time[idx+1])

def convergence_file(p,filename,N):
    data = genfromtxt(filename,
                     delimiter=',',
                     usecols=(0,3,4),
                     )
    return(convergence(p,data[:,1],data[:,2],data[:,0],N))

print(convergence_file(0.05,'data_results/mnist/hyper.csv',10))
