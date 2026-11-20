import numpy as np
from scipy.stats import t

def interval(x,family=1):
    n=len(x);mean=float(np.mean(x));se=float(np.std(x,ddof=1)/np.sqrt(n));q=t.ppf(1-.025/family,n-1)
    return mean,se,mean-q*se,mean+q*se
