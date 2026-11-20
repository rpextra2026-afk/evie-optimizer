import numpy as np

def calibrate(s,sizes=None,w=.85):
    if sizes is None: mean=np.mean(s,axis=-1)
    else: mean=np.mean([np.mean(v,axis=-1) for v in np.split(s,np.cumsum(sizes)[:-1],axis=-1)],axis=0)
    return np.clip((1/w-1)/np.maximum(mean,1e-12),10,5000)
def gate(z,method,w=.85):
    if method.startswith('ungated'):return np.ones_like(z)
    if method=='evie':return np.sqrt(np.maximum(1/(1+z),1e-6))
    return np.sqrt(np.clip((1+z)/(w*(1+2*z)),1e-6,8))
