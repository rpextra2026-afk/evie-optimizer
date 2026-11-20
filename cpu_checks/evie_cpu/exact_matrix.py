import numpy as np
from .exact_scalar import closed

def spectral(H,K,Sigma,C,N):
    h,V=np.linalg.eigh(H);half=(V*np.sqrt(h))@V.T;inv=(V/np.sqrt(h))@V.T
    t,U=np.linalg.eigh(half@K@half);u,w=closed(t,N)
    return .5*np.sum(np.diag(U.T@half@C@half@U)*u+np.diag(U.T@inv@Sigma@inv@U)*w)
