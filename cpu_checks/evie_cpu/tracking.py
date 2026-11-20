import numpy as np
from scipy.linalg import solve_discrete_lyapunov

def certificate(A,W):
    radius=float(max(abs(np.linalg.eigvals(A))))
    if radius>=1:return dict(status='condition_not_met',reason='reference_stability_condition_not_met',radius=radius)
    M=solve_discrete_lyapunov(A.T,np.eye(len(A)));v,U=np.linalg.eigh(M)
    if min(v)<=0:return dict(status='inconclusive_precision',radius=radius)
    half=(U*np.sqrt(v))@U.T;inv=(U/np.sqrt(v))@U.T
    rho=float(np.linalg.norm(half@A@inv,2));delta=float(np.linalg.eigvalsh(inv@W@inv).max())
    return dict(status='passed' if rho<1 else 'inconclusive_precision',radius=radius,M=M,rho=rho,delta=delta)
