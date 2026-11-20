import numpy as np

def sums(t,N):
    t=np.asarray(t);a=1-t;n=np.arange(N);k=np.arange(max(0,N-1))
    return np.sum(a[...,None]**(2*n),axis=-1),t*t*np.sum((N-1-k)*a[...,None]**(2*k),axis=-1)
def closed(t,N):
    t=np.asarray(t,dtype=float);U=np.full_like(t,float(N));W=np.zeros_like(t)
    if N<=1:return U,W
    mask=(t!=0)&(t!=2)
    tt=t[mask];q=(1-tt)**2
    with np.errstate(divide='ignore',invalid='ignore'):
        logq=np.where(tt<1,2*np.log1p(-tt),2*np.log(np.abs(1-tt)));z=-np.expm1(N*logq);U[mask]=z/(tt*(2-tt));W[mask]=N*tt/(2-tt)-z/(2-tt)**2
    small=(N*np.minimum(np.abs(t),np.abs(2-t))<1e-3)&(t!=0)
    if np.any(small):
        # Positive recurrence of a scalar moment/cost segment avoids subtraction.
        from .segments import cost
        a=(1-t[small])[...,None,None];q=(t[small]**2)[...,None,None]
        W[small]=cost(a,q,np.ones_like(a),np.zeros_like(a),N)
    if np.any(t==2):U[t==2],W[t==2]=sums(t[t==2],N)
    return U,W
def loss(t,N,h=1,x2=1,s=1):
    u,w=closed(t,N);return h*x2*u/2+s*w/(2*h)
