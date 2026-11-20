import numpy as np

def model(H,K,Sigma,C,beta=0):
    d=len(H);I=np.eye(d)
    if beta==0: return I-K@H,K@Sigma@K,H/2,C,-K
    A=np.block([[I-(1-beta)*K@H,-beta*K],[(1-beta)*H,beta*I]])
    B=np.vstack((-(1-beta)*K,(1-beta)*I))
    W=np.zeros((2*d,2*d));W[:d,:d]=H/2
    S=np.zeros_like(W);S[:d,:d]=C
    return A,B@Sigma@B.T,W,S,B

def diagonal(h,k,s,x2,beta):
    h,k,s,x2=np.broadcast_arrays(h,k,s,x2)
    A=np.zeros(h.shape+(2,2));A[...,0,0]=1-(1-beta)*k*h;A[...,0,1]=-beta*k;A[...,1,0]=(1-beta)*h;A[...,1,1]=beta
    B=np.stack((-(1-beta)*k,np.full_like(k,1-beta)),axis=-1)
    Q=B[..., :,None]*B[...,None,:]*s[...,None,None]
    W=np.zeros_like(A);W[...,0,0]=h/2
    S=np.zeros_like(A);S[...,0,0]=x2
    return A,Q,W,S

def spd(rng,d,lo,hi):
    u,_=np.linalg.qr(rng.normal(size=(d,d)));return (u*rng.uniform(lo,hi,d))@u.T
