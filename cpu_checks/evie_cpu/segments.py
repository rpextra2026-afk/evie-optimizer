"""Batched moment/cost monoid; trailing two axes are state matrices."""
import numpy as np

def tr(a,b): return np.einsum('...ij,...ji->...',a,b)
def transpose(a): return np.swapaxes(a,-1,-2)
def compose(a,b):
    T,R,U,c=a; V,S,Z,d=b
    return V@T,V@R@transpose(V)+S,U+transpose(T)@Z@T,c+d+tr(Z,R)
def segment(A,Q,W,N):
    I=np.broadcast_to(np.eye(A.shape[-1]),A.shape).copy();zero=np.zeros_like(A)
    out=(I,zero,zero,np.zeros(A.shape[:-2]));base=(A,Q,W,np.zeros(A.shape[:-2]))
    while N:
        if N&1: out=compose(out,base)
        N//=2
        if N: base=compose(base,base)
    return out
def cost(A,Q,W,S,N):
    T,R,U,c=segment(A,Q,W,N)
    return tr(U,S)+c
