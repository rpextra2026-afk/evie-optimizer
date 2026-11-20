import numpy as np
from .segments import tr

def direct(A,Q,W,S,N):
    S=S.copy();value=np.zeros(S.shape[:-2]);mineig=0.
    for n in range(N):
        value+=tr(W,S);S=A@S@np.swapaxes(A,-1,-2)+Q
    return value,S
