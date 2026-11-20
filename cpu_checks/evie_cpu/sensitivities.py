import numpy as np

def derivatives(t,N,h=1,x2=1,s=1):
    t,h,x2,s=np.broadcast_arrays(t,h,x2,s);y=x2.copy();dy=np.zeros_like(t);ddy=dy.copy();out=np.zeros(t.shape+(3,))
    a=(1-t)**2;da=-2*(1-t)
    for _ in range(N):
        out+=np.stack((y,dy,ddy),axis=-1)*h[...,None]/2
        y,dy,ddy=a*y+t*t*s/h**2,a*dy+da*y+2*t*s/h**2,a*ddy+2*da*dy+2*y+2*s/h**2
    return out
