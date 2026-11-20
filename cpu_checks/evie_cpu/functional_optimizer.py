"""Independent NumPy optimizer, batched replicas retain their own statistics."""
import numpy as np
from .calibration import gate,calibrate
class Functional:
    def __init__(self,x,method,lr,wd=0,sizes=(2,6)):
        self.x=np.array(x,copy=True);self.method=method;self.lr=np.asarray(lr);self.wd=wd;self.sizes=sizes
        self.m=np.zeros_like(x);self.v=self.m.copy();self.fm=self.m.copy();self.fv=self.m.copy();self.gamma=np.ones(x.shape[:-1]);self.t=0;self.calibrated=False
    def step(self,g):
        self.t+=1;t=self.t
        self.m=.9*self.m+.1*g;self.v=.999*self.v+.001*g*g;self.fm=.9*self.fm+.1*g;self.fv=.9*self.fv+.1*g*g
        mc=self.m/(1-.9**t);vc=self.v/(1-.999**t);self.s=np.maximum(vc-mc*mc,0)
        self.G=gate(self.gamma[...,None]*self.s,self.method);self.denom=np.maximum(np.sqrt(vc),1e-8)
        self.x*=1-self.lr[...,None]*self.wd;self.x-=self.lr[...,None]*mc*self.G/self.denom
    def calibrate(self):
        if not self.calibrated:
            self.gamma=calibrate(np.maximum(self.fv-self.fm*self.fm,0),self.sizes);self.calibrated=True
