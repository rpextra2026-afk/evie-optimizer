import unittest
import numpy as np
from evie_cpu.models import model,spd,diagonal
from evie_cpu.segments import cost,segment,compose,tr
from evie_cpu.moments import direct
from evie_cpu.exact_matrix import spectral
from evie_cpu.sensitivities import derivatives
from evie_cpu.exact_scalar import sums,loss

class Engines(unittest.TestCase):
    def test_noncommuting_and_singular_initialization(self):
        rng=np.random.default_rng(42)
        for d in [2,5]:
            H,S,K=[spd(rng,d,.3,1) for _ in range(3)];K*=.3;x=rng.normal(size=d);C=np.outer(x,x)
            for beta in [0,.9]:
                A,Q,W,X,_=model(H,K,S,C,beta)
                for N in [0,1,2,17,128]:
                    v,Z=direct(A,Q,W,X,N);self.assertAlmostEqual(float(cost(A,Q,W,X,N)),float(v),places=10)
                    if beta==0:self.assertAlmostEqual(spectral(H,K,S,C,N),float(v),places=10)
                    for a,b in [(3,14),(14,3)]:
                        _,_,U,c=compose(segment(A,Q,W,a),segment(A,Q,W,b));self.assertAlmostEqual(float(tr(U,X)+c),float(cost(A,Q,W,X,17)),places=10)
                    self.assertGreaterEqual(np.linalg.eigvalsh(Z).min(),-1e-10)
    def test_sensitivities(self):
        for t in [.01,.5,1,1.9]:
            f,d,dd=derivatives(t,23,h=.7,x2=1.3,s=.4);eps=1e-5
            self.assertAlmostEqual(d,float((loss(t+eps,23,.7,1.3,.4)-loss(t-eps,23,.7,1.3,.4))/(2*eps)),delta=2e-5*max(1,abs(d)))
            self.assertGreater(dd,0)
    def test_indexing(self):
        self.assertEqual(float(loss(.1,0)),0);self.assertEqual(float(loss(.1,1)),.5)
        u,w=sums(1,8);self.assertEqual(u,1);self.assertEqual(w,7)
if __name__=='__main__':unittest.main()
