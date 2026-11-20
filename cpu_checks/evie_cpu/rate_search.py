import numpy as np
from scipy.optimize import minimize_scalar,brentq

def search(evaluator,upper,beta=0,derivative=None):
    # evaluator accepts a vector of rates. Keep both declared grids and offset validation.
    u=np.unique(np.r_[np.geomspace(1e-12,.1,257),np.linspace(.1,1-1e-10,257)])
    offset=np.unique(np.r_[np.geomspace(1e-12,.1,513),np.linspace(.1,1-1e-10,513)])
    allu=np.unique(np.r_[u,offset]);values=np.asarray(evaluator(allu*upper));inds=np.where((values[1:-1]<=values[:-2])&(values[1:-1]<=values[2:]))[0]+1
    brackets=[(allu[max(0,i-1)],allu[min(len(allu)-1,i+1)]) for i in inds]+[(allu[0],allu[1]),(allu[-2],allu[-1])]
    candidates=[(0.,float(evaluator(np.array([0.]))[0])),(allu[-1]*upper,float(values[-1]))]
    for a,b in brackets:
        opt=minimize_scalar(lambda z:float(evaluator(np.array([z*upper]))[0]),bounds=(a,b),method='bounded',options={'xatol':1e-15})
        candidates.append((opt.x*upper,opt.fun))
    if derivative is not None and beta==0:
        if derivative(0)<0 and derivative(upper*(1-1e-10))>0:
            r=brentq(derivative,0,upper*(1-1e-10),xtol=1e-15);candidates.append((r,float(evaluator(np.array([r]))[0])))
    eta,value=min(candidates,key=lambda v:v[1]);status='boundary_infimum' if eta==0 else ('certified_convex_1d' if derivative is not None and beta==0 else 'numerical_multistart_1d')
    step=max(eta*1e-4,upper*1e-8)
    residual=float((evaluator(np.array([eta+step]))[0]-evaluator(np.array([max(0,eta-step)]))[0])/(step+min(eta,step)))
    return dict(eta=float(eta),loss=float(value),solver_status=status,stationary_residual=residual,candidates=candidates,zero_cost=candidates[0][1],grid_points=len(allu))
