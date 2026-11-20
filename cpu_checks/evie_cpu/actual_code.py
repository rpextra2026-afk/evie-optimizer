import ast,importlib.util
from pathlib import Path
import numpy as np
from .functional_optimizer import Functional
FILES=['cpu_checks/reference_optimizers/evie.py','cpu_checks/reference_optimizers/evie_amplify.py','finance/benchmark/benchmark_partA.py','vision/benchmark/benchmark_partA.py','language/benchmark/benchmark_partA.py']
def classes(root):
    import torch
    for i,rel in enumerate(FILES):
        path=root/rel
        if i<2:
            spec=importlib.util.spec_from_file_location('audit'+str(i),path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            name=['EvieOptimizer','EvieAmplifyOptimizer'][i];yield rel,name,getattr(mod,name),bool(i)
        else:
            tree=ast.parse(path.read_text())
            for name,amp in [('EvieOptimizer',False),('EvieAmplifyOptimizer',True)]:
                node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
                ns=dict(torch=torch,np=np,TARGET_WP=.85,WELL_POS_MAX=8.,FAST_EMA_BETA=.9,log=lambda *a,**kw:None)
                exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
                yield rel,name,ns[name],amp

def correspondence(root):
    import torch
    torch.set_num_threads(1);rows=[]
    for rel,name,cls,amp in classes(root):
      for wd in [0,.01]:
       for fixture in ['prescribed','zero','floor']:
        rng=np.random.default_rng(7266);x=rng.normal(size=8);ps=[torch.nn.Parameter(torch.tensor(x[:2],dtype=torch.float64)),torch.nn.Parameter(torch.tensor(x[2:].reshape(3,2),dtype=torch.float64))]
        opt=cls(ps,lr=.003,weight_decay=wd,target_wp=.85);f=Functional(x,'evie_amplify' if amp else 'evie',.003,wd)
        err=state_err=gamma_err=0.;noop=True
        for t in range(1,221):
            g=np.r_[rng.normal(.01*np.cos(t/40),.02,2),rng.normal(.01*np.cos(t/40),.04,6)]
            if fixture=='zero':g*=0
            if fixture=='floor':g*=1e-14
            ps[0].grad=torch.tensor(g[:2],dtype=torch.float64);ps[1].grad=torch.tensor(g[2:].reshape(3,2),dtype=torch.float64)
            opt.step();f.step(g)
            err=max(err,float(np.max(np.abs(np.concatenate([p.detach().numpy().ravel() for p in ps])-f.x))))
            for key,val in [('m1',f.m),('m2',f.v),('fast_m1',f.fm),('fast_m2',f.fv)]:
                got=np.concatenate([opt.state[p][key].numpy().ravel() for p in ps]);state_err=max(state_err,float(np.max(np.abs(got-val))))
            assert all(opt.state[p]['step']==t for p in ps)
            if t==126:
                opt.calibrate_gamma0();f.calibrate();gamma_err=abs(opt.param_groups[0]['gamma0']-float(f.gamma));old=opt.param_groups[0]['gamma0'];opt.calibrate_gamma0();noop=opt.param_groups[0]['gamma0']==old
        rows.append(dict(implementation=rel+':'+name,weight_decay=wd,fixture=fixture,steps=220,max_parameter_error=err,max_moment_error=state_err,calibration_error=gamma_err,repeated_calibration_noop=noop,status='passed' if err<=2e-12 and gamma_err<=2e-10 and noop and state_err<2e-12 else 'failed',extraction='module' if rel in FILES[:2] else 'AST class only; injected torch,np,TARGET_WP=.85,WELL_POS_MAX=8,FAST_EMA_BETA=.9,log=noop'))
    from .calibration import gate
    for method in ['evie','evie_amplify']:
        z=np.array([0.,1e-12,1.,1e6,1e20]);G=gate(z,method)
        expected=np.sqrt(np.clip(1/(1+z) if method=='evie' else (1+z)/(.85*(1+2*z)),1e-6,np.inf if method=='evie' else 8))
        rows.append(dict(implementation='standalone gate algebra',fixture=method,exposures=z.tolist(),gate=G.tolist(),status='passed' if np.array_equal(G,expected) else 'failed'))
    return rows
