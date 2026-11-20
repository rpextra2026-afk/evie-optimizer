import itertools as it
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from .segments import cost,segment,compose,tr
from .models import model,diagonal,spd
from .moments import direct
from .exact_scalar import sums,closed,loss
from .exact_matrix import spectral
from .sensitivities import derivatives
from .rate_search import search
from .calibration import gate,calibrate

def check(a,b):return abs(float(a)-float(b))<=1e-11+1e-9*max(abs(float(a)),abs(float(b)))
def r0(cfg,emit):
    for h,x,s,N,t in it.product([.5,1,2],[0,1,2],[0,.1,1],[0,1,2,3,8,32,128],[0,1e-12,1e-8,.01,.5,1,1.5,1.99,2-1e-10]):
        A,Q,W,S=diagonal(h,t/h,s,x,0);v=float(cost(A,Q,W,S,N));d=float(direct(A,Q,W,S,N)[0]);u,w=sums(t,N);raw=float(h*x*u/2+s*w/(2*h));c=float(loss(t,N,h,x,s));err=max(abs(v-d),abs(v-raw),abs(v-c))
        emit('engine_agreement',dict(h=h,x2=x,s=s,N=N,theta=t,segment=v,direct=d,scalar_sum=raw,closed=c,max_error=err,status='passed' if all(check(v,z) for z in [d,raw,c]) else 'failed'))
    for beta,N,t in it.product([0,.5,.9],[0,1,2,8,128],[.01,.5,1.5,2.1,3.]):
        if t>=2*(1+beta)/(1-beta):continue
        A,Q,W,S=diagonal(1,t,.3,1,beta);v=cost(A,Q,W,S,N);d,Sn=direct(A,Q,W,S,N);a=N//3;b=N-a
        ca=compose(segment(A,Q,W,a),segment(A,Q,W,b));cb=compose(segment(A,Q,W,b),segment(A,Q,W,a));err=max(abs(float(v-d)),abs(float(v-tr(ca[2],S)-ca[3])),abs(float(v-tr(cb[2],S)-cb[3])))
        emit('engine_agreement',dict(beta=beta,N=N,theta=t,max_error=err,min_eigenvalue=float(np.linalg.eigvalsh(Sn).min()),status='passed' if check(v,d) and err<1e-8 and np.linalg.eigvalsh(Sn).min()>-1e-11 else 'failed'))
    for d in [2,5,8]:
        rng=np.random.default_rng(np.random.SeedSequence([20261120,0,d]))
        H,Sigma,K=[spd(rng,d,.5,2) for _ in range(3)];K*=.1;x=rng.normal(size=d)
        for C in [np.outer(x,x),spd(rng,d,.5,2)]:
            for N in [0,1,2,32,128]:
                A,Q,W,S,B=model(H,K,Sigma,C,0);v,Sn=direct(A,Q,W,S,N);seg=cost(A,Q,W,S,N);sp=spectral(H,K,Sigma,C,N)
                emit('engine_agreement',dict(fixture='random_independent_eigenbases',d=d,N=N,max_error=float(max(abs(v-seg),abs(v-sp))),status='passed' if check(v,seg) and check(v,sp) else 'failed'))
    import mpmath as mp
    mp.mp.dps=80
    for N,t in it.product([2,32,128],[1e-12,1e-8,.01,.5,1.,1.99]):
        mt=mp.mpf(float(t));fn=lambda z:sum((1-z)**(2*n) for n in range(N))/2+mp.mpf('.15')*z*z*sum((N-1-j)*(1-z)**(2*j) for j in range(N-1))
        got=derivatives(t,N,s=.3);want=[float(fn(mt)),float(mp.diff(fn,mt)),float(mp.diff(fn,mt,2))];err=float(np.max(np.abs(got-want)))
        emit('engine_agreement',dict(N=N,theta=t,fixture='80_decimal_derivatives',max_error=err,status='passed' if all(check(a,b) for a,b in zip(got,want)) else 'failed'))

def r2(cfg,emit):
    c=cfg['r2']
    cases=list(it.product(c['horizons'],c['dimensions'],c['log_std_spreads'],c['noise_std_bases'],c['normalizers'],c['betas']))
    for N,d,r,base,norm,beta in cases:
      for calibration in (['population_reference','expected_fast_stat_reference'] if N==2048 and d==16 and r==1.5 and norm=='constant' else ['population_reference']):
        s=(base*np.exp(np.linspace(-r,r,d)))**2;v=np.ones(d) if norm=='constant' else 1+s
        ref=s if calibration=='population_reference' else (1-.9**126)*.9**126+s*((1-.9**126)-(1-.9)/(1+.9)*(1-.9**252))
        gamma=float(calibrate(ref,[d//2,d//2]));baseline=None
        for method in ['ungated_reference','evie','evie_amplify']:
            G=gate(gamma*s,method);D=G/np.sqrt(v);upper=2*(1+beta)/(1-beta)/max(D)
            def evaluator(rates):
                A,Q,W,S=diagonal(np.ones(d),rates[:,None]*D,s,np.ones(d),beta);return np.sum(cost(A,Q,W,S,N),axis=-1)
            # Independently differentiated scalar sums supply convex certification.
            def deriv(eta):return float(np.sum(derivatives(eta*D,N,s=s)[...,1]*D))
            result=search(evaluator,upper,beta,deriv if beta==0 else None)
            if baseline is None:baseline=result['loss']
            row=dict(N=N,d=d,spread=r,noise_base=base,normalizer=norm,beta=beta,method=method,calibration=calibration,gamma=gamma,v=v.tolist(),s=s.tolist(),gate=G.tolist(),effective_steps=(result['eta']*D).tolist(),stability_margin=1-result['eta']/upper,relative_gain=(baseline-result['loss'])/baseline,a_D=float(sum(1/D)),b_D=float(sum(s*D)),leading_loss=float(np.sqrt(N*sum(1/D)*sum(s*D))/2),status='descriptive',**result)
            emit('calibrated_gate_comparison',row);emit('rate_search_diagnostics',row)

def fixtures(c):
    for seed,d in it.product(c['seeds'],c['dimensions']):
        rng=np.random.default_rng(np.random.SeedSequence([20261120,3,seed,d]));H,C,S=[spd(rng,d,.5,2) for _ in range(3)];P,D=[spd(rng,d,.5,1.5) for _ in range(2)];yield seed,d,H,C,S,P,D

def r3(cfg,emit):
    from .references import anchors
    for row in anchors(cfg['r3']['horizons'][-2:],cfg['r3']['betas']):emit('precision_anchors',row)
    for seed,d,H,C,S,P,D in fixtures(cfg['r3']):
      for beta,N in it.product(cfg['r3']['betas'],cfg['r3']['horizons']):
        A,Q,W,X,B=model(H,P/np.sqrt(N),S,C,beta);exact=float(cost(A,Q,W,X,N));kap=(1-beta)/(1+beta)
        L=np.trace(np.linalg.solve(P,C)+S@P)/4;R=(np.trace(H@C)/kap+kap*np.trace(H@P@S@P)-np.trace(np.linalg.solve(H,S)))/8
        common=dict(seed=seed,d=d,beta=beta,N=N,H=H.tolist(),C=C.tolist(),Sigma=S.tolist(),P=P.tolist(),D=D.tolist(),commutator_norm=float(np.linalg.norm(H@S-S@H)),status='descriptive')
        emit('loss_expansion',dict(**common,exact=exact,leading=float(np.sqrt(N)*L),corrected=float(np.sqrt(N)*L+R),leading_error=float(exact-np.sqrt(N)*L),corrected_error=float(exact-np.sqrt(N)*L-R),scaled_corrected_error=float(np.sqrt(N)*(exact-np.sqrt(N)*L-R)),transient_radius=float(max(abs(np.linalg.eigvals(A)))**N)))
        h,U=np.linalg.eigh(H);half=(U*np.sqrt(h))@U.T;upper=2*(1+beta)/(1-beta)/max(np.linalg.eigvalsh(half@D@half))
        def evaluate(rates):
            # Batch dense augmented matrices over rates.
            models=[model(H,float(eta)*D,S,C,beta)[:4] for eta in rates];aa,qq,ww,ss=map(np.stack,zip(*models));return cost(aa,qq,ww,ss,N)
        result=search(evaluate,upper,beta);a=np.trace(np.linalg.solve(D,C));b=np.trace(S@D);c=np.trace(H@D@S@D);lead=np.sqrt(a/(N*b));corrected=lead-kap*a*c/(2*b*b*N)
        Ropt=(np.trace(H@C)/kap+kap*a*c/b-np.trace(np.linalg.solve(H,S)))/8
        emit('optimal_rate_expansion',dict(**common,**result,eta_leading=float(lead),eta_corrected=float(corrected),leading_error=float(result['eta']-lead),corrected_error=float(result['eta']-corrected),scaled_corrected_error=float(N**1.5*(result['eta']-corrected)),optimized_loss_prediction=float(np.sqrt(N*a*b)/2+Ropt)))
        emit('rate_search_diagnostics',dict(group='R3',seed=seed,d=d,beta=beta,N=N,**result))

def r4(cfg,emit):
    c=cfg['r4']
    for seed,d in it.product(c['seeds'],c['dimensions']):
      rng=np.random.default_rng(np.random.SeedSequence([20261120,4,seed,d]));H,C,S,K=[spd(rng,d,.5,2) for _ in range(4)];h,V=np.linalg.eigh(H);half=(V*np.sqrt(h))@V.T;K*=.8/max(np.linalg.eigvalsh(half@K@half))
      for beta,N in it.product(c['betas'],c['horizons']):
        A,Q,W,X,B=model(H,K,S,C,beta);v,Sn=direct(A,Q,W,X,N);seg=float(cost(A,Q,W,X,N));errors=[abs(v-seg)];energy=spec=None
        if beta==0:
            spec=spectral(H,K,S,C,N);errors.append(abs(v-spec));T=C.copy();summ=0
            eig=np.linalg.eigvalsh(half@K@half);lam=float(eig.min());delta=float(eig.max());init=np.trace(np.linalg.solve(K,C));noise=np.trace(K@S);proxy=(init+N*noise)/4;lower=proxy-(np.exp(-2*N*lam)*init+noise/(lam*(2-lam)))/4;upper=proxy/(1-delta/2)
            emit('proxy_bounds',dict(seed=seed,d=d,N=N,lower=float(lower),exact=float(v),upper=float(upper),lambda_min=lam,delta_max=delta,status='passed' if lower<=v+1e-10 and v<=upper+1e-10 else 'failed'))
            for _ in range(N):summ+=np.trace(H@K@H@T);T=A@T@A.T+Q
            energy=np.trace(np.linalg.solve(K,C))+N*np.trace(K@S)-np.trace(np.linalg.solve(K,Sn))+summ;errors.append(abs(4*v-energy)/4)
        stationary=solve_discrete_lyapunov(A,Q);lyap=np.linalg.norm(stationary-A@stationary@A.T-Q)
        emit('matrix_identity_checks',dict(seed=seed,d=d,beta=beta,N=N,exact=float(v),spectral=spec,energy=energy,max_error=float(max(errors)),lyapunov_residual=float(lyap),H=H.tolist(),C=C.tolist(),Sigma=S.tolist(),K=K.tolist(),min_eigenvalue=float(np.linalg.eigvalsh(Sn).min()),status='passed' if max(errors)<1e-9*max(1,abs(v)) and lyap<1e-10 else 'failed'))
        varying=X.copy();vv=0.;delta=0.;bound=0.
        for j in range(N):
            vv+=np.trace(W@varying);a=1+.25*np.sin(2*np.pi*j/max(2,N-1));varying=A@varying@A.T+a*Q
            if j<N-1:
                U=segment(A,Q,W,N-1-j)[2];weight=B.T@U@B;diff=(a-1)*S;delta+=np.trace(weight@diff);bound+=np.linalg.norm(weight,2)*sum(np.linalg.svd(diff,compute_uv=False))
        err=abs(vv-v-delta)
        emit('covariance_variation',dict(seed=seed,d=d,beta=beta,N=N,fixed_cost=float(v),varying_cost=float(vv),predicted_difference=float(delta),nuclear_bound=float(bound),max_error=float(err),status='passed' if err<1e-9*max(1,abs(v)) and abs(delta)<=bound+1e-10 else 'failed'))

def online(N,h,seeds,rates,method,tracking=False):
    from .functional_optimizer import Functional
    from .tracking import certificate
    rates=np.asarray(rates);count=len(seeds);d=len(h);sig=.25*np.exp(np.linspace(-1,1,d));rngs=[np.random.default_rng(np.random.SeedSequence([20261120,5,int(seed)])) for seed in seeds]
    x=np.ones((count,len(rates),d));f=Functional(x,method,np.broadcast_to(rates,(count,len(rates))),1e-4);J=np.zeros(x.shape[:-1]);div=np.zeros_like(J,dtype=bool);records=[];checks=[]
    if tracking:assert len(rates)==1
    jr=np.zeros(count);ja=np.zeros(count);bt=np.zeros(count);rm=np.zeros(count);trackerr=np.zeros(count);certs=[]
    for n in range(N):
        raw=np.stack([r.normal(size=d) for r in rngs])*sig;noise=raw[:,None,:];before=f.x.copy();mom=f.m.copy()
        value=np.sum(h*f.x*f.x,axis=-1)/2;J+=value;div|=(~np.isfinite(value))|(value>1e50)
        g=h*f.x+noise;norm=np.linalg.norm(g,axis=-1,keepdims=True);g*=np.minimum(1,5/np.maximum(norm,1e-300))
        f.step(g)
        if tracking and n>=126:
            z=np.concatenate((before[:,0],mom[:,0]),axis=-1);zn=np.concatenate((f.x[:,0],f.m[:,0]),axis=-1)
            expected=np.einsum('bij,bj->bi',AA,z)+np.einsum('bij,bj->bi',BB,raw);res=zn-expected
            e=z-ref;newref=np.einsum('bij,bj->bi',AA,ref)+np.einsum('bij,bj->bi',BB,raw);en=zn-newref
            checke=en-np.einsum('bij,bj->bi',AA,e)-res;trackerr=np.maximum(trackerr,np.max(np.abs(checke),axis=-1))
            ja+=np.sum(h*z[:,:d]**2,axis=-1)/2;jr+=np.sum(h*ref[:,:d]**2,axis=-1)/2;bt+=np.sum(h*e[:,:d]**2,axis=-1)/2
            for i,c in enumerate(certs):
                if c['status']=='passed':rm[i]+=res[i]@c['M']@res[i]
            ref=newref
        if n+1==126:
            f.calibrate()
            if tracking:
                # Freeze the post-update, calibrated gain from current moments.
                G=gate(f.gamma[...,None]*f.s,method);k=rates[0]*G[:,0]/f.denom[:,0];models=[model(np.diag(h),np.diag(ki),np.diag(sig**2),np.eye(d),.9) for ki in k];AA=np.stack([m[0] for m in models]);BB=np.stack([m[4] for m in models]);WW=models[0][2];ref=np.concatenate((f.x[:,0],f.m[:,0]),axis=-1)
                certs=[certificate(a,WW) for a in AA]
                conditional=np.array([float(cost(m[0],m[1],m[2],np.outer(z,z),N-126)) for m,z in zip(models,ref)])
        if n+1 in set([1,2,125,126,127,N//2,N]):
            records.append(dict(update=n+1,loss_mean=float(np.mean(value)),loss_std=float(np.std(value)),gamma_mean=float(np.mean(f.gamma)),gate_quantiles=np.quantile(f.G,[0,.25,.5,.75,1]).tolist(),denominator_floor_count=int(np.sum(f.denom==1e-8)),gate_floor_count=int(np.sum(f.G==.001)),gate_ceiling_count=int(np.sum(f.G==np.sqrt(8))),clipped_count=int(np.sum(norm>5)),variance_oracle_mean_absolute_error=float(np.mean(np.abs(f.s-sig**2)))))
    if tracking:
        for i,c in enumerate(certs):
            bound=c['delta']*rm[i]/(1-c['rho'])**2 if c['status']=='passed' else None
            err=abs(ja[i]-jr[i]);tight=2*np.sqrt(jr[i]*bt[i])+bt[i];loose=2*np.sqrt(jr[i]*bound)+bound if bound is not None else None
            ok=trackerr[i]<1e-10 and err<=tight+1e-8 and (bound is None or (bt[i]<=bound+1e-8 and err<=loose+1e-8))
            checks.append(dict(seed=int(seeds[i]),method=method,rate=float(rates[0]),actual_segment_cost=float(ja[i]),reference_segment_cost=float(jr[i]),conditional_reference_expectation=float(conditional[i]),B_track=float(bt[i]),B_res=bound,tracking_comparison_error=float(tight),residual_comparison_error=loose,tracking_recurrence_error=float(trackerr[i]),reference_spectral_radius=c['radius'],rho_M=c.get('rho'),delta_M=c.get('delta'),status='failed' if not ok else c['status'],reason=c.get('reason',''),bound_informative=bool(loose is not None and loose<jr[i])))
    return np.where(div,np.inf,J),checks,records

def r5(cfg,emit):
    from .statistics import interval
    c=cfg['r5'];methods=['ungated_matched','evie','evie_amplify'];tseeds=np.arange(c['tuning_seed_start'],c['tuning_seed_start']+c['tuning_seed_count']);eseeds=np.arange(c['evaluation_seed_start'],c['evaluation_seed_start']+c['evaluation_seed_count'])
    for N,profile in it.product(c['horizons'],c['hessian_profiles']):
        h=np.ones(8) if profile=='identity' else np.geomspace(1,8,8);rates=list(c['rate_grid']);tuned={}
        for extension in range(2):
            boundarylow=boundaryhigh=False
            for method in methods:
                costs,_,_=online(N,h,tseeds,rates,method);means=np.mean(costs,axis=0);best=int(np.argmin(means));tuned[method]=(rates[best],costs)
                boundarylow|=best==0;boundaryhigh|=best==len(rates)-1
                for j,rate in enumerate(rates):emit('online_tuning',dict(N=N,hessian_profile=profile,method=method,rate=rate,extension=extension,mean_cost=float(means[j]),divergent_count=int(sum(~np.isfinite(costs[:,j]))),seed_costs=costs[:,j].tolist(),status='descriptive'))
            if extension==0 and (boundarylow or boundaryhigh):rates=sorted(set(rates+([1e-6,3e-6] if boundarylow else [])+([3.,10.] if boundaryhigh else [])))
            else:break
        outcomes={};alltracking={}
        for method in methods:
            rate=tuned[method][0];costs,tracking,records=online(N,h,eseeds,[rate],method,True);outcomes[method]=costs[:,0];alltracking[method]=tracking
            for row in tracking:emit('tracking_and_transfer',dict(N=N,hessian_profile=profile,**row))
            for row in records:emit('online_checkpoints',dict(N=N,hessian_profile=profile,method=method,**row))
        for method in methods[1:]:
            for baseline,gated in zip(alltracking['ungated_matched'],alltracking[method]):
                gap=baseline['reference_segment_cost']-gated['reference_segment_cost'];err=baseline['tracking_comparison_error']+gated['tracking_comparison_error'];actual=baseline['actual_segment_cost']-gated['actual_segment_cost']
                emit('reference_advantage_transfer',dict(N=N,hessian_profile=profile,seed=baseline['seed'],method=method,reference_gap=gap,comparison_budget=err,actual_segment_gain=actual,sufficient_advantage_condition=gap>err,status='passed' if actual>=gap-err-1e-8 else 'failed'))
        for method in methods:
            values=outcomes[method];gains=outcomes['ungated_matched']-values
            if np.all(np.isfinite(gains)):mean,se,lo,hi=interval(gains);_,_,slo,shi=interval(gains,8)
            else:mean=se=lo=hi=slo=shi=None
            for i,seed in enumerate(eseeds):emit('online_evaluation',dict(N=N,hessian_profile=profile,method=method,seed=int(seed),rate=tuned[method][0],cost=float(values[i]),paired_gain=float(gains[i]),gain_mean=mean,gain_se=se,pointwise_95_low=lo,pointwise_95_high=hi,bonferroni_95_low=slo,bonferroni_95_high=shi,rate_grid_boundary=tuned[method][0] in [rates[0],rates[-1]],status='descriptive' if np.isfinite(values[i]) else 'condition_not_met'))

def r6(cfg,emit):
    from scipy.optimize import brentq
    for N in cfg['r6']['local_horizons']:
        h=.7;x2=1.3;s=np.array([.03,.4,2,10]);t=brentq(lambda t:np.sum(derivatives(t,N,h,x2,s)[...,1]),0,1,xtol=1e-15);l=derivatives(t,N,h,x2,s);p=t*l[:,1];A=sum(s*p)/2
        # independent noise derivative from finite sums, avoiding derivative at zero.
        k=np.arange(N-1);kk=k[1:];w=N-1-k;a=1-t;wp=2*t*sum(w*a**(2*k))-2*t*t*sum(w[1:]*kk*a**(2*kk-1));prediction=t*wp/(4*h)*sum((s-s.mean())**2)
        cap=1/max(s);q=np.maximum(abs(1-t/np.sqrt(1+cap*s)),abs(1-t));j=np.arange(1,N)
        M=[]
        for qi,si in zip(q,s):
            u1=2*sum(j*qi**(2*j-1));u2=2*sum(j*(2*j-1)*qi**(2*j-2));w1=sum(2*t*w*qi**(2*k))+sum(2*kk*t*t*w[1:]*qi**(2*kk-1));w2=sum(2*w*qi**(2*k))+sum(w[1:]*(8*kk*t*qi**(2*kk-1)+2*kk*(2*kk-1)*t*t*qi**(2*kk-2)));M.append((h*x2*u1/2+si*w1/(2*h),h*x2*u2/2+si*w2/(2*h)))
        M=np.array(M);base=sum(l[:,0])
        for method,c2 in [('evie',.75),('evie_amplify',1.75)]:
            B=sum(s*s*(t*t*M[:,1]/4+c2*t*M[:,0]));gamma=min(cap,A/B);G=gate(gamma*s,method)*(np.sqrt(.85) if method=='evie_amplify' else 1);fixed=sum(loss(t*G,N,h,x2,s));upper=2/max(G)
            result=search(lambda rates:np.sum(loss(rates[:,None]*G,N,h,x2,s),axis=-1),upper,0,lambda rate:sum(derivatives(rate*G,N,h,x2,s)[:,1]*G));guaranteed=A*gamma/2
            ok=check(A,prediction) and abs(sum(p))<1e-9 and base-fixed>=guaranteed and base-result['loss']>=guaranteed
            emit('local_advantage_bound',dict(N=N,method=method,theta=t,A=float(A),A_prediction=float(prediction),stationarity=float(sum(p)),B=float(B),gamma_safe=float(gamma),guaranteed_gain=float(guaranteed),fixed_gain=float(base-fixed),retuned_gain=float(base-result['loss']),status='passed' if ok else 'failed'))
    for p,c in it.product(cfg['r6']['proxy_exponents'],cfg['r6']['proxy_calibrations']):
        r=np.geomspace(1e-8,1e8,401);m=(1+c*r)**(-p);oracle=1/np.sqrt(r);bounded=np.minimum(1,oracle);ratio=(1/m+r*m)/(2*np.sqrt(r));cosh=np.cosh(np.log(m/oracle));bounded_ratio=(1/m+r*m)/(1/bounded+r*bounded)
        emit('supporting_scalar_checks',dict(fixture='proxy',p=p,c=c,r=r.tolist(),unconstrained_ratios=ratio.tolist(),shrink_only_ratios=bounded_ratio.tolist(),max_cosh_error=float(max(abs(ratio-cosh))),status='passed' if np.allclose(ratio,cosh,rtol=1e-12) else 'failed'))
    for rho,mu in it.product([.004,.04,.4,4,40,400],[np.sqrt(2),1.4779672430090043]):
        N=4000;s=rho/N;u=2/(1+np.sqrt(1+4*rho));t=min(1,mu/np.sqrt(1+4*rho));opt=brentq(lambda z:derivatives(z,N,s=s)[1],0,1,xtol=1e-15);exact=float(loss(t,N,s=s)/loss(opt,N,s=s));C=max((mu*mu+3)/(2*(mu+1)),mu/4+1/mu);eps=4*np.exp(-N*u/2)/(2-u/4)+rho*u/N;surrogate=u*(1+rho*t*t)/(t*(2-t));ok=surrogate<=C+1e-12 and (eps>=1 or exact<=C/(1-eps)+1e-10)
        emit('supporting_scalar_checks',dict(fixture='capped_scalar_benchmark',rho=rho,mu=mu,exact_ratio=exact,surrogate_ratio=surrogate,C=C,epsilon=eps,finite_N_bound=C/(1-eps) if eps<1 else None,mu_polynomial=mu**3-mu**2+2*mu-4,status='failed' if not ok else ('passed' if eps<1 else 'condition_not_met')))
