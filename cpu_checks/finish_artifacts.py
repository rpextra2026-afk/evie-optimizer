from pathlib import Path
import json,csv,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);report=json.loads((root/'verification_report.json').read_text())
def read(name):
    with (root/'tables'/f'{name}.csv').open() as f:return list(csv.DictReader(f))
def save(fig,name):
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(root/'figures'/f'{name}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
r=read('calibrated_gate_comparison');fig,axs=plt.subplots(2,2,figsize=(9,6),sharex=True)
for a,noise in enumerate([.1,1.]):
 for b,beta in enumerate([0.,.9]):
    ax=axs[a,b]
    for method in ['evie','evie_amplify']:
        v=sorted([x for x in r if int(x['d'])==16 and float(x['spread'])==1.5 and x['normalizer']=='constant' and x['calibration']=='population_reference' and float(x['noise_base'])==noise and float(x['beta'])==beta and x['method']==method],key=lambda x:int(x['N']))
        ax.plot([int(x['N']) for x in v],[100*float(x['relative_gain']) for x in v],'.-',label=method)
    ax.axhline(0,color='gray',lw=.5);ax.set(xscale='log',title=f'Noise SD base {noise}; beta {beta}',xlabel='Horizon N',ylabel='Gain over tuned ungated (%)');ax.legend(fontsize=8)
fig.suptitle('Exact frozen-reference moments; each rate independently optimized',fontsize=11);save(fig,'calibrated_gate_gain')
slopes=[]
for table,label in [('loss_expansion','Loss'),('optimal_rate_expansion','Optimal rate')]:
    rows=read(table);groups={}
    for row in rows:groups.setdefault((row['seed'],row['d'],row['beta']),[]).append(row)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for key,rs in groups.items():
        rs.sort(key=lambda r:int(r['N']));N=np.array([int(r['N']) for r in rs]);lead=np.abs([float(r['leading_error']) for r in rs]);corr=np.abs([float(r['corrected_error']) for r in rs]);axs[0].loglog(N,lead,alpha=.35);axs[1].loglog(N,corr,alpha=.35)
        # Last four points only; optima carry a conservative optimization-resolution floor.
        scale=np.array([float(r['exact']) if table=='loss_expansion' else float(r['eta']) for r in rs]);floor=np.maximum(1e-13,abs(scale)* (1e-11 if table=='loss_expansion' else 2e-7))
        good=np.where(corr>floor)[0];tail=good[-4:]
        slopes.append(dict(table=table,seed=key[0],d=key[1],beta=key[2],tail_horizons=N[tail].tolist(),slope=float(np.polyfit(np.log(N[tail]),np.log(corr[tail]),1)[0]) if len(tail)>=4 else None,status='descriptive' if len(tail)>=4 else 'inconclusive_precision',corrected_tail_better=bool(np.median(corr[-4:])<np.median(lead[-4:]))))
    for ax,title in zip(axs,['Leading prediction','Corrected prediction']):ax.set(xlabel='Horizon N',ylabel=f'Absolute {label.lower()} error',title=title+'; every seeded matrix fixture')
    save(fig,'loss_expansion_errors' if table=='loss_expansion' else 'rate_expansion_errors')
(root/'tables'/'tail_slopes.json').write_text(json.dumps(slopes,indent=2))
claims=[]
for group,table,statement,assumptions in [
('R0','engine_agreement','Independent moment, scalar, spectral and segment engine agreement','Declared finite-horizon linear reference fixtures'),
('R1','implementation_correspondence','Original optimizer correspondence','Supplied-gradient and floor fixtures; eight original classes; two weight decays'),
('R2','calibrated_gate_comparison','Calibrated shape gain after independent tuning','Fixed normalization; population or expected-fast-stat calibration; stable constant rate'),
('R3','loss_expansion','Candidate finite-horizon momentum loss expansion','SPD H,C,Sigma,P; zero initial momentum; normalized EMA'),
('R3','optimal_rate_expansion','Candidate optimal-rate correction','Fixed SPD shape; numerical global search; exact moments'),
('R4','matrix_identity_checks','Noncommuting energy and spectral identities','SPD reference matrices; no momentum for spectral and energy identity'),
('R4','covariance_variation','Covariance-variation identity and norm bound','Declared deterministic covariance sequence; martingale noise'),
('R5','online_evaluation','Sampled adaptive synthetic gain','Independent tuning/evaluation namespaces; Student-t pointwise or Bonferroni intervals'),
('R5','tracking_and_transfer','Pathwise tracking and transfer budget','Stable frozen reference for Lyapunov bound; actual anchor x and nonzero momentum'),
('R6','local_advantage_bound','Quantitative local gate advantage','Homogeneous signal; heterogeneous variances; safe-strength bound; beta=0'),
('R6','supporting_scalar_checks','Scalar proxy and capped benchmark checks','Declared calibration grid; capped benchmark is distinct from actual Evie')]:
    rows=read(table)
    for row in rows:claims.append(dict(group=group,statement=statement,case_id=row['case_id'],table='tables/'+table+'.csv',assumptions=assumptions,proof_required='yes',status=row['status']))
with (root/'claim_evidence.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(claims[0]));w.writeheader();w.writerows(claims)
(root/'figures'/'captions.md').write_text('''Calibrated gates: exact expected cumulative quadratic losses, initialized at ones, d=16, log-standard-deviation spread 1.5, constant normalizer. Each shape independently optimizes its rate under the same stability/search rules. Panels identify noise base and normalized momentum. Gains of every sign are retained.

Expansion errors: every predeclared matrix seed/dimension/momentum fixture is shown. Values are exact moments for loss, numerical optima for rates; leading and corrected errors use the same horizons. Tail fits retain at least four points above a declared numerical floor and are observations, not asymptotic proofs. Floor: max(1e-13, scale*1e-11) for losses, max(1e-13, scale*2e-7) for numerical rates.

Online tracking: complete prespecified adaptive family, sampled evaluation costs. Paired gain is relative to ungated_matched. The table contains pointwise Student-t and family Bonferroni intervals; the original overview plot uses 1.96 standard errors and labels this explicitly. Tracking uses actual post-calibration anchors and retained bound magnitudes. These are a posteriori diagnostics.
''')
print('Updated plots, captions, tail fits and row-level claim mapping:',root)
# Replace the overview with readable labels and the retained Student-t intervals.
rows=read('online_evaluation');groups={}
for r in rows:
    if r['method']!='ungated_matched':groups.setdefault((int(r['N']),r['hessian_profile'],r['method']),r)
fig,axs=plt.subplots(1,2,figsize=(11,5),gridspec_kw={'width_ratios':[1.3,1]});labels=[]
for i,(key,r) in enumerate(sorted(groups.items())):
    mean=float(r['gain_mean']);lo=float(r['pointwise_95_low']);hi=float(r['pointwise_95_high']);axs[0].errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color='C0' if key[2]=='evie' else 'C1');labels.append(f"N={key[0]}, {'identity' if key[1]=='identity' else 'diagonal H'}, {key[2].replace('evie_amplify','Amplify').replace('evie','Evie')}")
axs[0].set_yticks(range(len(labels)),labels,fontsize=8);axs[0].axvline(0,color='gray',lw=.7);axs[0].set(xlabel='Paired cumulative gain over ungated_matched',title='Sampled means; pointwise 95% Student-t intervals')
tracks=read('tracking_and_transfer')
for method,color in [('ungated_matched','C2'),('evie','C0'),('evie_amplify','C1')]:
    rs=[r for r in tracks if r['method']==method and r['B_res'] not in ['', 'None']];axs[1].scatter([float(r['B_track']) for r in rs],[float(r['B_res']) for r in rs],s=6,alpha=.4,label=method,color=color)
axs[1].set(xscale='log',yscale='log',xlabel='Measured tracking energy',ylabel='Residual sufficient bound',title='Post-calibration tracking budgets');axs[1].legend(fontsize=7);save(fig,'online_tracking')
import hashlib
manifest=json.loads((root/'manifest.json').read_text());manifest['postprocessor_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();manifest['postprocessor_command']=f'python3 {Path(__file__).resolve()} {root}';(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
