import csv,json,os
from pathlib import Path
import numpy as np

def write_json(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=lambda o:o.item() if isinstance(o,np.generic) else str(o)));os.replace(tmp,path)
def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row));tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
    os.replace(tmp,path)
def plots(out,tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for table,name,y,x in [('calibrated_gate_comparison','calibrated_gate_gain','relative_gain','N'),('loss_expansion','loss_expansion_errors','corrected_error','N'),('optimal_rate_expansion','rate_expansion_errors','corrected_error','N')]:
        rows=tables.get(table,[])
        if not rows:continue
        fig,ax=plt.subplots(figsize=(7,4));groups={}
        for row in rows:
            if table=='calibrated_gate_comparison' and not (row['d']==16 and row['spread']==1.5 and row['normalizer']=='constant' and row['calibration']=='population_reference'):continue
            key=(row.get('method'),row.get('noise_base'),row.get('beta'),row.get('seed'),row.get('d'));groups.setdefault(str(key),[]).append(row)
        if not groups:
            for row in rows:groups.setdefault(str((row.get('method'),row.get('spread'),row.get('beta'))),[]).append(row)
        for key,rs in groups.items():
            rs.sort(key=lambda r:r[x]);ax.plot([r[x] for r in rs],[abs(r[y]) if 'error' in y else r[y] for r in rs],'.-',alpha=.6,label=key)
        ax.set_xscale('log');ax.set_xlabel('Horizon N (initial state included)');ax.set_ylabel(y.replace('_',' '));ax.set_title('Exact moments and independent numerical rate optima')
        if 'error' in y:ax.set_yscale('log')
        if len(groups)<14:ax.legend(fontsize=6)
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/'figures'/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    rows=tables.get('online_evaluation',[]);track=tables.get('tracking_and_transfer',[])
    if rows:
        fig,axs=plt.subplots(1,2,figsize=(10,4));groups={}
        for r in rows:groups.setdefault((r['N'],r['hessian_profile'],r['method']),[]).append(r)
        for i,(key,rs) in enumerate(groups.items()):axs[0].errorbar(i,rs[0]['gain_mean'],yerr=1.96*rs[0]['gain_se'] if rs[0]['gain_se'] is not None else 0,fmt='o');axs[0].annotate(str(key),(i,rs[0]['gain_mean']),rotation=90,fontsize=5)
        axs[0].set_ylabel('Paired cumulative gain over ungated matched');axs[0].set_title('Sampled estimates; error bars = 1.96 SE')
        good=[r for r in track if r['B_res'] is not None];axs[1].scatter([r['B_track'] for r in good],[r['B_res'] for r in good],s=7);axs[1].set(xscale='log',yscale='log',xlabel='Measured tracking energy',ylabel='Residual sufficient bound',title='Post-calibration tracking diagnostics');fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/'figures'/f'online_tracking.{ext}',dpi=180)
        plt.close(fig)
