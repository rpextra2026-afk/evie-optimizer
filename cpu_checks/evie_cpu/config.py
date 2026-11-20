import json
from pathlib import Path

def load(path):
    c=json.loads(Path(path).read_text());schema=json.loads((Path(__file__).parents[1]/'configs/core.json').read_text())
    def validate(a,b,p=''):
        if not isinstance(a,type(b)):raise ValueError('Invalid type: '+p)
        if isinstance(b,dict):
            if set(a)!=set(b):raise ValueError('Unknown/missing keys: '+p+str(set(a)^set(b)))
            for k in a:validate(a[k],b[k],p+'.'+k)
        elif isinstance(a,list):
            if not a:raise ValueError('Empty list: '+p)
        elif isinstance(a,(int,float)) and not isinstance(a,bool) and a<0:raise ValueError('Negative value: '+p)
    validate(c,schema)
    if c['device']!='cpu' or c['dtype']!='float64' or c['workers']>4:raise ValueError('CPU float64; maximum 4 workers')
    if c['workers']<1 or c['threads_per_worker']!=1 or c['wall_time_seconds']<=0:raise ValueError('Invalid resource configuration')
    if set(c['groups'])-set(schema['groups']):raise ValueError('Unknown group')
    for name in ['r2','r3','r4']:
        if any(n<2 or not isinstance(n,int) for n in c[name]['horizons']):raise ValueError('Invalid horizon')
        if any(d<1 or not isinstance(d,int) for d in c[name]['dimensions']):raise ValueError('Invalid dimension')
        if any(not 0<=b<1 for b in c[name]['betas']):raise ValueError('Invalid momentum')
    if any(d%2 for d in c['r2']['dimensions']):raise ValueError('R2 requires even dimension')
    if any(v<=0 for v in c['r2']['noise_std_bases']) or any(v<0 for v in c['r2']['log_std_spreads']):raise ValueError('Invalid noise grid')
    if set(c['r2']['normalizers'])-{'constant','total_second_moment'}:raise ValueError('Unknown normalizer')
    if not 0<c['r2']['target_w']<=1 or c['r2']['gamma_min']>c['r2']['gamma_max']:raise ValueError('Invalid gate calibration')
    if c['r5']['dimension']!=sum(c['r5']['tensor_sizes']) or min(c['r5']['tensor_sizes'])<1:raise ValueError('Invalid partition')
    if any(n<=126 for n in c['r5']['horizons']) or any(r<=0 for r in c['r5']['rate_grid']):raise ValueError('Invalid online horizons/rates')
    if c['r5']['evaluation_seed_count']<2 or c['r5']['tuning_seed_count']<1:raise ValueError('Invalid seed counts')
    if not 0<c['rate_search']['u_min']<c['rate_search']['u_join']<c['rate_search']['u_max']<1:raise ValueError('Invalid search interval')
    for field in ['h_c_sigma_eigenvalue_range','p_d_eigenvalue_range']:
        lo,hi=c['r3'][field]
        if not 0<lo<=hi:raise ValueError('Invalid eigenvalue range')
    return c
