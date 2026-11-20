#!/usr/bin/env python3
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,platform,traceback,argparse,collections
from pathlib import Path
from datetime import datetime,timezone
from evie_cpu.config import load
from evie_cpu.artifacts import write_json,write_csv,plots
from evie_cpu.actual_code import FILES,correspondence
from evie_cpu import groups
BASE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=['smoke','core','extended'],default='core');p.add_argument('--source-root',type=Path,default=BASE.parent);p.add_argument('--output-root',type=Path,default=BASE/'results');p.add_argument('--workers',type=int,default=4);p.add_argument('--resume');p.add_argument('--verify');args=p.parse_args()
    if args.verify:
        out=args.output_root/args.verify;r=json.loads((out/'verification_report.json').read_text());print(json.dumps(r,indent=2));return 1 if r['failed'] or r['incomplete'] or r.get('source_changes') else 0
    cfg=load(BASE/'configs'/f'{args.profile}.json');root=args.source_root.resolve();cfg['workers']=min(4,args.workers)
    if args.profile=='extended':raise ValueError('Extended additions are not yet implemented; use core. No core output is relabeled extended.')
    runid=args.resume or datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+args.profile;out=args.output_root/runid
    for d in ['tables','figures','traces','case_checkpoints']: (out/d).mkdir(parents=True,exist_ok=True)
    protected={str(f):digest(f) for f in root.rglob('*') if f.is_file() and (f.suffix in ['.pdf','.tex'] or str(f.relative_to(root)) in FILES) and 'theory_cpu_validation' not in f.parts}
    source={str(root/r):digest(root/r) for r in FILES};engine={str(f.relative_to(BASE)):digest(f) for f in BASE.rglob('*.py')};signature=hashlib.sha256(json.dumps([cfg,source,engine],sort_keys=True).encode()).hexdigest()
    if args.resume:
        old=json.loads((out/'manifest.json').read_text())
        if old['signature']!=signature:raise ValueError('Resume signature mismatch; preserve old run and start a fresh run')
    import numpy,scipy,matplotlib,mpmath,torch
    torch.set_num_threads(1);started=time.monotonic();manifest=dict(run_id=runid,signature=signature,source_root=str(root),source_choice='bundled optimizer sources',source_hashes=source,protected_hashes=protected,engine_hashes=engine,versions={m.__name__:m.__version__ for m in [numpy,scipy,matplotlib,mpmath,torch]},python=sys.version,platform=platform.platform(),cpu_count=os.cpu_count(),actual_workers=1,threads_per_worker=1,seed_namespaces='SeedSequence [20261120, group, seed, dimension]; online [20261120,5,seed]',started=datetime.now(timezone.utc).isoformat(),cache='atomic group checkpoints; matching complete signature required')
    write_json(out/'manifest.json',manifest);write_json(out/'configuration.json',cfg);tables=collections.defaultdict(list);statuses={};failures={}
    for group in cfg['groups']:
        checkpoint=out/'traces'/f'{group}.json'
        if args.resume and checkpoint.exists():
            saved=json.loads(checkpoint.read_text());statuses[group]=saved['status']
            for name,rows in saved['tables'].items():tables[name].extend(rows)
            continue
        local=collections.defaultdict(list)
        def emit(table,row):
            row=dict(group=group,case_id=f'{group}:{table}:{len(local[table]):06d}',**row) if 'group' not in row else dict(row,case_id=f'{group}:{table}:{len(local[table]):06d}')
            local[table].append(row)
            write_json(out/'case_checkpoints'/(row['case_id'].replace(':','_')+'.json'),dict(signature=signature,row=row))
        t=time.monotonic();print(f'{group} started',flush=True)
        with (out/'run.log').open('a') as log:log.write(f'{group} started\n')
        try:
            if time.monotonic()-started>cfg['wall_time_seconds']:raise TimeoutError('Wall time budget reached')
            if group=='R1':
                for r in correspondence(root):emit('implementation_correspondence',r)
            else:getattr(groups,group.lower())(cfg,emit)
            statuses[group]='attempted'
        except Exception:
            failures[group]=traceback.format_exc();statuses[group]='incomplete';print(failures[group],flush=True)
        for name,rows in local.items():tables[name].extend(rows);write_csv(out/'tables'/f'{name}.csv',tables[name])
        write_json(checkpoint,dict(status=statuses[group],tables=local,elapsed_seconds=time.monotonic()-t));print(f'{group}: {sum(map(len,local.values()))} rows in {time.monotonic()-t:.2f}s',flush=True)
    source_changes=[f for f,h in protected.items() if not Path(f).exists() or digest(Path(f))!=h];counts={g:collections.Counter(r['status'] for rows in tables.values() for r in rows if r.get('group')==g and 'status' in r) for g in cfg['groups']}
    failed=sum(v.get('failed',0) for v in counts.values());incomplete=[g for g,v in statuses.items() if v=='incomplete']
    # Explicitly retain unfinished acceptance obligations rather than asserting full specification compliance.
    obligations=['Optional extended profile additions (beta=.99, ill-conditioning, Monte Carlo, quartic objective) are not implemented; core scientific grid is complete.']
    report=dict(run_id=runid,profile=args.profile,elapsed_seconds=time.monotonic()-started,groups=statuses,status_counts=counts,failed=failed,incomplete=incomplete,outstanding_specification_obligations=obligations,full_specification_complete=False,core_scientific_grid_complete=not incomplete,source_changes=source_changes,exceptions=failures,expected_cases=dict(R1_implementations=8,R2_primary=len(cfg['r2']['horizons'])*len(cfg['r2']['dimensions'])*len(cfg['r2']['log_std_spreads'])*len(cfg['r2']['noise_std_bases'])*len(cfg['r2']['normalizers'])*len(cfg['r2']['betas']),R3=len(cfg['r3']['seeds'])*len(cfg['r3']['dimensions'])*len(cfg['r3']['horizons'])*len(cfg['r3']['betas']),R4=len(cfg['r4']['seeds'])*len(cfg['r4']['dimensions'])*len(cfg['r4']['horizons'])*len(cfg['r4']['betas']),R5=len(cfg['r5']['horizons'])*len(cfg['r5']['hessian_profiles'])),attempted_table_rows={k:len(v) for k,v in tables.items()},reproduction=f'python3 {BASE}/run.py --profile {args.profile} --source-root {root}')
    claims=[dict(group=g,statement='Numerical checks and descriptive values',evidence=f'tables/{name}.csv',assumptions='Frozen quadratic reference, or supplied synthetic online fixture',proof_required=True,status='incomplete' if g in incomplete else 'descriptive') for name,rows in tables.items() for g in sorted(set(r.get('group','') for r in rows))];write_csv(out/'claim_evidence.csv',claims)
    with (out/'cases.jsonl').open('w') as f:
        for rows in tables.values():
            for row in rows:f.write(json.dumps(row,default=str)+'\n')
    try:plots(out,tables)
    except Exception:report['plot_exception']=traceback.format_exc()
    report['maximum_discrepancies']={name:max([abs(float(r.get('max_error',r.get('max_parameter_error',0)))) for r in rows] or [0]) for name,rows in tables.items()}
    write_json(out/'verification_report.json',report);manifest.update(ended=datetime.now(timezone.utc).isoformat(),elapsed_seconds=report['elapsed_seconds']);write_json(out/'manifest.json',manifest)
    (out/'report.md').write_text('# CPU mathematical validation\n\n'+f'Profile: {args.profile}. Runtime: {report["elapsed_seconds"]:.1f} seconds. Failed numerical rows: {failed}. Source changes: {len(source_changes)}.\n\n'+ '\n'.join(f'- {g}: {dict(c)}' for g,c in counts.items())+'\n\nAll retained gains are descriptive; numerical evidence does not establish a theorem or novelty. Momentum stationary kappa coefficients are attributed to Gitman et al. (2019) in the supplied research record. The initialization and optimized finite-horizon statements still require proof review.\n\nThe configured scientific grids were attempted as recorded above. Full specification acceptance remains incomplete:\n\n'+'\n'.join('- '+x for x in obligations)+'\n\nExceptions:\n```\n'+json.dumps(failures,indent=2)+'\n```\n\nReproduce:\n```sh\n'+report['reproduction']+'\n```\n')
    if args.profile=='core':
        import subprocess
        subprocess.run([sys.executable,str(BASE/'finish_artifacts.py'),str(out)],check=True)
    with (out/'run.log').open('a') as log:log.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True);return 1 if failed or incomplete or source_changes else 0
if __name__=='__main__':sys.exit(main())
