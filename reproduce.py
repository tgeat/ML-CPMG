"""Portable entry points for the archived ML-CPMG implementation.

Generated files go to outputs/. Archived measurements are never overwritten.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,random,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parent;W=ROOT/'workspace';OUT=ROOT/'outputs'
CODE=W/'results/paper_revision_v16/code'
for p in [W,CODE,W/'results/multilevel_component_separation_v1',W/'results/brugge_feasibility_v1/code']:
    sys.path.insert(0,str(p))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def save(name,value):
    OUT.mkdir(exist_ok=True);p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2),encoding='utf8');print(p)

def integrity():
    rows=json.loads((ROOT/'provenance.json').read_text())['archived_code_and_evidence']
    for r in rows:assert hashlib.sha256((ROOT/r['file']).read_bytes()).hexdigest()==r['sha256'],r['file']
    return {'verified_archived_files':len(rows)}

def smoke():
    import numpy as np
    from scipy.sparse.linalg import cg
    from threadpoolctl import threadpool_limits
    import darcy_core as dc,ml_cpmg_v16 as mg
    from active_graph import ActiveGraph
    reports=[];cfg=dict(mg._BASE_CONTROLS,max_basins=256,stall_ratio=.33,prolong_omega=.45)
    with threadpool_limits(1):
        for graph_case in [False,True]:
            shape=(10,9,8) if graph_case else (16,16,16)
            xyz=np.indices(shape).reshape(3,-1).T;n=len(xyz)
            coords=xyz/np.asarray(shape);k=np.exp(4*np.exp(-np.sum((coords-.5)**2,axis=1)/.06)).reshape(shape)
            e=dc.edges(k)
            if graph_case:
                e=(np.r_[e[0],0],np.r_[e[1],n-1],np.r_[e[2],.7])
                g=ActiveGraph.from_arrays(xyz,*e,shape);d=np.zeros(n);d[0]=1
                A,M,meta=mg.build_active_graph(g,k.ravel(),d,config=cfg,coarse_size=32,max_direct=512)
            else:
                built=mg.build_rectangular(k,e,config=cfg);A,M,meta=built['A'],built['M'],built['metadata']
            exact=np.sin(2*np.pi*coords[:,0])+np.cos(2*np.pi*coords[:,1]);exact-=exact.mean();b=A@exact
            count=[0]
            def callback(_):count[0]+=1
            x,info=cg(A,b,M=M,rtol=1e-11,atol=0,maxiter=500,callback=callback)
            if not graph_case:x-=x.mean()
            delta=x-exact;res=float(np.linalg.norm(b-A@x)/np.linalg.norm(b))
            err=float(np.sqrt(max(0,delta@(A@delta))/(exact@(A@exact))))
            assert info==0 and res<1e-9 and err<1e-7,(info,res,err)
            reports.append({'case':'grounded_graph_with_NNC' if graph_case else 'Cartesian_Neumann','unknowns':n,'iterations':count[0],'relative_residual':res,'relative_energy_error':err,'passed':True})
    save('smoke.json',reports)

def algebra():
    from threadpoolctl import threadpool_limits
    m=load('publication_algebra',CODE/'check_algebra.py')
    with threadpool_limits(1):rows=[m.check(g,s,line) for g in (False,True) for s in (7,11,29) for line in (False,True)]
    save('algebra_checks.json',rows)

def tables():
    import shutil
    m=load('publication_tables',W/'results/paper_revision_v16/build/summarize.py')
    target=OUT/'table_inputs';target.mkdir(parents=True,exist_ok=True)
    for p in (W/'results/paper_revision_v16/data').glob('*.json'):shutil.copy2(p,target/p.name)
    m.DATA=target;m.OUT=OUT;m.TAB=OUT/'tables';m.main()

def figures(names):
    m=load('publication_figures',W/'results/paper_revision_v18_fig10_median/code/figures.py')
    m.OUT=OUT/'figures';m.QA=OUT/'figure_audit';m.QA.mkdir(parents=True,exist_ok=True)
    for name in names:
        required={'spe10':'spe10_sections.npz','brugge':'brugge_geometry.npz'}.get(name)
        if required and not (m.SOURCE/required).exists():
            raise SystemExit(f'{name} requires the external display input {required}. See DATASETS.md and external_inputs.json.')
    rows=json.loads((m.DATA/'summary.json').read_text());S={(r['case'],r['method']):r for r in rows}
    funcs={'stencil':m.stencil,'block':m.block,'fields':m.fields,'segmentation':m.segmentation,'performance':lambda:m.performance(S),'pareto':m.pareto,'reservoir':lambda:m.reservoir(S),'spe10':m.spe10,'brugge':m.brugge}
    for name in names:funcs[name]()

def benchmark(args):
    import psutil,shutil
    from threadpoolctl import threadpool_limits
    import benchmark as b
    selected=json.loads((CODE.parent/'data/selected_config.json').read_text());cap=selected['max_basins'];stall=selected['stall_ratio']
    if args.suite in ('dynamic','brugge','static'):
        records=json.loads((ROOT/'external_inputs.json').read_text())
        prefix={'static':'results/spe10_model2_singlephase/','dynamic':'results/spe10_model2_','brugge':'results/brugge_validation_v2/'}[args.suite]
        missing=[r['workspace_path'] for r in records if r['workspace_path'].startswith(prefix) and not (W/r['workspace_path']).exists()]
        if missing:raise SystemExit('External inputs required (see DATASETS.md):\n'+'\n'.join(missing))
    process=psutil.Process();affinity=None
    if hasattr(process,'cpu_affinity'):
        allowed=process.cpu_affinity();cpu=args.cpu if args.cpu is not None else allowed[0]
        if cpu not in allowed:raise SystemExit(f'CPU {cpu} unavailable; allowed CPUs: {allowed}')
        process.cpu_affinity([cpu]);affinity=cpu
    elif args.cpu is not None:raise SystemExit('CPU pinning is unavailable on this platform.')
    methods=[args.method] if args.method else (b.METHODS+['Algebraic-only'] if args.suite=='brugge' else b.PRIMARY)
    b.DATA=OUT/'reruns'/args.suite;b.DATA.mkdir(parents=True,exist_ok=True);rows=[];matched=0
    with threadpool_limits(1):
        for case in b.cases(args.suite):
            if args.case and case['id']!=args.case:continue
            matched+=1
            for rep in range(args.repeats):
                sequence=methods.copy();random.Random(20261115+rep).shuffle(sequence)
                for method in sequence:
                    row=b.run_one(case,method,rep,cap,stall,save_pressure=rep==0,force_line=args.suite in ('static','dynamic'),reliable=True)
                    assert row['converged'] and row['gate_1e9_pass'],row
                    rows.append(row);b.save_json(b.DATA/'measurements.json',rows)
    if not matched:raise SystemExit('No case matched --case.')
    save(f'reruns/{args.suite}/run_environment.json',{'python':sys.version,'cpu':affinity,'repetitions':args.repeats,'archived_measurements_modified':False})

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ['verify','smoke','algebra','tables']:sub.add_parser(name)
    f=sub.add_parser('figures');f.add_argument('--only',nargs='+',default=['stencil','block','fields','segmentation','performance','pareto','reservoir'],choices=['stencil','block','fields','segmentation','performance','pareto','reservoir','spe10','brugge'])
    b=sub.add_parser('benchmark');b.add_argument('--suite',choices=['synthetic','static','dynamic','brugge'],default='synthetic');b.add_argument('--method',choices=['ML-CPMG','Min-threshold','Single-median','Algebraic-only','SA-0','SA-0.04','SA-0.1','RS-0.25']);b.add_argument('--case');b.add_argument('--repeats',type=int,default=3);b.add_argument('--cpu',type=int)
    args=p.parse_args();OUT.mkdir(exist_ok=True);integrity()
    if args.command=='verify':save('integrity.json',integrity())
    elif args.command=='smoke':smoke()
    elif args.command=='algebra':algebra()
    elif args.command=='tables':tables()
    elif args.command=='figures':figures(args.only)
    elif args.command=='benchmark':
        if args.repeats<1:p.error('--repeats must be positive')
        benchmark(args)
if __name__=='__main__':main()
