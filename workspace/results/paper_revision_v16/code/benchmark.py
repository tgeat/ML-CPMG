"""Serial same-kernel revision experiments; original solver/data are read-only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import sys,time,json,gc,random,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; TASK=ROOT/'results/paper_revision_v16'
for p in (ROOT,ROOT/'.paper_vendor',ROOT/'results/multilevel_component_separation_v1',
          ROOT/'results/brugge_feasibility_v1/code',ROOT/'results/brugge_feasibility_v1/pressure',
          ROOT/'results/brugge_feasibility_v1/dynamic',ROOT/'results/brugge_validation_v2'):
    sys.path.append(str(p))
import numpy as np
import psutil
from scipy import sparse,linalg
from scipy.sparse.linalg import LinearOperator,splu
from threadpoolctl import threadpool_limits
import pyamg
import ml_cpmg_v16 as mg
from bench_common import pcg_lanczos_solve,extract_hierarchy_metrics,save_csv,save_json,dd_refine_neumann
from checkpoint_io import load_checkpoint
import spe10_problem,spe10_dynamic_problem

DATA=TASK/'data'; DATA.mkdir(exist_ok=True)
METHODS=['ML-CPMG','Min-threshold','Single-median','SA-0','SA-0.04','SA-0.1','RS-0.25']
PRIMARY=['ML-CPMG','Min-threshold','Single-median','Algebraic-only','SA-0.04','RS-0.25']

def conf(method,cap=256,stall=.33):
    mode={'ML-CPMG':'ml_cpmg','Min-threshold':'legacy_k_gt_min','Single-median':'single_median','Cliff-only':'cliff_only','Multi-bin':'naive_multibin'}[method]
    return dict(mg._BASE_CONTROLS,phase_mode=mode,max_basins=cap,stall_ratio=stall,prolong_omega=.45)

def external(A,method,quotient):
    smooth=('gauss_seidel',{'sweep':'symmetric','iterations':1})
    common=dict(presmoother=smooth,postsmoother=smooth,max_levels=20,max_coarse=10 if method.startswith('SA') else 512,keep=False)
    np.random.seed(1729)
    if method.startswith('SA'):
        h=pyamg.smoothed_aggregation_solver(sparse.csr_array(A),B=np.ones((A.shape[0],1)),symmetry='hermitian',
          strength=('symmetric',{'theta':float(method.split('-')[1])}),**common)
    else:
        h=pyamg.ruge_stuben_solver(sparse.csr_array(A),strength=('classical',{'theta':.25}),**common)
    C=h.levels[-1].A
    print('  AMG levels:',[v.A.shape[0] for v in h.levels],flush=True)
    if quotient:
        # Restrict to the orthogonal complement of the transported constant.
        z=np.ones(A.shape[0])
        for lv in h.levels[:-1]:
            # SA supplies its own coarse candidate; RS reproduces constants.
            pass
        if hasattr(h.levels[-1],'B'): z=np.asarray(h.levels[-1].B[:,0]).ravel()
        else: z=np.ones(C.shape[0])
        U=linalg.null_space(z[None,:])
        cf=linalg.cho_factor(U.T@C.toarray()@U,lower=True,check_finite=False)
        def coarse(_,b): return U@linalg.cho_solve(cf,U.T@b,check_finite=False)
    else:
        cf=splu(C.tocsc())
        def coarse(_,b): return cf.solve(b)
    h.coarse_solver=pyamg.coarse_grid_solver(coarse)
    raw=h.aspreconditioner(cycle='V')
    if quotient:
        def apply(x):
            y=np.asarray(raw@(x-np.mean(x))).ravel();return y-np.mean(y)
        M=LinearOperator(A.shape,matvec=apply,dtype=float)
    else: M=raw
    return M,dict(level_sizes=[v.A.shape[0] for v in h.levels],level_nnz=[v.A.nnz for v in h.levels],library_version=pyamg.__version__)

def build(case,method,cap=256,stall=.33,force_line=False):
    start=time.perf_counter()
    if method=='Algebraic-only':
        grounded=case['d'] is not None
        A=mg.assemble_cartesian_fast(case['k'].shape,case['edge']) if case['kind']=='cartesian' else mg._assemble_general_edges(case['k'].size,case['edge'])
        if grounded: A.setdiag(A.diagonal()+case['d'])
        levels,meta=mg._construct_hierarchy(A,None,None,conf('ML-CPMG',cap,stall),grounded=grounded,
            cartesian_fine=case['kind']=='cartesian' and not grounded,rank_safe_damping=case['kind']=='graph')
        if case['kind']=='cartesian':
            line=mg._build_anisotropic_line_smoother(case['k'].shape,case['edge'],A.diagonal(),omega=levels[0]['omega'],forced_axis=0 if force_line else None)
            if line is not None: levels[0]['line_smoother']=line
            meta['anisotropic_line_smoother']=line is not None
        apply,_=mg._make_cycle_callable(levels,grounded=grounded,cycle_repeats=1,fine_A=A)
        M=LinearOperator(A.shape,matvec=apply,dtype=float);meta['high_component_count']=0
    elif method in METHODS[:3] or method in ('Cliff-only','Multi-bin'):
        if case['kind']=='cartesian':
            st=mg.build_rectangular(case['k'],case['edge'],d=case['d'],config=conf(method,cap,stall),line_axis=0 if force_line else None)
            A,M,meta=st['A'],st['M'],st['metadata']
        else:
            A,M,meta=mg.build_active_graph(case['graph'],case['k'],case['d'],config=conf(method,cap,stall),coarse_size=512,max_direct=512)
    else:
        if case['kind']=='cartesian': A=mg.assemble_cartesian_fast(case['k'].shape,case['edge'])
        else: A=mg._assemble_general_edges(case['k'].size,case['edge'])
        if case['d'] is not None: A.setdiag(A.diagonal()+case['d'])
        M,meta=external(A,method,case['d'] is None)
    return A,M,meta,time.perf_counter()-start

def cases(which):
    if which=='static':
        k,b,edge=spe10_problem.graph()
        yield dict(id='SPE10-static',kind='cartesian',k=k,edge=edge,d=None,b=b)
    elif which=='dynamic':
        k,phi=spe10_dynamic_problem.load();p=spe10_dynamic_problem.Problem(k,phi)
        folder=ROOT/'results/spe10_model2_dynamic/component_lump_1_1_c512'
        for day in (0.,520.04,2000.):
            S=np.full(p.shape,.2) if day==0 else load_checkpoint(folder/f'snapshot_{day:09.2f}.npz.lz4')['S']
            kt,e,d,b=p.pressure_system(S)
            yield dict(id=f'SPE10-{day:g}d',kind='cartesian',k=kt,edge=e,d=d,b=b,S=S)
    elif which=='brugge':
        from active_graph import ActiveGraph
        from benchmark_pressure import dynamic_system
        folder=ROOT/'results/brugge_validation_v2'
        with np.load(folder/'inputs.npz') as z:
            g=ActiveGraph.from_arrays(z['xyz'],z['tail'],z['head'],z['trans'],tuple(map(int,z['shape'])))
            inp=dict(graph=g,kx=z['kx'],injection=z['injection'],well_base=z['well_base'])
        for day in (0.,1000.,2000.):
            with np.load(folder/f'runs/candidate/snapshots/snapshot_day_{day:08.2f}.npz') as z: S=z['saturation']
            g,k,d,b,_=dynamic_system(inp,S)
            yield dict(id=f'Brugge-{day:g}d',kind='graph',graph=g,k=k,edge=(g.tail,g.head,g.trans),d=d,b=b)
    elif which=='synthetic':
        import darcy_core,multilevel_continuous_media as media
        for fam in media.FAMILIES:
            k,ref=media.geometry(64,fam,contrast=1e6,seed=20261102,return_meta=True)
            yield dict(id='Synthetic-'+fam,kind='cartesian',k=k,edge=darcy_core.edges(k),d=None,b=darcy_core.sources(64),ref=ref)
        from scipy.ndimage import gaussian_filter
        rng=np.random.default_rng(20261115)
        f=gaussian_filter(rng.standard_normal((64,64,64)),sigma=(4,7,2),mode='reflect')
        f=(f-f.min())/(f.max()-f.min());k=np.exp(np.log(1e6)*f)
        yield dict(id='Synthetic-unplanted_random',kind='cartesian',k=k,edge=darcy_core.edges(k),d=None,b=darcy_core.sources(64))
        x=(np.arange(64)+.5)/64;X,Y,Z=np.meshgrid(x,x,x,indexing='ij')
        f=np.exp(-((X-.5)**2+(Y-.5)**2+(Z-.5)**2)/.045);k=np.exp(np.log(1e6)*f)
        yield dict(id='Synthetic-smooth_gaussian',kind='cartesian',k=k,edge=darcy_core.edges(k),d=None,b=darcy_core.sources(64))

def reliable_finish(case,A,M,x):
    if case['d'] is None:
        hi,lo,_,meta=dd_refine_neumann(A,case['b'],case['edge'],M,x,target_gate=1e-9,inner_rtol=1e-7,max_outer=4)
        return hi,lo,meta
    import graph_precision as gp
    from spe10_dynamic_pressure import flux_and_residual
    started=time.perf_counter();b=case['b'];hi=x.copy();lo=np.zeros_like(x);extra=0;outer=0
    for outer in range(5):
        _,_,r=flux_and_residual(hi,lo,case['edge'],case['d'],b)
        rho=float(np.linalg.norm(r)/np.linalg.norm(b));eta=float(np.max(abs(r))/np.max(abs(b)))
        if max(rho,eta)<=1e-9 or outer==4:break
        delta,meta=pcg_lanczos_solve(A,r,M,quotient=False,rtol=1e-7,maxiter=1500,record_trace=False)
        extra+=meta['iterations'];hi,lo=gp.pair_update(hi,lo,delta)
    return hi,lo,dict(dd_outer_corrections=outer,dd_extra_cg_iterations=extra,dd_refine_seconds=time.perf_counter()-started,
        rho_dd=rho,eta_dd=eta,gate_1e9_pass=max(rho,eta)<=1e-9)

def run_one(case,method,rep,cap=256,stall=.33,save_pressure=False,force_line=False,reliable=False):
    gc.collect();start=time.perf_counter()
    print(f'Building {case["id"]} {method} cap={cap} stall={stall}',flush=True)
    A,M,meta,setup=build(case,method,cap,stall,force_line=force_line)
    x,cg=pcg_lanczos_solve(A,case['b'],M,quotient=case['d'] is None,rtol=1e-10,maxiter=2000,deadline=start+600)
    true=float(np.linalg.norm(case['b']-A@x)/np.linalg.norm(case['b']))
    row=dict(case=case['id'],method=method,rep=rep,cap=cap,stall=stall,setup_seconds=setup,
      total_seconds=setup+cg['solve_seconds'],true_residual=true,line_smoother=meta.get('anisotropic_line_smoother',False),
      **cg,**extract_hierarchy_metrics(meta))
    if reliable:
        hi,lo,quality=reliable_finish(case,A,M,x)
        row.update(quality)
        row['converged']=bool(cg['converged'] and quality['gate_1e9_pass'])
        row['pcg_initial_iterations']=cg['iterations'];row['pcg_initial_seconds']=cg['solve_seconds']
        row['iterations']=cg['iterations']+quality['dd_extra_cg_iterations']
        row['solve_seconds']=cg['solve_seconds']+quality['dd_refine_seconds']
        row['total_seconds']=setup+row['solve_seconds']
    if save_pressure:
        if reliable:np.savez_compressed(DATA/(case['id']+'_'+method+'_reliable.npz'),hi=hi,lo=lo,x=hi+lo)
        else:np.savez_compressed(DATA/(case['id']+'_'+method+'_pressure.npz'),x=x)
    if 'ref' in case and rep==0 and method in METHODS[:3]:
        labels,_,_=mg._extract_components_core(case['k'].ravel(),None,case['edge'],case['k'].shape,conf(method,cap,stall))
        ref=case['ref'];dom={};coverage=[];purity=[]
        for pid in range(1,ref['num_peaks']+1):
            v=labels.reshape(case['k'].shape)[ref['core_masks'][pid]];u,c=np.unique(v[v>0],return_counts=True)
            dom[pid]=int(u[np.argmax(c)]) if len(u) else 0
            coverage.append(float(np.count_nonzero(v)/max(1,len(v))))
            purity.append(float(max(c)/max(1,len(v))) if len(c) else 0.)
        row.update(planted_peaks=ref['num_peaks'],distinct_cores=len(set(dom.values())-{0}),
            close_pairs=len(ref['close_pairs']),separated_pairs=sum(dom[a]>0 and dom[b]>0 and dom[a]!=dom[b] for a,b in (z['pair'] for z in ref['close_pairs'])),
            mean_core_coverage=float(np.mean(coverage)),mean_core_purity=float(np.mean(purity)))
    print(f"{case['id']} {method} cap={cap} stall={stall} rep={rep}: {row['iterations']} it, {row['total_seconds']:.3f}s, Cop={row['operator_complexity']:.3f}, gate={row.get('gate_1e9_pass','not assessed')}",flush=True)
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('suite',choices=['calibrate','static','dynamic','brugge','synthetic']);p.add_argument('--repeats',type=int,default=3);p.add_argument('--reliable',action='store_true');a=p.parse_args()
    psutil.Process().cpu_affinity([19])
    rows=[];target=DATA/(a.suite+('_reliable' if a.reliable else '')+'.json')
    if target.exists(): rows=json.loads(target.read_text())
    def exists(case,m,rep,cap,stall): return any('error' not in r and r['case']==case and r['method']==m and r['rep']==rep and r['cap']==cap and r['stall']==stall for r in rows)
    with threadpool_limits(1):
        for case in cases('static' if a.suite=='calibrate' else a.suite):
            if a.suite=='calibrate':
                configs=[('ML-CPMG',cap,.33) for cap in (32,64,128,256,512)]+[('ML-CPMG',cap,.20) for cap in (64,256,512)]+[(m,64,.33) for m in ('Min-threshold','Single-median','SA-0','SA-0.04','SA-0.1','RS-0.25')]
            else:
                selected=json.loads((DATA/'selected_config.json').read_text());cap=selected['max_basins'];stall=selected['stall_ratio']
                configs=[(m,cap,stall) for m in (METHODS+['Algebraic-only'] if a.suite=='brugge' else PRIMARY)]
            for rep in range(a.repeats):
                seq=configs.copy();random.Random(20261115+rep).shuffle(seq)
                for m,cap,stall in seq:
                    if exists(case['id'],m,rep,cap,stall): continue
                    try: row=run_one(case,m,rep,cap,stall,save_pressure=rep==0,force_line=a.suite in ('static','dynamic'),reliable=a.reliable)
                    except Exception as exc:
                        row=dict(case=case['id'],method=m,rep=rep,cap=cap,stall=stall,error=repr(exc));print(row,flush=True)
                    rows.append(row);save_json(target,rows)
                    save_csv(target.with_suffix('.csv'),[{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows])
if __name__=='__main__': main()
