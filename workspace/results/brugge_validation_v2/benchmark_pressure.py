"""Serial, same-system pressure comparisons on the audited Brugge active graph.

Run under the root task's resource guard. This module never starts a trajectory
and never edits v1 code/data. Each saved saturation defines one *common* physical
matrix. All four methods use zero initial pressure, identical CG tolerances and
the same independently assembled sparse-direct reference. Three repetitions per
method are descriptive technical repetitions, not independent geological data.

The phase-only control changes only the initial component tags. Physical graph,
rank-safe damping, prolongation, positive-entry lumping and coarse limit match
the candidate. It is expected to be identical when the high phase is connected.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
             'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'results/brugge_feasibility_v1'
for _path in (HERE, OLD/'code', OLD/'pressure', OLD/'dynamic', ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import psutil
from scipy.sparse.linalg import cg, splu
from threadpoolctl import threadpool_limits, threadpool_info
from active_graph import ActiveGraph
from pressure_graph import assemble, build
from solve_pressure_core import _independent_matrix, _describe_solution, _comparison
from spe10_dynamic_transport import phase_properties

METHODS = ('component_rank_safe', 'phase_rank_safe', 'pyamg_rs', 'pyamg_sa')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''):h.update(block)
    return h.hexdigest()


def array_digest(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


class PhaseOnlyGraph(ActiveGraph):
    """Same graph type; replace component identities by a single high-phase tag."""
    def high_components(self, permeability):
        k = np.asarray(permeability, dtype=float)
        if k.shape != (len(self.xyz),) or np.any(k <= 0) or not np.isfinite(k).all():
            raise ValueError('invalid phase-only coefficient')
        tags = (k > k.min()).astype(np.int64)
        return tags, int(np.any(tags))


def check_resources(launch_gib=6, rss_gib=8):
    available = psutil.virtual_memory().available
    rss = psutil.Process().memory_info().rss
    if available < launch_gib*1024**3:
        raise MemoryError('less than launch free memory threshold')
    if rss > rss_gib*1024**3:
        raise MemoryError('benchmark process RSS exceeds threshold')
    return {'available_bytes':available,'process_rss_bytes':rss}


def dynamic_system(inputs, saturation):
    graph = inputs['graph']
    s = np.asarray(saturation, dtype=np.float64)
    if s.shape != (len(graph.xyz),) or not np.isfinite(s).all() or np.any((s < .2)|(s > .8)):
        raise ValueError('invalid saved saturation')
    start = time.perf_counter()
    mobility, _ = phase_properties(s)
    w = graph.trans * (2/(1/mobility[graph.tail]+1/mobility[graph.head]))
    d = np.asarray(inputs['well_base'])*mobility
    k = np.asarray(inputs['kx'])*mobility
    b = np.asarray(inputs['injection'], dtype=np.float64)
    dynamic = ActiveGraph.from_arrays(graph.xyz,graph.tail,graph.head,w,graph.shape)
    if any(v.shape!=(len(s),) or not np.isfinite(v).all() for v in (d,k,b)):
        raise ValueError('invalid physical cell vectors')
    if np.any(d < 0) or np.any(k <= 0) or np.any(b < 0) or not np.any(b):
        raise ValueError('invalid wells, permeability or zero injection')
    return dynamic,k,d,b,time.perf_counter()-start


def preconditioned_solve(method, graph, k, d, b, *, rtol, maxiter, seconds, coarse_size):
    """Timed assembly + hierarchy + CG; quality checks are timed separately."""
    started = time.perf_counter()
    if method in ('component_rank_safe','phase_rank_safe'):
        g = graph if method=='component_rank_safe' else PhaseOnlyGraph(
            graph.xyz,graph.tail,graph.head,graph.trans,graph.shape)
        A,M,description = build(g,k,d,coarse_size=coarse_size,max_direct=512,damping='rank_safe')
    else:
        import pyamg
        A,_,assembly_info = assemble(graph,d)
        smooth = ('gauss_seidel', {'sweep':'symmetric'})
        if method=='pyamg_rs':
            hierarchy = pyamg.ruge_stuben_solver(
                A,strength=('classical',{'theta':.25}),max_coarse=512,
                presmoother=smooth,postsmoother=smooth,coarse_solver='splu',keep=False)
            config={'family':'Ruge-Stuben','strength':'classical theta=.25','max_coarse':512,
                    'smoother':'symmetric Gauss-Seidel pre/post','coarse_solver':'splu'}
        elif method=='pyamg_sa':
            hierarchy = pyamg.smoothed_aggregation_solver(
                A,B=np.ones((A.shape[0],1)),symmetry='symmetric',
                strength=('symmetric',{'theta':0}),max_coarse=10,
                presmoother=smooth,postsmoother=smooth,coarse_solver='pinv',keep=False)
            config={'family':'smoothed aggregation','strength':'symmetric theta=0','max_coarse':10,
                    'smoother':'symmetric Gauss-Seidel pre/post','coarse_solver':'pinv',
                    'candidate_vectors':'constant'}
        else:raise ValueError('unknown method')
        M=hierarchy.aspreconditioner(cycle='V')
        description={'pyamg_version':pyamg.__version__,'configuration':config,
                     'levels':[{'size':int(l.A.shape[0]),'nnz':int(l.A.nnz)} for l in hierarchy.levels],
                     'assembly':assembly_info}
    setup_seconds=time.perf_counter()-started
    if setup_seconds > seconds:raise TimeoutError('setup exceeds common method budget')
    iterations=[0]
    def observe(_):
        iterations[0]+=1
        if time.perf_counter()-started > seconds:raise TimeoutError('CG exceeds common method budget')
    solve_start=time.perf_counter()
    x,status=cg(A,b,x0=np.zeros_like(b),M=M,rtol=rtol,atol=0,maxiter=maxiter,callback=observe)
    solve_seconds=time.perf_counter()-solve_start
    if time.perf_counter()-started > seconds:raise TimeoutError('solve exceeds common method budget')
    if not np.isfinite(x).all():raise FloatingPointError('nonfinite returned pressure')
    return A,x,{'cg_status':int(status),'iterations':iterations[0],
                'setup_and_assembly_seconds':setup_seconds,'cg_seconds':solve_seconds,
                'total_core_seconds':setup_seconds+solve_seconds,'hierarchy':description}


def benchmark_snapshot(inputs, saturation, output_dir, *, label, time_days,
                       repeats=3, methods=METHODS, rtol=1e-12, maxiter=5000,
                       seconds=900, coarse_size=512, save_vectors=True, seed=20261107,
                       source=None):
    output_dir=Path(output_dir)
    if output_dir.exists():raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    check_resources()
    with threadpool_limits(1):
        graph,k,d,b,coefficient_seconds=dynamic_system(inputs,saturation)
        ref_start=time.perf_counter()
        Aref=_independent_matrix(graph,d)
        reference=splu(Aref.tocsc()).solve(b)
        reference_seconds=time.perf_counter()-ref_start
        qref,qwref,rref,reference_metrics=_describe_solution(graph,Aref,d,b,reference)
        reference_ok=max(reference_metrics['rho_l2'],reference_metrics['eta_inf'],reference_metrics['balance_relative'])<=1e-10
        if not reference_ok:raise RuntimeError('direct numerical reference fails residual precondition')
        if save_vectors:
            np.savez_compressed(output_dir/'reference.npz',drawdown=reference,face_flux=qref,
                                production=qwref,rhs=b,well_diagonal=d,edge_weight=graph.trans,
                                saturation=saturation,time_days=float(time_days))
        _,high_count=graph.high_components(k)
        report={'schema':'brugge-same-system-pressure-comparison-v2','label':str(label),
                'time_days':float(time_days),'created_utc':datetime.now(timezone.utc).isoformat(),
                'source':source,'source_saturation_sha256':array_digest(saturation),
                'system_sha256':{n:array_digest(a) for n,a in [('rhs',b),('well_diagonal',d),('edge_weight',graph.trans)]},
                'active_cells':len(b),'edges':len(graph.trans),'repeats':int(repeats),
                'methods':list(methods),'high_component_count':int(high_count),
                'component_phase_identical_expected':bool(high_count<=1),
                'common_rtol':rtol,'common_maxiter':maxiter,'common_seconds_per_trial':seconds,
                'physical_target_residual':1e-9,'relative_error_target':1e-6,
                'common_coefficient_construction_seconds':coefficient_seconds,
                'reference_method':'SciPy SuperLU on independent incidence assembly',
                'reference_seconds':reference_seconds,'reference_metrics':reference_metrics,
                'reference_certified':False,
                'claim_limit':'Numerical same-binary64-system comparison, not an interval certificate or transport convergence proof.',
                'timing_scope':'Per-trial matrix assembly + hierarchy construction + zero-initialized CG. Quality and file IO excluded; common mobility construction reported separately.',
                'tuning':'All presets fixed before Brugge comparison; zero per-method Brugge tuning trials.',
                'threadpools':threadpool_info(),'input_audit':inputs.get('audit',{}),'trials':[]}
        rng=np.random.default_rng(seed)
        base=list(methods);rng.shuffle(base)
        for repeat in range(repeats):
            order=base[repeat%len(base):]+base[:repeat%len(base)]
            for position,method in enumerate(order):
                gc.collect()
                resources=check_resources()
                trial={'method':method,'repeat':repeat,'order_position':position,'resources_before':resources}
                try:
                    A,x,metrics=preconditioned_solve(method,graph,k,d,b,rtol=rtol,maxiter=maxiter,
                                                  seconds=seconds,coarse_size=coarse_size)
                    diff=(A-Aref).tocsr()
                    error=float(np.max(abs(diff.data),initial=0))/max(float(np.max(abs(Aref.data))),np.finfo(float).tiny)
                    if error>1e-12:raise ValueError('method physical matrix differs from common reference')
                    quality_start=time.perf_counter()
                    q,qw,residual,quality=_describe_solution(graph,Aref,d,b,x)
                    comparison=_comparison(graph,d,reference,x,qref,qwref,q,qw)
                    # Also expose face and producer terms separately so that
                    # a small contribution cannot hide inside their combined norm.
                    mask=d>0
                    face_num=math.fsum(map(float,(q-qref)**2/graph.trans))
                    face_den=math.fsum(map(float,qref**2/graph.trans))
                    well_num=math.fsum(map(float,(qw[mask]-qwref[mask])**2/d[mask]))
                    well_den=math.fsum(map(float,qwref[mask]**2/d[mask]))
                    face_relative=math.sqrt(face_num/face_den) if face_den>0 else (0. if face_num==0 else None)
                    well_relative=math.sqrt(well_num/well_den) if well_den>0 else (0. if well_num==0 else None)
                    comparison.update(relative_face_weighted_flux_difference=face_relative,
                                      relative_well_weighted_flux_difference=well_relative,
                                      max_absolute_pressure_difference_bar=float(np.max(abs(x-reference))),
                                      relative_drawdown_l2_difference=float(np.linalg.norm(x-reference)/np.linalg.norm(reference)))
                    accepted=(metrics['cg_status']==0 and max(quality['rho_l2'],quality['eta_inf'],quality['balance_relative'])<=1e-9
                              and comparison['energy_difference_pass'] and comparison['weighted_flux_difference_pass']
                              and face_relative is not None and well_relative is not None
                              and max(face_relative,well_relative)<=1e-6)
                    trial.update(metrics=metrics,quality=quality,comparison_to_numerical_reference=comparison,
                                 accepted=bool(accepted),assembly_relative_error=error,
                                 quality_seconds=time.perf_counter()-quality_start,
                                 total_with_common_coefficients_seconds=metrics['total_core_seconds']+coefficient_seconds)
                    if repeat==0 and save_vectors:
                        vec=output_dir/f'{method}.npz'
                        np.savez_compressed(vec,drawdown=x,face_flux=q,production=qw,residual=residual,
                                            saturation=saturation,time_days=float(time_days))
                        trial['vector_file']=vec.name;trial['vector_sha256']=digest(vec)
                except Exception as exc:
                    trial.update(accepted=False,error_type=type(exc).__name__,error=str(exc))
                report['trials'].append(trial)
                save_json(output_dir/f'trial_{repeat:02d}_{method}.json',trial)
        summaries={}
        for method in methods:
            trials=[t for t in report['trials'] if t['method']==method]
            accepted=[t for t in trials if t['accepted']]
            summaries[method]={'accepted':len(accepted),'attempted':len(trials),'all_accepted':len(accepted)==repeats,
                'median_core_seconds':float(np.median([t['metrics']['total_core_seconds'] for t in accepted])) if accepted else None,
                'median_total_with_coefficients_seconds':float(np.median([t['total_with_common_coefficients_seconds'] for t in accepted])) if accepted else None,
                'max_relative_energy_difference':max((t['comparison_to_numerical_reference']['reference_relative_energy_error'] for t in accepted),default=None),
                'max_relative_weighted_flux_difference':max((t['comparison_to_numerical_reference']['reference_relative_weighted_flux_error'] for t in accepted),default=None)}
        report['summary']=summaries
        baseline=[(m,v['median_core_seconds']) for m,v in summaries.items() if m!='component_rank_safe' and v['all_accepted']]
        candidate=summaries.get('component_rank_safe',{})
        if baseline and candidate.get('all_accepted'):
            fastest=min(baseline,key=lambda x:x[1])
            report['candidate_vs_fastest_qualified_baseline']={'method':fastest[0],
                'time_ratio':candidate['median_core_seconds']/fastest[1],
                'time_saving_fraction':1-candidate['median_core_seconds']/fastest[1],
                'claim_scope':'same snapshot, descriptive three technical repetitions, full setup+CG, quality passed'}
        save_json(output_dir/'summary.json',report)
        return report


def load_inputs(args):
    if args.input_npz:
        with np.load(args.input_npz,allow_pickle=False) as z:
            graph=ActiveGraph.from_arrays(z['xyz'],z['tail'],z['head'],z['trans'],tuple(map(int,z['shape'])))
            result={key:np.array(z[key],copy=True) for key in ('kx','injection','well_base')}
            result.update(graph=graph,audit={'input_npz':str(args.input_npz.resolve()),'sha256':digest(args.input_npz)})
            return result
    name='dynamic_inputs' if args.legacy_inputs else 'full_well_inputs'
    mod=importlib.import_module(name)
    expected=OLD/'dynamic/dynamic_inputs.py' if args.legacy_inputs else HERE/'full_well_inputs.py'
    if Path(mod.__file__).resolve()!=expected.resolve():raise ImportError('unexpected input builder module')
    return mod.load_inputs()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--snapshots-dir',type=Path)
    parser.add_argument('--snapshot',type=Path,action='append',default=[])
    parser.add_argument('--input-npz',type=Path)
    parser.add_argument('--legacy-inputs',action='store_true')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--seconds',type=float,default=900)
    parser.add_argument('--maxiter',type=int,default=5000)
    parser.add_argument('--rtol',type=float,default=1e-12)
    parser.add_argument('--no-vectors',action='store_true')
    args=parser.parse_args()
    if args.repeats<1 or args.maxiter<1 or args.seconds<=0 or not 0<args.rtol<1:parser.error('invalid numerical budget')
    if len(set(args.methods))!=len(args.methods):parser.error('duplicate method')
    inputs=load_inputs(args)
    files=list(args.snapshot)
    if args.snapshots_dir:files+=sorted(args.snapshots_dir.glob('snapshot*.npz'))
    files=list(dict.fromkeys(p.resolve() for p in files))
    cases=[]
    if not files:cases=[('initial',0.,np.full(len(inputs['kx']),.2),None)]
    for path in files:
        with np.load(path,allow_pickle=False) as z:
            day=float(np.asarray(z['time_days']).ravel()[0]) if 'time_days' in z.files else None
            if day is None:
                side=path.with_suffix('.json')
                if not side.is_file():raise ValueError('snapshot needs time_days or matching JSON')
                day=float(json.loads(side.read_text(encoding='utf-8'))['time_days'])
            cases.append((path.stem,day,np.array(z['saturation'],copy=True),{'path':str(path),'sha256':digest(path)}))
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    results=[]
    for label,day,saturation,source in cases:
        result=benchmark_snapshot(inputs,saturation,args.output/label,label=label,time_days=day,repeats=args.repeats,
                                  methods=tuple(args.methods),rtol=args.rtol,maxiter=args.maxiter,seconds=args.seconds,
                                  save_vectors=not args.no_vectors,source=source)
        results.append({'label':label,'time_days':day,'summary':result['summary']})
        print(json.dumps(results[-1],ensure_ascii=False),flush=True)
    save_json(args.output/'campaign_summary.json',{'cases':results,'source_sha256':digest(__file__),
              'completion_note':'Compare each actual acceptance flag; process exit alone is not scientific acceptance.'})


if __name__=='__main__':main()
