"""Independently assess stored first-repeat pressures; no alteration of timings."""
from benchmark import *
import spe10_accuracy as sa,spe10_dynamic_accuracy as da
from bench_common import dd_refine_neumann

def compact(q):
    return {k:q.get(k) for k in ('pressure_relative_upper','flux_relative_upper','accuracy_pass','pressure_true_residual','flux_true_residual','rho_upper','eta_upper','reason') if k in q}

def run():
    selected=json.loads((DATA/'selected_config.json').read_text());cap=selected['max_basins'];stall=selected['stall_ratio']
    out=DATA/'accuracy.json';records=json.loads(out.read_text()) if out.exists() else []
    for suite in ('static','synthetic','dynamic','brugge'):
        for case in cases(suite):
            cid=case['id']
            if any(r.get('case')==cid and r.get('complete_case') for r in records):continue
            print('Assessing stored pressures:',cid,flush=True)
            names=PRIMARY if suite!='brugge' else METHODS+['Algebraic-only']
            if suite=='brugge':
                A=mg._assemble_general_edges(case['k'].size,case['edge'],well_diagonal=case['d'])
                ref=splu(A.tocsc()).solve(case['b']);en=float(ref@(A@ref))
                for name in names:
                    f=DATA/f'{cid}_{name}_reliable.npz'
                    if not f.exists():continue
                    x=np.load(f)['x'];delta=x-ref
                    records.append(dict(case=cid,method=name,relative_energy_difference=float(np.sqrt(max(0,delta@(A@delta))/en)),
                        true_residual=float(np.linalg.norm(case['b']-A@x)/np.linalg.norm(case['b'])),comparison='direct SuperLU; numerical difference, not interval bound'))
            else:
                if suite=='static':
                    folder=ROOT/'results/spe10_model2_singlephase'
                    z=np.load(folder/'reference.npz');rh,rl=z['hi'],z['lo'];cert=json.loads((folder/'reference.json').read_text())
                elif suite=='dynamic' and cid!='SPE10-0d':
                    day=float(cid[6:-1]);folder=ROOT/'results/spe10_model2_dynamic/component_lump_1_1_c512'
                    z=load_checkpoint(folder/f'snapshot_{day:09.2f}.reference.npz.lz4');rh,rl=z['hi'],z['lo']
                    raw=json.loads((folder/f'snapshot_{day:09.2f}.reference.json').read_text());cert=raw.get('certificate',raw)
                else:
                    saved=np.load(DATA/f'{cid}_ML-CPMG_reliable.npz');x=(saved['hi'],saved['lo'])
                    A,M,_,_=build(case,'ML-CPMG',cap,stall,force_line=suite=='dynamic')
                    if case['d'] is None:
                        rh,rl,cert=sa.make_reference(case['k'].shape,case['b'],case['edge'],A,preconditioner=M,initial_pressure=x,relative_target=5e-9)
                    else:
                        rh,rl,cert=da.make_reference(case['k'].shape,case['b'],case['edge'],case['d'],A,preconditioner=M,initial_pressure=x,relative_target=5e-9)
                    np.savez_compressed(DATA/f'{cid}_reference.npz',hi=rh,lo=rl);save_json(DATA/f'{cid}_reference.json',cert)
                for name in names:
                    f=DATA/f'{cid}_{name}_reliable.npz'
                    if not f.exists():continue
                    saved=np.load(f);hi,lo=saved['hi'],saved['lo']
                    if case['d'] is None:q=sa.assess(case['k'].shape,case['b'],case['edge'],hi,lo,rh,rl,cert)
                    else:q=da.assess(case['k'].shape,case['b'],case['edge'],case['d'],hi,lo,rh,rl,cert)
                    records.append(dict(case=cid,method=name,**compact(q)))
                    print(name,compact(q),flush=True)
            records.append(dict(case=cid,complete_case=True));save_json(out,records)
    print('Pressure assessment complete.',flush=True)

if __name__=='__main__':
    psutil.Process().cpu_affinity([19])
    with threadpool_limits(1):run()
