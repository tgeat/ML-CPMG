"""Rebuild display arrays from the frozen v16/v17 inputs; no timing reruns."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
TASK=ROOT/'results/paper_revision_v18'
sys.path.insert(0,str(ROOT/'results/paper_revision_v16/code'))
from benchmark import cases,conf,mg

def main():
    arrays={}; report={}
    for case in cases('synthetic'):
        name=case['id'].removeprefix('Synthetic-'); k=case['k']
        arrays[name+'_logk']=np.log10(k).astype('float32')
        if name not in ('multilevel_nested_lobes','multilevel_braided_channels','smooth_gaussian'):
            continue
        info={}
        for method in ('Min-threshold','Single-median','ML-CPMG'):
            lab,count,meta=mg.extract_multilevel_components(k,edge=case['edge'],config=conf(method))
            arrays[name+'_'+method]=lab.astype('int32')
            data={'positive_labels_3d':count,'positive_labels_x32':int(np.unique(lab[32][lab[32]>0]).size)}
            if 'ref' in case:
                ref=case['ref']; peaks=ref['peak_id_grid']; dom=[]
                for n in range(1,int(peaks.max())+1):
                    vals=lab[peaks==n]; vals=vals[vals>0]
                    dom.append(int(np.bincount(vals).argmax()) if len(vals) else 0)
                data.update(prescribed_cores=len(dom),resolved_cores=len(set(dom)-{0}),dominant_labels=dom)
            info[method]=data
        report[name]=info
        print(name,info,flush=True)
    np.savez_compressed(TASK/'data/synthetic.npz',**arrays)
    report['slice']={'axis':0,'zero_based_index':32,'coordinate':32.5/64,'interpolation':'none'}
    case=next(cases('static'))
    lab,count,info=mg.extract_multilevel_components(case['k'],edge=case['edge'],config=conf('ML-CPMG'))
    np.savez_compressed(TASK/'data/spe10_labels.npz',labels=lab.astype('int32'))
    report['spe10']={'labels_3d':count,'shape':list(case['k'].shape)}
    source=ROOT/'results/paper_revision_v16/data';shape=case['k'].shape
    ref=np.load(ROOT/'results/spe10_model2_singlephase/reference.npz');rh,rl=ref['hi'],ref['lo'];scale=float(np.max(abs(rh+rl)))
    pp=np.load(source/'SPE10-static_ML-CPMG_reliable.npz');p=(pp['hi']+pp['lo']).reshape(shape)/scale
    slices=lambda a:[a[42].T,a[:,109,:],a[:,:,29]]
    sd={'label_count':count,'reference_scale':scale}
    for prefix,a in [('p',p),('k',np.log10(case['k'])),('labels',lab)]:
        for j,v in enumerate(slices(a)):sd[f'{prefix}{j}']=v
    maxima=[]
    for r,m in enumerate(['ML-CPMG','Single-median','Algebraic-only','SA-0.04']):
        z=np.load(source/f'SPE10-static_{m}_reliable.npz');err=(abs((z['hi']-rh)+(z['lo']-rl))/scale).reshape(shape);maxima.append(float(err.max()))
        for j,v in enumerate(slices(err)):sd[f'error{r}_{j}']=v
    sd['error_max']=max(maxima)
    np.savez_compressed(TASK/'data/spe10_sections.npz',**sd)
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'multilevel_continuous_media.py',ROOT/'results/paper_revision_v16/code/ml_cpmg_v16.py',ROOT/'results/paper_revision_v16/data/summary.json',ROOT/'results/paper_revision_v16/data/calibrate.json']}
    (TASK/'data/segmentation_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    build_brugge()

def build_brugge():
    sys.path.insert(0,str(ROOT/'results/brugge_feasibility_v1/model_audit'))
    from audit_grdecl import parse_keyword,DATA,PREFIX,NX,NY,NZ
    g=np.load(ROOT/'results/brugge_validation_v2/inputs.npz')
    old=np.load(ROOT/'results/brugge_feasibility_v1/pressure/processed_graph_provisional.npz')
    for key in ['xyz','tail','head','global_index','trans']:
        assert np.array_equal(g[key],old[key]),key
    xyz=g['xyz']; i,j,k=xyz.T
    coord=parse_keyword(DATA/f'{PREFIX}_GRID.GRDECL','COORD').reshape(NY+1,NX+1,2,3)
    ma=parse_keyword(DATA/f'{PREFIX}_GRID.INC','MAPAXES').reshape(3,2)
    origin=ma[1]; ux=(ma[2]-origin)/np.linalg.norm(ma[2]-origin); uy=(ma[0]-origin)/np.linalg.norm(ma[0]-origin)
    coord[...,:2]=origin+coord[...,0,None]*ux+coord[...,1,None]*uy
    dep=parse_keyword(DATA/f'{PREFIX}_GRID.GRDECL','ZCORN').reshape(2*NZ,2*NY,2*NX)
    vertices=[]
    for di,dj,dk in [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]:
        z=dep[2*k+dk,2*j+dj,2*i+di]; p=coord[j+dj,i+di]
        t=(z-p[:,0,2])/(p[:,1,2]-p[:,0,2]); xy=p[:,0,:2]+t[:,None]*(p[:,1,:2]-p[:,0,:2])
        vertices.append(np.column_stack((xy,z)))
    corners=np.stack(vertices,axis=1)
    tail,head=g['tail'],g['head']; nnc=np.abs(xyz[tail]-xyz[head]).sum(axis=1)!=1
    z=np.load(ROOT/'results/paper_revision_v16/data/Brugge-0d_ML-CPMG_reliable.npz')
    p=z['hi']+z['lo']
    assert len(corners)==43474 and np.isfinite(corners).all()
    np.savez_compressed(TASK/'data/brugge_geometry.npz',corners=corners,xyz=xyz,tail=tail[nnc],head=head[nnc],pressure=p,shape=g['shape'],trans=g['trans'][nnc])
    audit={'active_cells':len(xyz),'edge_count':len(tail),'nnc_logical_nonadjacent':int(nnc.sum()),'nnc_edge_types':{str(t):int(np.sum(nnc&(old['edge_type']==t))) for t in np.unique(old['edge_type'])},'pressure_state_day':0,'pressure_range':p[[p.argmin(),p.argmax()]].tolist(),'coordinate_units':'m','geometry':'source COORD/ZCORN with MAPAXES, exact active ordering','edges':'positive transmissibility graph, L1 logical distance != 1'}
    (TASK/'data/brugge_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(audit,flush=True)

if __name__=='__main__':main()
