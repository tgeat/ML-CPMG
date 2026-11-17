"""Small-system tests of the rank and non-Galerkin V-cycle argument."""
import os
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[v]='1'
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))
import numpy as np
from scipy import sparse,linalg
from threadpoolctl import threadpool_limits
import ml_cpmg_v16 as m
import darcy_core

def check(grounded,seed,line=False):
    rng=np.random.default_rng(seed);shape=(6,6,6);n=np.prod(shape)
    k=np.exp(rng.uniform(0,np.log(100),shape));e=darcy_core.edges(k)
    A=m.assemble_cartesian_fast(shape,e)
    if grounded: A.setdiag(A.diagonal()+np.linspace(.1,.3,n))
    xyz=np.indices(shape).reshape(3,-1).T
    tags=(k.ravel()>np.median(k)).astype(np.int64)
    cfg=dict(m._BASE_CONTROLS,coarse_size=16,max_direct=512)
    levels,meta=m._construct_hierarchy(A,xyz,tags,cfg,grounded=grounded,cartesian_fine=not grounded)
    if line:
        levels[0]['line_smoother']=m._build_anisotropic_line_smoother(shape,e,A.diagonal(),omega=levels[0]['omega'],forced_axis=0)
    details=[]
    for lv,nxt in zip(levels[:-1],levels[1:]):
        P=lv['P'].toarray();G=P.T@lv['A'].toarray()@P;C=nxt['A'].toarray()
        rank=int(np.linalg.matrix_rank(P));diff=linalg.eigvalsh(C-G)
        assert rank==P.shape[1]
        assert np.max(abs(P.sum(axis=1)-1))<1e-12
        assert diff.min()>-1e-9*max(1,linalg.norm(C,2))
        details.append(dict(n=lv['A'].shape[0],nc=P.shape[1],rank=rank,
                            min_energy_increment=float(diff.min()),lumped=bool(np.max(abs(C-G))>1e-10)))
    apply,_=m._make_cycle_callable(levels,grounded=grounded,cycle_repeats=1,fine_A=A)
    B=np.column_stack([apply(np.eye(n)[:,i]) for i in range(n)])
    assert linalg.norm(B-B.T)/linalg.norm(B)<1e-11
    U=np.eye(n) if grounded else linalg.null_space(np.ones((1,n)))
    Ar=U.T@A.toarray()@U;Br=U.T@B@U
    w,V=linalg.eigh(Ar);Ah=(V*np.sqrt(w))@V.T
    spec=linalg.eigvalsh(Ah@Br@Ah)
    assert spec.min()>0 and spec.max()<1+1e-10
    return dict(grounded=grounded,seed=seed,line=line,levels=details,
                symmetry_defect=float(linalg.norm(B-B.T)/linalg.norm(B)),
                min_preconditioned_eigenvalue=float(spec.min()),max_preconditioned_eigenvalue=float(spec.max()),
                energy_contraction=float(1-spec.min()))

if __name__=='__main__':
    with threadpool_limits(1): rows=[check(g,s,line) for g in (False,True) for s in (7,11,29) for line in (False,True)]
    out=ROOT/'results/paper_revision_v16/data/algebra_checks.json'
    out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
