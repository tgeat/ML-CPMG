"""Prospective stronger legacy MG baseline; not attached to the locked round.

The four configurations use nu=1/2 and coarse_size=96/512. Boolean material
phase, np.unique geometric aggregation, ordinary Darcy CSR assembly, Jacobi
smoothed prolongation and Galerkin products retain the legacy mechanism.
No component labels, packed aggregate encoding, lumping, or candidate kernels
are imported. No numerical work is performed on import.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, splu
from threadpoolctl import threadpool_limits
import darcy_core as dc

CONFIGS = {
    'legacy_nu1_c96': {'nu': 1, 'coarse_size': 96},
    'legacy_nu2_c96': {'nu': 2, 'coarse_size': 96},
    'legacy_nu1_c512': {'nu': 1, 'coarse_size': 512},
    'legacy_nu2_c512': {'nu': 2, 'coarse_size': 512},
}


def _project(x):
    return x-np.mean(x)


def _hierarchy(A, k, nu, coarse_size):
    xyz = np.indices(k.shape).reshape(3, -1).T
    phase = (k.ravel() > np.min(k)).astype(np.int64)
    levels, sizes, nnz = [], [], []
    while True:
        A = A.tocsr()
        diagonal = A.diagonal()
        if np.any(diagonal <= 0):
            raise ValueError('nonpositive legacy multigrid diagonal')
        dinv = 1.0/diagonal
        bound = float(np.max(np.asarray(abs(A).sum(axis=1)).ravel()*dinv))
        omega = min(2.0/3.0, 4.0/(3.0*bound))
        sizes.append(int(A.shape[0]))
        nnz.append(int(A.nnz))
        level = {'A': A, 'di': omega*dinv}
        levels.append(level)
        if A.shape[0] <= coarse_size:
            level['factor'] = splu(A[1:, 1:].tocsc())
            break
        blocks = xyz//2
        tags = np.column_stack((blocks, phase))
        unique, inverse = np.unique(tags, axis=0, return_inverse=True)
        nc = unique.shape[0]
        if nc >= A.shape[0] or len(levels) >= 8:
            level['factor'] = splu(A[1:, 1:].tocsc())
            break
        Z = sparse.csr_matrix((np.ones(A.shape[0]),
                              (np.arange(A.shape[0]), inverse)),
                              shape=(A.shape[0], nc))
        P = (Z-(A @ Z).multiply((omega*dinv)[:, None])).tocsr()
        rowsum = np.asarray(P.sum(axis=1)).ravel()
        P = P.multiply((1.0/rowsum)[:, None]).tocsr()
        level['P'], level['R'] = P, P.T.tocsr()
        coarse = level['R'] @ A @ P
        coarse = ((coarse+coarse.T)*0.5).tocsr()
        rowdrift = np.asarray(coarse.sum(axis=1)).ravel()
        coarse.setdiag(coarse.diagonal()-rowdrift)
        coarse.eliminate_zeros()
        A = coarse
        xyz, phase = unique[:, :3], unique[:, 3]

    def cycle(index, rhs):
        level = levels[index]
        rhs = _project(rhs)
        if 'factor' in level:
            result = np.zeros_like(rhs)
            if len(result) > 1:
                result[1:] = level['factor'].solve(rhs[1:])
            return _project(result)
        matrix, damped_inverse = level['A'], level['di']
        result = damped_inverse*rhs
        for _ in range(nu-1):
            result += damped_inverse*(rhs-matrix @ result)
        residual = _project(rhs-matrix @ result)
        result += level['P'] @ cycle(index+1, level['R'] @ residual)
        for _ in range(nu):
            result += damped_inverse*(rhs-matrix @ result)
        return _project(result)

    M = LinearOperator(levels[0]['A'].shape,
                       matvec=lambda rhs: cycle(0, rhs), dtype=np.float64)
    coarse_factor = levels[-1]['factor']
    return M, {'level_sizes': sizes, 'level_nnz': nnz,
               'operator_complexity': float(sum(nnz)/nnz[0]),
               'coarse_direct_size': sizes[-1],
               'coarse_factor_nnz': int(coarse_factor.L.nnz+coarse_factor.U.nnz)}


def build(k, config=None):
    """Return {'A','M','metadata','krylov':'cg'} for the common precision kernel.

    All coefficient-dependent validation, fine assembly and hierarchy setup
    belong to this call. The driver charges imports, I/O and common refinement.
    The four native parameter combinations must be frozen in a separate round;
    the module does not change any currently locked plan or source.
    """
    started = time.perf_counter()
    if config is None:
        parameters = dict(CONFIGS['legacy_nu2_c96'])
        name = 'legacy_nu2_c96'
    elif isinstance(config, str):
        parameters = dict(CONFIGS[config])
        name = config
    else:
        supplied = dict(config)
        name = supplied.pop('preset', None)
        parameters = dict(CONFIGS[name]) if name is not None else {}
        parameters.update(supplied)
        unexpected = set(parameters)-{'nu', 'coarse_size'}
        if unexpected:
            raise ValueError('only nu/coarse_size are native tuning parameters')
        parameters.setdefault('nu', 2)
        parameters.setdefault('coarse_size', 96)
        name = name or 'legacy_nu%d_c%d'%(parameters['nu'], parameters['coarse_size'])
    nu, coarse_size = parameters['nu'], parameters['coarse_size']
    if nu not in (1, 2) or coarse_size not in (96, 512):
        raise ValueError('strong legacy domain is nu=1/2 x coarse_size=96/512')
    nu, coarse_size = int(nu), int(coarse_size)
    name = 'legacy_nu%d_c%d'%(nu, coarse_size)
    with threadpool_limits(limits=1):
        k = np.asarray(k, dtype=np.float64)
        if k.ndim != 3 or len(set(k.shape)) != 1 or k.shape[0] < 2:
            raise ValueError('expected cubic permeability with n>=2')
        if not np.all(np.isfinite(k)) or np.any(k <= 0):
            raise ValueError('finite positive permeability required')
        A = dc.assemble(k)
        assembled = time.perf_counter()
        M, hierarchy_meta = _hierarchy(A, k, nu, coarse_size)
        ended = time.perf_counter()
    metadata = {'name': name, 'family': 'strong_legacy_mg',
        'config': {'nu': int(nu), 'coarse_size': int(coarse_size)},
        'krylov': 'cg', 'coordinate_system': 'physical zero-mean pressure quotient',
        'nu': int(nu), 'pre_sweeps': int(nu), 'post_sweeps': int(nu),
        'coarse_size': int(coarse_size), 'assembly_seconds': assembled-started,
        'hierarchy_seconds': ended-assembled, 'setup_seconds': ended-started,
        'all_coefficient_setup_included': True,
        'mechanism': 'legacy boolean-phase np.unique; ordinary fine CSR; smoothed Galerkin; symmetric Jacobi V-cycle',
        'ordinary_fine_assembly': 'darcy_core.assemble',
        'component_preservation': False, 'candidate_fast_aggregation': False,
        'coarse_lumping': False, 'symmetric_preconditioner': True,
        'attached_to_current_locked_round': False,
        **hierarchy_meta}
    return {'A': A, 'M': M, 'metadata': metadata, 'krylov': 'cg'}
