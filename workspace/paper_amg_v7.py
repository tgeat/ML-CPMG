"""Prospective PyAMG 5.4.1 baselines for the independent-paper campaign.

Eight declared native configurations: SA/RS x two strength thresholds x
max_coarse=10/512. All use the physical quotient and one symmetric GS sweep.
Both AMG families use QR only when the candidate null mode and its
complete prolongation identities are actually verified; otherwise the full
SPD coarse operator is solved without discarding physical modes.
No numerical work, installation, or private benchmark runs occur on import.
The public fine operator uses the same fast CSR kernel as the candidate;
the configured hierarchy is never grounded or shifted.
The experiment controller must qualify symmetry/positivity and solver checks
before admitting any configuration to a CG performance comparison.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import time
from importlib import metadata as package_metadata
import numpy as np
from scipy import sparse
from scipy.linalg import qr, cho_factor, cho_solve, eigvalsh
from scipy.sparse.linalg import LinearOperator
from threadpoolctl import threadpool_limits
from darcy_core import edges
from path_multigrid_v3 import _fine_csr


EXPECTED_PYAMG_VERSION = '5.4.1'
SETUP_RANDOM_SEED = 20261101
PRESETS = {
    f'{family}_theta{theta_tag}_c{max_coarse}': {
        'family': family, 'gauge_strategy': 'quotient', 'nu': 1,
        'strength_theta': theta, 'max_coarse': max_coarse,
        'max_levels': 10 if family == 'sa' else 30,
        'cycle': 'V', 'setup_random_seed': SETUP_RANDOM_SEED,
    }
    for family, theta_choices in (
        ('sa', (('0', 0.0), ('01', 0.1))),
        ('rs', (('025', 0.25), ('05', 0.5))),
    )
    for theta_tag, theta in theta_choices
    for max_coarse in (10, 512)
}
# extension_worker_v2 expands named ctx-builder parameters as keyword args.
CONFIGS = {name: {'config': name} for name in PRESETS}


def _project(vector):
    vector = np.asarray(vector, dtype=np.float64).ravel()
    return vector-np.mean(vector)


def _quotient_coarse_solver(ml, family, require_time):
    """Choose a fixed coarse inverse from verified hierarchy identities.

    SA propagated B and RS candidate ones are hypotheses to verify. Strength
    filtering or aggregation/interpolation can lose their reproduction, so
    Galerkin coarse matrices can be SPD despite a singular physical fine A.
    Every decision and factor is fixed during setup, independent of an RHS.
    """
    require_time()
    last = ml.levels[-1]
    n = int(last.A.shape[0])
    if n < 1 or n > 2048:
        raise MemoryError('declared dense quotient coarse size guard')
    raw = np.asarray(last.A.toarray(), dtype=np.float64)
    symmetric = (raw+raw.T)*0.5
    if not np.all(np.isfinite(symmetric)):
        raise ValueError('nonfinite coarse matrix')
    if family == 'sa':
        modes = np.asarray(last.B, dtype=np.float64)
        if modes.ndim != 2 or modes.shape != (n, 1):
            raise ValueError('SA quotient expects exactly one propagated candidate mode')
        null = modes[:, 0].copy()
        mode_source = 'SA candidate B[:,0]; removed only after identity verification'
    else:
        null = np.ones(n, dtype=np.float64)
        mode_source = 'RS candidate ones; removed only after identity verification'
    norm_null = float(np.linalg.norm(null))
    if not np.all(np.isfinite(null)) or not np.isfinite(norm_null):
        raise ValueError('nonfinite coarse candidate mode')
    candidate_mode_available = norm_null > 0
    if candidate_mode_available:
        null /= norm_null
    norm_matrix = float(np.linalg.norm(symmetric, ord='fro'))
    residual = float(np.linalg.norm(symmetric @ null)) if candidate_mode_available else None
    null_relative = (residual/max(norm_matrix, np.finfo(np.float64).tiny)
                     if candidate_mode_available else None)
    raw_asymmetry = float(np.linalg.norm(raw-raw.T, ord='fro') /
                         max(float(np.linalg.norm(raw, ord='fro')), np.finfo(np.float64).tiny))

    propagation = []
    for index, level in enumerate(ml.levels[:-1]):
        require_time()
        if family == 'sa':
            fine_mode = np.asarray(level.B, dtype=np.float64)[:, 0]
            coarse_mode = np.asarray(ml.levels[index+1].B, dtype=np.float64)[:, 0]
        else:
            fine_mode = np.ones(level.A.shape[0], dtype=np.float64)
            coarse_mode = np.ones(ml.levels[index+1].A.shape[0], dtype=np.float64)
        if not np.all(np.isfinite(fine_mode)) or not np.all(np.isfinite(coarse_mode)):
            raise ValueError('nonfinite propagated candidate mode')
        difference = np.asarray(level.P @ coarse_mode).ravel()-fine_mode
        propagation.append(float(np.linalg.norm(difference) /
                                 max(float(np.linalg.norm(fine_mode)), np.finfo(np.float64).tiny)))
    tolerance = 1e-10  # unchanged identity guard, never a solver accuracy gate
    propagation_ok = max(propagation, default=0.0) <= tolerance
    # A singleton with a verified complete constant-reproduction chain has
    # no physical complement. Its floating scalar need not be exactly zero.
    # Compare that residual with the preceding Galerkin multiplication scale,
    # not the scalar's own near-zero norm. The new roundoff guard is at least
    # as strict as the unchanged1e-10 identity guard; it is not an eigencutoff.
    singleton = {}
    if n == 1:
        if len(ml.levels) < 2:
            raise ValueError('AMG singleton requires a preceding transfer identity')
        previous = ml.levels[-2]
        parent_frobenius = float(np.linalg.norm(previous.A.data))
        transfer_frobenius = float(np.linalg.norm(previous.P.data))
        parent_product_scale = parent_frobenius*transfer_frobenius**2
        if not np.isfinite(parent_product_scale):
            raise ValueError('Nonfinite AMG singleton preceding Galerkin scale')
        scalar_residual = float(abs(raw[0, 0]))
        relative_to_parent = scalar_residual/max(parent_product_scale, np.finfo(np.float64).tiny)
        roundoff_relative_limit = min(tolerance, 64*np.finfo(np.float64).eps*
                                     max(1, int(previous.A.nnz)))
        singleton_roundoff_ok = bool(relative_to_parent <= roundoff_relative_limit)
        singleton = {
            'singleton_parent_size': int(previous.A.shape[0]),
            'singleton_parent_nnz': int(previous.A.nnz),
            'singleton_parent_frobenius_norm': parent_frobenius,
            'singleton_transfer_frobenius_norm': transfer_frobenius,
            'singleton_parent_Galerkin_product_scale': parent_product_scale,
            'singleton_absolute_coarse_residual': scalar_residual,
            'singleton_residual_relative_to_parent_scale': relative_to_parent,
            'singleton_roundoff_relative_limit': roundoff_relative_limit,
            'singleton_roundoff_absolute_limit': roundoff_relative_limit*parent_product_scale,
            'singleton_roundoff_guard_pass': singleton_roundoff_ok,
            'singleton_structural_null_verified': bool(candidate_mode_available and propagation_ok and singleton_roundoff_ok),
        }
    coarse_null_ok = bool(candidate_mode_available and
                          (singleton['singleton_roundoff_guard_pass'] if n == 1
                           else null_relative <= tolerance))
    known_null_ok = bool(propagation_ok and coarse_null_ok)
    branch_reason = ('candidate_mode_unavailable' if not candidate_mode_available else
                     'prolongation_candidate_identity_not_verified' if not propagation_ok else
                     'coarse_candidate_null_identity_not_verified' if not coarse_null_ok else
                     'verified_structural_null_direction')
    diagnostics = (f'n={n}, propagation={propagation!r}, '
                   f'Ac_null_relative={null_relative!r}, Ac_null_absolute={residual!r}, '
                   f'Ac_frobenius={norm_matrix!r}, singleton={singleton!r}')

    require_time()
    if known_null_ok:
        branch = 'known_null_QR_complement_Cholesky'
        basis = qr(null[:, None], mode='full', check_finite=True)[0][:, 1:]
        removed_null_modes = 1
    else:
        # Galerkin Ac=P.T A P can be SPD when constants are outside range(P).
        # A full factor preserves every actual coarse mode. Failure is final;
        # no shift, eigenvalue clipping, pinv fallback or retry is permitted.
        branch = 'full_SPD_coarse_Cholesky_no_mode_removed'
        basis = np.eye(n, dtype=np.float64)
        removed_null_modes = 0
    dimension = int(basis.shape[1])
    if dimension == 0:
        factor = None
        minimum = None
        inverse_residual = 0.0
    else:
        reduced = basis.T @ symmetric @ basis
        reduced = (reduced+reduced.T)*0.5
        minimum = (float(eigvalsh(reduced, subset_by_index=(0, 0), check_finite=True)[0])
                   if dimension <= 64 else None)
        try:
            factor = cho_factor(reduced, lower=True, overwrite_a=False, check_finite=True)
        except np.linalg.LinAlgError as error:
            raise ValueError('AMG coarse Cholesky failed; branch='+branch+', '
                             +diagnostics+', minimum_eigenvalue='+repr(minimum)) from error
        if dimension <= 64:
            identity = np.eye(dimension, dtype=np.float64)
            inverse = cho_solve(factor, identity, check_finite=True)
            inverse_residual = float(np.linalg.norm(reduced @ inverse-identity, ord='fro') /
                                     max(float(np.linalg.norm(identity, ord='fro')), np.finfo(np.float64).tiny))
        else:
            inverse_residual = None
    require_time()

    def solve(_matrix, rhs):
        require_time()
        rhs = np.asarray(rhs, dtype=np.float64).ravel()
        if rhs.size != n or not np.all(np.isfinite(rhs)):
            raise ValueError('invalid quotient coarse residual')
        if factor is None:
            return np.zeros_like(rhs)
        return basis @ cho_solve(factor, basis.T @ rhs, check_finite=False)

    metadata = {
        'coarse_solve_branch': branch,
        'coarse_solve_branch_reason': branch_reason,
        'coarse_candidate_mode_available': candidate_mode_available,
        'coarse_candidate_mode_norm_before_normalization': norm_null,
        'coarse_actual_dimension': n,
        'known_coarse_null_mode_source': mode_source,
        'known_coarse_null_mode_verified': known_null_ok,
        'coarse_null_residual_absolute': residual,
        'coarse_null_residual_relative_frobenius': null_relative,
        'coarse_raw_asymmetry_relative_frobenius': raw_asymmetry,
        'known_null_prolongation_relative_errors': propagation,
        'known_null_prolongation_identity_pass': bool(propagation_ok),
        'known_null_coarse_identity_pass': bool(coarse_null_ok),
        'known_null_identity_tolerance': tolerance,
        'coarse_complement_dimension': dimension,
        'coarse_null_modes_removed': removed_null_modes,
        'coarse_complement_min_eigenvalue_if_size_le64': minimum,
        'coarse_complement_inverse_residual_relative_frobenius': inverse_residual,
        'coarse_nonzero_modes_discarded': 0,
        'coarse_eigenvalue_clipping': False,
        'coarse_spectral_pinv_threshold_used': False,
        'coarse_nullspace_factorization': branch,
        **singleton,
    }
    return solve, metadata


def build(ctx, config='sa_theta0_c10'):
    """Return A/M/metadata/krylov for the frozen common precision kernel.

    quotient: M = P M_amg(A) P, verified-null or full-SPD coarse solve.
    grounded: M = P E M_amg(A[1:,1:]) E.T P, Cholesky coarse solve.
    P is physical arithmetic zero-mean projection; E inserts zero at anchor0.
    No shift/penalty is added to the public operator and no RHS is changed.
    Near-nullspace ones are supplied to SA, not inferred from the reference.
    """
    started = time.perf_counter()
    if not isinstance(config, str) or config not in PRESETS:
        raise ValueError('choose one of the eight declared native PyAMG presets')
    parameters = dict(PRESETS[config])
    version = package_metadata.version('pyamg')
    if version != EXPECTED_PYAMG_VERSION:
        raise RuntimeError('freeze the planned PyAMG version before executing')
    import pyamg

    deadline = ctx.get('deadline')
    def require_time():
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('PyAMG builder exceeded common deadline')

    require_time()
    with threadpool_limits(limits=1):
        # Match the candidate exactly: harmonic face construction and the
        # shared direct seven-point CSR kernel are both charged here.
        k = np.asarray(ctx['k'], dtype=np.float64)
        if k.ndim != 3 or len(set(k.shape)) != 1 or k.shape[0] < 2:
            raise ValueError('expected cubic permeability with n>=2')
        if not np.all(np.isfinite(k)) or np.any(k <= 0):
            raise ValueError('finite positive permeability required')
        edge = edges(k)
        A = sparse.csr_array(_fine_csr(k, edge), dtype=np.float64, copy=False)
        if A.shape[0] != A.shape[1] or A.shape[0] < 2:
            raise ValueError('square fine operator with at least two unknowns required')
        if not np.all(np.isfinite(A.data)):
            raise ValueError('nonfinite fine operator')
        A.sum_duplicates()
        A.sort_indices()
        size = int(A.shape[0])
        assembled = time.perf_counter()
        gauge = parameters['gauge_strategy']
        H = A if gauge == 'quotient' else A[1:, 1:].tocsr()
        coarse_solver = 'pinv' if gauge == 'quotient' else 'cholesky'
        smoother = ('gauss_seidel', {
            'sweep': 'symmetric', 'iterations': parameters['nu'],
        })
        # The documented Jacobi prolongation smoother estimates a spectral
        # radius from a random vector. Freeze and restore the process RNG;
        # neither a reference solution nor a test-dependent seed is used.
        random_state = np.random.get_state()
        try:
            np.random.seed(parameters['setup_random_seed'])
            require_time()
            common = dict(
                presmoother=smoother, postsmoother=smoother,
                max_levels=parameters['max_levels'],
                max_coarse=parameters['max_coarse'],
                coarse_solver=coarse_solver, keep=False,
            )
            if parameters['family'] == 'sa':
                ml = pyamg.smoothed_aggregation_solver(
                    H, B=np.ones((H.shape[0], 1), dtype=np.float64),
                    symmetry='hermitian',
                    strength=('symmetric', {'theta': parameters['strength_theta']}),
                    aggregate='standard',
                    smooth=('jacobi', {'omega': 4.0/3.0}),
                    improve_candidates=(
                        ('block_gauss_seidel', {'iterations': 4, 'sweep': 'symmetric'}),
                        None,
                    ),
                    **common,
                )
            else:
                ml = pyamg.ruge_stuben_solver(
                    H, strength=('classical', {'theta': parameters['strength_theta']}),
                    CF=('RS', {'second_pass': False}),
                    interpolation='classical', **common,
                )
        finally:
            np.random.set_state(random_state)
        require_time()
        hierarchy_ended = time.perf_counter()
        coarse_A = ml.levels[-1].A
        # PyAMG's coarse solver factorizes lazily. Guard any unexpected
        # failure to coarsen before requesting dense pinv/Cholesky, and charge
        # actual first factorization to setup for both gauge strategies.
        if coarse_A.shape[0] > 2048:
            raise MemoryError('declared dense coarse-solver guard: size exceeds 2048')
        coarse_metadata = {}
        if gauge == 'quotient':
            fixed_solver, coarse_metadata = _quotient_coarse_solver(ml, parameters['family'], require_time)
            ml.coarse_solver = pyamg.coarse_grid_solver(fixed_solver)
            coarse_solver = coarse_metadata['coarse_solve_branch']
        ml.coarse_solver(coarse_A, np.zeros(coarse_A.shape[0], dtype=np.float64))
        require_time()
        raw_M = ml.aspreconditioner(cycle=parameters['cycle'])

        def apply(rhs):
            rhs = _project(rhs)
            if rhs.size != size or not np.all(np.isfinite(rhs)):
                raise ValueError('finite physical residual with one value per cell required')
            require_time()
            if gauge == 'quotient':
                result = np.asarray(raw_M @ rhs, dtype=np.float64).ravel()
            else:
                result = np.zeros(size, dtype=np.float64)
                result[1:] = raw_M @ rhs[1:]
            return _project(result)

        M = LinearOperator(A.shape, matvec=apply, dtype=np.float64)
        ended = time.perf_counter()
        level_sizes = [int(level.A.shape[0]) for level in ml.levels]
        level_nnz = [int(level.A.nnz) for level in ml.levels]
        metadata = {
            'name': config, 'family': 'pyamg_'+parameters['family'],
            'config': parameters, 'library': 'PyAMG', 'library_version': version,
            'krylov': 'cg', 'coordinate_system': 'physical zero-mean pressure quotient',
            'gauge_strategy': gauge,
            'internal_ground_anchor': 0 if gauge == 'grounded' else None,
            'public_operator_grounded': False, 'public_operator_shifted': False,
            'shared_fast_fine_assembly': True,
            'fine_assembly': 'path_multigrid_v3._fine_csr(k, darcy_core.edges(k))',
            'fine_graph_rebuilt_inside_setup': True,
            'generic_ctx_assemble_used': False,
            'near_nullspace': 'ones' if parameters['family'] == 'sa' else 'classical interpolation',
            'cycle': parameters['cycle'], 'coarse_solver': coarse_solver,
            'pre_sweeps': parameters['nu'], 'post_sweeps': parameters['nu'],
            'smoother': 'symmetric scalar Gauss-Seidel',
            'strength_theta': parameters['strength_theta'],
            'strength_measure': 'symmetric' if parameters['family'] == 'sa' else 'classical',
            'native_search_axes': ['strength_theta', 'max_coarse'],
            'library_symmetric_smoothing': bool(ml.symmetric_smoothing),
            'symmetric_preconditioner_design': True,
            'cg_qualification_required': True,
            'max_coarse': parameters['max_coarse'],
            'level_sizes': level_sizes, 'level_nnz': level_nnz,
            'physical_size': size, 'hierarchy_fine_size': int(H.shape[0]),
            'operator_complexity': float(ml.operator_complexity()),
            'grid_complexity': float(ml.grid_complexity()),
            'coarse_direct_size': level_sizes[-1],
            'assembly_seconds': assembled-started,
            'hierarchy_seconds': hierarchy_ended-assembled,
            'coarse_factor_and_adapter_seconds': ended-hierarchy_ended,
            'setup_seconds': ended-started,
            'all_coefficient_setup_included': True,
            'coarse_factorization_eagerly_charged': True,
            'uses_reference_information': False,
            'is_petsc_gamg_reproduction': False,
            'source_api': 'PyAMG5.4.1 public SA/RS/MultilevelSolver/coarse_grid_solver APIs',
            'adapter_revision': 'v7_both_families_verify_null_or_full_SPD',
            **coarse_metadata,
        }
    return {'A': A, 'M': M, 'metadata': metadata, 'krylov': 'cg'}
