"""Serial RT0/trapezoidal pressure solver on a uniform unit cube.

Unknowns are cell pressures, RHS entries are cell-integrated sources, and
edge fluxes are face-integrated flows oriented from the lower cell index.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.fft import dctn, idctn
from scipy.ndimage import label
from scipy.sparse.linalg import LinearOperator, cg, gmres, splu, spilu
from threadpoolctl import threadpool_limits

METHODS = ['jacobi_cg', 'superlu', 'ilu_gmres', 'dct_cg', 'component_cg']
BASELINES = METHODS[:3]
CANDIDATES = METHODS[3:]


def geometry(n, family, contrast):
    """Fixed physical geometry. Width 1/4; formal n=16 resolves four cells."""
    c = (np.arange(n) + 0.5) / n
    x, y, z = np.meshgrid(c, c, c, indexing='ij')
    if family == 'channels':
        mask = (((abs(y - .5) < .125) & (abs(z - .5) < .125)) |
                ((abs(x - .5) < .125) & (abs(z - .5) < .125)) |
                ((abs(x - .5) < .125) & (abs(y - .5) < .125)))
    elif family == 'inclusions':
        mask = np.zeros((n, n, n), dtype=bool)
        for a in (.25, .75):
            for b in (.25, .75):
                for d in (.25, .75):
                    mask |= (abs(x-a) < .125) & (abs(y-b) < .125) & (abs(z-d) < .125)
    elif family == 'constant':
        mask = np.zeros((n, n, n), dtype=bool)
    else:
        raise ValueError(family)
    return np.where(mask, float(contrast), 1.0)


def edges(k):
    n = k.shape[0]
    ids = np.arange(k.size).reshape(k.shape)
    tail, head, weights = [], [], []
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(lo), tuple(hi)
        tail.append(ids[a].ravel())
        head.append(ids[b].ravel())
        weights.append((2.0 / n / (1 / k[a] + 1 / k[b])).ravel())
    return np.concatenate(tail), np.concatenate(head), np.concatenate(weights)


def assemble(k):
    i, j, t = edges(k)
    rows = np.concatenate((i, j, i, j))
    cols = np.concatenate((i, j, j, i))
    data = np.concatenate((t, t, -t, -t))
    return sparse.coo_matrix((data, (rows, cols)), shape=(k.size, k.size)).tocsr()


def flux(p, edge):
    i, j, t = edge
    return t * (p[i] - p[j])


def divergence(q, edge, size):
    i, j, _ = edge
    return np.bincount(i, weights=q, minlength=size) - np.bincount(j, weights=q, minlength=size)


def action(p, edge):
    return divergence(flux(p, edge), edge, len(p))


def metrics(p, b, edge):
    r = b - action(p, edge)
    norm = np.linalg.norm(b)
    return {'rho': float(np.linalg.norm(r) / norm) if norm else float(np.linalg.norm(r)),
            'eta': float(np.max(abs(r)) / np.max(abs(b))) if norm else float(np.max(abs(r))),
            'mean_pressure': float(np.mean(p))}


def sources(n):
    c = (np.arange(n) + .5) / n
    x, y, z = np.meshgrid(c, c, c, indexing='ij')
    left = (x < .25) & (abs(y-.5) < .125) & (abs(z-.5) < .125)
    right = (x > .75) & (abs(y-.5) < .125) & (abs(z-.5) < .125)
    b = np.zeros((n, n, n))
    b[left] = 1 / left.sum()
    b[right] = -1 / right.sum()
    return b.ravel()


def manufactured(n):
    c = (np.arange(n) + .5) / n
    avg = np.cos(np.pi * c) * np.sinc(1 / (2 * n))
    p = np.einsum('i,j,k->ijk', avg, avg, avg).ravel()
    p -= p.mean()
    return p, 3 * np.pi**2 / n**3 * p


def build_preconditioner(A, k, method):
    N = A.shape[0]
    diagonal = A.diagonal()
    if method == 'jacobi_cg':
        return LinearOperator(A.shape, matvec=lambda r: r / diagonal)
    if method == 'dct_cg':
        n = k.shape[0]
        lamb = 4 / n * np.sin(np.pi * np.arange(n) / (2*n))**2
        eigenvalues = lamb[:, None, None] + lamb[None, :, None] + lamb[None, None, :]
        eigenvalues[0, 0, 0] = 1
        def apply(r):
            transformed = dctn(r.reshape(k.shape), type=2, norm='ortho')
            transformed /= eigenvalues
            transformed[0, 0, 0] = 0
            return idctn(transformed, type=2, norm='ortho').ravel()
        return LinearOperator(A.shape, matvec=apply)
    if method == 'component_cg':
        # A distinct mechanism: resolve each conducting component as a coarse
        # unknown, plus geometric background aggregates, with Jacobi smoothing.
        components, number = label(k > 1)
        n = k.shape[0]
        coords = np.indices(k.shape)
        agg = (coords[0] // 4) * ((n+3)//4)**2 + (coords[1] // 4) * ((n+3)//4) + coords[2] // 4
        tags = np.where(components > 0, components-1, number+agg).ravel()
        _, inverse = np.unique(tags, return_inverse=True)
        count = np.bincount(inverse)
        Z = sparse.csr_matrix((1/np.sqrt(count[inverse]), (np.arange(N), inverse)),
                              shape=(N, len(count)))
        E = (Z.T @ A @ Z).toarray()
        eigenvalues, vectors = np.linalg.eigh((E+E.T)/2)
        # Quotient solve: discard only the one known constant null vector.
        order = np.argsort(eigenvalues)
        keep = order[1:]
        if len(keep) and np.min(eigenvalues[keep]) <= 0:
            raise ValueError('coarse quotient matrix is not positive definite')
        V = vectors[:, keep]
        inverse_coarse = (V/eigenvalues[keep]) @ V.T
        def apply(r):
            out = r/diagonal + Z @ (inverse_coarse @ (Z.T @ r))
            return out-out.mean()
        return LinearOperator(A.shape, matvec=apply)
    raise ValueError(method)


class BudgetStop(Exception):
    pass


def solve(k, b, method, *, maxiter=5000, deadline=None, checkpoint_path=None):
    """One complete construction+solve. deadline is an absolute perf_counter."""
    started = time.perf_counter()
    A = assemble(k)
    edge = edges(k)
    assembled = time.perf_counter()
    trace = []
    last = None
    iterations = 0
    fixed_budget = deadline is not None
    finalization_reserve = 0.025
    checkpoint_serial = 0
    def save_checkpoint(p, q, values):
        nonlocal checkpoint_serial
        saved_p, saved_q = p.copy(), q.copy()
        path = None
        if checkpoint_path:
            checkpoint_serial += 1
            # Immutable files: a late/incomplete write cannot destroy an earlier
            # accepted checkpoint. Only the path selected below is evaluated.
            path = str(checkpoint_path)+f'.checkpoint{checkpoint_serial:05d}.npz'
            np.savez(path, p=saved_p, q=saved_q, rho=values['rho'], eta=values['eta'])
        completed = time.perf_counter()
        return (saved_p, saved_q, values, completed, path) if completed < deadline else None
    def observe(x, force=False):
        nonlocal iterations, last
        iterations += 1
        if force or iterations == 1 or iterations % 10 == 0:
            p = np.asarray(x)-np.mean(x)
            q = flux(p, edge)
            values = metrics(p, b, edge)
            trace.append({'iteration': iterations, 'seconds': time.perf_counter()-started, **values})
            if fixed_budget and values['rho'] <= 1e-9 and values['eta'] <= 1e-9:
                if time.perf_counter() < deadline-finalization_reserve:
                    saved = save_checkpoint(p, q, values)
                    if saved is not None:
                        last = saved
        if fixed_budget and time.perf_counter() >= deadline-finalization_reserve:
            raise BudgetStop()
    p = np.zeros_like(b)
    info = 0
    setup_done = assembled
    with threadpool_limits(limits=1):
        if not np.any(b):
            setup_done = time.perf_counter()
        elif method == 'superlu':
            # Ground one cell, solve the SPD quotient, restore zero mean.
            factor = splu(A[1:, 1:].tocsc())
            setup_done = time.perf_counter()
            p[1:] = factor.solve(b[1:])
            p -= p.mean()
        elif method == 'ilu_gmres':
            reduced = A[1:, 1:].tocsc()
            ilu = spilu(reduced, drop_tol=1e-4, fill_factor=10)
            preconditioner = LinearOperator(reduced.shape, matvec=ilu.solve)
            setup_done = time.perf_counter()
            def callback_reduced(x):
                full = np.zeros_like(b)
                full[1:] = x
                observe(full, force=True)
            try:
                p[1:], info = gmres(reduced, b[1:], M=preconditioner, x0=np.zeros(len(b)-1),
                                    rtol=0 if fixed_budget else 1e-12, atol=0,
                                    maxiter=maxiter, restart=50, callback=callback_reduced,
                                    callback_type='x')
                p -= p.mean()
            except BudgetStop:
                info = -99
        else:
            preconditioner = build_preconditioner(A, k, method)
            project = lambda r: r-r.mean()
            raw_preconditioner = preconditioner
            preconditioner = LinearOperator(A.shape,
                matvec=lambda r: project(raw_preconditioner @ project(r)))
            quotient_operator = LinearOperator(A.shape, matvec=lambda r: project(A @ project(r)))
            setup_done = time.perf_counter()
            try:
                p, info = cg(quotient_operator, project(b), M=preconditioner, x0=np.zeros(len(b)),
                             rtol=0 if fixed_budget else 1e-12, atol=0,
                             maxiter=maxiter, callback=observe)
                p -= p.mean()
            except BudgetStop:
                info = -99
    solve_done = time.perf_counter()
    if fixed_budget:
        p -= p.mean()
        q = flux(p, edge)
        values = metrics(p, b, edge)
        if (time.perf_counter() < deadline-finalization_reserve and
                values['rho'] <= 1e-9 and values['eta'] <= 1e-9):
            saved = save_checkpoint(p, q, values)
            if saved is not None:
                last = saved
        if last is None:
            return None, {'status': 'no_conservation_compliant_checkpoint_before_deadline',
                          'iterations': iterations, 'info': int(info), 'trace': trace}
        p, q, values, checkpoint_completed, accepted_path = last
    else:
        q = flux(p, edge)
        values = metrics(p, b, edge)
        checkpoint_completed = None
        accepted_path = None
    ended = time.perf_counter()
    return (p, q), {'status': 'completed', 'iterations': iterations, 'info': int(info),
                    'iterations_unit': 'restart cycles (at most 50 inner iterations)' if method == 'ilu_gmres' else 'CG iterations',
                    'assembly_seconds': assembled-started,
                    'setup_seconds': setup_done-assembled,
                    'paper_setup_solve_seconds': solve_done-assembled,
                    'inprocess_total_seconds': ended-started,
                    'checkpoint_completed_perf_counter': checkpoint_completed,
                    'accepted_checkpoint_path': accepted_path,
                    'trace': trace, **values}
