"""Strict 2x2 mechanism ablation over the frozen extension MG implementation.

All four cells use the SAME direct seven-point CSR assembly, packed integer
aggregation, fine/coarse sweep counts, coarse stopping threshold, damping
rule, symmetric cycle and physical-coordinate adapter.  Only the two named
mechanisms differ: high-phase face-component labels versus binary phase,
and the existing cutoff-1 positive-coarse-coupling energy lump versus exact
Galerkin coarse construction.  Construction still occurs afresh per call.

This file does not alter extension_multigrid_v2.py or run an experiment on
import.  Root must use the common supervised worker and independent checks.
"""

from extension_multigrid_v2 import build as _frozen_build


_COMMON = {
    'fine_sweeps': 1,
    'coarse_sweeps': 1,
    'coarse_size': 512,
    'lump_start_level': 1,
    'cycle_repeats': 1,
}

CONFIGS = {
    'phase_exact_1_1_c512': dict(
        _COMMON, preserve_components=False, lump_max_relative=None),
    'component_exact_1_1_c512': dict(
        _COMMON, preserve_components=True, lump_max_relative=None),
    'phase_lump_1_1_c512': dict(
        _COMMON, preserve_components=False, lump_max_relative=1.0),
    'component_lump_1_1_c512': dict(
        _COMMON, preserve_components=True, lump_max_relative=1.0),
}

DEFAULT_CONFIG = 'component_lump_1_1_c512'


def _resolve(config):
    """Accept a registered name or one exact registered parameter mapping.

    Free parameter overrides are deliberately excluded: these are four fixed
    ablation cells, rather than another hyperparameter search.
    """
    if config is None:
        name = DEFAULT_CONFIG
    elif isinstance(config, str):
        if config not in CONFIGS:
            raise ValueError('Unknown fixed ablation cell: ' + config)
        name = config
    elif isinstance(config, dict):
        matches = [key for key, value in CONFIGS.items() if config == value]
        if len(matches) != 1:
            raise ValueError('Ablation parameters must exactly match one registered cell')
        name = matches[0]
    else:
        raise TypeError('Ablation config must be a registered name or exact dict')
    return name, dict(CONFIGS[name])


def build(k, config=None, deadline=None):
    """Build one fixed ablation cell using the unchanged numerical kernel."""
    name, parameters = _resolve(config)
    state = _frozen_build(k, config=parameters, deadline=deadline)
    components = parameters['preserve_components']
    lump = parameters['lump_max_relative'] is not None
    # Metadata records the actual two factors and common controls.  The
    # underlying state.config remains exactly the frozen kernel parameters.
    state.setup_meta = dict(state.setup_meta)
    state.setup_meta.update({
        'paper_ablation_revision': 'strict_2x2_v1',
        'paper_ablation_cell': name,
        'paper_ablation_factors': {
            'preserve_high_face_components': components,
            'positive_coarse_energy_lump': lump,
        },
        'paper_ablation_common_controls': dict(_COMMON),
        'paper_ablation_frozen_kernel': 'extension_multigrid_v2.build',
        'paper_ablation_fast_assembly': 'path_multigrid_v3._fine_csr',
        'paper_ablation_integer_aggregation':
            'extension_multigrid_v2._aggregate_components',
        'coordinate_system': 'physical zero-mean pressure quotient',
        'mechanism': (
            ('face-component-preserving' if components else 'binary-phase')
            + ' packed SA; '
            + ('cutoff-1 positive coarse energy lump' if lump else 'exact Galerkin coarse energies')
            + '; common direct fine CSR and symmetric one-sweep V-cycle'
        ),
    })
    state.krylov = 'cg'
    state.coordinate_system = 'physical zero-mean pressure quotient'
    return state
