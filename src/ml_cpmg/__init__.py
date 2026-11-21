"""ML-CPMG: Multi-Level Persistence Component-Preserving Multigrid for Darcy flow."""
from .grid import ActiveGraph, action, edges, sources
from .solver import (
    DEFAULT_CONFIG,
    PARAMETER_DOMAIN,
    PHASE_MODES,
    Multigrid,
    assemble_cartesian_fast,
    build,
    build_active_graph,
    build_rectangular,
    extract_multilevel_components,
    solve,
)

__all__ = [
    "ActiveGraph",
    "DEFAULT_CONFIG",
    "Multigrid",
    "PARAMETER_DOMAIN",
    "PHASE_MODES",
    "action",
    "assemble_cartesian_fast",
    "build",
    "build_active_graph",
    "build_rectangular",
    "edges",
    "extract_multilevel_components",
    "solve",
    "sources",
]
