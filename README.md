# ML-CPMG

Research implementation and numerical evidence accompanying manuscript **v18** on persistent-component multigrid for high-contrast Darcy pressure equations. The solver supports Cartesian TPFA operators and positive-transmissibility active graphs with non-neighbor connections (NNCs).

The code is released under the [MIT license](LICENSE). Third-party reservoir datasets and Python dependencies retain their own terms; they are not relicensed by this repository.

## Quick start

The archived environment used Python 3.13.9. From the repository root:

```sh
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py smoke
python reproduce.py algebra
python reproduce.py tables
python reproduce.py figures
```

`smoke` solves manufactured Cartesian Neumann and grounded NNC graph problems. `algebra` checks twelve small hierarchies. `tables` regenerates seven numerical tables from archived measurements and copies the fixed protocol-settings table (`configuration.tex`), providing all eight manuscript tables. `figures` regenerates Figures 1–6 and 10. These commands need no reservoir downloads. Outputs are written to `outputs/`, leaving the archived evidence unchanged. Matching figure typography requires locally installed Arial and Times New Roman; fonts are not distributed here.

To run a fresh, conservation-checked 64³ synthetic pressure solve:

```sh
python reproduce.py benchmark --suite synthetic --case Synthetic-smooth_gaussian --method ML-CPMG --repeats 1
```

Omit `--case` to run all six synthetic fields. Omit `--method` to run the primary comparison methods. The default is three repetitions. Reservoir reruns and Figures 7–9 require external inputs described in [DATASETS.md](DATASETS.md); cloning this repository alone does not reproduce those solves or spatial plots.

## Implementation and evidence

| Location | Purpose |
| --- | --- |
| `reproduce.py` | Portable entry points and separate output directories |
| `workspace/results/paper_revision_v16/code/ml_cpmg_v16.py` | Solver used for the v18 numerical results |
| `workspace/multilevel_continuous_media.py` | Continuous synthetic fields |
| `workspace/results/paper_revision_v16/code/benchmark.py` | Common solver, timing and physical-conservation protocol |
| `workspace/results/paper_revision_v16/data/` | 252 timed repetitions, calibration, accuracy and algebraic evidence |
| `workspace/results/paper_revision_v16/build/summarize.py` | Table generation |
| `workspace/results/paper_revision_v18_fig10_median/code/figures.py` | Final v18 plotting source |
| `provenance.json` | Software versions, original timing environment and SHA-256 hashes |
| `external_inputs.json` | Exact identities of omitted reservoir inputs |

Historical directory names are retained so the archived implementation and its relative imports remain unchanged. Use `reproduce.py` rather than the original benchmark command-line driver, whose CPU selection and output paths were specific to the original machine.

## Numerical scope

The archived results used one pinned CPU and one BLAS thread, three sequential repetitions in permuted method order, and a physical-conservation gate of 10⁻⁹. The wrapper preserves the solver and acceptance criterion and chooses an available CPU; use `--cpu` to specify one. New wall times depend on hardware and software. Regenerating tables from archived JSON is distinct from rerunning numerical experiments.

SPE10 and Brugge comparisons solve pressure equations at prescribed, frozen waterflood states. They are not independently advanced ML-CPMG transport trajectories. No field-production data are included.

## Citation

This repository accompanies the authors' manuscript v18. Until publication metadata are available, cite the repository URL and the specific Git commit used. No journal acceptance, DOI or archived dataset DOI is implied.
