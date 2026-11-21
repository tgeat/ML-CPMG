# ML-CPMG

The `main` branch accompanies manuscript **v20** on multilevel persistence component-preserving multigrid for heterogeneous Darcy pressure systems. The implementation supports Cartesian TPFA and faulted corner-point active graphs with non-neighbour connections. The v18.1 tag preserves the earlier Brugge experiments. [v18.1 source](https://github.com/tgeat/ML-CPMG/tree/v18.1)

## Quick start

The archived experiments used Python 3.13.9. Create and activate a Python 3.13 virtual environment, then run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python run.py smoke
python run.py exp6
python run.py tables
python run.py figures
```

The smoke tests cover a Cartesian Neumann system and a grounded active graph with a non-neighbour connection. Experiment 6 runs 16 checks for prolongator rank, symmetry and V-cycle contraction. The table command generates all eight manuscript tables under `outputs/tables/`. The figure command generates plots under `outputs/figures/` and skips the SPE10 and OLYMPUS spatial plates if their display arrays are unavailable. These scripts regenerate numerical plots and schematics. The final v20 manuscript uses separate figure typesetting. Matching the plotting fonts requires locally installed Arial and Times New Roman.

Run a fresh synthetic pressure solve with the conservation checks enabled.

```sh
python run.py exp2 --case Synthetic-smooth_gaussian --method ML-CPMG --repeats 1
```

## Experiments and archived results

| Command | Scope |
| --- | --- |
| `python run.py exp1` | SPE10 parameter calibration |
| `python run.py exp2` | Six synthetic permeability fields |
| `python run.py exp3` | SPE10 static five-spot pressure |
| `python run.py exp4` | SPE10 pressure at three prescribed mobility states |
| `python run.py exp5` | OLYMPUS_1 pressure in three fault configurations |
| `python run.py exp6` | Small-system algebraic and spectral checks |

The solver is under `src/ml_cpmg/`. Benchmark, table and plotting scripts are under `experiments/`. Archived records under `data/results/` cover 252 timed solves, 16 algebraic checks, 60 Cartesian interval certifications and 24 direct-solution comparisons on OLYMPUS_1. The timing protocols use one logical CPU where affinity is available and limit numerical libraries to one thread. Rerun timings depend on the machine. SPE10 and OLYMPUS experiments operate on prescribed pressure systems, including frozen waterflood states.

Use `python run.py --help` for the full command list. Generated outputs go under `outputs/`. Reservoir reruns and fresh accuracy certification require local inputs or checkpoints. [Data sources and instructions](DATASETS.md)

The source code is provided under the [MIT license](LICENSE). Cite the manuscript and the Git commit used. The reservoir inputs retain their original terms and attribution. [Dataset attribution](DATASETS.md)
