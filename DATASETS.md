# Data sources and reproduction scope

The repository includes our synthetic field realization, synthetic generators, solver source, parameters, numerical measurements and plotting source. It does **not** redistribute SPE10/Brugge input arrays, derived reservoir geometry, or the frozen reservoir state files. The MIT license applies to the authors' code; it does not grant rights to third-party data.

## SPE10

The original input source is [OPM/opm-data, spe10model2](https://github.com/OPM/opm-data/tree/2e810769abfc001ca2cf32d1b674c4bce7a918cb/spe10model2), revision `2e810769abfc001ca2cf32d1b674c4bce7a918cb`. The files are `SPE10MODEL2_PERM.INC` and `SPE10MODEL2_PHI.INC`. Consult that provider for applicable terms and attribution. The permeability download SHA-256 is `1829e331b64d2efb416fc40f4d68388087a11e4f7ff5ac6866eb8f1a88f8993d`.

After obtaining the source inputs in accordance with the provider's terms, the existing preparation routines download/parse the pinned files and create the local NumPy arrays:

```sh
cd workspace
python -c "import spe10_problem; spe10_problem.prepare()"
python -c "import spe10_dynamic_problem; spe10_dynamic_problem.prepare()"
cd ..
python reproduce.py benchmark --suite static --method ML-CPMG
```

The dynamic pressure suite additionally requires the frozen saturation states at 520.04 and 2000 days. The zero-day state is initialized in the benchmark. Downloading permeability and porosity alone does not reconstruct these frozen states.

## Brugge

The source is [TNO/Brugge](https://github.com/TNO/Brugge/tree/4e282a11ab7ba4794bebcb639ee6632d5afe48e7), revision `4e282a11ab7ba4794bebcb639ee6632d5afe48e7`, archive `BRUGGE60K_FY-SF-KM-1-1_JAN2017.tar.gz`, SHA-256 `e32b7efeea1fa1f77638876d0611d4fc4a2e4aae3d6dc532fa876a3c6f14086b`. Acknowledge TNO and consult its original README and data-use terms, including its noncommercial-use condition.

The manuscript uses this single synthetic realization, with 43,474 active cells, 122,543 graph edges and 152 NNCs. Our graph and pressure code is included. The processed graph/rock input and frozen states at 0, 1000 and 2000 days are separate external inputs. This public repository is not a complete reservoir-state generation pipeline.

## Exact external inputs

`external_inputs.json` lists each required file, its path relative to `workspace/`, byte count and SHA-256. Existing holders of the original reproducibility inputs can place them at those paths and verify their checksums before using:

```sh
python reproduce.py benchmark --suite dynamic --method ML-CPMG
python reproduce.py benchmark --suite brugge --method ML-CPMG
python reproduce.py figures --only spe10 brugge
```

The last command requires the two named display arrays and produces Figures 7–9. Source downloads alone do not supply the processed display arrays. The manifest records input identity, not a public download endpoint or a promise of unrestricted access. Request any unavailable processed inputs from the manuscript authors, subject to the original providers' terms.

The archived solver measurements and table/summary plots remain reproducible from this repository without these external inputs. They should not be described as newly rerun reservoir experiments.
