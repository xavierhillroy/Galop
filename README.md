# Galop — warp-per-individual GPU linear genetic programming

Galop is a hybrid CPU–GPU linear genetic programming (LGP) system built around one
design decision: **one individual per CUDA warp**. The 32 lanes of a warp execute the
same program over 32 fitness cases in lockstep, population parallelism is exposed
across warps, and selection and variation stay on the CPU. This repository contains
the system, the experiment scripts, the measured results and the figure code behind
the paper *GALOP: warp-level evaluation for population-scale genetic programming on
GPUs* (EuroGP 2027).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./reproduce.sh check      # build + CPU–GPU differential test + unit tests + golden regression (~5 min, needs CUDA)
./reproduce.sh numbers    # every number in the paper, recomputed from the shipped result CSVs (seconds)
./reproduce.sh figures    # the paper's figures and its table, from the same CSVs (seconds)
```

`./reproduce.sh` with no argument lists every stage, including the ones that re-run
the experiments (local GPU stages and SLURM jobs for an H100). All stages are
resume-safe. Details, software pins, the measurement protocol and what to expect
from a rerun are in [REPRODUCE.md](REPRODUCE.md).

## What is here

| path | content |
|---|---|
| `src/`, `tests/`, `Makefile` | the engine (CUDA evaluator, ISA, evolution loop, configuration), the whole-run driver, the differential test, the residency probe, the unit tests |
| `golden/`, `tools/check_golden.sh` | golden CPU regression histories and the gate that diffs against them |
| `datasets/`, `tools/gen_*` | every dataset used, plus the deterministic generators that produced them (md5s in REPRODUCE.md) |
| `tools/*.sh`, `tools/*.py` | the local experiment scripts (RTX 5070 Ti) and the three read-outs that print the paper's numbers |
| `tools/cluster/*.sbatch` | the SLURM jobs (one H100 80 GB per job) |
| `harness/`, `tools/baselines/`, `tools/beagle_adapter/` | how the baselines were driven: the EvoGP harness and the SRGPU, Kozax and Beagle adapters. Baseline engines are fetched at pinned commits and never modified |
| `results/`, `results_h100/` | the measured rows every figure and number is computed from, unedited (`RELEASE_MANIFEST.md` lists each file with its md5 and row count) |
| `report/plot_paper.py`, `report/figures/` | the figure code and the rendered figures |

## Baselines

EvoGP (commit `5538d75`, built from source), Beagle (Noblis `beagle-v1.x`, commit
`f96a2d1`, with the adapter in `tools/beagle_adapter/`), SRGPU (from a retained
source snapshot; the upstream repository is no longer public — available on request)
and Kozax 0.1.4 (pip). Configuration parity and the anchor protocol are described in the
paper and in REPRODUCE.md.

## License

See [LICENSE](LICENSE).
