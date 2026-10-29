# Reproducing the paper

One entry point: `./reproduce.sh` (no argument = the stage list). Every figure and
every measured number in the paper traces to a stage here and to a result file in
`results/` or `results_h100/`; `RELEASE_MANIFEST.md` lists those files with their
md5 and row counts.

```
./reproduce.sh check      # build + gates (~5 min)          ./reproduce.sh numbers   # all paper numbers (seconds)
./reproduce.sh figures    # fig3 + Table 2, fig5, fig6      ./reproduce.sh datasets  # regenerate + md5-verify datasets
```

## Layout

```
src/ tests/ Makefile golden/        engine, drivers, tests, golden histories
datasets/ tools/gen_*               shipped datasets + their generators
tools/*.sh tools/*.py               local experiment scripts + read-outs (speed/quality/capacity_readout.py)
tools/cluster/*.sbatch              SLURM jobs (one H100 per job)
harness/ tools/baselines/ tools/beagle_adapter/   EvoGP harness; SRGPU, Kozax, Beagle adapters
results/ results_h100/              measured rows (RTX / H100), unedited
report/plot_paper.py                the paper's figures from the CSVs
```

## Hardware and software

- **Local track** (speed grid, quality, CPU anchor, RTX residency and flagship runs):
  one CUDA GPU — paper: RTX 5070 Ti 16 GB on a Ryzen 9 host, `OMP_NUM_THREADS=16`;
  CUDA 13.1 (`nvcc` on PATH, or set `NVCC=`), gcc with OpenMP; Python 3.12 with
  `requirements.txt` (`PYTHON=` overrides the default `.venv/bin/python`);
  torch 2.9.0+cu130 and EvoGP built from source at commit `5538d75` for the EvoGP arms.
  Set-up and what each stage needs: "Running the local track" below.
- **Cluster track** (H100 grid, anchors, anchor quality, capacity): SLURM, one
  NVIDIA H100 80 GB per job — paper: an Alliance (DRAC) cluster, `module load
  StdEnv/2023 python/3.11 cuda/13.2`, host venv `.venv_cluster` (torch from the
  site wheelhouse, EvoGP built from source). Set-up and submission, step by step:
  "Running the cluster track" below.
- **Baselines**: EvoGP at the pinned commit; Kozax 0.1.4 (pip) via `~/.venv_kozax`; SRGPU from
  the retained source snapshot at `~/galop_srgpu_src` (`tools/baselines/setup_srgpu.sh`;
  upstream is no longer public, snapshot on request); Beagle stock checkout at
  `~/beagle_stock` (commit `f96a2d1`, `tools/baselines/setup_beagle.sh`) with the
  adapter installed by `tools/beagle_adapter/install_beagle_adapter.sh`. **No baseline
  engine is modified**: adapters and harnesses only.

## Running the local track

Everything runs on one machine with one CUDA GPU, from the repository root. Each
stage compiles its own binaries with `-arch=native`, i.e. for the GPU that is
present, so there is nothing to configure for the card. The timed stages assume
the machine is otherwise idle.

1. **Toolchain.** A CUDA toolkit whose `nvcc` accepts `-std=c++20` (paper: 13.1),
   gcc with OpenMP (paper: 13.3), `make`, and for the EvoGP build the Python
   headers (`python3-dev` on Debian/Ubuntu). `nvidia-smi` must list the GPU. If
   `nvcc` is not on `PATH` the scripts look in `/usr/local/cuda/bin`; `NVCC=<path>`
   overrides.

2. **Python environment (once).** The analysis path:

   ```
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```

   This is enough for `numbers`, `figures`, `datasets` and `cpu-anchor`. The
   stages with an EvoGP arm (`speed-rtx`, `quality-rtx`, `residency-rtx`) also
   need torch and EvoGP in the same venv: a torch wheel whose CUDA major version
   matches your `nvcc` (paper: 2.9.0+cu130), then EvoGP built from source at the
   pinned commit against that torch:

   ```
   .venv/bin/pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130
   .venv/bin/pip install setuptools wheel ninja
   .venv/bin/pip install --no-build-isolation git+https://github.com/EMI-Group/evogp.git@5538d75470ee0fa444ee6bbe5ea98b2eed6e99f8
   .venv/bin/python -c "import torch, evogp; print(torch.__version__, torch.cuda.is_available())"
   ```

   `PYTHON=<interpreter>` points every stage at another environment.

3. **Gate first.**

   ```
   ./reproduce.sh check
   ```

   Builds the GPU binaries and the CPU tests, runs the CPU-vs-GPU differential
   test, the unit tests and the golden regression (~5 min). `cuda_diff` is the
   binding correctness gate; the golden gate compares against histories recorded
   with the paper's toolchain, and a mismatch there on a different compiler is
   not by itself a defect.

4. **The quick path (seconds).** `./reproduce.sh numbers` and `./reproduce.sh
   figures` recompute every number, table and figure from the shipped CSVs in
   `results/` and `results_h100/`; `./reproduce.sh datasets` regenerates every
   dataset and prints the md5s to compare with the list below (its California
   Housing file is fetched by scikit-learn the first time, so that stage needs
   internet once).

5. **Re-measuring.** Each stage writes one CSV under `results/` and is
   resume-safe: a rerun fills only the missing rows, so a stage run on top of the
   shipped file does nothing. To re-measure, move the shipped CSV aside first.
   `./reproduce.sh local-all` runs every local stage in order (about 15 h).

   | stage | needs | about |
   |---|---|---|
   | `speed-rtx` | GPU idle; `.venv` with torch + EvoGP | 3 h |
   | `host-share-rtx` | GPU idle | 12 min |
   | `quality-rtx` | GPU; `.venv` with torch + EvoGP | 9 h |
   | `effective-length-rtx` | GPU; `results/quality_galop_rtx.csv` (shipped or re-measured) | 40–60 min |
   | `cpu-anchor` | GPU idle; `.venv` (gplearn) | 1 h |
   | `residency-rtx` | GPU; `.venv` with torch + EvoGP | 10 min |
   | `flagship-rtx` | GPU idle; ~22 GB of free host RAM (checked by the script) | 15 min |

6. **Another GPU.** The stages build for whatever card is present, so they run on
   any CUDA GPU; the wall-clocks then belong to that card, not the paper's
   (ratios and orderings are the claims, see "What to expect"), `residency-rtx`
   reports that card's ceiling, and `flagship-rtx` is sized by host RAM
   (P=3.2e7 peaks near 20 GB on the host).

## Running the cluster track

Two facts shape every step. The login node has internet but no GPU; the compute
nodes have a GPU but no internet. So everything that downloads happens once on
the login node, and everything that needs the GPU (compiling the CUDA binaries
with `-arch=native`, every measurement) happens inside a job. Each job compiles
its own binaries on its node, so nothing GPU-side is built by hand. (`./reproduce.sh
check` is a local-GPU stage: on a login node it stops with "no CUDA devices".)

1. **Set up (login node, once, ~15 min).** From the root of a fresh clone:

   ```
   bash tools/cluster/setup_login.sh sm_90        # sm_80 for an A100
   ```

   This loads `StdEnv/2023 python/3.11 cuda/13.2`, creates `.venv_cluster` with
   torch and scipy from the site wheelhouse and EvoGP built from source at the
   pinned commit `5538d75`, compiles the CPU-side tests and dataset generators,
   and runs the golden gate. It is all the GALOP and EvoGP jobs need:
   `speed-h100`, `breakdown-h100`, `residency-h100`, the `galop` / `cap50` /
   `evogp` stages of `capacity-h100` and the `galop` stage of `scale-h100`.

2. **Anchor systems (login node, once; skip unless you run `anchors-h100`,
   `quality-h100`, the `beagle` stage of `scale-h100`, `beagle_wall`, or the
   `kozax` / `beagle` / `srgpu` stages of `capacity-h100`).** The jobs expect the
   three installs at fixed paths under `$HOME`:

   ```
   # Kozax 0.1.4: the jobs activate ~/.venv_kozax under python/3.13 + cuda/12.6
   module load StdEnv/2023 python/3.13 cuda/12.6
   python -m venv ~/.venv_kozax && source ~/.venv_kozax/bin/activate
   pip install --no-index jax jaxlib jax_cuda12_plugin jax_cuda12_pjrt jaxtyping optax sympy
   pip install kozax==0.1.4 --no-deps
   deactivate

   # Beagle: the stock checkout at commit f96a2d1 plus the adapter (engine untouched)
   curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --channel 10.0
   export PATH=$HOME/.dotnet:$PATH DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
   git clone https://github.com/Noblis/beagle-v1.x ~/beagle_stock && git -C ~/beagle_stock checkout f96a2d1
   bash tools/beagle_adapter/install_beagle_adapter.sh ~/beagle_stock

   # SRGPU: upstream is no longer public. Put the retained snapshot (on request,
   # commit e7f0eb8) at ~/galop_srgpu_src; the jobs patch and build it themselves.
   ```

3. **Allocation and partition (login shell, each session).** The job files carry
   neither. If your cluster requires them, export them before submitting; SLURM
   reads both from the environment, so no job file is edited:

   ```
   export SBATCH_ACCOUNT=def-yourprof             # and SBATCH_PARTITION=<partition> if needed
   ```

4. **Submit (login node, repository root).** A stage submits one or more jobs;
   `STAGE=` runs one part of a job (the parts are listed in the stage table and at
   the top of each job file):

   ```
   ./reproduce.sh speed-h100                                        # one stage
   ./reproduce.sh h100-all                                          # every cluster job
   sbatch --export=ALL,STAGE=grid tools/cluster/speed_h100.sbatch   # one part of one job
   ```

   Every GALOP job first compiles its binaries on its node and runs the quick
   `cuda_diff` gate, then measures. Rows are appended to `results_h100/*.csv`;
   SLURM output goes to `results_h100/*_slurm_<jobid>.log` / `.err`. Producers are
   resume-safe: a job that times out or is killed is simply resubmitted, and a
   stage submitted on top of the shipped CSV does nothing. To re-measure, move the
   shipped CSV aside first.

5. **Read the results.** `./reproduce.sh numbers` and `figures` run wherever the
   CSVs are: at home after `rsync -av <cluster>:<clone>/results_h100/ results_h100/`,
   or on the login node with `PYTHON=.venv_cluster/bin/python` (`figures` also
   needs `pip install --no-index matplotlib` in that venv). The jobs append to
   tracked files in the clone: copy them out before any `git reset --hard` there,
   and never reset while a job runs.

6. **Another cluster.** Three places to adapt. In `tools/cluster/setup_login.sh`,
   the module line and the `--no-index` installs (the Alliance wheelhouse):
   elsewhere drop `--no-index` and install a torch that matches the CUDA module.
   In each job file, the `module load` lines and `#SBATCH --gres=gpu:h100:1`
   (whatever your site calls its H100s). And the resource requests: most jobs ask
   for 16 CPUs and 250 GB; `breakdown` 64 GB, `residency` 8 CPUs / 120 GB,
   `capacity` and `beagle_wall` 1000 GB, `scale` 48 CPUs (its optional `p1e9` /
   `pcap` stages are submitted with `--mem=600G` / `--mem=1000G`). Match them to a
   node you can get; the capacity criterion itself is defined at 1 TB of host
   memory, so a smaller node changes which wall that stage reports.

## Stages

| stage | runs | writes | feeds |
|---|---|---|---|
| `check` | build; `cuda_diff` (GPU vs CPU oracle); `lgp_test`; `tools/check_golden.sh` | — | the validation sentence (Sec. 3) |
| `numbers` | `tools/speed_readout.py`, `quality_readout.py`, `capacity_readout.py` | stdout | every number in Sec. 5: the pair grids, the D extension, the forced-hybrid arm, the anchor grids, the CPU sentence; the 22-cell verdicts, solve rates, the sensitivity arm, Nguyen 9/10/12, the anchor quality medians; the ladder, gaps, residency, whole runs and GALOP's program sizes (Table 3: `tools/footprint/`) |
| `figures` | `report/plot_paper.py` | `report/figures/paper_short/` | fig3 + Table 2, fig5, fig6 |
| `datasets` | the generators | `datasets/**` + md5s | Sec. 4.2 |
| `speed-rtx` | `tools/speed_grid.sh` | `results/speed_grid_rtx.csv` | fig3a, Table 2 |
| `quality-rtx` | `tools/quality_galop.sh`, `quality_evogp.sh`, `quality_sensitivity.sh` | `results/quality_galop_rtx.csv`, `quality_evogp_rtx.csv`, `quality_sensitivity_rtx.csv` | fig5, Sec. 5.2 |
| `host-share-rtx` | `tools/host_share.sh` | `results/host_share_rtx.csv` | where a run's time goes as P grows: the bench's timers split the evaluator call into host packing, buffer growth, H2D upload, kernel (launch to synchronize) and D2H download, and the host work outside it into top-k selection, elite copy, selection/variation and statistics; D=32 and 1,024 at P=1e3-1e6 and D=32,768 at P=1e5, five seeds (Sec. 5.1) |
| `breakdown-h100` | `tools/cluster/breakdown_h100.sbatch` (the same producer on one H100; `DLIST="32 1024 32768 131072"` adds the grid's fourth D) | `results_h100/breakdown_h100.csv` | the same split on the H100 at D in {32, 1,024, 32,768} x P in {1e3, 1e4, 1e5, 1e6}, five seeds (Sec. 5.1) |
| `effective-length-rtx` | `tools/effective_length.sh` | `results/effective_length_rtx.csv` | effective program length per workload (Sec. 5.3); reruns the P=1e5 quality cells with the intron-removed layout and requires each run to reproduce its quality row |
| `cpu-anchor` | `tools/cpu_anchor.sh` | `results/cpu_anchor_rtx.csv` | the gplearn sentence |
| `residency-rtx` | `tools/residency_rtx.py` | `results/residency_rtx.csv` | 448 M on the RTX |
| `flagship-rtx` | `tools/scale_rtx.sh` | `results/scale_rtx.csv` | 151.5 s at P=3.2e7 |
| `speed-h100` | `tools/cluster/speed_h100.sbatch` (STAGE = grid, redo, boundary, hybrid, modes) | `results_h100/speed_grid_h100.csv`, `speed_boundary_h100.csv`, `speed_hybrid_h100.csv`, `speed_evogp_modes_h100.txt` | fig3b/c, Table 2, Sec. 5.1 |
| `anchors-h100` | `tools/cluster/anchors_h100.sbatch` (STAGE = srgpu, kozax, beagle) | `results_h100/anchors_grid_h100.csv` | anchor ratios |
| `quality-h100` | `tools/cluster/quality_h100.sbatch` (STAGE = srgpu, kozax, beagle) | `results_h100/quality_anchors_h100.csv`, `quality_beagle_h100.csv` | anchor quality sentences |
| `capacity-h100` | `tools/cluster/capacity_h100.sbatch` (STAGE = evogp, kozax, beagle, srgpu, galop, cap50; `cap50` = both GALOP ladders with every program at the 50-instruction cap; `sustain` = the intron-removed ceiling rung run for ten true generations, never part of `all`, submitted with `--time=08:00:00`; `pinit` = one generation at the ceiling with the shipped serial initializer and with the `-DGALOP_PARALLEL_INIT` build flag (initialization time only; the flag's population is a different random population and is never mixed with the shipped results); `cap50thr` = the at-cap intron-removed ladder with `THREADS` host threads instead of the criterion's 16, a labelled robustness row, submitted with `--cpus-per-task=$THREADS`), `residency_h100.sbatch`, `scale_h100.sbatch` (STAGE = galop, beagle, p1e9 = P=1e9 x 100 generations, never part of `all`, submitted with `--mem=600G`; pcap = the same run at the intron-removed ceiling 2^31-2, one seed, `--mem=1000G`, may end in a device OOM by design), `beagle_wall_h100.sbatch` | `results_h100/capacity_h100.csv`, `residency_h100.csv`, `scale_h100.csv`, `beagle_wall_h100.*` | fig6, Sec. 5.3 |

`local-all` and `h100-all` run or submit every stage of a track. Every producer is
resume-safe (per-row guards): rerunning fills only missing rows, so a stage run on top of
the shipped file does nothing. To re-measure a stage, move its shipped CSV aside first; the
producer then rebuilds it row by row and the readouts print from whichever file is in place
(`tools/speed_grid.sh` and `tools/host_share.sh` also accept `SPEED_CSV` / `HS_CSV` for a
different output path). Cluster jobs take
`STAGE=` (via `sbatch --export=ALL,STAGE=...`) to run one part. The shipped result files
were measured in several sessions of these producers; each capacity row carries its node
and date, and the two speed-grid cells re-measured in a dedicated session carry
`session=redo`.

## The protocol (applies to every timed cell)

Adaptive warm-up (≥2 s cumulative, ≤3 runs, discarded), then **5 timed runs (seeds
70–74** where the system is seedable; independent repetitions otherwise), single
session and device per job, **medians** reported. A cell whose spread
(max−min)/median exceeds 25% is not measurable (n.m.); the two H100 P=10³ cells that
stalled on a shared node were re-measured in a dedicated session (`STAGE=redo` of the
speed job) and the re-measurement is reported; for the anchors such cells are excluded. Anchor grids use
probe-timeout discipline: outcomes `OK / TIMEOUT / OOM_DEVICE / OOM_HOST / CRASH /
unsupported` — every cell ends as a number or an attributed cause. Timers: GALOP
brackets its evolution loop (initial population included); EvoGP its pipeline run;
Beagle its own log clock at generation 100; SRGPU its library-reported regression
time; Kozax process wall-clock including JAX start-up and compilation.

Quality: n=100 seeds (0–99) per cell, two-sided Mann–Whitney U with Holm correction
within five predeclared families (per-P Nguyen ×3, California, Feynman) and
Vargha–Delaney A12 (`tools/matched_verdicts.py` prints the scoreboard); solve = test
MSE < 1e-12. All quality is scored by our evaluator on held-out test CSVs, except
Beagle, whose best genomes are scored by Beagle's own VM through the adapter.

Capacity: one criterion for every system — complete one full generation of its own
pipeline on the shared D=64 quartic under one 80 GB H100, 1 TB host memory and a
20-minute cap per rung; doubling ladder with finer steps near each wall, one attempt
per rung, every ladder end recorded. The device-residency probe (`tests/cuda_capacity.cu`)
is a separately labelled fact: allocation plus one evaluation pass, no evolution.

## Workloads and dataset md5s

Speed and capacity use the Nguyen-2 quartic x⁴+x³+x²+x sampled U[−1,1] (CPython MT,
seed 1234) at each D. **Legacy filenames: `datasets/gridsynth/nguyen1_D*.csv` and
`datasets/sweep/nguyen1_D*.csv` contain that quartic — verify by md5, not by name.**
Quality uses `datasets/sweep` (Nguyen 1–6, U[−1,1], 100 train / 200 test),
`datasets/big` (California Housing 80/20 standardized on train; Feynman I.9.18
sampled U[1,2]/U[3,4], 1024/2048) and `datasets/ext` (Nguyen 9/10/12, U[0,1],
100/200). The golden gate uses the root `datasets/nguyen*_{train,test}.csv`.

```
a800e5ff3e2969e2a75fdc980d373eae  gridsynth/nguyen1_D32_train.csv      0d3f677062f8c9c7f8dcb07c4e095ffd  gridsynth/nguyen1_D64_train.csv
5d35560081531d0746f799a37f98f0ee  gridsynth/nguyen1_D1024_train.csv    cc509c8d01a94eaf8f293b10033a3fe3  gridsynth/nguyen1_D32768_train.csv
7873e91655f844d79477a45763558487  sweep/nguyen1_train.csv   9f936b4b10d26edceecdc15ad909db70  sweep/nguyen1_test.csv
d4f256083bd1aabad299cd6b85f7dcbe  sweep/nguyen2_train.csv   d4ac21ec9c81347b233a1c6f2f627831  sweep/nguyen2_test.csv
77e03d29b9108e77560600d081ed125e  sweep/nguyen3_train.csv   790b85e660d60c9bd6b4fb0bc8e43e23  sweep/nguyen3_test.csv
271f2fe0d93ccbb99dc9537a486170da  sweep/nguyen4_train.csv   e2056613c947e983915785243d43f1a8  sweep/nguyen4_test.csv
6f2b238d386dc91b49b75e7f2de1f33f  sweep/nguyen5_train.csv   1e59b817c51f247151fafba0fde93ebd  sweep/nguyen5_test.csv
6fe0cd599ccff8a40b03ff54d30a025f  sweep/nguyen6_train.csv   aeebe2890af920a2d818838bf6a58831  sweep/nguyen6_test.csv
83a6f7011b790fa61e3f88d4f9c7ffae  big/california_train.csv  5bd3a5ef8484422bdcf348b17ec457cd  big/california_test.csv
1a6c28523882418762c3d14442907bd4  big/feynman_i_9_18_train.csv  25143006ef73f991d54e01ad3d77b37e  big/feynman_i_9_18_test.csv
f51bf11f11cba5d459f52325cccb3cff  ext/nguyen9_train.csv     36552b2bc45926da9f7f10baa537b5da  ext/nguyen9_test.csv
ba01499e506ba1c6eed44e7786240527  ext/nguyen10_train.csv    7bcf4f5c1e69da3d8b6e51766452b6b9  ext/nguyen10_test.csv
569121cb6c06826e84d638c5aa118182  ext/nguyen12_train.csv    81d408acd09d094ce568c3f8e65b5450  ext/nguyen12_test.csv
```

## What to expect from a rerun

| what | expectation |
|---|---|
| GALOP fitness values | **bit-exact** (deterministic per-pair PCG32 at any host thread count; the golden gate enforces it) |
| dataset files | bit-exact (md5s above) |
| matched-P verdicts / A12 | statistical: the aggregate is a near-tie with no direction flips; boundary cells (p ≈ .05) may cross the significance line on a fresh EvoGP draw |
| EvoGP run values | not seed-stable across driver / torch versions (a measured property) — compare distributions, not seeds |
| timing medians | within a few percent session to session (RTX ±5–10% across sessions; H100 0.95–1.08× across sessions); ratios and orderings are the claims, not digits |
| ceilings / walls | same wall class at the same or adjacent rung |
| Beagle | distributions only (no seed control, by its design) |
