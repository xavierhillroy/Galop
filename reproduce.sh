#!/usr/bin/env bash
# =============================================================================
# GALOP paper reproduction — the one entry point.
#   ./reproduce.sh              list the stages
#   ./reproduce.sh <stage>      run a local stage / submit the SLURM jobs of a stage
# Quick path (seconds, no experiments): numbers, figures — every number and
# figure in the paper recomputed from the shipped result CSVs.
# Local stages need one CUDA GPU (paper: RTX 5070 Ti) and the Python env of
# requirements.txt (PYTHON=... overrides .venv/bin/python; NVCC=... overrides
# the compiler). Cluster stages submit SLURM jobs (paper: one 80 GB H100 per
# job): run them on the login node from the repo root, and edit the
# module/venv lines at the top of a job for another cluster.
# Every stage is resume-safe: rerunning fills only missing rows.
# Pins, protocol, outputs and expectations: REPRODUCE.md
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}

declare -A LOCAL SLURM DESC
order=(check numbers figures datasets
       speed-rtx host-share-rtx quality-rtx effective-length-rtx cpu-anchor residency-rtx flagship-rtx local-all
       speed-h100 breakdown-h100 anchors-h100 quality-h100 capacity-h100 h100-all)

LOCAL[check]='make gpu && make all && GALOP_DIFF_QUICK=1 ./cuda_diff | tail -1 && ./lgp_test | tail -1 && bash tools/check_golden.sh | tail -1'
DESC[check]='build + CPU-GPU differential test + unit tests + golden regression (~5 min; run first)'
LOCAL[numbers]="python3 tools/speed_readout.py && $PY tools/quality_readout.py && python3 tools/capacity_readout.py"
DESC[numbers]='every number in the paper, recomputed from the shipped CSVs (seconds)'
LOCAL[figures]="$PY report/plot_paper.py && ls report/figures/paper_short/"
DESC[figures]='fig3 + Table 2, fig4, fig5, fig6 from the shipped CSVs (seconds)'
LOCAL[datasets]="make gen_datasets gen_sweep_datasets && ./gen_datasets && ./gen_sweep_datasets && $PY tools/gen_big_datasets.py && $PY tools/gen_ext_datasets.py && md5sum datasets/*.csv datasets/sweep/nguyen[1-6]_*.csv datasets/big/*.csv datasets/ext/*.csv datasets/gridsynth/*.csv"
DESC[datasets]='regenerate every dataset and print md5s (must match REPRODUCE.md; ~2 min)'

LOCAL[speed-rtx]='bash tools/speed_grid.sh'
DESC[speed-rtx]='the RTX speed grid, both arms (fig3a, Table 2; ~3 h)'
LOCAL[quality-rtx]='bash tools/quality_galop.sh && bash tools/quality_evogp.sh && bash tools/quality_sensitivity.sh'
DESC[quality-rtx]='GALOP arms + EvoGP references + Nguyen 9/10/12 + sensitivity arm, n=100 (fig5; ~9 h)'
LOCAL[cpu-anchor]='bash tools/cpu_anchor.sh'
DESC[cpu-anchor]='GALOP and gplearn at the CPU-sentence cells (~1 h)'
LOCAL[residency-rtx]="make gpu && $PY tools/residency_rtx.py"
DESC[residency-rtx]='device-residency probe on the local GPU (~10 min)'
LOCAL[flagship-rtx]='bash tools/scale_rtx.sh'
DESC[flagship-rtx]='P=3.2e7 x 100 generations on the local GPU (~15 min)'
LOCAL[host-share-rtx]='bash tools/host_share.sh'
DESC[host-share-rtx]='time split of the D=32 and D=1,024 cells across P and of D=32,768 at P=1e5: evaluator call (packing, buffer growth, H2D, kernel, D2H) and host work by phase, five seeds (Sec. 5.1; ~12 min)'
LOCAL[effective-length-rtx]='bash tools/effective_length.sh'
DESC[effective-length-rtx]='effective program length on the quality workloads at P=1e5, n=100, each run checked against its quality row (Sec. 5.3; needs quality-rtx; ~40 min)'
LOCAL[local-all]='for s in check datasets speed-rtx host-share-rtx quality-rtx effective-length-rtx cpu-anchor residency-rtx flagship-rtx; do ./reproduce.sh $s || exit 1; done'
DESC[local-all]='every local stage, in order'

SLURM[speed-h100]='speed_h100.sbatch'
DESC[speed-h100]='[SLURM] H100 grid + D extension + forced-hybrid arm + execute-mode fingerprint (fig3b/c, Table 2; ~17 h; STAGE=grid|redo|boundary|hybrid|modes)'
SLURM[breakdown-h100]='breakdown_h100.sbatch'
DESC[breakdown-h100]='[SLURM] the same time split on one H100 at D in {32, 1,024, 32,768} x P=1e3-1e6, five seeds (Sec. 5.1; ~35 min; DLIST="32 1024 32768 131072" adds the fourth D, ~75 min more)'
SLURM[anchors-h100]='anchors_h100.sbatch'
DESC[anchors-h100]='[SLURM] SRGPU / Kozax / Beagle anchor grids (~20 h; STAGE=srgpu|kozax|beagle)'
SLURM[quality-h100]='quality_h100.sbatch'
DESC[quality-h100]='[SLURM] SRGPU / Kozax / Beagle quality (~40 h; STAGE=srgpu|kozax|beagle)'
SLURM[capacity-h100]='capacity_h100.sbatch residency_h100.sbatch scale_h100.sbatch beagle_wall_h100.sbatch'
DESC[capacity-h100]='[SLURM] the capacity ladder (all five systems), the residency probe, the P=1e8 runs, the Beagle wall attribution (fig6; ~4.5 h incl. the cap-width arm + ~1 h + ~18 h + ~20 min)'
SLURM[h100-all]="${SLURM[speed-h100]} ${SLURM[breakdown-h100]} ${SLURM[anchors-h100]} ${SLURM[quality-h100]} ${SLURM[capacity-h100]}"
DESC[h100-all]='[SLURM] submit every cluster job'

if [ $# -eq 0 ]; then
  echo "GALOP reproduction stages (details: REPRODUCE.md):"
  for k in "${order[@]}"; do printf "  %-14s %s\n" "$k" "${DESC[$k]}"; done
  exit 0
fi
k=$1
if [ -n "${LOCAL[$k]:-}" ]; then
  echo ">> [$k] ${DESC[$k]}"; eval "${LOCAL[$k]}"
elif [ -n "${SLURM[$k]:-}" ]; then
  echo ">> [$k] ${DESC[$k]}"
  for j in ${SLURM[$k]}; do sbatch "tools/cluster/$j"; done
else
  echo "unknown stage '$k' — run with no argument for the list"; exit 1
fi
