#!/usr/bin/env bash
# =============================================================================
# Achieved occupancy of Galop's warp kernel as the population grows (Sec. 5.1:
# does the shrinking small-D speedup come with falling occupancy?). Builds the
# CPU-GPU differential test at P = 1e3, 1e4, 1e5, 1e6 (D=32, quick phase) and
# profiles the warp kernel's first launch with Nsight Compute, as
# tools/occupancy_probe.sh does for the block-vs-warp comparison at P=1e5.
#   -> results/occupancy/galop_warp_P<P>.csv   (Nsight CSV; metric line = the value)
# Needs GPU performance-counter permissions: run with sudo, or enable
# NVreg_RestrictProfilingToAdminUsers=0. Uses the CUDA-13 ncu (see occupancy_probe.sh).
#   sudo bash tools/occupancy_vs_p.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
NCU=${NCU:-$( [ -x /usr/local/cuda/bin/ncu ] && echo /usr/local/cuda/bin/ncu || command -v ncu )}
NVCC_BIN=${NVCC_BIN:-$( [ -x /usr/local/cuda/bin/nvcc ] && echo /usr/local/cuda/bin/nvcc || command -v nvcc )}
M=sm__warps_active.avg.pct_of_peak_sustained_active
mkdir -p results/occupancy
export GALOP_DIFF_QUICK=1
for P in 1000 10000 100000 1000000; do
  bin=/tmp/cuda_diff_occ_$P
  [ -x $bin ] || $NVCC_BIN -std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native -DPOPULATION_SIZE_OVERRIDE=$P \
      tests/cuda_diff.cu src/EvaluatorCuda.cu src/EvaluatorCpu.cpp src/Evaluator.cpp src/Interpreter.cpp src/Dataset.cpp src/LGPEngine.cpp -o $bin >/dev/null 2>&1 \
      || { echo "build failed at P=$P"; exit 1; }
  $NCU --metrics $M -k "regex:^evaluate_kernel$" --launch-count 1 --csv $bin > results/occupancy/galop_warp_P$P.csv 2>&1
  v=$(grep "sm__warps_active" results/occupancy/galop_warp_P$P.csv | tail -1 | awk -F'","' '{print $NF}' | tr -d '"')
  grep -q "ERR_NVGPUCTRPERM" results/occupancy/galop_warp_P$P.csv && { echo "no GPU-counter permission: run with sudo"; rm -f results/occupancy/galop_warp_P$P.csv; exit 1; }
  echo "  P=$P: achieved occupancy ${v:-?}%"
done
echo "OCCUPANCY VS P COMPLETE -> results/occupancy/galop_warp_P*.csv"
