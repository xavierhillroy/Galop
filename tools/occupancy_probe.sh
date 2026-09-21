#!/usr/bin/env bash
# Achieved occupancy (sm__warps_active % of peak) for both systems' eval
# kernels via Nsight Compute. Needs GPU-counter permissions: run with sudo.
#
# Notes from debugging:
#  - MUST use the CUDA-13 ncu (/usr/local/cuda/bin/ncu). /usr/bin/ncu is a
#    2024 build that predates Blackwell (gb203) -> "Profiling is not supported".
#  - Profile cuda_diff, NOT cuda_bench: ncu kernel replay breaks cudaEvent
#    timing, making cuda_bench's ERR() exit with cudaErrorInvalidValue (11).
#    cuda_diff launches the same two kernels with plain synchronize.
#  - EvoGP: filter to their fitness kernels (treeGP*FitnessKernel); otherwise
#    the capture window catches torch data-prep kernels.
set -uo pipefail
cd "$(dirname "$0")/.."
# Prefer the CUDA-13 ncu locally; on module-based clusters fall back to PATH.
NCU=${NCU:-$( [ -x /usr/local/cuda/bin/ncu ] && echo /usr/local/cuda/bin/ncu || command -v ncu )}
NVCC_BIN=${NVCC_BIN:-$( [ -x /usr/local/cuda/bin/nvcc ] && echo /usr/local/cuda/bin/nvcc || command -v nvcc )}
M=sm__warps_active.avg.pct_of_peak_sustained_active
mkdir -p results/occupancy

[ -x cuda_diff_occ ] || {
echo "[occupancy] building cuda_diff at P=1e5 (saturated regime -- the P=1000
# default measures launch-limited occupancy ~28%, not the design difference)..."
$NVCC_BIN -std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native \
    -DPOPULATION_SIZE_OVERRIDE=100000 \
    tests/cuda_diff.cu src/EvaluatorCuda.cu src/EvaluatorCpu.cpp src/Evaluator.cpp \
    src/Interpreter.cpp src/Dataset.cpp src/LGPEngine.cpp -o cuda_diff_occ >/dev/null 2>&1
}
export GALOP_DIFF_QUICK=1

echo "[occupancy] GALOP warp kernel..."
$NCU --metrics $M -k "regex:^evaluate_kernel$" --launch-count 1 --csv \
    ./cuda_diff_occ > results/occupancy/galop_warp.csv 2>&1

echo "[occupancy] GALOP block kernel..."
$NCU --metrics $M -k "regex:^evaluate_kernel_block$" --launch-count 1 --csv \
    ./cuda_diff_occ > results/occupancy/galop_block.csv 2>&1

echo "[occupancy] EvoGP fitness kernels..."
$NCU --metrics $M --target-processes all -k "regex:.*FitnessKernel" \
    --launch-skip 2 --launch-count 4 --csv \
    .venv/bin/python tools/evogp_eval_once.py > results/occupancy/evogp.csv 2>&1

echo "[occupancy] summary (achieved % of peak warps):"
grep -h "sm__warps_active" results/occupancy/*.csv | head -8 \
    || echo "  (no rows -- check per-file logs in results/occupancy/)"
