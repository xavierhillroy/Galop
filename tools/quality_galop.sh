#!/usr/bin/env bash
# =============================================================================
# GALOP quality runs (fig5, Sec. 5.2): 100 seeded runs per cell, G=100, best-of-run
# train/test MSE scored by the engine on the held-out test file.
#   Nguyen 1-6 x P in {1e3,1e4,1e5}: 8 registers (single-input arity rule)
#   California, Feynman I.9.18 x P in {1e4,1e5}: 16 registers (multi-input rule)
#   Nguyen 9, 10, 12 x P in {1e3,1e4,1e5}: 16 registers (descriptive, not in the
#   inferential families)
# Standard configuration throughout (INPUT_PROTECT, 3 macro inserts). Quality runs
# make no timing claims and tolerate a busy desktop. Resume-safe per (problem,P,seed).
#   bash tools/quality_galop.sh          (NVCC=, QUALITY_NSEEDS= optional)
# -> results/quality_galop_rtx.csv   problem,P,seed,train_mse,test_mse
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=16
NVCC=${NVCC:-$( [ -x /usr/local/cuda/bin/nvcc ] && echo /usr/local/cuda/bin/nvcc || command -v nvcc )}
SRCS="tests/cuda_evo_bench.cu src/EvaluatorCuda.cu src/Dataset.cpp src/LGPEngine.cpp src/EvaluatorCpu.cpp src/Evaluator.cpp src/Interpreter.cpp"
NVF="-std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native -Xcompiler -fopenmp"
STD8="-DINPUT_PROTECT_OVERRIDE=1 -DMACRO_INSERTS_OVERRIDE=3"
STD16="-DNUM_REGISTERS_OVERRIDE=16 -DINPUT_PROTECT_OVERRIDE=1 -DMACRO_INSERTS_OVERRIDE=3"
NSEEDS=${QUALITY_NSEEDS:-100}; LAST=$((NSEEDS-1))
CSV=results/quality_galop_rtx.csv
mkdir -p results; [ -f $CSV ] || echo "problem,P,seed,train_mse,test_mse" > $CSV
python3 tools/gen_ext_datasets.py >/dev/null
run_cells() {   # FLAGS P dir problems...
  local FLAGS=$1 P=$2 dir=$3; shift 3
  local bin=/tmp/ceb_quality_${P}_$(echo "$FLAGS" | md5sum | cut -c1-6)
  for t in "$@"; do
    grep -q "^$t,$P,$LAST," $CSV 2>/dev/null && { echo "  galop $t P=$P SKIP"; continue; }
    [ -x $bin ] || $NVCC $NVF $FLAGS -DPOPULATION_SIZE_OVERRIDE=$P -DMAX_GENERATIONS_OVERRIDE=100 $SRCS -o $bin
    for s in $(seq 0 $LAST); do
      grep -q "^$t,$P,$s," $CSV && continue
      line=$($bin datasets/$dir/${t}_train.csv $s warp datasets/$dir/${t}_test.csv 2>/dev/null)
      [ -n "$line" ] || { echo "  galop $t P=$P seed $s: EMPTY OUTPUT — aborting"; exit 1; }
      echo "$t,$P,$s,$(echo "$line"|cut -d, -f8),$(echo "$line"|cut -d, -f9)" >> $CSV
    done
    echo "  galop $t P=$P done"
  done
}
for P in 1000 10000 100000; do run_cells "$STD8"  $P sweep nguyen1 nguyen2 nguyen3 nguyen4 nguyen5 nguyen6; done
for P in 10000 100000;      do run_cells "$STD16" $P big   california feynman_i_9_18; done
for P in 1000 10000 100000; do run_cells "$STD16" $P ext   nguyen9 nguyen10 nguyen12; done
echo "GALOP QUALITY COMPLETE -> $CSV"
