#!/usr/bin/env bash
# =============================================================================
# Effective program length on the quality workloads (Sec. 5.3: the intron-removed
# capacity ceiling depends on how much effective code a workload retains).
# Re-runs the Galop quality cells at P=1e5 for the seeds of
# results/quality_galop_rtx.csv with the intron-removed layout (GALOP_EFF_LOG).
# That layout changes only what is uploaded to the device; fitness is bitwise
# identical, so every run must reproduce the train and test MSE of its quality
# row to all printed digits — otherwise the script stops. Recorded per run: the
# mean program length and the mean effective length at the last generation and
# the peak mean effective length during the run (instructions per individual).
#   -> results/effective_length_rtx.csv
#      problem,P,seed,final_length,final_effective,peak_effective,train_mse,test_mse
# Resume-safe (rerun fills missing rows). About 1 h on the RTX 5070 Ti.
#   bash tools/effective_length.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
NVCC=$(command -v nvcc)
SRCS="tests/cuda_evo_bench.cu src/EvaluatorCuda.cu src/Dataset.cpp src/LGPEngine.cpp src/EvaluatorCpu.cpp src/Evaluator.cpp src/Interpreter.cpp"
NVF="-std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native -Xcompiler -fopenmp"
STD8="-DINPUT_PROTECT_OVERRIDE=1 -DMACRO_INSERTS_OVERRIDE=3"
STD16="-DNUM_REGISTERS_OVERRIDE=16 -DINPUT_PROTECT_OVERRIDE=1 -DMACRO_INSERTS_OVERRIDE=3"
NSEEDS=${EFFLEN_NSEEDS:-100}; LAST=$((NSEEDS-1)); P=100000
Q=results/quality_galop_rtx.csv; CSV=results/effective_length_rtx.csv
[ -f $Q ] || { echo "missing $Q — run tools/quality_galop.sh first"; exit 1; }
[ -f $CSV ] || echo "problem,P,seed,final_length,final_effective,peak_effective,train_mse,test_mse" > $CSV
python3 tools/gen_ext_datasets.py >/dev/null
LOG=$(mktemp)
run_cells() {   # FLAGS dir problems...
  local FLAGS=$1 dir=$2; shift 2
  local bin=/tmp/ceb_quality_${P}_$(echo "$FLAGS" | md5sum | cut -c1-6)
  for t in "$@"; do
    grep -q "^$t,$P,$LAST," $CSV && { echo "  $t SKIP"; continue; }
    [ -x $bin ] || $NVCC $NVF $FLAGS -DPOPULATION_SIZE_OVERRIDE=$P -DMAX_GENERATIONS_OVERRIDE=100 $SRCS -o $bin
    for s in $(seq 0 $LAST); do
      grep -q "^$t,$P,$s," $CSV && continue
      line=$(GALOP_EFF_LOG=$LOG $bin datasets/$dir/${t}_train.csv $s warp datasets/$dir/${t}_test.csv 2>/dev/null)
      [ -n "$line" ] || { echo "  $t seed $s: EMPTY OUTPUT — aborting"; exit 1; }
      tr=$(echo "$line" | cut -d, -f8); te=$(echo "$line" | cut -d, -f9)
      # exact string match against the quality row (the shipped file has CRLF line ends, so no grep anchor)
      awk -F, -v t="$t" -v P="$P" -v s="$s" -v a="$tr" -v b="$te" '{ sub(/\r$/, "") } ($1 "") == t && ($2 "") == P && ($3 "") == s && ($4 "") == a && ($5 "") == b { f = 1 } END { exit !f }' $Q \
        || { echo "  $t seed $s: MSE $tr / $te does not reproduce its quality row — aborting"; exit 1; }
      shape=$(awk -F, -v P=$P 'NR > 1 { L = $2; E = $3; if ($3 > X) X = $3 } END { printf "%.3f,%.3f,%.3f", L / P, E / P, X / P }' $LOG)
      echo "$t,$P,$s,$shape,$tr,$te" >> $CSV
    done
    echo "  $t done"
  done
}
run_cells "$STD8"  sweep nguyen1 nguyen2 nguyen3 nguyen4 nguyen5 nguyen6
run_cells "$STD16" ext   nguyen9 nguyen10 nguyen12
run_cells "$STD16" big   feynman_i_9_18 california
rm -f $LOG
echo "EFFECTIVE LENGTH COMPLETE -> $CSV"
