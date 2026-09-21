#!/usr/bin/env bash
# Device-residency probe (Sec. 5.3, a separately labelled fact, never a ladder
# row): the largest population GALOP can hold and evaluate on one H100 —
# allocation of an evolved-shaped population plus one evaluation pass, no
# evolution. tests/cuda_capacity.cu launches the unchanged engine kernels in
# <=2^31 windows so the probe can climb past the int32 index bound.
# Effective layout (intron-stripped, bitwise-identical fitness — the ladder's
# top rung), evolved-population shape (30 instr, 16.6% effective), D=64,
# one untimed proof launch then the median of 3 timed launches per rung; every
# ladder end attributed.
# -> results_h100/residency_h100.csv   P,outcome,eval_ms,evals_per_s,note
# Submit from the login node at the repo root:  sbatch tools/cluster/residency_h100.sbatch
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=galop-residency
#SBATCH --output=results_h100/residency_slurm_%j.log
#SBATCH --gres=gpu:h100:1
#SBATCH --error=results_h100/residency_slurm_%j.err
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load StdEnv/2023 python/3.11 cuda/13.2 2>/dev/null || module load StdEnv/2023 python/3.11 cuda
NVCC=$(command -v nvcc)
echo "NODE: $(hostname)"
mkdir -p results_h100
CSV=results_h100/residency_h100.csv
[ -f $CSV ] || echo "P,outcome,eval_ms,evals_per_s,note" > $CSV
BIN=/tmp/cap_true
$NVCC -std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native -Xcompiler -fopenmp \
  tests/cuda_capacity.cu src/EvaluatorCuda.cu src/Dataset.cpp src/LGPEngine.cpp \
  src/EvaluatorCpu.cpp src/Evaluator.cpp src/Interpreter.cpp -o $BIN 2>&1 | tail -3
[ -x $BIN ] || { echo "BUILD FAILED"; exit 1; }
LAYOUT=effective   # intron-removed storage, the ladder's top rung
MEANLEN=30         # evolved-population shape: mean program length ...
EFFFRAC=0.166      # ... and effective fraction (measured at P=1e4 x 1000 generations)
D=64               # fitness cases, as in the ladder
REPS=3             # timed launches per rung after the untimed proof launch; the probe reports their median
for P in 16000000 64000000 256000000 1024000000 2048000000 2200000000 2300000000 2400000000 2500000000 2560000000 2700000000 3072000000; do
  grep -q "^$P," $CSV && continue
  ERR=$(mktemp)
  OUT=$($BIN $LAYOUT $P $MEANLEN $EFFFRAC $D $REPS 2>$ERR | tail -1); RC=$?
  if [ $RC -eq 0 ] && [[ "$OUT" == OK* ]]; then
    MS=$(echo "$OUT" | grep -o "eval_ms=[0-9.]*" | cut -d= -f2)
    ES=$(echo "$OUT" | grep -o "evals_per_s=[0-9.e+]*" | cut -d= -f2)
    echo "$P,OK,$MS,$ES," >> $CSV
    echo "  P=$P OK (${MS}ms, ${ES}/s)"
  else
    NOTE=$(tr -d ',\n' < $ERR | grep -io "out of memory[^;]*\|cudaError[^;]*" | head -c 100)
    echo "$P,FAIL_rc$RC,,,${NOTE:-$(tr -d ',\n' < $ERR | tail -c 80)}" >> $CSV
    echo "  P=$P FAILED: $NOTE"
    rm -f $ERR; break
  fi
  rm -f $ERR
done
echo "RESIDENCY H100 COMPLETE -> $CSV"
