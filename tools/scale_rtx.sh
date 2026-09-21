#!/usr/bin/env bash
# =============================================================================
# CONSUMER-CARD FLAGSHIP: P=3.2e7 x G=100 on the RTX 5070 Ti — the "you don't
# need a datacenter GPU" row. Same workload family as the H100 1e8 flagship
# (Nguyen-2 quartic, D=64, aligned), same definitive protocol: adaptive
# warm-up (discarded) + 5 timed runs seeds 70-74, median, spread rule.
# P chosen by HOST RAM, not the GPU: 1e8 needs ~62 GB host (2x dense ping-pong
# 40 GB + 20 GB pinned prealloc); 3.2e7 fits a 32 GB machine (~20 GB peak).
# RUN DEDICATED: close browsers/apps; the GPU must be otherwise idle.
# -> results/scale_rtx.csv   P,D,seed,wall_s      Run: bash tools/scale_rtx.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=16
NVCC=${NVCC:-$( [ -x /usr/local/cuda/bin/nvcc ] && echo /usr/local/cuda/bin/nvcc || command -v nvcc )}
P=32000000
SRCS="tests/cuda_evo_bench.cu src/EvaluatorCuda.cu src/Dataset.cpp src/LGPEngine.cpp src/EvaluatorCpu.cpp src/Evaluator.cpp src/Interpreter.cpp"
NVF="-std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native -Xcompiler -fopenmp -DGALOP_STANDARD"

echo "=== GATES ==="
make gpu >/dev/null 2>&1; GALOP_DIFF_QUICK=1 ./cuda_diff | tail -1 || { echo "GATE FAILED"; exit 1; }

mkdir -p results datasets/gridsynth
python3 - <<'GSEOF'
import random, os
random.seed(1234)
p = "datasets/gridsynth/nguyen1_D64_train.csv"   # legacy filename: the content is the NGUYEN-2 quartic
if not os.path.exists(p):
    with open(p, "w") as f:
        f.write("x,y\n")
        for _ in range(64):
            x = random.uniform(-1, 1); f.write(f"{x},{x**4+x**3+x**2+x}\n")
GSEOF
TR=datasets/gridsynth/nguyen1_D64_train.csv

FREE_GB=$(free -g | awk '/^Mem:/{print $7}')
[ "$FREE_GB" -ge 22 ] || { echo "only ${FREE_GB} GB host RAM available — close applications (need ~22 GB)"; exit 1; }

bin=/tmp/ceb_rtx_flagship
[ -x $bin ] || $NVCC $NVF -DPOPULATION_SIZE_OVERRIDE=$P -DMAX_GENERATIONS_OVERRIDE=100 $SRCS -o $bin

CSV=results/scale_rtx.csv
[ -f $CSV ] || echo "P,D,seed,wall_s" > $CSV

echo "=== adaptive warm-up (discarded) ==="
T0=$(date +%s.%N); $bin $TR 42 warp >/dev/null; TW=$(echo "$(date +%s.%N) - $T0" | bc)
echo "  warm-up: ${TW}s"

for s in 70 71 72 73 74; do
  grep -q "^$P,64,$s," $CSV && { echo "  seed $s SKIP"; continue; }
  T0=$(date +%s.%N)
  $bin $TR $s warp >/dev/null
  W=$(echo "$(date +%s.%N) - $T0" | bc)
  echo "$P,64,$s,$W" >> $CSV
  echo "  seed $s: ${W}s"
done

python3 - <<'PYEOF'
import csv
w = sorted(float(r["wall_s"]) for r in csv.DictReader(open("results/scale_rtx.csv")))
if len(w) >= 5:
    med = w[len(w)//2]; spread = (w[-1] - w[0]) / med * 100
    print(f"RTX FLAGSHIP: P=3.2e7 x G100 median {med:.1f}s ({med/60:.1f} min), spread {spread:.0f}%"
          + (" — n.m. (>25%)" if spread > 25 else " — protocol-clean"))
PYEOF
