#!/usr/bin/env bash
# =============================================================================
# CPU anchor (the gplearn sentence of Sec. 5.1), local desktop: GALOP and gplearn
# on the same machine and datasets, wall of the evolutionary run only
# (construction excluded), 3 seeded runs per cell, G=100, tournament 3.
#   GALOP   standard configuration; nguyen2 @ P in {1e3,1e4,1e6}, feynman @ {1e3,1e6}
#           (16 registers for the 9-input problem)
#   gplearn its expressible 6 of the 8 primitives, MSE metric, n_jobs=16 (the
#           uniform host-thread setting), other settings default;
#           nguyen2 @ P in {1e3,1e4}, feynman @ 1e3
# Paper: GALOP's median at nguyen2 P=1e6 (4.3 s) against gplearn's at P=1e3 (66.6 s).
#   bash tools/cpu_anchor.sh             (NVCC=, PYTHON= optional; resume-safe)
# -> results/cpu_anchor_rtx.csv   system,problem,P,seed,wall_s
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=16
NVCC=${NVCC:-$( [ -x /usr/local/cuda/bin/nvcc ] && echo /usr/local/cuda/bin/nvcc || command -v nvcc )}
SRCS="tests/cuda_evo_bench.cu src/EvaluatorCuda.cu src/Dataset.cpp src/LGPEngine.cpp src/EvaluatorCpu.cpp src/Evaluator.cpp src/Interpreter.cpp"
NVF="-std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native -Xcompiler -fopenmp -DGALOP_STANDARD -DMAX_GENERATIONS_OVERRIDE=100"
CSV=results/cpu_anchor_rtx.csv
mkdir -p results; [ -f $CSV ] || echo "system,problem,P,seed,wall_s" > $CSV
for spec in "nguyen2:datasets/sweep/nguyen2_train.csv:1000:" \
            "nguyen2:datasets/sweep/nguyen2_train.csv:10000:" \
            "nguyen2:datasets/sweep/nguyen2_train.csv:1000000:" \
            "feynman_i_9_18:datasets/big/feynman_i_9_18_train.csv:1000:-DNUM_REGISTERS_OVERRIDE=16" \
            "feynman_i_9_18:datasets/big/feynman_i_9_18_train.csv:1000000:-DNUM_REGISTERS_OVERRIDE=16"; do
  IFS=: read -r prob tr P extra <<< "$spec"
  [ -f "$tr" ] || { echo "missing $tr (run ./reproduce.sh datasets)"; exit 1; }
  bin=/tmp/ceb_cpuanchor_${prob}_$P
  [ -x $bin ] || $NVCC $NVF $extra -DPOPULATION_SIZE_OVERRIDE=$P $SRCS -o $bin 2>/dev/null || { echo "compile failed ($prob P=$P)"; exit 1; }
  for s in 0 1 2; do
    grep -q "^galop,$prob,$P,$s," $CSV && continue
    W=$($bin $tr $s warp 2>/dev/null | cut -d, -f6)
    echo "galop,$prob,$P,$s,$W" >> $CSV; echo "  galop $prob P=$P seed $s: ${W}s"
  done
done
${PYTHON:-.venv/bin/python} - <<'PYEOF'
import csv, os, time
from gplearn.genetic import SymbolicRegressor
CSV = "results/cpu_anchor_rtx.csv"
done = {(r["system"], r["problem"], r["P"], r["seed"]) for r in csv.DictReader(open(CSV))}
f = open(CSV, "a", newline=""); w = csv.writer(f)
def load(path):
    rows = [l.strip().split(",") for l in open(path)][1:]
    return [[float(v) for v in r[:-1]] for r in rows], [float(r[-1]) for r in rows]
for prob, path, P in (("nguyen2", "datasets/sweep/nguyen2_train.csv", 1000),
                      ("nguyen2", "datasets/sweep/nguyen2_train.csv", 10000),
                      ("feynman_i_9_18", "datasets/big/feynman_i_9_18_train.csv", 1000)):
    X, y = load(path)
    for seed in (0, 1, 2):
        if ("gplearn", prob, str(P), str(seed)) in done: continue
        est = SymbolicRegressor(population_size=P, generations=100, tournament_size=3,
                                function_set=("add", "sub", "mul", "div", "sin", "cos"),
                                metric="mse", n_jobs=16, random_state=seed, verbose=0)
        t0 = time.perf_counter(); est.fit(X, y); wall = time.perf_counter() - t0
        w.writerow(["gplearn", prob, P, seed, f"{wall:.2f}"]); f.flush()
        print(f"  gplearn {prob} P={P} seed {seed}: {wall:.1f}s", flush=True)
PYEOF
echo "CPU ANCHOR COMPLETE -> $CSV"
