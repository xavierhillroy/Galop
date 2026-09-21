#!/usr/bin/env bash
# =============================================================================
# Speed grid — the paper's whole-run protocol, both cards (fig3, Table 2).
# Every (arm, P, D) cell: discarded warm-up executions until >=2 s cumulative
# device activity (cap 3), then 5 timed complete 100-generation runs (seeds
# 70-74), single session, single device; the paper reports medians of the 5.
# Arms: galop_std (the engine, -DGALOP_STANDARD) and evogp (parity config).
# Workload: the shared Nguyen-2 quartic file at each D (legacy filenames
# datasets/gridsynth/nguyen1_D*_train.csv; generated in place if missing).
#   bash tools/speed_grid.sh                        # RTX: D in {32,1024,32768}
# Environment (all optional):
#   SPEED_CSV      output (default results/speed_grid_rtx.csv)
#   SPEED_DLIST    D values (default "32 1024 32768"; the H100 job adds 131072)
#   SPEED_PLIST    P values (default "1000 10000 100000 1000000")
#   SPEED_SESSION  session tag written on every row (default grid)
#   SPEED_CELLS    "arm:P:D ..." — measure only these cells; with
#                  SPEED_SESSION=redo this is the dedicated-session
#                  re-measurement of cells whose five-run spread exceeded 25%
#   PYTHON, NVCC   interpreter / compiler (defaults .venv/bin/python, CUDA nvcc)
# Resume-safe: a row already present for (arm, P, D, seed, session) is skipped.
# -> arm,P,D,seed,wall_s,session
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=16
NVCC=${NVCC:-${NVCC_BIN:-$( [ -x /usr/local/cuda/bin/nvcc ] && echo /usr/local/cuda/bin/nvcc || command -v nvcc )}}
SRCS="tests/cuda_evo_bench.cu src/EvaluatorCuda.cu src/Dataset.cpp src/LGPEngine.cpp src/EvaluatorCpu.cpp src/Evaluator.cpp src/Interpreter.cpp"
NVF="-std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native -Xcompiler -fopenmp -DGALOP_STANDARD"
export SPEED_CSV=${SPEED_CSV:-results/speed_grid_rtx.csv}
export SPEED_DLIST=${SPEED_DLIST:-"32 1024 32768"}
export SPEED_PLIST=${SPEED_PLIST:-"1000 10000 100000 1000000"}
export SPEED_SESSION=${SPEED_SESSION:-grid}
export SPEED_CELLS=${SPEED_CELLS:-}
CSV=$SPEED_CSV; SES=$SPEED_SESSION
wanted() { [ -z "$SPEED_CELLS" ] || [[ " $SPEED_CELLS " == *" $1:$2:$3 "* ]]; }
mkdir -p "$(dirname "$CSV")" datasets/gridsynth
python3 - <<'GSEOF'
import random, os
random.seed(1234)
for D in [int(x) for x in os.environ["SPEED_DLIST"].split()]:
    p = f"datasets/gridsynth/nguyen1_D{D}_train.csv"      # legacy filename: the content is the Nguyen-2 quartic
    if os.path.exists(p): continue
    with open(p, "w") as f:
        f.write("x,y\n")
        for _ in range(D):
            x = random.uniform(-1, 1); f.write(f"{x},{x**4+x**3+x**2+x}\n")
GSEOF
make gpu 2>&1 | tail -1
GALOP_DIFF_QUICK=1 ./cuda_diff | tail -1 || { echo "GATE FAILED"; exit 1; }
[ -f "$CSV" ] || echo "arm,P,D,seed,wall_s,session" > "$CSV"

# ---------------- galop_std: one binary per P, warm-up + 5 timed runs per cell ----------------
for P in $SPEED_PLIST; do
  bin=/tmp/ceb_speed_$P
  for D in $SPEED_DLIST; do
    wanted galop_std $P $D || continue
    grep -q "^galop_std,$P,$D,74,.*,$SES$" "$CSV" && { echo "  [galop_std] P=$P D=$D SKIP"; continue; }
    [ -x $bin ] || $NVCC $NVF -DPOPULATION_SIZE_OVERRIDE=$P -DMAX_GENERATIONS_OVERRIDE=100 $SRCS -o $bin 2>/dev/null
    GS=datasets/gridsynth/nguyen1_D${D}_train.csv
    wtot=0                      # adaptive warm-up, discarded: >=2 s cumulative device activity, cap 3
    for wn in 1 2 3; do
      wv=$($bin $GS 99 warp 2>/dev/null | cut -d, -f6); wv=${wv:-0}
      wtot=$(awk -v a="$wtot" -v b="$wv" 'BEGIN{print a+b}')
      [ "$(awk -v t="$wtot" 'BEGIN{print (t>=2.0)?1:0}')" = 1 ] && break
    done
    for s in 70 71 72 73 74; do
      grep -q "^galop_std,$P,$D,$s,.*,$SES$" "$CSV" && continue
      w=$($bin $GS $s warp 2>/dev/null | cut -d, -f6)
      echo "galop_std,$P,$D,$s,$w,$SES" >> "$CSV"
    done
    echo "  [galop_std] P=$P D=$D done"
  done
done

# ---------------- evogp: parity configuration, same protocol ----------------
${PYTHON:-${PYBIN:-.venv/bin/python}} - <<'PYEOF'
import torch, time, sys, csv, os
sys.path.insert(0, "harness")
from evogp_run import make_descriptor, make_algorithm, make_problem
from evogp.pipeline import StandardPipeline
cfg = {"max_tree_len":101, "features":1, "using_funcs":["+","-","*","/","sin","cos","<",">"],
       "const_samples":[0.,1.,-1.,3.14159,0.5,2.,10.,100.], "tournament":3,
       "survivor_rate":1.0, "elite_rate":0.1, "mutation_rate":0.8, "execute_mode":"auto"}
# survivor_rate=1.0 is the parity value: EvoGP's parents are tournament winners drawn over the
# full population, matching GALOP's fresh tournament per parent (in EvoGP's step() it is the
# mating pool, not a retention rate). Measured wall-neutral against 0.1 and 0.5.
desc = make_descriptor(cfg)
CSV, SES = os.environ["SPEED_CSV"], os.environ["SPEED_SESSION"]
PL = [int(x) for x in os.environ["SPEED_PLIST"].split()]; DL = [int(x) for x in os.environ["SPEED_DLIST"].split()]
cells = os.environ.get("SPEED_CELLS", "").split()
wanted = lambda P, D: not cells or f"evogp:{P}:{D}" in cells
done = set()
for r in csv.DictReader(open(CSV)): done.add((r["arm"], r["P"], r["D"], r["seed"], r.get("session", "grid")))
w = csv.writer(open(CSV, "a", newline=""))
for P in PL:
    for D in DL:
        if not wanted(P, D): continue
        if ("evogp", str(P), str(D), "74", SES) in done:
            print(f"  [evogp] P={P} D={D} SKIP", flush=True); continue
        X = torch.rand(D, 1, device="cuda") * 2 - 1                # EvoGP samples its own D points of the same quartic
        y = (X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0]).unsqueeze(1)
        prob = make_problem(cfg, X, y)
        wtot = 0.0                                                  # adaptive warm-up, discarded
        for _ in range(3):
            torch.manual_seed(99); t0 = time.perf_counter()
            StandardPipeline(make_algorithm(cfg, desc, P), prob, generation_limit=100, is_show_details=False).run()
            torch.cuda.synchronize(); wtot += time.perf_counter() - t0
            if wtot >= 2.0: break
        for s in (70, 71, 72, 73, 74):
            if ("evogp", str(P), str(D), str(s), SES) in done: continue
            torch.manual_seed(s); alg = make_algorithm(cfg, desc, P)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            StandardPipeline(alg, prob, generation_limit=100, is_show_details=False).run()
            torch.cuda.synchronize()
            w.writerow(["evogp", P, D, s, f"{time.perf_counter()-t0:.4f}", SES])
        print(f"  [evogp] P={P} D={D} done", flush=True)
PYEOF
echo "SPEED GRID COMPLETE -> $CSV"
