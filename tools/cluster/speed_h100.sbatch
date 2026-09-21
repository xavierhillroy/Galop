#!/usr/bin/env bash
# =============================================================================
# H100 speed measurements (fig3b/c, Table 2, Sec. 5.1) — one job, four stages:
#   grid      the 16-cell pair grid, D in {32,1024,32768,131072}, through the
#             shared protocol script tools/speed_grid.sh (warm-up + seeds 70-74)
#             -> results_h100/speed_grid_h100.csv
#   redo      the dedicated-session re-measurement of cells whose five-run
#             spread exceeded 25% (rows tagged session=redo; the paper reports
#             the re-measurement): resubmit with
#             sbatch --export=ALL,STAGE=redo,SPEED_REDO_CELLS="galop_std:1000:32 evogp:1000:32 galop_std:1000:1024 evogp:1000:1024" tools/cluster/speed_h100.sbatch
#   boundary  D in {262144, 524288} for both arms at P <= 1e5 and galop_std at
#             1e6; the evogp arms at 1e6 are time-attributed (TIMEBOUND rows),
#             projected ~20 h -> results_h100/speed_boundary_h100.csv
#   hybrid    EvoGP with execute_mode "hybrid parallel" forced at the five cells
#             where its heuristic picks data-parallel against its own interest
#             -> results_h100/speed_hybrid_h100.csv
#   modes     EvoGP evaluation time under each execute mode at three shapes
#             (which mode "auto" resolves to) -> results_h100/speed_evogp_modes_h100.txt
# STAGE=all (default) runs grid, boundary, hybrid, modes. Resume-safe throughout.
# Submit from the login node at the repo root:  sbatch tools/cluster/speed_h100.sbatch
# =============================================================================
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --job-name=galop-speed
#SBATCH --output=results_h100/speed_slurm_%j.log
#SBATCH --error=results_h100/speed_slurm_%j.err
#SBATCH --gres=gpu:h100:1
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load StdEnv/2023 python/3.11 cuda/13.2 2>/dev/null || module load StdEnv/2023 python/3.11 cuda
source .venv_cluster/bin/activate
export NVCC=$(command -v nvcc) PYTHON=.venv_cluster/bin/python OMP_NUM_THREADS=16
STAGE=${STAGE:-all}
echo "NODE: $(hostname)  STAGE: $STAGE"
SRCS="tests/cuda_evo_bench.cu src/EvaluatorCuda.cu src/Dataset.cpp src/LGPEngine.cpp src/EvaluatorCpu.cpp src/Evaluator.cpp src/Interpreter.cpp"
NVF="-std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native -Xcompiler -fopenmp -DGALOP_STANDARD"
mkdir -p results_h100 datasets/gridsynth
gridsynth() {   # generate the shared quartic file(s) for the given D values if missing
  DL="$*" python3 - <<'GSEOF'
import random, os
random.seed(1234)
for D in [int(x) for x in os.environ["DL"].split()]:
    p = f"datasets/gridsynth/nguyen1_D{D}_train.csv"      # legacy filename: the content is the Nguyen-2 quartic
    if os.path.exists(p): continue
    with open(p, "w") as f:
        f.write("x,y\n")
        for _ in range(D):
            x = random.uniform(-1, 1); f.write(f"{x},{x**4+x**3+x**2+x}\n")
GSEOF
}
echo "=== GATES ==="; make gpu 2>&1 | tail -1
GALOP_DIFF_QUICK=1 ./cuda_diff | tail -1 || { echo "GATE FAILED"; exit 1; }

# ---------------- grid / redo: the shared protocol script ----------------
if [ "$STAGE" = all ] || [ "$STAGE" = grid ]; then
  SPEED_CSV=results_h100/speed_grid_h100.csv SPEED_DLIST="32 1024 32768 131072" bash tools/speed_grid.sh
fi
if [ "$STAGE" = redo ]; then
  [ -n "${SPEED_REDO_CELLS:-}" ] || { echo "STAGE=redo needs SPEED_REDO_CELLS='arm:P:D ...'"; exit 1; }
  SPEED_CSV=results_h100/speed_grid_h100.csv SPEED_DLIST="32 1024 32768 131072" SPEED_SESSION=redo SPEED_CELLS="$SPEED_REDO_CELLS" bash tools/speed_grid.sh
fi

# ---------------- boundary: D extension ----------------
if [ "$STAGE" = all ] || [ "$STAGE" = boundary ]; then
  gridsynth 262144 524288
  CSV=results_h100/speed_boundary_h100.csv
  [ -f $CSV ] || echo "arm,P,D,seed,wall_s" > $CSV
  for P in 1000 10000 100000 1000000; do
    bin=/tmp/ceb_bnd_$P
    [ -x $bin ] || $NVCC $NVF -DPOPULATION_SIZE_OVERRIDE=$P -DMAX_GENERATIONS_OVERRIDE=100 $SRCS -o $bin 2>/dev/null
    for D in 262144 524288; do
      grep -q "^galop_std,$P,$D,74," $CSV && continue
      GS=datasets/gridsynth/nguyen1_D${D}_train.csv
      wtot=0
      for wn in 1 2 3; do
        wv=$($bin $GS 99 warp 2>/dev/null | cut -d, -f6); wv=${wv:-0}
        wtot=$(awk -v a="$wtot" -v b="$wv" 'BEGIN{print a+b}')
        [ "$(awk -v t="$wtot" 'BEGIN{print (t>=2.0)?1:0}')" = 1 ] && break
      done
      for s in 70 71 72 73 74; do
        grep -q "^galop_std,$P,$D,$s," $CSV && continue
        w=$($bin $GS $s warp 2>/dev/null | cut -d, -f6)
        echo "galop_std,$P,$D,$s,$w" >> $CSV
      done
      echo "  [galop_std] P=$P D=$D done"
    done
  done
  $PYTHON - <<'PYEOF'
import torch, time, sys, csv
sys.path.insert(0, "harness")
from evogp_run import make_descriptor, make_algorithm, make_problem
from evogp.pipeline import StandardPipeline
cfg = {"max_tree_len":101, "features":1, "using_funcs":["+","-","*","/","sin","cos","<",">"],
       "const_samples":[0.,1.,-1.,3.14159,0.5,2.,10.,100.], "tournament":3,
       "survivor_rate":1.0, "elite_rate":0.1, "mutation_rate":0.8, "execute_mode":"auto"}
desc = make_descriptor(cfg); CSV = "results_h100/speed_boundary_h100.csv"
done = {(r["arm"], r["P"], r["D"], r["seed"]) for r in csv.DictReader(open(CSV))}
fh = open(CSV, "a", newline=""); w = csv.writer(fh)
for P in (1000, 10000, 100000):
    for D in (262144, 524288):
        if ("evogp", str(P), str(D), "74") in done: print(f"  [evogp] P={P} D={D} SKIP", flush=True); continue
        try:
            X = torch.rand(D, 1, device="cuda") * 2 - 1
            y = (X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0]).unsqueeze(1)
            prob = make_problem(cfg, X, y); wtot = 0.0
            for _ in range(3):
                torch.manual_seed(99); t0 = time.perf_counter()
                StandardPipeline(make_algorithm(cfg, desc, P), prob, generation_limit=100, is_show_details=False).run()
                torch.cuda.synchronize(); wtot += time.perf_counter() - t0
                if wtot >= 2.0: break
            for s in (70, 71, 72, 73, 74):
                if ("evogp", str(P), str(D), str(s)) in done: continue
                torch.manual_seed(s); alg = make_algorithm(cfg, desc, P)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                StandardPipeline(alg, prob, generation_limit=100, is_show_details=False).run()
                torch.cuda.synchronize(); w.writerow(["evogp", P, D, s, f"{time.perf_counter()-t0:.4f}"]); fh.flush()
            print(f"  [evogp] P={P} D={D} done", flush=True)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            w.writerow(["evogp", P, D, "fail", f"OOM:{str(e)[:60]}"]); fh.flush(); torch.cuda.empty_cache()
PYEOF
  # evogp at P=1e6: time-attributed, not run — projected ~20 h for the pair from the ~2.6x ratio
  # on galop_std's measured medians x 6 runs per cell; bounded by the P=1e5 row (no crossing)
  for D in 262144 524288; do
    grep -q "^evogp,1000000,$D," $CSV || echo "evogp,1000000,$D,fail,TIMEBOUND_wall12h_projected20h" >> $CSV
  done
fi

# ---------------- hybrid: EvoGP's hybrid mode forced ----------------
if [ "$STAGE" = all ] || [ "$STAGE" = hybrid ]; then
  gridsynth 32768 131072
  $PYTHON - <<'PYEOF'
import torch, time, sys, csv, os
sys.path.insert(0, "harness")
from evogp_run import make_descriptor, make_algorithm, make_problem
from evogp.pipeline import StandardPipeline
cfg = {"max_tree_len":101, "features":1, "using_funcs":["+","-","*","/","sin","cos","<",">"],
       "const_samples":[0.,1.,-1.,3.14159,0.5,2.,10.,100.], "tournament":3,
       "survivor_rate":1.0, "elite_rate":0.1, "mutation_rate":0.8, "execute_mode":"hybrid parallel"}
desc = make_descriptor(cfg); CSV = "results_h100/speed_hybrid_h100.csv"
if not os.path.exists(CSV): open(CSV, "w").write("arm,P,D,seed,wall_s\n")
done = {(r["arm"], r["P"], r["D"], r["seed"]) for r in csv.DictReader(open(CSV))}
fh = open(CSV, "a", newline=""); w = csv.writer(fh)
for P, D in ((10000, 32768), (100000, 32768), (100000, 131072), (1000000, 32768), (1000000, 131072)):
    if ("evogp_hybrid", str(P), str(D), "74") in done: print(f"  [evogp_hybrid] P={P} D={D} SKIP", flush=True); continue
    try:
        X = torch.rand(D, 1, device="cuda") * 2 - 1
        y = (X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0]).unsqueeze(1)
        prob = make_problem(cfg, X, y); wtot = 0.0
        for _ in range(3):
            torch.manual_seed(99); t0 = time.perf_counter()
            StandardPipeline(make_algorithm(cfg, desc, P), prob, generation_limit=100, is_show_details=False).run()
            torch.cuda.synchronize(); wtot += time.perf_counter() - t0
            if wtot >= 2.0: break
        for s in (70, 71, 72, 73, 74):
            if ("evogp_hybrid", str(P), str(D), str(s)) in done: continue
            torch.manual_seed(s); alg = make_algorithm(cfg, desc, P)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            StandardPipeline(alg, prob, generation_limit=100, is_show_details=False).run()
            torch.cuda.synchronize(); w.writerow(["evogp_hybrid", P, D, s, f"{time.perf_counter()-t0:.4f}"]); fh.flush()
        print(f"  [evogp_hybrid] P={P} D={D} done", flush=True)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        w.writerow(["evogp_hybrid", P, D, "fail", f"OOM:{str(e)[:60]}"]); fh.flush(); torch.cuda.empty_cache()
PYEOF
fi

# ---------------- modes: which execute mode EvoGP's heuristic resolves to ----------------
if [ "$STAGE" = all ] || [ "$STAGE" = modes ]; then
  [ -s results_h100/speed_evogp_modes_h100.txt ] || $PYTHON - <<'PYEOF'
import torch, time, statistics, sys
sys.path.insert(0, "harness")
from evogp_run import make_descriptor
from evogp.tree import Forest
from evogp.problem import SymbolicRegression
cfg = {"max_tree_len":101, "features":1, "using_funcs":["+","-","*","/","sin","cos","<",">"],
       "const_samples":[0.,1.,-1.,3.14159,0.5,2.,10.,100.]}
desc = make_descriptor(cfg)
out = open("results_h100/speed_evogp_modes_h100.txt", "w")
for P, D in ((10000, 1024), (1000000, 32768), (1000000, 131072)):
    torch.manual_seed(1234)
    X = torch.rand(D, 1, device="cuda") * 2 - 1
    y = (X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0]).unsqueeze(1)
    forest = Forest.random_generate(pop_size=P, descriptor=desc)
    line = [f"P={P} D={D}:"]
    for mode in ("auto", "hybrid parallel", "data parallel", "tree parallel"):
        try:
            prob = SymbolicRegression(datapoints=X, labels=y, execute_mode=mode)
            for _ in range(2): prob.evaluate(forest)
            torch.cuda.synchronize(); ts = []
            for _ in range(5):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                prob.evaluate(forest); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
            line.append(f"{mode}={statistics.median(ts)*1e3:.1f}ms")
        except Exception as e:
            line.append(f"{mode}=FAIL({str(e)[:40]})")
        torch.cuda.empty_cache()
    print(" ".join(line), file=out, flush=True)
PYEOF
fi
echo "SPEED H100 ($STAGE) COMPLETE"
