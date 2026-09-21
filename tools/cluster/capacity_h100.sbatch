#!/usr/bin/env bash
# =============================================================================
# CAPACITY LADDER — all five systems in one job, one criterion, one file
# (fig6a, Sec. 5.3): ceiling = largest OK rung; the next rung's outcome names
# the wall. Every row carries the node and the date it was measured on.
#
# The criterion (identical for every system): complete ONE FULL GENERATION of
# the system's own pipeline on the shared D=64 Nguyen-2 quartic file
# (datasets/gridsynth/nguyen1_D64_train.csv, md5 0d3f6770...), one 80 GB H100,
# 1 TB host (--mem=1000G), 16 host threads, 1,200 s cap on the whole process per
# rung (EvoGP: an in-process 1,200 s alarm; Beagle: the adapter runner kills at
# the cap). Ladders stop at the first non-OK rung. Resume-safe: a (system,P)
# already in the CSV is skipped, so a killed job is simply resubmitted.
#
# Per-system settings (the size caps of Table 1; each system's own initializer):
#   GALOP   -DGALOP_STANDARD, MAX_GENERATIONS_OVERRIDE=2 = initial evaluation +
#           one vary() + child evaluation (one TRUE generation); two shipped
#           layouts: packed (rungs 1.0e9, 1.4e9, 1.8e9) and intron-removed via
#           GALOP_EFF_LOG (rungs 1.4e9, 1.8e9, 2.1e9, 2^31-2, then the engine's
#           own walls recorded in-file: 2^31-1 BUILD_FAIL, 2.2e9 INT32_WALL).
#   SRGPU   retained snapshot, SRGPU_DEPTH=7 (parity cap), initialization capped
#           at the same depth (harness default since 2026-09-20), forced rebuild
#           + binary guard; rungs 4e6, 8e6, 16e6, 32e6.
#   Kozax   KOZAX_MAX_NODES=101 (parity cap; executable for one generation),
#           KOZAX_OPS=8; rungs 2e4, 3e4, 4e4, 6e4.
#   EvoGP   parity configuration (101-node cap, 8 functions, fixed constant
#           pool, tournament 3, survivor_rate 1.0, elite 0.1, mutation 0.8,
#           execute_mode auto), StandardPipeline(generation_limit=1);
#           rungs 4e6, 8e6, 2^24, 2^25.
#   Beagle  stock checkout + adapter, EPG 64 (full batch), generation cut 1;
#           colonies 4e8, 4.8e8, 5.36e8, 5.6e8 (536,870,911 = int32/4 is its
#           source-derived bound).
# -> results_h100/capacity_h100.csv
#      system,P,outcome,wall_process_s,evolve_s,node,date,note   (evolve_s: GALOP only)
#    per-rung stdout/stderr in results_h100/capacity_logs/
# Submit from the login node at the repo root:  sbatch tools/cluster/capacity_h100.sbatch
# Runtime: ~3 h (GALOP ~100 min, SRGPU ~50 min, Beagle ~15 min, Kozax/EvoGP minutes).
# =============================================================================
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000G
#SBATCH --time=06:00:00
#SBATCH --job-name=galop-capacity
#SBATCH --output=results_h100/capacity_slurm_%j.log
#SBATCH --error=results_h100/capacity_slurm_%j.err
#SBATCH --gres=gpu:h100:1
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load StdEnv/2023 python/3.11 cuda/13.2 cmake 2>/dev/null || module load StdEnv/2023 python/3.11 cuda cmake
NVCC=$(command -v nvcc); export OMP_NUM_THREADS=16
NODE=$(hostname); DATE=$(date +%F)
echo "NODE: $NODE"; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "ENGINE: $(git rev-parse HEAD) $(git status --short src tests tools | wc -l) local modifications in src/tests/tools"
mkdir -p results_h100/capacity_logs datasets/gridsynth
python3 - <<'GSEOF'
import random, os, hashlib
random.seed(1234)
p = "datasets/gridsynth/nguyen1_D64_train.csv"   # legacy filename: the content is the Nguyen-2 quartic
if not os.path.exists(p):
    with open(p, "w") as f:
        f.write("x,y\n")
        for _ in range(64):
            x = random.uniform(-1, 1); f.write(f"{x},{x**4+x**3+x**2+x}\n")
print("workload md5:", hashlib.md5(open(p,"rb").read()).hexdigest(), "(expected 0d3f677062f8c9c7f8dcb07c4e095ffd)")
GSEOF
TR=datasets/gridsynth/nguyen1_D64_train.csv
CSV=results_h100/capacity_h100.csv
LOGD=results_h100/capacity_logs
[ -f $CSV ] || echo "system,P,outcome,wall_process_s,evolve_s,node,date,note" > $CSV
CAP=1200
done_row() { grep -q "^$1,$2," $CSV; }
classify() {   # RC ERRFILE -> outcome word (shared by GALOP, SRGPU, Kozax)
  local RC=$1 ERR=$2
  if   [ $RC -eq 0 ];   then echo OK
  elif [ $RC -eq 124 ]; then echo TIMEOUT
  elif grep -qi "out of memory\|cudaErrorMemoryAllocation\|RESOURCE_EXHAUSTED\|OOM" $ERR; then echo OOM_DEVICE
  elif grep -qi "bad_alloc\|Cannot allocate\|Killed" $ERR; then echo OOM_HOST
  else echo "CRASH_rc$RC"; fi
}

# ============================ EvoGP (one pipeline step) ============================
echo "=== EvoGP ==="
source .venv_cluster/bin/activate 2>/dev/null
NODE=$NODE DATE=$DATE .venv_cluster/bin/python - <<'PYEOF'
import torch, time, sys, csv, os, signal
sys.path.insert(0, "harness")
from evogp_run import make_descriptor, make_algorithm, make_problem, load_csv
from evogp.pipeline import StandardPipeline
cfg = {"max_tree_len":101, "features":1, "using_funcs":["+","-","*","/","sin","cos","<",">"],
       "const_samples":[0.,1.,-1.,3.14159,0.5,2.,10.,100.], "tournament":3,
       "survivor_rate":1.0, "elite_rate":0.1, "mutation_rate":0.8, "execute_mode":"auto"}
PATH = "results_h100/capacity_h100.csv"; NODE = os.environ["NODE"]; DATE = os.environ["DATE"]
done = {l.split(",")[1] for l in open(PATH) if l.startswith("evogp,")}
fh = open(PATH, "a", newline=""); w = csv.writer(fh)
Xtr, ytr = load_csv("datasets/gridsynth/nguyen1_D64_train.csv")
desc = make_descriptor(cfg); prob = make_problem(cfg, Xtr, ytr)
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError('rung exceeded 1200 s')))
for P in (4000000, 8000000, 16777216, 33554432):
    if str(P) in done: continue
    try:
        signal.alarm(1200)
        torch.manual_seed(70); t0 = time.perf_counter()
        StandardPipeline(make_algorithm(cfg, desc, P), prob, generation_limit=1, is_show_details=False).run()
        torch.cuda.synchronize(); signal.alarm(0)
        w.writerow(["evogp", P, "OK", f"{time.perf_counter()-t0:.0f}", "", NODE, DATE, "generation_limit=1"]); fh.flush()
        print(f"  evogp P={P} -> OK", flush=True)
    except Exception as e:
        signal.alarm(0)
        msg = str(e).replace(",", " ")[:90]
        o = ("TIMEOUT" if isinstance(e, TimeoutError) else "OOM_DEVICE" if "out of memory" in msg.lower() else "CRASH_framework")
        w.writerow(["evogp", P, o, "0", "", NODE, DATE, msg]); fh.flush()
        print(f"  evogp P={P} -> {o}: {msg[:60]}", flush=True)
        torch.cuda.empty_cache(); break
PYEOF
deactivate 2>/dev/null || true

# ============================ Kozax (one generation, 101-node parity cap) ============================
echo "=== Kozax ==="
source ~/.venv_kozax/bin/activate
export KOZAX_MAX_NODES=101 KOZAX_OPS=8
if python -c "import jax; assert any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in jax.devices())" 2>/dev/null; then
  for P in 20000 30000 40000 60000; do
    done_row kozax $P && continue
    ERR=$LOGD/kozax_$P.err; T0=$(date +%s)
    timeout $CAP python tools/baselines/kozax_bench.py $TR $TR $P 1 0 >$LOGD/kozax_$P.out 2>$ERR; RC=$?; W=$(( $(date +%s) - T0 ))
    O=$(classify $RC $ERR)
    echo "kozax,$P,$O,$W,,$NODE,$DATE,max_nodes 101 $(tr -d ',\n' < $ERR | tail -c 80)" >> $CSV
    echo "  kozax P=$P -> $O (${W}s)"; [ "$O" = "OK" ] || break
  done
else echo "kozax env: no GPU — skipped"; fi
deactivate 2>/dev/null || true

# ============================ Beagle (one generation, EPG 64) ============================
echo "=== Beagle ==="
DLL=$HOME/beagle_stock/Beagle/Run/bin/Release/net10.0/Run.dll
if grep -q "GALOP-adapter" $HOME/beagle_stock/Beagle/Run/Program.cs 2>/dev/null; then
  for C in 400000000 480000000 536000000 560000000; do
    done_row beagle $C && continue
    OUT=$(bash tools/beagle_adapter/run_beagle_cell.sh $DLL $TR $C 64 1 $CAP $LOGD/beagle_$C.log)
    case "$OUT" in
      OK*) W=$(echo "$OUT" | cut -f2); echo "beagle,$C,OK,$W,,$NODE,$DATE,EPG 64 gen cut 1" >> $CSV; echo "  beagle C=$C -> OK (${W}s)";;
      *)   echo "beagle,$C,$(echo "$OUT" | tr ' \t' '__'),0,,$NODE,$DATE,$(tail -3 $LOGD/beagle_$C.log | tr -d ',\n' | tail -c 90)" >> $CSV
           echo "  beagle C=$C -> $OUT"; break;;
    esac
  done
else echo "beagle adapter missing — section skipped"; fi

# ============================ SRGPU (one generation, initialization under the cap) ============================
echo "=== SRGPU ==="
SRC=$HOME/galop_srgpu_src
if [ -d $SRC ]; then
  grep -q SRGPU_INIT_DEPTH_HI tools/baselines/srgpu_bench.cu || { echo "harness predates the init-depth fix"; exit 1; }
  ( cd $SRC
    sed -i 's/ experi_benchmark.cu//' CMakeLists.txt 2>/dev/null || true
    sed -i 's/set(CMAKE_CUDA_STANDARD 14)/set(CMAKE_CUDA_STANDARD 17)/' CMakeLists.txt
    cp "$SLURM_SUBMIT_DIR/tools/baselines/srgpu_bench.cu" . && touch srgpu_bench.cu
    grep -q srgpu_bench CMakeLists.txt || cat >> CMakeLists.txt <<'CM'
add_executable(srgpu_bench src/prefix.cu src/regression.cu src/fit_eval.cu src/program.cu srgpu_bench.cu)
set_target_properties(srgpu_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
CM
    rm -f build/srgpu_bench
    mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native .. >/dev/null && make srgpu_bench -j8 2>&1 | tail -1 )
  [ -x $SRC/build/srgpu_bench ] || { echo "srgpu_bench build failed"; exit 1; }
  grep -a -q SRGPU_INIT_DEPTH_HI $SRC/build/srgpu_bench || { echo "binary predates the init-depth fix — refusing to measure"; exit 1; }
  export SRGPU_DEPTH=7; unset SRGPU_INIT_DEPTH_HI          # initial depth 4..7
  for P in 4000000 8000000 16000000 32000000; do
    done_row srgpu $P && continue
    ERR=$LOGD/srgpu_$P.err; T0=$(date +%s)
    timeout $CAP $SRC/build/srgpu_bench $TR $TR $P 1 >$LOGD/srgpu_$P.out 2>$ERR; RC=$?; W=$(( $(date +%s) - T0 ))
    O=$(classify $RC $ERR)
    echo "srgpu,$P,$O,$W,,$NODE,$DATE,depth cap 7 init 4-7 $(tr -d ',\n' < $ERR | tail -c 80)" >> $CSV
    echo "  srgpu P=$P -> $O (${W}s)"; [ "$O" = "OK" ] || break
  done
else echo "srgpu snapshot missing — skipped"; fi

# ============================ GALOP (one TRUE generation, both layouts) ============================
echo "=== GALOP: gates ==="; make gpu 2>&1 | tail -1
GALOP_DIFF_QUICK=1 ./cuda_diff | tail -1 || { echo "GATE FAILED"; exit 1; }
SRCS="tests/cuda_evo_bench.cu src/EvaluatorCuda.cu src/Dataset.cpp src/LGPEngine.cpp src/EvaluatorCpu.cpp src/Evaluator.cpp src/Interpreter.cpp"
NVF="-std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native -Xcompiler -fopenmp -DGALOP_STANDARD"
rung() {   # rung MODE P -> one CSV row; echoes the outcome word
  local MODE=$1 P=$2 SYS=galop_$1
  done_row $SYS $P && { echo "  $SYS P=$P SKIP (done)"; grep "^$SYS,$P," $CSV | cut -d, -f3; return; }
  if [ "$P" -gt 2147483647 ]; then
    echo "$SYS,$P,INT32_WALL,0,,$NODE,$DATE,POPULATION_SIZE is constexpr int: beyond 2^31-1 the engine cannot be built" >> $CSV; echo INT32_WALL; return; fi
  local bin=/tmp/ceb_final_${MODE}_$P ERR=$LOGD/${MODE}_${P}.err
  if ! $NVCC $NVF -DPOPULATION_SIZE_OVERRIDE=$P -DMAX_GENERATIONS_OVERRIDE=2 $SRCS -o $bin 2>$ERR; then
    echo "$SYS,$P,BUILD_FAIL,0,,$NODE,$DATE,$(tr -d ',\n' < $ERR | tail -c 90)" >> $CSV; echo BUILD_FAIL; return; fi
  if [ "$MODE" = eff ]; then export GALOP_EFF_LOG=$LOGD/${MODE}_${P}.efflog; else unset GALOP_EFF_LOG; fi
  local T0=$(date +%s); timeout $CAP $bin $TR 70 warp >$LOGD/${MODE}_${P}.out 2>$ERR; local RC=$?; local W=$(( $(date +%s) - T0 ))
  local O=$(classify $RC $ERR)
  local EV=$(grep -o 'total=[0-9.]*' $ERR | tail -1 | cut -d= -f2); EV=${EV:-0}
  echo "$SYS,$P,$O,$W,$EV,$NODE,$DATE,MAX_GENERATIONS=2 $(tr -d ',\n' < $ERR | tail -c 100)" >> $CSV
  echo "  $SYS P=$P -> $O (process ${W}s, evolve ${EV}s)"; rm -f $bin; echo $O
}
echo "=== GALOP packed ==="
for P in 1000000000 1400000000 1800000000; do r=$(rung packed $P | tail -1); [ "$r" = OK ] || break; done
echo "=== GALOP intron-removed ==="
for P in 1400000000 1800000000 2100000000 2147483646; do r=$(rung eff $P | tail -1); [ "$r" = OK ] || break; done
[ "$r" = OK ] && { rung eff 2147483647 >/dev/null; rung eff 2200000000 >/dev/null; }   # the engine's own walls, recorded in-file
unset GALOP_EFF_LOG
echo "CAPACITY H100 COMPLETE -> $CSV (logs: $LOGD/)"
