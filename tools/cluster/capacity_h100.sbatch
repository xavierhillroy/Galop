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
#   GALOP at cap width (systems galop_packed_cap50 / galop_eff_cap50): the same
#           ladders with every program initialized at the 50-instruction cap
#           (-DSTARTING_PROGRAM_SIZE_OVERRIDE=50), i.e. the largest footprint the
#           configuration allows, so that GALOP's ceilings are also reported at
#           the cap width EvoGP and Kozax pay for every individual; packed rungs
#           2e8, 3e8, 3.5e8, 4e8, 5e8; intron-removed 1.4e9, 1.8e9, 2.1e9, 2^31-2.
#           Reported as an additional, labelled arm — never replaces the
#           initializer arm.
#   Sustained generations at the ceiling (system galop_eff_10gen; STAGE=sustain
#           only, never part of 'all'): the intron-removed ceiling rung 2^31-2 run
#           for TEN true generations (MAX_GENERATIONS_OVERRIDE=11) under the same
#           build, data, host memory and threads. Instruments, none touching the
#           engine: GALOP_EFF_LOG (per-generation instruction totals, written by
#           the bench at the end of the run), GALOP_PHASE_TIMERS (host phase totals
#           printed once at the end, stderr), and nvidia-smi sampling device memory
#           every 10 s into a .gpumem trace — the only live one: a run that dies
#           mid-way leaves the trace, the .err and the SLURM log. Cap 27,000 s
#           (7.5 h): submit with --time=08:00:00. Answers whether the ceiling is a
#           working population or a first-generation upper bound.
#   Host-thread robustness row (system galop_eff_cap50_<T>thr; STAGE=cap50thr only,
#           never part of 'all'): the at-cap intron-removed ladder again from 1.8e9
#           (1.8e9, 2.1e9, 2^31-2) with T host threads instead of the criterion's 16
#           (THREADS=T, submitted with --cpus-per-task=T; the job refuses a mismatch).
#           Tests whether the time-bound at-cap ceiling is the host's; the criterion
#           rows above stay at 16 threads. The packed at-cap ladder is memory-bound
#           and is not repeated.
#   Parallel-initializer check (STAGE=pinit only, never part of 'all'): the ceiling rung
#           2^31-2 once with the shipped serial initializer (system galop_eff_sinit, a
#           same-node control with the memory trace) and once built with
#           -DGALOP_PARALLEL_INIT (galop_eff_pinit): one true generation each, the
#           10 s nvidia-smi trace dating the time to the first evaluation. The flag
#           changes initialization only (per-program PCG32 streams in an OpenMP loop,
#           buffers allocated without the zero pass); its population is a different
#           random population, so the flag is never mixed with the shipped results.
#           ~45 min:  sbatch --export=ALL,STAGE=pinit tools/cluster/capacity_h100.sbatch
# STAGE = evogp | kozax | beagle | srgpu | galop | cap50 | all (default) | sustain | cap50thr | pinit.
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
# Runtime: ~4.5 h for all (GALOP ~100 min + cap-width arm ~90 min, SRGPU ~50 min, Beagle
# ~15 min, Kozax/EvoGP minutes); the cap-width arm alone:  sbatch --export=ALL,STAGE=cap50 tools/cluster/capacity_h100.sbatch
# STAGE=sustain alone (~3 h):  sbatch --time=08:00:00 --export=ALL,STAGE=sustain tools/cluster/capacity_h100.sbatch
# STAGE=cap50thr (~1 h):       sbatch --cpus-per-task=48 --export=ALL,STAGE=cap50thr,THREADS=48 tools/cluster/capacity_h100.sbatch
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
NVCC=$(command -v nvcc); THREADS=${THREADS:-16}; export OMP_NUM_THREADS=$THREADS   # 16 = the criterion; other values only for the labelled cap50thr stage
[ "${SLURM_CPUS_PER_TASK:-$THREADS}" = "$THREADS" ] || { echo "THREADS=$THREADS but the allocation has ${SLURM_CPUS_PER_TASK} cpus: submit with --cpus-per-task=$THREADS"; exit 1; }
NODE=$(hostname); DATE=$(date +%F); STAGE=${STAGE:-all}
run_stage() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }
echo "NODE: $NODE  STAGE: $STAGE"; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
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
if run_stage evogp; then
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
fi

# ============================ Kozax (one generation, 101-node parity cap) ============================
if run_stage kozax; then
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
fi

# ============================ Beagle (one generation, EPG 64) ============================
if run_stage beagle; then
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
fi

# ============================ SRGPU (one generation, initialization under the cap) ============================
if run_stage srgpu; then
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
fi

# ============================ GALOP (one TRUE generation, both layouts) ============================
if run_stage galop || run_stage cap50 || run_stage sustain || run_stage cap50thr || run_stage pinit; then
echo "=== GALOP: gates ==="; make gpu 2>&1 | tail -1
GALOP_DIFF_QUICK=1 ./cuda_diff | tail -1 || { echo "GATE FAILED"; exit 1; }
fi
SRCS="tests/cuda_evo_bench.cu src/EvaluatorCuda.cu src/Dataset.cpp src/LGPEngine.cpp src/EvaluatorCpu.cpp src/Evaluator.cpp src/Interpreter.cpp"
NVF="-std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native -Xcompiler -fopenmp -DGALOP_STANDARD"
EXTRA=""; SUFFIX=""; GENS=2; RCAP=$CAP; THR_NOTE=""; XNOTE=""   # cap-width arm sets EXTRA/SUFFIX/XNOTE; sustain sets GENS (MAX_GENERATIONS), RCAP (its cap), SUFFIX; cap50thr sets THR_NOTE; pinit sets EXTRA/SUFFIX/XNOTE
rung() {   # rung MODE P -> one CSV row; progress lines go to the log; the outcome is read back from the CSV by last()
  local MODE=$1 P=$2 SYS=galop_$1$SUFFIX
  done_row $SYS $P && { echo "  $SYS P=$P SKIP (done: $(grep "^$SYS,$P," $CSV | tail -1 | cut -d, -f3))"; return; }
  if [ "$P" -gt 2147483647 ]; then
    echo "$SYS,$P,INT32_WALL,0,,$NODE,$DATE,POPULATION_SIZE is constexpr int: beyond 2^31-1 the engine cannot be built" >> $CSV; echo "  $SYS P=$P -> INT32_WALL"; return; fi
  local bin=/tmp/ceb_final_${MODE}${SUFFIX}_$P ERR=$LOGD/${MODE}${SUFFIX}_${P}.err
  if ! $NVCC $NVF $EXTRA -DPOPULATION_SIZE_OVERRIDE=$P -DMAX_GENERATIONS_OVERRIDE=$GENS $SRCS -o $bin 2>$ERR; then
    echo "$SYS,$P,BUILD_FAIL,0,,$NODE,$DATE,$(tr -d ',\n' < $ERR | tail -c 90)" >> $CSV; echo "  $SYS P=$P -> BUILD_FAIL (see $ERR)"; return; fi
  if [ "$MODE" = eff ]; then export GALOP_EFF_LOG=$LOGD/${MODE}${SUFFIX}_${P}.efflog; else unset GALOP_EFF_LOG; fi
  local T0=$(date +%s); timeout $RCAP $bin $TR 70 warp >$LOGD/${MODE}${SUFFIX}_${P}.out 2>$ERR; local RC=$?; local W=$(( $(date +%s) - T0 ))
  local O=$(classify $RC $ERR)
  local EV=$(grep -o 'total=[0-9.]*' $ERR | tail -1 | cut -d= -f2); EV=${EV:-0}
  echo "$SYS,$P,$O,$W,$EV,$NODE,$DATE,MAX_GENERATIONS=$GENS$XNOTE$THR_NOTE $(tr -d ',\n' < $ERR | tail -c 100)" >> $CSV
  echo "  $SYS P=$P -> $O (process ${W}s, evolve ${EV}s)"; rm -f $bin
}
last() { grep "^galop_$1$SUFFIX,$2," $CSV | tail -1 | cut -d, -f3; }   # outcome word of the (mode, P) row
if run_stage galop; then
echo "=== GALOP packed ==="
for P in 1000000000 1400000000 1800000000; do rung packed $P; [ "$(last packed $P)" = OK ] || break; done
echo "=== GALOP intron-removed ==="
for P in 1400000000 1800000000 2100000000 2147483646; do rung eff $P; [ "$(last eff $P)" = OK ] || break; done
[ "$(last eff 2147483646)" = OK ] && { rung eff 2147483647; rung eff 2200000000; }   # the engine's own walls, recorded in-file
unset GALOP_EFF_LOG
fi
if run_stage cap50; then
EXTRA="-DSTARTING_PROGRAM_SIZE_OVERRIDE=50"; SUFFIX="_cap50"; XNOTE=" STARTING_PROGRAM_SIZE=50"
echo "=== GALOP at cap width: packed ==="
for P in 200000000 300000000 350000000 400000000 500000000; do rung packed $P; [ "$(last packed $P)" = OK ] || break; done
echo "=== GALOP at cap width: intron-removed ==="
for P in 1400000000 1800000000 2100000000 2147483646; do rung eff $P; [ "$(last eff $P)" = OK ] || break; done
unset GALOP_EFF_LOG; EXTRA=""; SUFFIX=""; XNOTE=""
fi
if run_stage cap50thr; then
[ "$THREADS" != 16 ] || { echo "cap50thr is the host-thread robustness row: set THREADS to a value other than 16 (e.g. --cpus-per-task=48 --export=ALL,STAGE=cap50thr,THREADS=48)"; exit 1; }
EXTRA="-DSTARTING_PROGRAM_SIZE_OVERRIDE=50"; SUFFIX="_cap50_${THREADS}thr"; XNOTE=" STARTING_PROGRAM_SIZE=50"; THR_NOTE=" THREADS=$THREADS"
echo "=== GALOP at cap width, intron-removed, $THREADS host threads (robustness row; the criterion is 16) ==="
for P in 1800000000 2100000000 2147483646; do rung eff $P; [ "$(last eff $P)" = OK ] || break; done
unset GALOP_EFF_LOG; EXTRA=""; SUFFIX=""; XNOTE=""; THR_NOTE=""
fi
if run_stage sustain; then
GENS=11; RCAP=${SUSTAIN_CAP:-27000}; SUFFIX="_10gen"; P=2147483646
echo "=== GALOP intron-removed ceiling P=$P: ten true generations (MAX_GENERATIONS=$GENS, cap ${RCAP}s) ==="
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 10 > $LOGD/eff${SUFFIX}_${P}.gpumem 2>/dev/null &
SMI=$!
export GALOP_PHASE_TIMERS=1; rung eff $P; unset GALOP_PHASE_TIMERS
kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
unset GALOP_EFF_LOG; GENS=2; RCAP=$CAP; SUFFIX=""
fi
if run_stage pinit; then
P=2147483646
for ARM in sinit pinit; do
  if [ $ARM = pinit ]; then EXTRA="-DGALOP_PARALLEL_INIT"; XNOTE=" GALOP_PARALLEL_INIT"; else EXTRA=""; XNOTE=" serial initializer (control)"; fi
  SUFFIX="_$ARM"
  echo "=== GALOP intron-removed ceiling P=$P, one generation, $ARM ==="
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 10 > $LOGD/eff${SUFFIX}_${P}.gpumem 2>/dev/null &
  SMI=$!
  rung eff $P
  kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
done
unset GALOP_EFF_LOG; EXTRA=""; SUFFIX=""; XNOTE=""
fi
echo "CAPACITY H100 COMPLETE -> $CSV (logs: $LOGD/)"
