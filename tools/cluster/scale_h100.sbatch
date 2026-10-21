#!/usr/bin/env bash
# =============================================================================
# Whole runs at the top of the ladders (fig6b, Sec. 5.3): P=1e8 x 100
# generations on the shared D=64 quartic, one H100, for the systems whose
# ceilings admit it. Speed-experiment protocol: a discarded warm-up, then 5
# timed runs (seeds 70-74 where seedable), median reported.
#   galop   standard configuration; two host-thread settings: 16 threads (the
#           uniform setting of the comparisons) and 48 (the full node)
#   beagle  stock checkout + adapter, EPG 64, correlation fitness, generation
#           cut 100, wall = its own log clock; one discarded warm-up (colony
#           1e6, 5 generations), then 5 runs under a 10,800 s cap each
#   p1e9    (STAGE=p1e9 only, never part of 'all') the order-of-magnitude step:
#           P=1e9 x 100 generations at the uniform 16 threads, intron-removed
#           layout (the packed layout's device footprint at the evolved program
#           shape would not fit 1e9), seeds 70-72, no discarded warm-up (a run is
#           ~1.3 h); the P=1e8 trio in the same layout is the layout control for
#           the per-individual-generation comparison with the packed rows above.
#           Per run: efflog (per-generation instruction totals, written at the
#           end) and a 10 s nvidia-smi memory trace (dates the one-time setup) in
#           results_h100/scale_logs/. Needs --mem=600G (host stride 410 GB):
#           sbatch --mem=600G --export=ALL,STAGE=p1e9 tools/cluster/scale_h100.sbatch  (~4.5 h)
#   pcap    (STAGE=pcap only, never part of 'all') the same 100-generation run at the
#           intron-removed ceiling P=2^31-2, one seed (SEEDS=70 by default), same layout,
#           threads and instruments. A gamble by design: effective code grows during
#           the run and the device budget at this P allows about 5 effective
#           instructions per individual, so the run may end in a device OOM; a FAIL
#           row with the error, the memory trace and the SLURM log are then the record
#           (the efflog is written only at the end). Needs --mem=1000G (host stride
#           880 GB):  sbatch --mem=1000G --export=ALL,STAGE=pcap tools/cluster/scale_h100.sbatch  (~3 h per seed)
#           Both stages take SEEDS (a space-separated list) and can be submitted one
#           job per seed to run concurrently: rows are keyed (P, seed), logs are per
#           (P, seed), each job builds its own binary:
#             for s in 73 74; do STAGE=p1e9 SEEDS=$s sbatch --mem=600G --export=ALL tools/cluster/scale_h100.sbatch; done
#             for s in 71 72 73 74; do STAGE=pcap SEEDS=$s sbatch --mem=1000G --export=ALL tools/cluster/scale_h100.sbatch; done
# STAGE = galop | beagle | all (default) | p1e9 | pcap. Resume-safe (resubmit until COMPLETE).
# -> results_h100/scale_h100.csv   system,P,D,threads,rep,outcome,wall_s,note
# Submit from the login node at the repo root:  sbatch tools/cluster/scale_h100.sbatch
# =============================================================================
#SBATCH --cpus-per-task=48
#SBATCH --mem=250G
#SBATCH --time=20:00:00
#SBATCH --job-name=galop-scale
#SBATCH --output=results_h100/scale_slurm_%j.log
#SBATCH --error=results_h100/scale_slurm_%j.err
#SBATCH --gres=gpu:h100:1
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
STAGE=${STAGE:-all}
echo "NODE: $(hostname)  STAGE: $STAGE"
mkdir -p results_h100 datasets/gridsynth /tmp/scale
python3 - <<'GSEOF'
import random, os
random.seed(1234)
p = "datasets/gridsynth/nguyen1_D64_train.csv"      # legacy filename: the content is the Nguyen-2 quartic
if not os.path.exists(p):
    with open(p, "w") as f:
        f.write("x,y\n")
        for _ in range(64):
            x = random.uniform(-1, 1); f.write(f"{x},{x**4+x**3+x**2+x}\n")
GSEOF
TR=datasets/gridsynth/nguyen1_D64_train.csv
CSV=results_h100/scale_h100.csv
[ -f $CSV ] || echo "system,P,D,threads,rep,outcome,wall_s,note" > $CSV
P=100000000

# ============================ GALOP: 16 threads, then the full node ============================
if [ "$STAGE" = all ] || [ "$STAGE" = galop ]; then
  module load StdEnv/2023 python/3.11 cuda/13.2 2>/dev/null || module load StdEnv/2023 python/3.11 cuda
  NVCC=$(command -v nvcc)
  SRCS="tests/cuda_evo_bench.cu src/EvaluatorCuda.cu src/Dataset.cpp src/LGPEngine.cpp src/EvaluatorCpu.cpp src/Evaluator.cpp src/Interpreter.cpp"
  NVF="-std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native -Xcompiler -fopenmp -DGALOP_STANDARD"
  echo "=== GATES ==="; make gpu 2>&1 | tail -1
  GALOP_DIFF_QUICK=1 ./cuda_diff | tail -1 || { echo "GATE FAILED"; exit 1; }
  bin=/tmp/ceb_scale_$P
  [ -x $bin ] || $NVCC $NVF -DPOPULATION_SIZE_OVERRIDE=$P -DMAX_GENERATIONS_OVERRIDE=100 $SRCS -o $bin 2>/dev/null
  for T in 16 $SLURM_CPUS_PER_TASK; do
    export OMP_NUM_THREADS=$T
    grep -q "^galop,$P,64,$T,74," $CSV && { echo "  galop $T threads SKIP"; continue; }
    wtot=0                                # adaptive warm-up, discarded (these cells need one run)
    for wn in 1 2 3; do
      wv=$($bin $TR 99 warp 2>/dev/null | cut -d, -f6); wv=${wv:-0}
      wtot=$(awk -v a="$wtot" -v b="$wv" 'BEGIN{print a+b}')
      [ "$(awk -v t="$wtot" 'BEGIN{print (t>=2.0)?1:0}')" = 1 ] && break
    done
    for s in 70 71 72 73 74; do
      grep -q "^galop,$P,64,$T,$s," $CSV && continue
      w=$($bin $TR $s warp 2>/dev/null | cut -d, -f6)
      echo "galop,$P,64,$T,$s,OK,$w," >> $CSV; echo "  galop $T threads seed $s: ${w}s"
    done
  done
fi

# ============================ GALOP: the order-of-magnitude step (STAGE=p1e9 only) ============================
if [ "$STAGE" = p1e9 ] || [ "$STAGE" = pcap ]; then
  if [ "$STAGE" = p1e9 ]; then PLIST="100000000 1000000000"; SEEDS=${SEEDS:-"70 71 72"}; NEEDMB=550000; NEEDG=600G
  else PLIST="2147483646"; SEEDS=${SEEDS:-70}; NEEDMB=950000; NEEDG=1000G; fi
  [ "${SLURM_MEM_PER_NODE:-0}" -ge $NEEDMB ] || { echo "STAGE=$STAGE needs --mem=$NEEDG (host stride 410 B per individual): sbatch --mem=$NEEDG --export=ALL,STAGE=$STAGE tools/cluster/scale_h100.sbatch"; exit 1; }
  module load StdEnv/2023 python/3.11 cuda/13.2 2>/dev/null || module load StdEnv/2023 python/3.11 cuda
  NVCC=$(command -v nvcc)
  SRCS="tests/cuda_evo_bench.cu src/EvaluatorCuda.cu src/Dataset.cpp src/LGPEngine.cpp src/EvaluatorCpu.cpp src/Evaluator.cpp src/Interpreter.cpp"
  NVF="-std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native -Xcompiler -fopenmp -DGALOP_STANDARD"
  echo "=== GATES ==="; make gpu 2>&1 | tail -1
  GALOP_DIFF_QUICK=1 ./cuda_diff | tail -1 || { echo "GATE FAILED"; exit 1; }
  mkdir -p results_h100/scale_logs; export OMP_NUM_THREADS=16
  for PP in $PLIST; do
    bin=/tmp/ceb_scale_${PP}_${SLURM_JOB_ID:-$$}   # same build as the packed rows (the layout is chosen at run time by GALOP_EFF_LOG); one binary per job, so jobs sharing a node never race on it
    [ -x $bin ] || $NVCC $NVF -DPOPULATION_SIZE_OVERRIDE=$PP -DMAX_GENERATIONS_OVERRIDE=100 $SRCS -o $bin 2>/dev/null
    for s in $SEEDS; do
      grep -q "^galop_eff,$PP,64,16,$s," $CSV && { echo "  galop_eff P=$PP seed $s SKIP"; continue; }
      LOG=results_h100/scale_logs/galop_eff_${PP}_s$s
      nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 10 > $LOG.gpumem 2>/dev/null & SMI=$!
      T0=$(date +%s); w=$(GALOP_EFF_LOG=$LOG.efflog $bin $TR $s warp 2>$LOG.err | tee $LOG.out | cut -d, -f6); W=$(( $(date +%s) - T0 ))
      kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
      if [ -n "$w" ]; then O=OK; NOTE="intron-removed layout; 100 generations; no discarded warm-up; node $(hostname -s)"
      else O=FAIL; w=$W; NOTE="intron-removed layout; run died, wall_s is the process wall; no discarded warm-up; node $(hostname -s)"; fi
      echo "galop_eff,$PP,64,16,$s,$O,$w,$NOTE; $(tr -d ',\n' < $LOG.err | tail -c 80)" >> $CSV
      echo "  galop_eff P=$PP seed $s: ${w}s ($O)"
    done
    rm -f $bin
  done
fi

# ============================ Beagle ============================
if [ "$STAGE" = all ] || [ "$STAGE" = beagle ]; then
  export PATH=$HOME/.dotnet:$PATH DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
  DLL=$HOME/beagle_stock/Beagle/Run/bin/Release/net10.0/Run.dll
  if grep -q "BEAGLE_FIT" $HOME/beagle_stock/Beagle/Run/Program.cs 2>/dev/null; then
    export BEAGLE_FIT=corr
    if ! grep -q "^beagle,$P,64,,warmup," $CSV; then
      OUT=$(bash tools/beagle_adapter/run_beagle_cell.sh $DLL $TR 1000000 64 5 600 /tmp/scale/beagle_warm.log)
      echo "beagle,$P,64,,warmup,$(echo "$OUT" | cut -f1 | tr ' ' '_'),$(echo "$OUT" | cut -f2 2>/dev/null || echo 0),discarded warm-up: colony 1e6 5 generations" >> $CSV
      echo "  beagle warm-up (discarded): $OUT"
    fi
    for r in 0 1 2 3 4; do
      grep -q "^beagle,$P,64,,$r," $CSV && continue
      OUT=$(bash tools/beagle_adapter/run_beagle_cell.sh $DLL $TR $P 64 100 10800 /tmp/scale/beagle_rep$r.log)
      case "$OUT" in
        OK*) W=$(echo "$OUT" | cut -f2); B=$(echo "$OUT" | cut -f3); echo "beagle,$P,64,,$r,OK,$W,births_g100=$B" >> $CSV; echo "  beagle rep $r = ${W}s";;
        *)   echo "beagle,$P,64,,$r,$(echo "$OUT" | tr ' \t' '__'),0," >> $CSV; echo "  beagle rep $r -> $OUT";;
      esac
    done
  else echo "beagle adapter missing — skipped"; fi
fi
echo "SCALE H100 ($STAGE) COMPLETE -> $CSV"
