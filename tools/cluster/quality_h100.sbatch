#!/usr/bin/env bash
# =============================================================================
# Anchor quality on the H100 (Sec. 5.2, descriptive anchors): SRGPU, Kozax and
# Beagle on the quality problems, 100 generations, best-of-run test MSE on the
# shared held-out files. Probe-timeout discipline per (system, problem, P).
#   SRGPU   parity primitives {+,-,*,/,sin,cos}, depth cap 7, MSE objective; no
#           seed control -> independent runs (25 at P=1e4, 10 at P=1e5); test
#           MSE by its own library on our test file; recorded with
#           SRGPU_INIT_DEPTH_HI=10 (export 7 for a cap-respecting rerun)
#   Kozax   8-op parity set, tournament 3, 15-node default cap (more than one
#           generation is not executable above ~51 nodes); seeds 70-79 at 1e4,
#           70-74 at 1e5; the P=1e5 tier is expected to fail attributed
#   Beagle  stock checkout + adapter, both published fitness arms (corr, std),
#           full-batch EPG, generation cut 100, 100 runs at 1e4 / 50 at 1e5 per
#           arm, California excluded by its cases-per-generation cap; test MSE
#           scored by Beagle's own VM through the BatchExecuteGenome shim; the
#           best genome of every run is archived in the CSV
# Problems: Nguyen 1-6, California (not Beagle), Feynman I.9.18, Nguyen 9/10/12.
# STAGE = srgpu | kozax | beagle | all (default). Resume-safe (resubmit until COMPLETE).
# -> results_h100/quality_anchors_h100.csv  system,problem,P,rep,outcome,wall_s,test_mse
# -> results_h100/quality_beagle_h100.csv   problem,P,fitfunc,rep,wall_to_g100,births_to_g100,test_mse,outcome,gcl
# Submit from the login node at the repo root:  sbatch tools/cluster/quality_h100.sbatch
# =============================================================================
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=48:00:00
#SBATCH --job-name=galop-quality
#SBATCH --output=results_h100/quality_slurm_%j.log
#SBATCH --error=results_h100/quality_slurm_%j.err
#SBATCH --gres=gpu:h100:1
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
STAGE=${STAGE:-all}
echo "NODE: $(hostname)  STAGE: $STAGE"
mkdir -p results_h100 /tmp/quality
python3 tools/gen_ext_datasets.py >/dev/null
CSV=results_h100/quality_anchors_h100.csv
[ -f $CSV ] || echo "system,problem,P,rep,outcome,wall_s,test_mse" > $CSV
PROBS="nguyen1:sweep nguyen2:sweep nguyen3:sweep nguyen4:sweep nguyen5:sweep nguyen6:sweep feynman_i_9_18:big california:big nguyen9:ext nguyen10:ext nguyen12:ext"

# ============================ SRGPU ============================
if [ "$STAGE" = all ] || [ "$STAGE" = srgpu ]; then
  module load StdEnv/2023 python/3.11 cuda/13.2 cmake 2>/dev/null || module load StdEnv/2023 python/3.11 cuda cmake
  SRC=$HOME/galop_srgpu_src
  if [ -d $SRC ]; then
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
    BIN=$SRC/build/srgpu_bench; [ -x $BIN ] || { echo "srgpu_bench build failed"; exit 1; }
    export SRGPU_DEPTH=7
    export SRGPU_INIT_DEPTH_HI=${SRGPU_INIT_DEPTH_HI:-10}   # as recorded (initialization above the cap, their README example); export 7 for a cap-respecting rerun
    for PP in "10000:25:1200" "100000:10:2400"; do
      P=${PP%%:*}; N=$(echo $PP | cut -d: -f2); CAP=${PP##*:}; LAST=$((N-1))
      for pr in $PROBS; do
        prob=${pr%%:*}; dir=${pr##*:}
        TR=datasets/$dir/${prob}_train.csv; TE=datasets/$dir/${prob}_test.csv
        grep -q "^srgpu,$prob,$P,$LAST," $CSV && continue
        if ! grep -q "^srgpu,$prob,$P,probe,OK" $CSV; then
          ERR=$(mktemp); T0=$(date +%s)
          OUT=$(timeout $CAP $BIN $TR $TE $P 100 2>$ERR | tail -1); RC=$?; W=$(( $(date +%s) - T0 ))
          if   [ $RC -eq 0 ] && [ -n "$OUT" ]; then O="OK"
          elif [ $RC -eq 124 ]; then O="TIMEOUT"
          elif grep -qi "out of memory\|cuda" $ERR; then O="OOM_DEVICE"
          else O="CRASH_rc$RC"; fi
          rm -f $ERR
          echo "srgpu,$prob,$P,probe,$O,$W,$(echo "$OUT" | cut -d, -f5)" >> $CSV; echo "  srgpu probe $prob P=$P -> $O"
          [ "$O" = "OK" ] || continue
        fi
        for r in $(seq 0 $LAST); do
          grep -q "^srgpu,$prob,$P,$r," $CSV && continue
          OUT=$(timeout $CAP $BIN $TR $TE $P 100 2>/dev/null | tail -1)
          [ -n "$OUT" ] && echo "srgpu,$prob,$P,$r,OK,$(echo "$OUT" | cut -d, -f3),$(echo "$OUT" | cut -d, -f5)" >> $CSV \
                        || echo "srgpu,$prob,$P,$r,FAIL,0,nan" >> $CSV
        done
        echo "  [srgpu] $prob P=$P done"
      done
    done
  else echo "srgpu snapshot missing — skipped"; fi
fi

# ============================ Kozax ============================
if [ "$STAGE" = all ] || [ "$STAGE" = kozax ]; then
  module purge 2>/dev/null; module load StdEnv/2023 python/3.13 cuda/12.6 2>/dev/null || module load StdEnv/2023 python/3.13 cuda
  source ~/.venv_kozax/bin/activate
  export KOZAX_OPS=8 KOZAX_MAX_NODES=15
  if python -c "import jax; assert any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in jax.devices())" 2>/dev/null; then
    for PP in "10000:70 71 72 73 74 75 76 77 78 79:1800" "100000:70 71 72 73 74:3600"; do
      P=${PP%%:*}; SEEDS=$(echo "$PP" | cut -d: -f2); CAP=${PP##*:}; LASTSEED=$(echo $SEEDS | awk '{print $NF}')
      for pr in $PROBS; do
        prob=${pr%%:*}; dir=${pr##*:}
        TR=datasets/$dir/${prob}_train.csv; TE=datasets/$dir/${prob}_test.csv
        grep -q "^kozax,$prob,$P,$LASTSEED," $CSV && continue
        if ! grep -q "^kozax,$prob,$P,probe,OK" $CSV; then
          ERR=$(mktemp); T0=$(date +%s)
          OUT=$(timeout $CAP python tools/baselines/kozax_bench.py $TR $TE $P 100 99 2>$ERR | grep "^KOZAXROW" | tail -1); RC=$?; W=$(( $(date +%s) - T0 ))
          if   [ -n "$OUT" ]; then O="OK"
          elif [ $RC -eq 124 ]; then O="TIMEOUT"
          elif grep -qi "RESOURCE_EXHAUSTED\|out of memory\|OOM" $ERR; then O="OOM_DEVICE"
          elif grep -qi "bad_alloc\|Killed" $ERR; then O="OOM_HOST"
          else O="CRASH_rc$RC"; fi
          rm -f $ERR
          echo "kozax,$prob,$P,probe,$O,$W,nan" >> $CSV; echo "  kozax probe $prob P=$P -> $O"
          [ "$O" = "OK" ] || continue
        fi
        for s in $SEEDS; do
          grep -q "^kozax,$prob,$P,$s," $CSV && continue
          OUT=$(timeout $CAP python tools/baselines/kozax_bench.py $TR $TE $P 100 $s 2>/dev/null | grep "^KOZAXROW" | tail -1)
          [ -n "$OUT" ] && echo "kozax,$prob,$P,$s,OK,$(echo "$OUT" | cut -d, -f5),$(echo "$OUT" | cut -d, -f7)" >> $CSV \
                        || echo "kozax,$prob,$P,$s,FAIL,0,nan" >> $CSV
        done
        echo "  [kozax] $prob P=$P done"
      done
    done
  else echo "kozax env: no GPU — skipped"; fi
  deactivate 2>/dev/null || true
fi

# ============================ Beagle ============================
if [ "$STAGE" = all ] || [ "$STAGE" = beagle ]; then
  export PATH=$HOME/.dotnet:$PATH DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
  DLL=$HOME/beagle_stock/Beagle/Run/bin/Release/net10.0/Run.dll
  if grep -q "BEAGLE_FIT" $HOME/beagle_stock/Beagle/Run/Program.cs 2>/dev/null; then
    SHIM=$(ls $HOME/beagle_stock/Beagle/Utils/BatchExecuteGenome/bin/Release/net*/BatchExecuteGenome.dll 2>/dev/null | head -1)
    if [ -z "$SHIM" ]; then
      cp -r tools/beagle_adapter/BatchExecuteGenome $HOME/beagle_stock/Beagle/Utils/
      dotnet build -c Release $HOME/beagle_stock/Beagle/Utils/BatchExecuteGenome -v q
      SHIM=$(ls $HOME/beagle_stock/Beagle/Utils/BatchExecuteGenome/bin/Release/net*/BatchExecuteGenome.dll | head -1)
    fi
    [ -f "$SHIM" ] || { echo "BatchExecuteGenome shim missing and build failed"; exit 1; }
    BCSV=results_h100/quality_beagle_h100.csv
    [ -f $BCSV ] || echo "problem,P,fitfunc,rep,wall_to_g100,births_to_g100,test_mse,outcome,gcl" > $BCSV
    for FIT in corr std; do
      export BEAGLE_FIT=$FIT
      for PP in "10000:100" "100000:50"; do
        P=${PP%%:*}; N=${PP##*:}; LAST=$((N-1))
        for pr in $PROBS; do
          prob=${pr%%:*}; dir=${pr##*:}
          [ "$prob" = california ] && continue          # excluded by Beagle's cases-per-generation cap
          TR=datasets/$dir/${prob}_train.csv; TE=datasets/$dir/${prob}_test.csv
          grep -q "^$prob,$P,$FIT,$LAST," $BCSV && continue
          for r in $(seq 0 $LAST); do
            grep -q "^$prob,$P,$FIT,$r," $BCSV && continue
            OUT=$(bash tools/beagle_adapter/run_beagle_cell.sh $DLL $TR $P 0 100 900 /tmp/quality/beagle_run.log)
            case "$OUT" in
              OK*) W=$(echo "$OUT" | cut -f2); B=$(echo "$OUT" | cut -f3); GCL=$(echo "$OUT" | cut -f4-)
                   M=$(python3 tools/beagle_adapter/shim_score.py "$GCL" $TE "$SHIM" 2>>/tmp/quality/score.err || echo "nan")
                   echo "$prob,$P,$FIT,$r,$W,$B,$M,OK,\"${GCL//\"/\"\"}\"" >> $BCSV;;
              *)   echo "$prob,$P,$FIT,$r,0,0,nan,${OUT%% *}," >> $BCSV;;
            esac
          done
          echo "  [beagle:$FIT] $prob P=$P done (n=$N)"
        done
      done
    done
  else echo "beagle adapter missing — skipped"; fi
fi
echo "QUALITY H100 ($STAGE) COMPLETE"
