#!/usr/bin/env bash
# =============================================================================
# Anchor grids on the H100 (Sec. 5.1, the descriptive anchors): SRGPU, Kozax
# and Beagle on the same 16 (P, D) cells as the pair grid, 100 generations,
# probe-timeout discipline per cell — a probe run under the cap decides whether
# the cell is feasible; if it is, one discarded warm-up run then 5 timed runs
# (seeds 70-74 where the system is seedable: Kozax; independent repetitions
# otherwise). Every cell ends as a number or an attributed outcome
# (OK / TIMEOUT / OOM_DEVICE / OOM_HOST / CRASH / ARCH_NA).
#   SRGPU   retained snapshot, parity primitives, depth cap 7; recorded with
#           SRGPU_INIT_DEPTH_HI=10 (initialization above the cap, its README
#           example) — drop that export for a cap-respecting rerun
#   Kozax   8-op parity set, tournament 3, 15-node default cap (more than one
#           generation is not executable above ~51 nodes: XLA sort shared memory)
#   Beagle  stock checkout + adapter, full-batch EPG=D for D <= 1024;
#           D > 1024 is ARCH_NA (its 1,024-cases-per-generation design cap);
#           wall = its own log clock at generation 100; correlation fitness
# STAGE = srgpu | kozax | beagle | all (default). Resume-safe.
# -> results_h100/anchors_grid_h100.csv   system,P,D,rep,outcome,wall_s,note
# Submit from the login node at the repo root:  sbatch tools/cluster/anchors_h100.sbatch
# =============================================================================
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --job-name=galop-anchors
#SBATCH --output=results_h100/anchors_slurm_%j.log
#SBATCH --error=results_h100/anchors_slurm_%j.err
#SBATCH --gres=gpu:h100:1
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
STAGE=${STAGE:-all}
echo "NODE: $(hostname)  STAGE: $STAGE"
mkdir -p results_h100 datasets/gridsynth /tmp/anchors
python3 - <<'PYEOF'
import random, os
random.seed(1234)
for D in (32, 1024, 32768, 131072):
    for part in ("train", "test"):
        p = f"datasets/gridsynth/nguyen1_D{D}_{part}.csv"      # legacy filename: the content is the Nguyen-2 quartic
        if os.path.exists(p): continue
        with open(p, "w") as f:
            f.write("x,y\n")
            for _ in range(D):
                x = random.uniform(-1, 1); f.write(f"{x},{x**4+x**3+x**2+x}\n")
PYEOF
CSV=results_h100/anchors_grid_h100.csv
[ -f $CSV ] || echo "system,P,D,rep,outcome,wall_s,note" > $CSV
classify() {   # RC ERRFILE -> outcome word
  local RC=$1 ERR=$2
  if   [ $RC -eq 0 ];   then echo OK
  elif [ $RC -eq 124 ]; then echo TIMEOUT
  elif grep -qi "out of memory\|cudaErrorMemoryAllocation\|RESOURCE_EXHAUSTED\|OOM\|cuda" $ERR; then echo OOM_DEVICE
  elif grep -qi "bad_alloc\|Cannot allocate\|Killed" $ERR; then echo OOM_HOST
  else echo "CRASH_rc$RC"; fi
}

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
    for P in 1000 10000 100000 1000000; do
      for D in 32 1024 32768 131072; do
        grep -q "^srgpu,$P,$D,5," $CSV && continue
        TR=datasets/gridsynth/nguyen1_D${D}_train.csv; TE=datasets/gridsynth/nguyen1_D${D}_test.csv
        if ! grep -q "^srgpu,$P,$D,probe,OK" $CSV; then
          ERR=$(mktemp); T0=$(date +%s)
          timeout 1200 $BIN $TR $TE $P 100 >/dev/null 2>$ERR; RC=$?; W=$(( $(date +%s) - T0 ))
          O=$(classify $RC $ERR); rm -f $ERR
          echo "srgpu,$P,$D,probe,$O,$W," >> $CSV; echo "  srgpu probe P=$P D=$D -> $O (${W}s)"
          [ "$O" = "OK" ] || continue
        fi
        for r in 0 1 2 3 4 5; do          # r=0 is the discarded warm-up
          grep -q "^srgpu,$P,$D,$r," $CSV && continue
          w=$(timeout 2400 $BIN $TR $TE $P 100 2>/dev/null | tail -1 | cut -d, -f3)
          [ $r -eq 0 ] || echo "srgpu,$P,$D,$r,OK,${w:-FAIL}," >> $CSV
        done
        echo "  [srgpu] P=$P D=$D done"
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
    kozax_cell() {   # P D seed timeout -> "outcome,wall"
      local ERR; ERR=$(mktemp); local T0=$(date +%s)
      timeout $4 python tools/baselines/kozax_bench.py datasets/gridsynth/nguyen1_D$2_train.csv datasets/gridsynth/nguyen1_D$2_test.csv $1 100 $3 >/dev/null 2>$ERR; local RC=$?
      local W=$(( $(date +%s) - T0 )); local O=$(classify $RC $ERR); rm -f $ERR; echo "$O,$W"
    }
    for P in 1000 10000 100000 1000000; do
      for D in 32 1024 32768 131072; do
        grep -q "^kozax,$P,$D,74," $CSV && continue
        if ! grep -q "^kozax,$P,$D,probe,OK" $CSV; then
          R=$(kozax_cell $P $D 99 1800); echo "kozax,$P,$D,probe,${R}," >> $CSV; echo "  kozax probe P=$P D=$D -> $R"
          [ "${R%%,*}" = "OK" ] || continue
        fi
        for s in 70 71 72 73 74; do
          grep -q "^kozax,$P,$D,$s," $CSV && continue
          R=$(kozax_cell $P $D $s 3600); echo "kozax,$P,$D,$s,${R}," >> $CSV
        done
        echo "  [kozax] P=$P D=$D done"
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
    export BEAGLE_FIT=corr
    for P in 1000 10000 100000 1000000; do
      for D in 32 1024 32768 131072; do
        if [ $D -gt 1024 ]; then
          grep -q "^beagle,$P,$D,arch," $CSV || echo "beagle,$P,$D,arch,ARCH_NA,0,design cap: <=1024 cases per generation" >> $CSV
          continue
        fi
        grep -q "^beagle,$P,$D,5," $CSV && continue
        TR=datasets/gridsynth/nguyen1_D${D}_train.csv
        for r in 0 1 2 3 4 5; do          # r=0 is the discarded warm-up
          grep -q "^beagle,$P,$D,$r," $CSV && continue
          OUT=$(bash tools/beagle_adapter/run_beagle_cell.sh $DLL $TR $P $D 100 1200 /tmp/anchors/beagle_${P}_${D}_${r}.log)
          case "$OUT" in
            OK*) W=$(echo "$OUT" | cut -f2); B=$(echo "$OUT" | cut -f3); [ $r -eq 0 ] || echo "beagle,$P,$D,$r,OK,$W,births=$B" >> $CSV;;
            *)   [ $r -eq 0 ] || echo "beagle,$P,$D,$r,${OUT%% *},0," >> $CSV;;
          esac
        done
        echo "  [beagle] P=$P D=$D done"
      done
    done
  else echo "beagle adapter missing — skipped (bash tools/beagle_adapter/install_beagle_adapter.sh ~/beagle_stock on the login node)"; fi
fi
echo "ANCHORS H100 ($STAGE) COMPLETE -> $CSV"
