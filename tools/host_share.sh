#!/usr/bin/env bash
# =============================================================================
# Where a Galop run's time goes (Sec. 5.1: the host side's share at small D).
# With GALOP_PHASE_TIMERS set the bench reports, per 100-generation run and on
# one clock: the evaluator call split into host packing of the population,
# (re)allocation of the packed buffers, host-to-device upload, the kernel from
# launch to synchronize, and the fitness download; and the engine's host work
# outside the evaluator split into top-k selection, elite copy, selection +
# variation, and statistics (the remainder is initialization and bookkeeping).
# Timing only: the hooks are off unless the variable is set and change nothing.
# Default cells: the H100 grid's populations at D=32 and D=1,024 plus P=1e5 at
# D=32,768; five seeds; OMP_NUM_THREADS=16 and the speed grid's discarded
# adaptive warm-up per cell (>= 2 s cumulative, at most three executions).
# Env: HS_CSV (output), HS_DLIST, HS_PLIST, HS_SEEDS, HS_P32768 (the populations
# run at D=32,768). Resume-safe (a cell is skipped once all its seeds are in).
#   -> results/host_share_rtx.csv  (tools/cluster/breakdown_h100.sbatch writes results_h100/breakdown_h100.csv)
#      P,D,seed,total_s,eval_s,host_other_s,pack_s,alloc_s,h2d_s,kernel_s,d2h_s,topk_s,elites_s,selvary_s,stats_s,node
# About 12 minutes on the RTX 5070 Ti.   bash tools/host_share.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
NVCC=$(command -v nvcc)
SRCS="tests/cuda_evo_bench.cu src/EvaluatorCuda.cu src/Dataset.cpp src/LGPEngine.cpp src/EvaluatorCpu.cpp src/Evaluator.cpp src/Interpreter.cpp"
NVF="-std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native -Xcompiler -fopenmp -DGALOP_STANDARD"
export OMP_NUM_THREADS=16                        # the speed grid's setting (tools/speed_grid.sh)
CSV=${HS_CSV:-results/host_share_rtx.csv}; mkdir -p "$(dirname $CSV)"
DLIST=${HS_DLIST:-"32 1024 32768"}; PLIST=${HS_PLIST:-"1000 10000 100000 1000000"}; SEEDS=${HS_SEEDS:-"70 71 72 73 74"}; P32768=${HS_P32768:-"100000"}
NODE=$(hostname -s)
[ -f $CSV ] || echo "P,D,seed,total_s,eval_s,host_other_s,pack_s,alloc_s,h2d_s,kernel_s,d2h_s,topk_s,elites_s,selvary_s,stats_s,node" > $CSV
ERR=$(mktemp)
for P in $PLIST; do
  bin=/tmp/ceb_hostshare_${P}_${SLURM_JOB_ID:-local}
  [ -x $bin ] || $NVCC $NVF -DPOPULATION_SIZE_OVERRIDE=$P -DMAX_GENERATIONS_OVERRIDE=100 $SRCS -o $bin
  for D in $DLIST; do
    [ "$D" = 32768 ] && ! echo " $P32768 " | grep -q " $P " && continue
    TR=datasets/gridsynth/nguyen1_D${D}_train.csv
    need=0; for s in $SEEDS; do grep -q "^$P,$D,$s," $CSV || need=1; done
    [ $need = 1 ] || { echo "  P=$P D=$D SKIP"; continue; }
    wtot=0                                # adaptive warm-up, discarded: >= 2 s cumulative, cap 3 (tools/speed_grid.sh)
    for wn in 1 2 3; do
      wv=$($bin $TR 99 warp 2>/dev/null | cut -d, -f6); wv=${wv:-0}
      wtot=$(awk -v a="$wtot" -v b="$wv" 'BEGIN{print a+b}')
      [ "$(awk -v t="$wtot" 'BEGIN{print (t>=2.0)?1:0}')" = 1 ] && break
    done
    for s in $SEEDS; do
      grep -q "^$P,$D,$s," $CSV && continue
      GALOP_PHASE_TIMERS=1 $bin $TR $s warp >/dev/null 2>$ERR
      B=$(sed -n 's/.*total=\([0-9.]*\)s eval=\([0-9.]*\)s.*host+other=\([0-9.]*\)s.*/\1,\2,\3/p' $ERR)
      E=$(sed -n 's/.*pack=\([0-9.]*\)s alloc=\([0-9.]*\)s h2d=\([0-9.]*\)s kernel=\([0-9.]*\)s d2h=\([0-9.]*\)s.*/\1,\2,\3,\4,\5/p' $ERR)
      PH=$(sed -n 's/.*topk=\([0-9.]*\)s elites=\([0-9.]*\)s sel+vary=\([0-9.]*\)s stats=\([0-9.]*\)s.*/\1,\2,\3,\4/p' $ERR)
      [ -n "$B" ] && [ -n "$E" ] && [ -n "$PH" ] || { echo "  P=$P D=$D seed $s: no breakdown — aborting"; cat $ERR | tail -3; exit 1; }
      echo "$P,$D,$s,$B,$E,$PH,$NODE" >> $CSV
    done
    echo "  P=$P D=$D done"
  done
  rm -f $bin
done
rm -f $ERR; echo "HOST SHARE COMPLETE -> $CSV"
