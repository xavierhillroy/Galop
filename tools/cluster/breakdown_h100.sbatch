#!/usr/bin/env bash
# =============================================================================
# Where a Galop run's time goes on the H100 (Sec. 5.1). Runs the shared producer
# tools/host_share.sh on one H100 at the speed grid's cells D in {32, 1,024,
# 32,768} x P in {1e3, 1e4, 1e5, 1e6}: the speed grid's configuration and
# protocol (OMP_NUM_THREADS=16, a discarded adaptive warm-up per cell, seeds
# 70-74), with the bench's timers switched on (GALOP_PHASE_TIMERS; timing only).
# Per run the evaluator call is split into host packing of the population,
# growth of the packed buffers, host-to-device upload, the kernel from launch to
# synchronize and the fitness download; the host work outside it into top-k
# selection, elite copy, selection + variation and statistics.
# Env: DLIST (default "32 1024 32768"; "32 1024 32768 131072" adds the grid's
# fourth D, about 75 min more), PLIST, SEEDS. Resume-safe (resubmit until COMPLETE).
# -> results_h100/breakdown_h100.csv  (the columns of results/host_share_rtx.csv, node per row)
# Submit from the login node at the repo root:  sbatch tools/cluster/breakdown_h100.sbatch   (~35 min)
# =============================================================================
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=galop-breakdown
#SBATCH --output=results_h100/breakdown_slurm_%j.log
#SBATCH --error=results_h100/breakdown_slurm_%j.err
#SBATCH --gres=gpu:h100:1
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load StdEnv/2023 python/3.11 cuda/13.2 2>/dev/null || module load StdEnv/2023 python/3.11 cuda
DLIST=${DLIST:-"32 1024 32768"}; PLIST=${PLIST:-"1000 10000 100000 1000000"}; SEEDS=${SEEDS:-"70 71 72 73 74"}
echo "NODE: $(hostname)  DLIST: $DLIST  PLIST: $PLIST  SEEDS: $SEEDS"
mkdir -p results_h100 datasets/gridsynth
DL="$DLIST" python3 - <<'GSEOF'
import random, os
random.seed(1234)
for D in [int(x) for x in os.environ["DL"].split()]:
    p = f"datasets/gridsynth/nguyen1_D{D}_train.csv"      # legacy filename: the content is the Nguyen-2 quartic (tools/speed_grid.sh)
    if os.path.exists(p): continue
    with open(p, "w") as f:
        f.write("x,y\n")
        for _ in range(D):
            x = random.uniform(-1, 1); f.write(f"{x},{x**4+x**3+x**2+x}\n")
GSEOF
echo "=== GATES ==="; make gpu 2>&1 | tail -1
GALOP_DIFF_QUICK=1 ./cuda_diff | tail -1 || { echo "GATE FAILED"; exit 1; }
HS_CSV=results_h100/breakdown_h100.csv HS_DLIST="$DLIST" HS_PLIST="$PLIST" HS_SEEDS="$SEEDS" HS_P32768="$PLIST" bash tools/host_share.sh || exit 1
echo "BREAKDOWN H100 COMPLETE -> results_h100/breakdown_h100.csv"
