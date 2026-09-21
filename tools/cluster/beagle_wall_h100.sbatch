#!/usr/bin/env bash
# =============================================================================
# Beagle wall attribution (Sec. 5.3) — names the resource that ends Beagle's ladder.
# The 5.6e8 rung dies with ~8x time headroom (gen-1 projects to ~157 s under a
# 1,200 s cap), so the wall is not time-bound; and their design batches around
# device memory (tutorial s2.2), so a device-OOM attribution would contradict
# their documented behavior. This captures WHAT actually ends it: full stderr,
# exit code, peak host RSS, and nvidia-smi device memory at 1 Hz.
# Runs the failing 5.6e8 rung under the full 1 TB budget with no time pressure
# and samples peak host RSS and device memory, so the outcome separates cleanly:
# completes -> the ladder's ceiling was a memory budget; dies with RSS far under
# the limit -> an implementation wall with a named signature (exit code + stderr).
# One rung, ~20 min. -> results_h100/beagle_wall_h100.txt (+ .log/.err/.smi)
# Submit from the login node at the repo root:  sbatch tools/cluster/beagle_wall_h100.sbatch
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000G
#SBATCH --time=01:00:00
#SBATCH --job-name=galop-beagle-wall
#SBATCH --output=results_h100/beagle_wall_slurm_%j.log
#SBATCH --gres=gpu:h100:1
#SBATCH --error=results_h100/beagle_wall_slurm_%j.err
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "NODE: $(hostname)"; free -g | head -2
mkdir -p results_h100 /tmp/bgdiag
python3 - <<'GSEOF'
import random, os
random.seed(1234)
p = "datasets/gridsynth/nguyen1_D64_train.csv"
if not os.path.exists(p):
    os.makedirs("datasets/gridsynth", exist_ok=True)
    with open(p, "w") as f:
        f.write("x,y\n")
        for _ in range(64):
            x = random.uniform(-1, 1); f.write(f"{x},{x**4+x**3+x**2+x}\n")
GSEOF
DLL=$HOME/beagle_stock/Beagle/Run/bin/Release/net10.0/Run.dll
grep -q "GALOP-adapter" $HOME/beagle_stock/Beagle/Run/Program.cs || { echo "adapter missing"; exit 1; }
export PATH=$HOME/.dotnet:$PATH DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
export BEAGLE_CSV=datasets/gridsynth/nguyen1_D64_train.csv BEAGLE_COLONY=560000000 BEAGLE_EPG=64
OUT=results_h100/beagle_wall_h100.txt
CG=$( { cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null; } | head -1)
dotnet $DLL NoEscMenu > results_h100/beagle_wall_h100.log 2> results_h100/beagle_wall_h100.err &
PID=$!
PEAK_KB=0
: > results_h100/beagle_wall_h100.smi
while kill -0 $PID 2>/dev/null; do
  R=$(awk '/VmHWM/{print $2}' /proc/$PID/status 2>/dev/null)
  [ -n "$R" ] && PEAK_KB=$R
  echo "$(date +%s) rss_kB=$PEAK_KB gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)" \
    >> results_h100/beagle_wall_h100.smi
  sleep 2
done
wait $PID; RC=$?
{
  echo "=== BEAGLE WALL DIAGNOSTIC: colony 5.6e8, D=64, no time pressure ==="
  echo "exit code: $RC"
  echo "host budget (cgroup bytes): $CG"
  echo "peak host RSS: $(awk -v k=$PEAK_KB 'BEGIN{printf \"%.1f GB\", k/1048576}')"
  echo "--- last 30 lines of stdout ---"; tail -30 results_h100/beagle_wall_h100.log
  echo "--- stderr (failure signature) ---"; tail -40 results_h100/beagle_wall_h100.err
  echo "--- last 3 samples (rss / device memory) ---"; tail -3 results_h100/beagle_wall_h100.smi
  echo "--- host memory ---"; free -g | head -2
} > $OUT
cat $OUT
echo "BEAGLE WALL COMPLETE -> $OUT"
