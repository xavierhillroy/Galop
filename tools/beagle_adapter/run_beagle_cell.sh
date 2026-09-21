#!/usr/bin/env bash
# Run one Beagle cell until generation GENCUT appears in the log (then kill),
# or until CAP seconds (then kill). Their StopAfterMin only checks between
# generations, so external supervision is mandatory.
# Usage: run_beagle_cell.sh <Run.dll> <train.csv> <colony> <epg> <gencut> <cap_s> <log>
# Stdout: "OK<TAB>wall_s<TAB>gcl"  |  "TIMEOUT"  |  "DIED rc=<rc>"
set -uo pipefail
DLL=$1; CSV=$2; COLONY=$3; EPG=$4; GENCUT=$5; CAP=$6; LOG=$7
SELF=$(cd "$(dirname "$0")" && pwd)
export BEAGLE_CSV=$CSV BEAGLE_COLONY=$COLONY
if [ "$EPG" != "0" ]; then export BEAGLE_EPG=$EPG; else unset BEAGLE_EPG; fi   # 0 = auto (row count, capped 1024)
export PATH=$HOME/.dotnet:$PATH DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
: > "$LOG"
dotnet "$DLL" NoEscMenu > "$LOG" 2>&1 &
PID=$!
T0=$(date +%s)
while true; do
  sleep 2
  # dispatch guard: the engine announces its setup early; anything but our
  # CsvMLSetup means the adapter didn't engage —
  # fail LOUDLY rather than time a wrong workload
  if grep -q "Running .*-CorrelationFitFunc" "$LOG" 2>/dev/null && ! grep -q "Running CsvMLSetup" "$LOG" 2>/dev/null; then
    kill $PID 2>/dev/null; wait $PID 2>/dev/null
    echo "DIED rc=WRONG-SETUP ($(grep -m1 -o 'Running [^ ]*' "$LOG"))"; exit 0
  fi
  # epg0-freeze guard: a legitimate ASR is a ratio <= 1. "Infinity"/"NaN" is
  # the fingerprint of MaxGenerationScore==0 (adapter ctor-order defect,
  # fixed 2026-08-31) — abort loudly rather than record a frozen genome.
  if grep -q "ASR = Infinity\|ASR = NaN" "$LOG"; then
    kill $PID 2>/dev/null; wait $PID 2>/dev/null
    echo "DIED rc=EPG0-FREEZE (MaxGenerationScore==0 fingerprint in log)"; exit 0
  fi
  if grep -q "Generation ${GENCUT} took\|Generation $(printf "%'d" ${GENCUT} 2>/dev/null) took" "$LOG" 2>/dev/null; then
    kill $PID 2>/dev/null; wait $PID 2>/dev/null
    OUT=$(python3 "$SELF/beagle_gen100.py" "$LOG" "$GENCUT")
    case "$OUT" in FAIL*) echo "DIED rc=parse";; *) echo -e "OK\t$OUT";; esac
    exit 0
  fi
  if ! kill -0 $PID 2>/dev/null; then echo "DIED rc=$(wait $PID 2>/dev/null; echo $?)"; exit 0; fi
  if [ $(( $(date +%s) - T0 )) -ge "$CAP" ]; then
    kill $PID 2>/dev/null; wait $PID 2>/dev/null; echo "TIMEOUT"; exit 0
  fi
done
