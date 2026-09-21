#!/usr/bin/env bash
# Install the GALOP campaign adapter into a Beagle checkout and rebuild.
# Harness-level only: adds CsvMLSetup.cs + one dispatch branch in Program.cs
# (env BEAGLE_CSV triggers it); the evolutionary engine is untouched.
# Idempotent. Usage: install_beagle_adapter.sh <beagle-checkout-dir>
set -euo pipefail
SELF=$(cd "$(dirname "$0")" && pwd)
B=${1:?beagle checkout dir}
cp "$SELF/CsvMLSetup.cs" "$B/Beagle/Run/MLSetups/CsvMLSetup.cs"
python3 - "$B/Beagle/Run/Program.cs" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
# 1) strip any previously inserted adapter branch (broken or stale)
s = re.sub(r"[ \t]*// GALOP-adapter dispatch.*?return;\s*\}\s*\n", "", s, flags=re.S)
# 2) mirror the Train(...) argument list from the stock RunFeynman call, so the
#    branch compiles against WHATEVER Train signature this upstream commit has
m = re.search(r"mlEngine\.Train\(([^;]*)\);", s)
assert m, "no stock mlEngine.Train call found to mirror"
targs = m.group(1)
anchor = "        #region RunFeynmen"
assert anchor in s, "Program.cs anchor not found"
branch = f"""        // GALOP-adapter dispatch: env BEAGLE_CSV routes to the CSV setup (engine untouched).
        // BEAGLE_FIT selects between Beagle's two published fitness functions:
        //   corr (default; their best-published config) | std (their point-to-point default)
        if (Environment.GetEnvironmentVariable("BEAGLE_CSV") != null)
        {{
            var fit = Environment.GetEnvironmentVariable("BEAGLE_FIT") ?? "corr";
            Console.WriteLine($"GALOP-adapter: CsvMLSetup + fitfunc={{fit}}");
            if (fit == "std")
            {{
                using var csvEngineS = new MLEngine<CsvMLSetup, StdFitFunc>(forceCPUAccelerator: false);
                csvEngineS.Train({targs});
            }}
            else
            {{
                using var csvEngine = new MLEngine<CsvMLSetup, CorrelationFitFunc>(forceCPUAccelerator: false);
                csvEngine.Train({targs});
            }}
            return;
        }}

{anchor}"""
s = s.replace(anchor, branch, 1)
open(p, "w").write(s)
print(f"Program.cs patched (Train args mirrored from stock: {targs.strip()})")
PYEOF
export PATH=$HOME/.dotnet:$PATH DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
cd "$B"
BLOG=$(mktemp)
if ! dotnet build Beagle/Run/Run.csproj -c Release > "$BLOG" 2>&1; then
  echo "========== BUILD FAILED — full compiler output below =========="
  grep -E "error|Error|FAILED" "$BLOG" | head -20
  tail -15 "$BLOG"
  rm -f "$BLOG"
  exit 1
fi
rm -f "$BLOG"
DLL="$B/Beagle/Run/bin/Release/net10.0/Run.dll"
if strings "$DLL" 2>/dev/null | grep -q "CsvMLSetup"; then
  echo "VERIFIED: compiled binary contains CsvMLSetup"
else
  echo "BUILD SUCCEEDED BUT BINARY LACKS CsvMLSetup — wrong output path? Investigate."
  exit 1
fi
echo "adapter installed in $B"
