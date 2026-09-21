#!/usr/bin/env bash
# Clone + patch + build SRGPU (Zhang/Lensen/Sun PRICAI'22, RayZhhh/SymbolicRegressionGPU)
# with our benchmark adapter. Idempotent. Usage: setup_srgpu.sh <workdir>
set -euo pipefail
SELF=$(cd "$(dirname "$0")" && pwd)   # resolve BEFORE any cd
W=${1:?workdir}
# Pinned revision: e7f0eb80bc38c23df11a06244649a2453d0fae74 — verified identical
# on the local snapshot AND the cluster tree (2026-08-31). Upstream has since
# gone dark (404); the retained private snapshot is the canonical source.
SRGPU_REV=e7f0eb80bc38c23df11a06244649a2453d0fae74
[ -d $W/srgpu ] || git clone https://github.com/RayZhhh/SymbolicRegressionGPU $W/srgpu \
  || { echo "upstream 404 — copy the retained snapshot to $W/srgpu (see REPRODUCE.md)"; exit 1; }
ACTUAL=$(git -C $W/srgpu rev-parse HEAD 2>/dev/null || echo none)
[ "$ACTUAL" = "$SRGPU_REV" ] || { echo "SRGPU tree at $ACTUAL != pinned $SRGPU_REV — refusing to proceed"; exit 1; }
cd $W/srgpu
sed -i 's/ experi_benchmark.cu//' CMakeLists.txt                       # phantom source
sed -i 's/set(CMAKE_CUDA_STANDARD 14)/set(CMAKE_CUDA_STANDARD 17)/' CMakeLists.txt  # thrust needs C++17
cp "$SELF/srgpu_bench.cu" .
grep -q srgpu_bench CMakeLists.txt || cat >> CMakeLists.txt <<'CM'
add_executable(srgpu_bench src/prefix.cu src/regression.cu src/fit_eval.cu src/program.cu srgpu_bench.cu)
set_target_properties(srgpu_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
CM
mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native .. >/dev/null && make srgpu_bench -j8 2>&1 | tail -1
echo "srgpu_bench ready: $W/srgpu/build/srgpu_bench"
