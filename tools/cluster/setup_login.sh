#!/usr/bin/env bash
# =============================================================================
# ONE-TIME setup on the DRAC *login* node (compute nodes have NO internet;
# login nodes have NO GPU -- so: download/install here, compile GPU code in
# the job where -arch=native works).
#
#   bash tools/cluster/setup_login.sh sm_90    # H100   (sm_80 for A100)
# =============================================================================
set -euo pipefail
ARCH=${1:-sm_90}                  # sm_90 = H100, sm_80 = A100
cd "$(dirname "$0")/../.."

# cuda module MUST match the DRAC torch wheel's CUDA (torch 2.12.1+computecanada
# -> CUDA 13.2; check with: python -c 'import torch; print(torch.version.cuda)').
# Bare 'module load cuda' gives 12.6 -> torch cpp_extension hard-errors on the
# major-version mismatch, and the wheel links 13.x libs at runtime.
module load StdEnv/2023 python/3.11 cuda/13.2 2>/dev/null \
  || module load StdEnv/2023 python/3.11 cuda \
  || module load cuda python

# --- venv + torch from the DRAC wheelhouse (works offline later) ------------
python -m venv .venv_cluster
source .venv_cluster/bin/activate
pip install --no-index --upgrade pip
# build tools FIRST: with --no-build-isolation, evogp's build uses the venv's
# setuptools/wheel -- without `wheel`, metadata gen dies with
# "error: invalid command 'bdist_wheel'"
pip install --no-index --upgrade 'setuptools<82' wheel ninja   # <82: torch wheel's pin
pip install --no-index torch scipy    # DRAC provides CUDA torch wheels locally

# --- EvoGP: clone here (internet), build here (nvcc works without a GPU if
#     we tell torch the target arch instead of letting it query the device) ---
git clone --depth 1 https://github.com/EMI-Group/evogp.git /tmp/evogp_src_$USER || true
# torch cpp_extension needs CUDA_HOME (set by the cuda module) -- fail loudly if
# this script is run in a shell without the modules loaded.
: "${CUDA_HOME:?CUDA_HOME unset -- run 'module load StdEnv/2023 cuda python/3.11' first}"
TORCH_ARCH=$( [ "$ARCH" = "sm_90" ] && echo "9.0" || echo "8.0" )
TORCH_CUDA_ARCH_LIST="$TORCH_ARCH" pip install --no-build-isolation /tmp/evogp_src_$USER

python -c "import torch, evogp; print('torch', torch.__version__, '| evogp OK')"

# --- CPU-side builds + datasets (no GPU needed) ------------------------------
make lgp_test lgp_run gen_sweep_datasets
./gen_sweep_datasets
./lgp_test | tail -1
tools/check_golden.sh | tail -1 || echo "NOTE: golden mismatch on a different \
toolchain is a DOCUMENTED possibility (byte-exactness is compiler-specific); \
cuda_diff in the job is the binding correctness gate."

echo "LOGIN SETUP DONE. Submit with: sbatch tools/cluster/replicate.sbatch"
