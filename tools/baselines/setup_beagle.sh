#!/usr/bin/env bash
# Clone + build Beagle (Noblis, Apache-2.0). Requires ~/.dotnet (dotnet-install --channel 10.0).
set -euo pipefail
W=${1:?workdir}
export PATH=$HOME/.dotnet:$PATH
[ -d $W/beagle ] || git clone --depth 1 https://github.com/Noblis/beagle-v1.x $W/beagle
cd $W/beagle && dotnet build Beagle/Run/Run.csproj -c Release 2>&1 | tail -1
echo "beagle ready: $W/beagle/Beagle/Run/bin/Release/net10.0/Run.dll"
