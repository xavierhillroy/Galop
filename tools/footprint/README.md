# Per-individual representation footprint (Table 3)

Each probe calls the system's own population initializer under the paper's configuration
and reports the mean initial program size and the bytes per individual of its representation.

| system | probe | needs |
|---|---|---|
| GALOP | `python3 tools/capacity_readout.py` (section "GALOP program size"): raw and effective instructions per individual from the ladder's per-generation logs in `results_h100/capacity_logs/`; bytes = 4 per instruction + 8 offset + 1 length + 4 fitness on the device, 2 x 50 x 4 on the host | nothing |
| EvoGP | `python tools/footprint/evogp_initsize.py` | the local Python env (torch + EvoGP) |
| Kozax | `KOZAX_OPS=8 python tools/footprint/kozax_initsize.py` | an env with `kozax==0.1.4` and a CUDA jax |
| SRGPU | `nvcc -std=c++17 -O2 -arch=native -I<snapshot>/src tools/footprint/srgpu_initsize.cu <snapshot>/src/{prefix,regression,fit_eval,program}.cu -o initsize && ./initsize 20000 7 7` (arguments: P, initial-depth upper bound, depth cap) | the retained SRGPU snapshot |
| Beagle | copy `tools/footprint/beagle_initsize/` and `tools/beagle_adapter/CsvMLSetup.cs` into `<beagle>/Beagle/Utils/InitSizeProbe/`, `dotnet build -c Release`, then `BEAGLE_CSV=datasets/gridsynth/nguyen1_D64_train.csv BEAGLE_COLONY=1000 BEAGLE_EPG=64 BEAGLE_FIT=corr dotnet bin/Release/net10.0/InitSizeProbe.dll 20000 1` | the Beagle checkout (.NET 10) |

Kozax's transient allocation in its constant-simplification step is read off the ladder row that
ends its ladder (requested bytes / population), not from a probe.
