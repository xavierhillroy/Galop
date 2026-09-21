#!/usr/bin/env python3
"""
residency_rtx.py — device-residency probe on the local GPU: the largest population each
layout can hold and evaluate once (allocation + one evaluation pass, no evolution).

Probes four configurations on evolved-population shapes (mean length 30,
effective fraction 16.6% -- measured at P=1e4 x 1000 gens):

  galop-dense      P x MAX_PROGRAM_SIZE(50) padded slots        (~205 B/indiv)
  galop-packed     CSR, Sigma L words + int64 offsets           (~132 B/indiv)
  galop-effective  CSR over effective code only                 (~ 32 B/indiv)
  evogp            forest tensors at the op-parity cap (101)    (~0.8 KB/tree)

GALOP probes run via cuda_capacity (device-side setup: not host-RAM-bound, so
the cap is high). Doubling until OOM; last-ok is the reported ceiling.

Writes results/residency_rtx.csv:  system,P,status
Run: python tools/residency_rtx.py   (needs torch + evogp; build ./cuda_capacity first: make gpu)
"""
import os, csv, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

GALOP_CAP = 2_048_000_000   # int32 n_programs ceiling; on 80GB this is ~5% under the memory ceiling anyway
EVOGP_CAP = 1_024_000_000

rows = []

def probe_galop(layout):
    P = 1_000_000
    while P <= GALOP_CAP:
        r = subprocess.run(["./cuda_capacity", layout, str(P)],
                           cwd=ROOT, capture_output=True, text=True)
        ok = r.returncode == 0
        rows.append((f"galop-{layout}", P, "ok" if ok else "oom"))
        print(f"galop-{layout:<10} P={P:>13,}: {'ok' if ok else 'OOM'}", flush=True)
        if not ok:
            return
        P *= 2

def probe_evogp():
    import torch
    from evogp.tree import Forest, GenerateDescriptor
    from evogp.problem import SymbolicRegression
    desc = GenerateDescriptor(max_tree_len=101, input_len=1, output_len=1,
        using_funcs=["+","-","*","/","sin","cos","<",">"], max_layer_cnt=6,
        const_samples=[0.,1.,-1.,3.14159,0.5,2.,10.,100.])
    X = torch.rand((32,1), device="cuda")*2-1
    y = (X[:,0]**3 + X[:,0]**2 + X[:,0]).reshape(-1,1)
    prob = SymbolicRegression(datapoints=X, labels=y, execute_mode="auto")
    P = 1_000_000
    while P <= EVOGP_CAP:
        try:
            f = Forest.random_generate(pop_size=P, descriptor=desc)
            prob.evaluate(f)
            torch.cuda.synchronize()
            del f
            ok = True
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            ok = False
        torch.cuda.empty_cache()
        rows.append(("evogp", P, "ok" if ok else "oom"))
        print(f"{'evogp':<16} P={P:>13,}: {'ok' if ok else 'OOM'}", flush=True)
        if not ok:
            return
        P *= 2

for lay in ("dense", "packed", "effective"):
    probe_galop(lay)
probe_evogp()

with open(os.path.join(OUT, "residency_rtx.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["system", "P", "status"])
    w.writerows(rows)
print("wrote results/oom/oom.csv")
