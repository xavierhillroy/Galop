#!/usr/bin/env python3
"""Recompute every §5.3 capacity number from the result files, in plain code (companion of
speed_readout.py / quality_readout.py). Unified criterion: one full generation of the
system's own pipeline, D=64 quartic, one 80 GB H100, 1 TB host, 20-minute cap per rung;
ceiling = largest OK rung, the next rung's outcome names the wall.
Files: results_h100/capacity_h100.csv (the ladder), residency_h100.csv + results/residency_rtx.csv
(device-residency probe), results_h100/scale_h100.csv + results/scale_rtx.csv (whole runs).
Run: python3 tools/capacity_readout.py   (stdlib only)"""
import csv, os, statistics as st
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
def rows(p):
    p = os.path.join(R, p)
    if not os.path.exists(p): print(f"  (absent: {os.path.relpath(p, R)})"); return []
    return [r for r in csv.DictReader(open(p)) if r and list(r.values())[0] not in (None, "", list(r.keys())[0])]
def med(v): return st.median(v)
def spread(v): return (max(v) - min(v)) / med(v)

print("=== Unified-criterion ladder (results_h100/capacity_h100.csv): largest OK rung = ceiling; next rung = the wall ===")
lad = {}
for r in rows("results_h100/capacity_h100.csv"): lad.setdefault(r["system"], []).append(r)
ceil = {}
for sysname in ("kozax", "srgpu", "evogp", "beagle", "galop_packed", "galop_eff"):
    rs = sorted(lad.get(sysname, []), key=lambda r: int(r["P"]))
    ok = [r for r in rs if r["outcome"] == "OK"]
    if not ok: print(f"  {sysname:13s} no OK rung"); continue
    top = ok[-1]; ceil[sysname] = int(top["P"])
    nxt = [r for r in rs if int(r["P"]) > int(top["P"])]
    wall = f"{nxt[0]['outcome']} at P={int(nxt[0]['P']):,} ({nxt[0]['note'][:60].strip()})" if nxt else "no higher rung attempted"
    ev = f", evolve {float(top['evolve_s']):.0f} s" if top.get("evolve_s") else ""
    print(f"  {sysname:13s} ceiling P={ceil[sysname]:>13,}  process wall {float(top['wall_process_s']):6.0f} s{ev}   next: {wall}")
if "galop_eff" in ceil:
    others = {k: v for k, v in ceil.items() if not k.startswith("galop")}
    ru = max(others, key=others.get)
    print(f"  gap: galop_eff / runner-up {ru} = {ceil['galop_eff']/others[ru]:.1f}x ; / evogp = {ceil['galop_eff']/others['evogp']:.0f}x"
          f" ; galop_packed / {ru} = {ceil['galop_packed']/others[ru]:.1f}x")

print("\n=== Device-residency probe (allocation + one evaluation pass, no evolution; never a comparison row) ===")
tc = rows("results_h100/residency_h100.csv")
ok = [r for r in tc if r["outcome"] == "OK"]
if ok:
    top = max(ok, key=lambda r: int(r["P"]))
    es = [float(r["evals_per_s"]) for r in ok]; lo = min(int(r["P"]) for r in ok)
    print(f"  H100: max OK P={int(top['P']):,}; evals/s over the {len(ok)} OK rungs from {lo:,}: median {med(es):.3e},"
          f" +/-{(max(es)-min(es))/2/med(es)*100:.1f}% around the median; first FAIL: "
          + next((f"P={int(r['P']):,} {r['outcome']}" for r in sorted(tc, key=lambda r: int(r['P'])) if r['outcome'] != 'OK'), "none"))
oom = rows("results/residency_rtx.csv")
for s in sorted({r["system"] for r in oom}):
    oks = [int(r["P"]) for r in oom if r["system"] == s and r["status"] == "ok"]
    if oks: print(f"  RTX 5070 Ti ({s}): max ok P={max(oks):,}")

print("\n=== Whole runs at the top of the ladders (median of timed runs, spread) ===")
def wr(label, rs, P, per=False):
    v = [float(r["wall_s"]) for r in rs if r["outcome"] == "OK" and r.get("rep", r.get("seed", "")) != "warmup"]
    if not v: print(f"  {label}: no rows"); return None
    m = med(v); extra = f"; {m/(P*100)*1e9:.1f} ns per individual-generation" if per else ""
    print(f"  {label}: n={len(v)} median {m:,.1f} s (spread {spread(v)*100:.1f}%){extra}"); return m
sc = rows("results_h100/scale_h100.csv")
a = wr("GALOP P=1e8 x 100 gens, D=64, 16 host threads (H100)", [r for r in sc if r["system"] == "galop" and r["threads"] == "16"], 100000000, per=True)
b = wr("GALOP P=1e8 x 100 gens, D=64, 48 host threads (H100)", [r for r in sc if r["system"] == "galop" and r["threads"] == "48"], 100000000)
c = wr("Beagle P=1e8 x 100 gens, D=64 (H100)", [r for r in sc if r["system"] == "beagle"], 100000000)
if a and c: print(f"  Beagle / GALOP: {c/a:.1f}x (16 thr){f', {c/b:.1f}x (48 thr)' if b else ''}")
d = wr("GALOP P=3.2e7 x 100 gens (RTX 5070 Ti)", [dict(r, outcome="OK") for r in rows("results/scale_rtx.csv")], 32000000, per=True)
if a and d: print(f"  RTX per-individual-generation vs H100: {((d/(3.2e7*100))/(a/(1e8*100))-1)*100:+.1f}%")

print("\n=== GALOP program size at the cited rungs (results_h100/capacity_logs/*.efflog: per-generation instruction totals) ===")
import glob
for f in sorted(glob.glob(os.path.join(R, "results_h100/capacity_logs/eff_*.efflog"))):
    P = int(os.path.basename(f)[4:-7])
    for r in csv.DictReader(open(f)):
        raw, eff = int(r["sum_L"]) / P, int(r["sum_L_eff"]) / P
        print(f"  P={P:,} generation {r['generation']}: {raw:.3f} instructions per individual ({raw*4:.1f} B), {eff:.3f} effective ({eff*4:.1f} B);"
              f" device footprint packed {raw*4+13:.0f} B, intron-removed {eff*4+13:.0f} B (+8 B offset, 1 B length, 4 B fitness); host {2*50*4+10} B")
