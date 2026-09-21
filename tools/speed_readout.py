#!/usr/bin/env python3
"""Recompute every §5.1 speed number from the raw CSVs with the paper's rules, in plain code.
Rules (Sec. 4.3/4.4 of the paper):
  * a cell = one (system, P, D); five timed runs (seeds 70-74 / reps 0-4); the MEDIAN is reported
  * spread = (max - min) / median; a cell whose spread exceeds 25% is not reliable
  * H100 pair grid: results_h100/speed_grid_h100.csv (the two P=1e3 cells re-measured in a
    dedicated session carry session=redo)
  * speedup = median(EvoGP) / median(galop_std)      (>1 favours Galop; galop_std = the paper's arm)
  * anchors (SRGPU, Kozax, Beagle) are compared to galop_std on the H100 grid; their 'probe'
    rows are warm-ups and are excluded; only OK rows count
Run: python3 tools/speed_readout.py   (stdlib only)
"""
import csv, statistics as st
from collections import defaultdict

def load_pair(path):
    """arm,P,D,seed,wall_s -> {(arm,P,D): [walls]}"""
    d = defaultdict(list)
    for r in csv.DictReader(open(path)):
        d[(r["arm"], int(r["P"]), int(r["D"]))].append(float(r["wall_s"]))
    return d

def load_anchor(path, system):
    """system,P,D,rep,outcome,wall_s,note -> {(P,D): [walls of OK timed runs]}, {(P,D): failure} for one system"""
    ok, fail = defaultdict(list), {}
    for r in csv.DictReader(open(path)):
        if r["system"] != system: continue
        rep = r["rep"]
        k = (int(r["P"]), int(r["D"]))
        if r["outcome"] != "OK":
            fail[k] = r["outcome"]                          # a failed probe still attributes the cell
        elif rep != "probe":
            ok[k].append(float(r["wall_s"]))                # probes are warm-ups, never timed runs
    return ok, fail

def med(v):    return st.median(v)
def spread(v): return (max(v) - min(v)) / med(v)

PS, DS_RTX, DS_H100 = [1000, 10000, 100000, 1000000], [32, 1024, 32768], [32, 1024, 32768, 131072]

rtx = load_pair("results/speed_grid_rtx.csv")
h100 = load_pair("results_h100/speed_grid_h100.csv")

def grid(name, dd, Ds):
    print(f"\n=== {name}: median wall (s) of 5 timed runs, spread, EvoGP/galop_std ===")
    print(f"{'P':>8} {'D':>7} | {'galop_std':>10} {'spr':>5} | {'evogp':>10} {'spr':>5} | {'ratio':>6}")
    ratios = {}
    for P in PS:
        for D in Ds:
            g, e = dd[("galop_std", P, D)], dd[("evogp", P, D)]
            r = med(e) / med(g); ratios[(P, D)] = r
            flag = "  <- spread >25%: n.m." if max(spread(g), spread(e)) > 0.25 else ""
            print(f"{P:>8} {D:>7} | {med(g):>10.4f} {spread(g)*100:>4.0f}% | {med(e):>10.4f} {spread(e)*100:>4.0f}% | {r:>6.2f}{flag}")
    faster = sum(r > 1 for r in ratios.values())
    print(f"cells: {len(ratios)}, Galop faster in {faster}; ratios {min(ratios.values()):.2f}-{max(ratios.values()):.2f}x"
          f" (excluding cells <1x: {min(r for r in ratios.values() if r > 1):.2f}-{max(ratios.values()):.2f}x)")
    return ratios

grid("RTX 5070 Ti  (results/speed_grid_rtx.csv)", rtx, DS_RTX)
h = grid("H100  (results_h100/speed_grid_h100.csv)", h100, DS_H100)

print("\n=== H100 growth with P: wall ratio for each tenfold step (10 = linear) ===")
for arm in ("galop_std", "evogp"):
    for D in DS_H100:
        m = [med(h100[(arm, P, D)]) for P in PS]
        steps = " ".join(f"x{m[i+1]/m[i]:.2f}" for i in range(3))
        ns = med(h100[(arm, 1000000, D)]) / (1e6 * 100) * 1e9
        print(f"  {arm:9s} D={D:<6d} {steps}   per-individual-generation at P=1e6: {ns:8.1f} ns")

galop_h100 = {k[1:]: med(v) for k, v in h100.items() if k[0] == "galop_std"}
for name, system in (("SRGPU", "srgpu"), ("Kozax", "kozax"), ("Beagle", "beagle")):
    ok, fail = load_anchor("results_h100/anchors_grid_h100.csv", system)
    print(f"\n=== {name} vs galop_std on the H100 grid  ({system} rows of results_h100/anchors_grid_h100.csv) ===")
    print(f"{'P':>8} {'D':>7} | {'anchor':>9} {'n':>2} {'spr':>5} | {'galop':>9} | {'ratio':>8}")
    allr, cleanr = [], []
    for P in PS:
        for D in DS_H100:
            k = (P, D)
            if k in ok:
                r = med(ok[k]) / galop_h100[k]; s = spread(ok[k]); allr.append(r)
                if s <= 0.25: cleanr.append(r)
                print(f"{P:>8} {D:>7} | {med(ok[k]):>9.3f} {len(ok[k]):>2} {s*100:>4.0f}% | {galop_h100[k]:>9.4f} | {r:>8.1f}x{'  <- spread >25%' if s > 0.25 else ''}{'  <- n<5' if len(ok[k]) < 5 else ''}")
            else:
                print(f"{P:>8} {D:>7} | {fail.get(k, 'not run'):>9} {'':>2} {'':>5} | {galop_h100[k]:>9.4f} |")
    if allr:
        print(f"range over {len(allr)} feasible cells: {min(allr):.1f}-{max(allr):,.0f}x;"
              f" over the {len(cleanr)} cells meeting the 25% spread rule: {min(cleanr):.1f}-{max(cleanr):,.0f}x")

# ---------------------------------------------------------------------------------------------
# D extension (fig3c): the boundary cells, with the time-attributed evogp rows named as such
print("\n=== D extension (results_h100/speed_boundary_h100.csv): median walls and EvoGP/galop_std at D=262,144 and 524,288 ===")
bnd, attributed = defaultdict(list), {}
for r in csv.DictReader(open("results_h100/speed_boundary_h100.csv")):
    try:    bnd[(r["arm"], int(r["P"]), int(r["D"]))].append(float(r["wall_s"]))
    except ValueError: attributed[(r["arm"], int(r["P"]), int(r["D"]))] = r["wall_s"]   # e.g. TIMEBOUND_...: not a wall
for P in PS:
    for D in (262144, 524288):
        g, e = bnd.get(("galop_std", P, D)), bnd.get(("evogp", P, D))
        if g and e: print(f"{P:>8} {D:>7} | galop_std {med(g):>9.1f} s ({spread(g)*100:.0f}%) | evogp {med(e):>9.1f} s ({spread(e)*100:.0f}%) | ratio {med(e)/med(g):.2f}")
        elif g:     print(f"{P:>8} {D:>7} | galop_std {med(g):>9.1f} s ({spread(g)*100:.0f}%) | evogp {attributed.get(('evogp', P, D), 'absent')}")

# EvoGP with its hybrid mode forced (the five cells where auto picks data-parallel against its interest)
print("\n=== EvoGP forced hybrid mode (results_h100/speed_hybrid_h100.csv) vs galop_std and vs EvoGP auto, H100 ===")
hyb = load_pair("results_h100/speed_hybrid_h100.csv")
worse = 0
for (arm, P, D), v in sorted(hyb.items()):
    g, e = h100[("galop_std", P, D)], h100[("evogp", P, D)]
    r = med(v) / med(g); worse += r < 1
    print(f"{P:>8} {D:>7} | hybrid {med(v):>9.2f} s | auto {med(e):>9.2f} s | galop_std {med(g):>9.2f} s | hybrid/galop {r:.2f}x  auto/hybrid {med(e)/med(v):.2f}x")
print(f"cells where forced hybrid beats galop_std: {worse} of {len(hyb)}")

# the CPU sentence
print("\n=== CPU anchor (results/cpu_anchor_rtx.csv): medians of 3 runs, wall of the evolutionary run only ===")
cpu = defaultdict(list)
for r in csv.DictReader(open("results/cpu_anchor_rtx.csv")): cpu[(r["system"], r["problem"], int(r["P"]))].append(float(r["wall_s"]))
for k in sorted(cpu): print(f"  {k[0]:8s} {k[1]:15s} P={k[2]:<8d} median {med(cpu[k]):8.2f} s (n={len(cpu[k])})")
if ("galop", "nguyen2", 1000000) in cpu and ("gplearn", "nguyen2", 1000) in cpu:
    print(f"  paper: gplearn P=1e3 {med(cpu[('gplearn','nguyen2',1000)]):.1f} s vs GALOP P=1e6 {med(cpu[('galop','nguyen2',1000000)]):.1f} s on the same desktop")
