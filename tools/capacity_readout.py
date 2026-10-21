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
_base = ("kozax", "srgpu", "evogp", "beagle", "galop_packed", "galop_eff", "galop_packed_cap50", "galop_eff_cap50")
for sysname in _base + tuple(sorted(s for s in lad if s not in _base and s not in ("galop_eff_10gen", "galop_eff_sinit", "galop_eff_pinit"))):   # labelled extra arms (e.g. galop_eff_cap50_48thr) list after the ladder
    rs = sorted(lad.get(sysname, []), key=lambda r: int(r["P"]))
    if not rs: continue                     # a system with no rows in the file (e.g. an arm not run) is not listed
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
    if "galop_eff_cap50" in ceil or "galop_packed_cap50" in ceil:
        print("  at cap width (every program 50 instructions): " + " ; ".join(f"{k} / {ru} = {ceil[k]/others[ru]:.2f}x" for k in ceil if "cap50" in k))

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

def first_upload(smp, eff0_words):
    """Seconds from the first sample to generation 0's upload in a (time, MiB) memory trace. The packed device
    buffer is first allocated at 1.25 x the generation-0 effective total (4 B per word), so the upload is the
    first rise of that size (+/-15%); without an efflog total, the second rise above 1 GiB (the first being
    the evaluator's fixed arrays)."""
    t0 = smp[0][0]
    rises = [((smp[i][0] - t0).total_seconds(), smp[i][1] - smp[i-1][1]) for i in range(1, len(smp)) if smp[i][1] - smp[i-1][1] > 1000]
    if eff0_words:
        want = eff0_words * 1.25 * 4 / 2**20
        for t, d in rises:
            if abs(d - want) <= 0.15 * want: return t
        return None
    return rises[1][0] if len(rises) >= 2 else None
def read_trace(path):
    import datetime as _d
    out = []
    for l in open(path):
        if "MiB" in l:
            ts, mb = l.split(",")[0].strip(), float(l.split(",")[1].split()[0])
            out.append((_d.datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f"), mb))
    return out
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
ge8 = [r for r in sc if r["system"] == "galop_eff" and r["P"] == "100000000"]
ge9 = [r for r in sc if r["system"] == "galop_eff" and r["P"] == "1000000000"]
gecap = [r for r in sc if r["system"] == "galop_eff" and r["P"] == "2147483646"]
if ge8 or ge9 or gecap:
    e = wr("GALOP P=1e8 x 100 gens, D=64, 16 host threads, intron-removed layout (H100)", ge8, 100000000, per=True) if ge8 else None
    f = wr("GALOP P=1e9 x 100 gens, D=64, 16 host threads, intron-removed layout (H100)", ge9, 1000000000, per=True) if ge9 else None
    if e and f: print(f"  per-individual-generation cost, 1e8 -> 1e9 in the same layout (medians over all runs, several nodes): {e/1e10*1e9:.1f} -> {f/1e11*1e9:.1f} ns ({(f/1e11)/(e/1e10)*100-100:+.0f}%)")
    _pairs = []                      # the 1e8 and 1e9 runs of one seed ran back to back in one job on one node: the within-node step
    for r8 in ge8:
        r9 = [x for x in ge9 if x["rep"] == r8["rep"] and x["outcome"] == "OK"]
        if r9 and r8["outcome"] == "OK":
            n8, n9 = float(r8["wall_s"]) / 1e10 * 1e9, float(r9[0]["wall_s"]) / 1e11 * 1e9
            node = r9[0]["note"].split("node ")[1].split(";")[0].strip() if "node " in r9[0]["note"] else "-"
            _pairs.append(f"seed {r8['rep']} ({node}): {n8:.1f} -> {n9:.1f} ns ({n9/n8*100-100:+.0f}%)")
    if _pairs: print("  same-job pairs, 1e8 then 1e9 on one node: " + "; ".join(_pairs))
    if gecap:
        if any(r["outcome"] == "OK" for r in gecap): wr("GALOP P=2^31-2 x 100 gens, D=64, 16 host threads, intron-removed layout (H100)", gecap, 2147483646, per=True)
        for r in gecap:
            if r["outcome"] != "OK": print(f"  GALOP P=2^31-2 x 100 gens seed {r['rep']}: {r['outcome']} after {float(r['wall_s'] or 0):,.0f} s of process wall — {r['note'].split('warm-up; ')[-1].strip()}")
    for P9, _pl in ((100000000, "1e8"), (1000000000, "1e9"), (2147483646, "2^31-2")):
        rws = sorted([r for r in sc if r["system"] == "galop_eff" and r["P"] == str(P9)], key=lambda r: int(r["rep"]))
        if not rws: continue
        print(f"  --- P={_pl}, every run: seed | node | outcome | wall s | ns per ind-gen | setup s | peak device MiB | generations | final length | final effective | peak effective")
        per = []
        for r in rws:
            base = os.path.join(R, f"results_h100/scale_logs/galop_eff_{P9}_s{r['rep']}")
            d = {"seed": r["rep"], "out": r["outcome"], "wall": float(r["wall_s"] or 0), "node": r["note"].split("node ")[1].split(";")[0].strip() if "node " in r["note"] else "-"}
            rs = list(csv.DictReader(open(base + ".efflog"))) if os.path.exists(base + ".efflog") else []
            smp = read_trace(base + ".gpumem") if os.path.exists(base + ".gpumem") else []
            if rs: d.update(gens=len(rs), L=int(rs[-1]["sum_L"]) / P9, E=int(rs[-1]["sum_L_eff"]) / P9, X=max(int(x["sum_L_eff"]) for x in rs) / P9)
            if smp:
                t0 = smp[0][0]
                if P9 > 120000000: d["peak"] = max(m for _, m in smp)      # below that the evaluator preallocates: the trace is not the footprint
                if rs and P9 > 120000000:
                    up = first_upload(smp, int(rs[0]["sum_L_eff"]))
                    if up is not None: d["setup"] = up
                    cap, glast = 0, 0                                   # last buffer growth: replay the 1.25x rule on the efflog, read its time off the trace
                    for g, x in enumerate(rs):
                        w = int(x["sum_L_eff"])
                        if w > cap: cap = w + w // 4; glast = g
                    big = [(smp[i][0] - t0).total_seconds() for i in range(1, len(smp)) if smp[i][1] - smp[i-1][1] > 1000]
                    ends = [(smp[i][0] - t0).total_seconds() for i in range(1, len(smp)) if smp[i-1][1] - smp[i][1] > 10000]
                    if big and ends and len(rs) - 1 > glast: d["steady"] = (ends[-1] - big[-1]) / (len(rs) - 1 - glast)
                if not rs:
                    rises = [(smp[i][0] - t0).total_seconds() for i in range(1, len(smp)) if smp[i][1] - smp[i-1][1] > 1000]
                    d["died"] = f"trace spans {(smp[-1][0]-t0).total_seconds():,.0f} s, peak {max(m for _, m in smp):,.0f} MiB, buffer (re)allocations at " + ", ".join(f"+{t:,.0f} s" for t in rises[1:])
            per.append(d)
            f_ = lambda k, fmt: format(d[k], fmt) if k in d else "-"
            print(f"      {d['seed']:>4} | {d['node']:>4} | {d['out']:4} | {d['wall']:9,.1f} | {d['wall']/(P9*100)*1e9 if d['out'] == 'OK' else float('nan'):5.1f} | {f_('setup', ',.0f'):>6} | {f_('peak', ',.0f'):>7} | {f_('gens', 'd'):>3} | {f_('L', '.1f'):>5} | {f_('E', '.2f'):>5} | {f_('X', '.2f'):>5}" + (f"   {d['died']}" if "died" in d else ""))
        ok = [d for d in per if d["out"] == "OK"]
        if not ok: continue
        ws = [d["wall"] for d in ok]
        line = f"  P={_pl}: completed 100 generations in {len(ok)}/{len(per)} runs; wall median {med(ws):,.1f} s (min {min(ws):,.1f}, max {max(ws):,.1f}, spread {spread(ws)*100:.1f}%)"
        pk = [d["peak"] for d in ok if "peak" in d]
        if pk: line += f"; peak device memory median {med(pk):,.0f} MiB, max {max(pk):,.0f} MiB = {max(pk)/1024:.1f} GiB of {81559/1024:.1f}"
        print(line)
        st_ = [d["setup"] for d in ok if "setup" in d]; sd = [d["steady"] for d in ok if "steady" in d]
        if st_: print(f"  P={_pl}: one-time setup before the first evaluation (memory trace, +/-10 s): median {med(st_):,.0f} s (min {min(st_):,.0f}, max {max(st_):,.0f})")
        if sd: print(f"  P={_pl}: steady cost after the last buffer growth (trace-dated): median {med(sd):.0f} s per generation = {med(sd)/P9*1e9:.1f} ns per individual-generation")
        sh = [d for d in ok if "E" in d]
        if sh: print(f"  P={_pl}: program shape after 100 generations: mean length median {med([d['L'] for d in sh]):.1f}, effective median {med([d['E'] for d in sh]):.2f} per individual ({med([d['E'] for d in sh])/med([d['L'] for d in sh])*100:.1f}%), peak effective over all runs {max(d['X'] for d in sh):.2f}; intron-removed footprint {med([d['E'] for d in sh])*4+13:.0f} B")
d = wr("GALOP P=3.2e7 x 100 gens (RTX 5070 Ti)", [dict(r, outcome="OK") for r in rows("results/scale_rtx.csv")], 32000000, per=True)
if a and d: print(f"  RTX per-individual-generation vs H100: {((d/(3.2e7*100))/(a/(1e8*100))-1)*100:+.1f}%")

print("\n=== GALOP program size at the cited rungs (results_h100/capacity_logs/*.efflog: per-generation instruction totals) ===")
import glob
for f in sorted(glob.glob(os.path.join(R, "results_h100/capacity_logs/eff_*.efflog"))):
    P = int(os.path.basename(f)[:-7].split("_")[-1]); arm = os.path.basename(f)[:-7].rsplit("_", 1)[0]
    for r in csv.DictReader(open(f)):
        raw, eff = int(r["sum_L"]) / P, int(r["sum_L_eff"]) / P
        print(f"  {arm} P={P:,} generation {r['generation']}: {raw:.3f} instructions per individual ({raw*4:.1f} B), {eff:.3f} effective ({eff*4:.1f} B);"
              f" device footprint packed {raw*4+13:.0f} B, intron-removed {eff*4+13:.0f} B (+8 B offset, 1 B length, 4 B fitness); host {2*50*4+10} B")

sus = [r for r in rows("results_h100/capacity_h100.csv") if r["system"] == "galop_eff_10gen"]
if sus:
    import re
    print("\n=== Sustained generations at the intron-removed ceiling (galop_eff_10gen: the 2^31-2 rung run for ten true generations) ===")
    for r in sus:
        P = int(r["P"]); ev = float(r["evolve_s"] or 0)
        m = re.search(r"MAX_GENERATIONS=(\d+)", r["note"]); ng = int(m.group(1)) - 1 if m else 10
        g1 = [x for x in lad.get("galop_eff", []) if int(x["P"]) == P and x["outcome"] == "OK"]
        ref = f" (the ladder's one generation at this P: {float(g1[0]['evolve_s']):.0f} s)" if g1 and g1[0].get("evolve_s") else ""
        per = f", {ev/ng:.0f} s per generation on average over {ng}{ref}" if r["outcome"] == "OK" and ev else ""
        print(f"  P={P:,}: {r['outcome']}, process wall {float(r['wall_process_s']):.0f} s, evolve {ev:.0f} s{per}  [{r['node']}, {r['date']}]")
        err = os.path.join(R, f"results_h100/capacity_logs/eff_10gen_{P}.err")
        if os.path.exists(err):
            for line in open(err):
                if line.startswith(("BREAKDOWN", "PHASES")): print("  " + line.strip())
        if g1 and g1[0].get("evolve_s") and r["outcome"] == "OK" and ev and ng > 1:
            e1v = float(g1[0]["evolve_s"]); marg = (ev - e1v) / (ng - 1)
            print(f"  marginal cost per generation (this evolve minus the ladder's one-generation evolve, over the {ng-1} added generations): {marg:.0f} s"
                  f" = {marg/P*1e9:.1f} ns per individual-generation; the one-generation figure is mostly one-time setup ({e1v-marg:.0f} s)")
        efl = os.path.join(R, f"results_h100/capacity_logs/eff_10gen_{P}.efflog")
        rs = list(csv.DictReader(open(efl))) if os.path.exists(efl) else []
        if rs:
            l0, l1 = int(rs[0]["sum_L"]) / P, int(rs[-1]["sum_L"]) / P
            e0, e1 = int(rs[0]["sum_L_eff"]) / P, int(rs[-1]["sum_L_eff"]) / P
            print(f"  {len(rs)} generation rows: mean length {l0:.2f} -> {l1:.2f} instructions, effective {e0:.3f} -> {e1:.3f} per individual;"
                  f" intron-removed footprint {e0*4+13:.0f} -> {e1*4+13:.0f} B (per-generation rows above)")
        gm = os.path.join(R, f"results_h100/capacity_logs/eff_10gen_{P}.gpumem")
        if os.path.exists(gm):
            import datetime as _dt
            smp = []
            for l in open(gm):
                if "MiB" not in l: continue
                ts, mb = l.split(",")[0].strip(), float(l.split(",")[1].split()[0])
                smp.append((_dt.datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f"), mb))
            if smp:
                t0 = smp[0][0]; secs = lambda t: (t - t0).total_seconds()
                print(f"  device memory (nvidia-smi every 10 s, {len(smp)} samples = {secs(smp[-1][0])/3600:.2f} h): peak {max(m for _, m in smp):,.0f} MiB of 81,559")
                rises = [(secs(t), m - smp[i-1][1]) for i, (t, m) in enumerate(smp) if i and m - smp[i-1][1] > 1000]
                # the packed device buffer is (re)allocated only when a generation's effective total exceeds its
                # capacity (ensure_packed_capacity: capacity = 1.25 x the total that overflowed it), so each rise in
                # the trace is the upload of a known generation: replay the rule on the efflog
                grow, cap = [], 0
                for g, x in enumerate(rs):
                    w = int(x["sum_L_eff"])
                    if w > cap: cap = w + w // 4; grow.append(g)
                if rs and len(rises) == len(grow) + 1: rises = rises[1:]   # the first rise is the evaluator's fixed arrays (fitness, offsets) at process start, not an upload
                if rs and len(rises) == len(grow):
                    marks = ", ".join(f"generation {g} uploaded at +{t:,.0f} s (+{d/1024:.1f} GiB)" for g, (t, d) in zip(grow, rises))
                    print(f"  buffer (re)allocations date the generations (+/-10 s): {marks}")
                    if len(grow) > 1:
                        (ga, (ta, _)), (gb, (tb, _)) = (grow[0], rises[0]), (grow[-1], rises[-1])
                        print(f"  in-run cost: generations {ga}->{gb} span {tb-ta:,.0f} s = {(tb-ta)/(gb-ga):.0f} s per generation = {(tb-ta)/(gb-ga)/P*1e9:.1f} ns per individual-generation;"
                              f" before the first evaluation: {ta:,.0f} s from the sampler start = compile + host setup (allocation, first touch, initial population)")
                else:
                    print(f"  trace rises >1 GiB at: {', '.join(f'+{t:,.0f} s' for t, _ in rises)} (efflog predicts (re)allocations at generations {grow})")

pin = {k: [r for r in rows("results_h100/capacity_h100.csv") if r["system"] == f"galop_eff_{k}"] for k in ("sinit", "pinit")}
if pin["sinit"] or pin["pinit"]:
    import datetime as _dt2
    print("\n=== Parallel initializer (build flag GALOP_PARALLEL_INIT; initialization only, never mixed with the shipped results): one generation at 2^31-2, serial control vs flag, same node ===")
    tfe = {}
    for k, lab in (("sinit", "serial initializer (shipped engine)"), ("pinit", "parallel initializer (flag)")):
        for r in pin[k]:
            P = int(r["P"]); ev = float(r["evolve_s"] or 0)
            gm = os.path.join(R, f"results_h100/capacity_logs/eff_{k}_{P}.gpumem")
            if os.path.exists(gm):
                smp = []
                for l in open(gm):
                    if "MiB" in l:
                        ts, mb = l.split(",")[0].strip(), float(l.split(",")[1].split()[0])
                        smp.append((_dt2.datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f"), mb))
                if len(smp) > 2:
                    t0 = smp[0][0]
                    rises = [(smp[i][0] - t0).total_seconds() for i in range(1, len(smp)) if smp[i][1] - smp[i-1][1] > 1000]
                    if len(rises) >= 2: tfe[k] = rises[1]
            print(f"  {lab:38s} P={P:,}: {r['outcome']}, process wall {float(r['wall_process_s']):,.0f} s, evolve {ev:,.0f} s" + (f", first evaluation at +{tfe[k]:,.0f} s (memory trace, incl. compile)" if k in tfe else "") + f"  [{r['node']}, {r['date']}]")
    if "sinit" in tfe and "pinit" in tfe:
        print(f"  time to the first evaluation: {tfe['sinit']:,.0f} s -> {tfe['pinit']:,.0f} s ({tfe['sinit']/tfe['pinit']:.1f}x less)")

el = os.path.join(R, "results/effective_length_rtx.csv")
if os.path.exists(el):
    print("\n=== Effective program length on the quality workloads (results/effective_length_rtx.csv: P=1e5, 100 generations, the seeds of the quality runs; every run reproduces its quality row's train and test MSE) ===")
    by = {}
    for r in csv.DictReader(open(el)): by.setdefault(r["problem"], []).append(r)
    CARD = 81559 * 2**20                      # the H100's device memory as nvidia-smi reports it
    stat = {}
    for prob, rs in by.items():
        e = sorted(float(r["final_effective"]) for r in rs)
        stat[prob] = (med(e), e[len(e) // 4], e[(3 * len(e)) // 4], e[0], e[-1], med([float(r["peak_effective"]) for r in rs]), max(float(r["peak_effective"]) for r in rs), med([float(r["final_length"]) for r in rs]), len(e))
    for prob in sorted(stat, key=lambda k: stat[k][0]):
        m, q1, q3, lo, hi, pm, px, L, n = stat[prob]
        cap = CARD / (13 + 5 * m)             # 13 B fixed + 4 B per effective instruction x the 1.25 buffer rule
        print(f"  {prob:15s} n={n:3d}  final length {L:.1f}  effective: median {m:5.2f}, quartiles {q1:.2f}-{q3:.2f}, range {lo:.2f}-{hi:.2f}; peak median {pm:.2f}, max {px:.2f}; storage-model ceiling at the median "
              + ("the index bound 2^31-2" if cap >= 2147483646 else f"{cap:.2e}"))
    if stat:
        ms = sorted((v[0], k) for k, v in stat.items())
        allv = [float(r["final_effective"]) for rs in by.values() for r in rs]
        print(f"  across the {len(stat)} workloads the median effective length ranges from {ms[0][0]:.1f} ({ms[0][1]}) to {ms[-1][0]:.1f} ({ms[-1][1]}) instructions; over all {len(allv)} runs {min(allv):.1f}-{max(allv):.1f}")
        print("  storage model (an estimate, not a measured ceiling): 13 B + 4 B x effective x 1.25 per individual on an 81,559 MiB card; the index bound holds while the mean effective length stays below "
              + f"{(CARD / 2147483646 - 13) / 5:.1f}")
