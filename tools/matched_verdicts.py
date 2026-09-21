#!/usr/bin/env python3
"""CANONICAL matched-population verdict computation — the paper's pillar-2
scoreboard, recomputed from raw CSVs under the paper's rules:
MW-U two-sided, Holm within five predeclared families (per-P Nguyen x3,
per-problem California/Feynman), undecided never reported as equivalence.
fig5 embeds identical logic; this prints the human-readable table.
Run: python tools/matched_verdicts.py   (needs scipy)"""
import csv, collections
from scipy.stats import mannwhitneyu

def load_gal():
    gal = {1000: collections.defaultdict(list), 10000: collections.defaultdict(list),
           100000: collections.defaultdict(list)}
    for r in csv.DictReader(open("results/quality_galop_rtx.csv")):
        gal[int(r["P"])][r["problem"]].append(float(r["test_mse"]))
    return gal

def a12_gt(g, e):   # A12 of (GALOP error > EvoGP error): <0.5 favours GALOP
    return 1 - sum((a < b) + 0.5 * (a == b) for a in g for b in e) / (len(g) * len(e))

def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0.0] * len(ps); mx = 0.0
    for rank, i in enumerate(idx):
        mx = max(mx, min(1.0, (len(ps) - rank) * ps[i])); out[i] = mx
    return out

gal = load_gal()
ev = collections.defaultdict(list)
for r in csv.DictReader(open("results/quality_evogp_rtx.csv")):
    ev[(r["problem"], int(r["P"]))].append(float(r["test_mse"]))

NG = [f"nguyen{i}" for i in range(1, 7)]
FAMS = ([("Nguyen@%g" % P, [(t, P) for t in NG]) for P in (1000, 10000, 100000)] +
        [("California", [("california", 10000), ("california", 100000)]),
         ("Feynman", [("feynman_i_9_18", 10000), ("feynman_i_9_18", 100000)])])
score = collections.Counter()
print(f"{'family':<14}{'cell':>16}{'P':>8}{'A12':>7}{'p_holm':>10}  verdict   (n=100 both arms)")
for fname, cells in FAMS:
    st = []
    for t, P in cells:
        g, e = gal[P][t], ev[(t, P)]
        assert len(g) == 100 and len(e) == 100, (t, P, len(g), len(e))
        st.append((t, P, a12_gt(g, e), mannwhitneyu(g, e, alternative="two-sided").pvalue))
    for (t, P, A, _), ph in zip(st, holm([x[3] for x in st])):
        v = "und" if ph >= 0.05 else ("GALOP" if A < 0.5 else "EvoGP")
        score[v] += 1
        print(f"{fname:<14}{t:>16}{P:>8}{A:>7.2f}{ph:>10.1e}  {v}")
print(f"\nAGGREGATE over 22 cells: GALOP {score['GALOP']} — {score['EvoGP']} EvoGP — {score['und']} undecided")

# Correction-scope sensitivity: the family structure is pre-declared, and the
# aggregate is shown insensitive to it — only the borderline Nguyen-3@1e4 cell
# (p_holm .048) moves under global correction; "neither dominates" holds at
# every scope. Run 2026-09-01: families 9-9-4, uncorrected 9-9-4,
# global Holm 8-9-5, global Bonferroni 8-9-5.
allc = [(t, P) for _, cs in FAMS for (t, P) in cs]
ps = {c: mannwhitneyu(gal[c[1]][c[0]], ev[c], alternative="two-sided").pvalue for c in allc}
A = {c: a12_gt(gal[c[1]][c[0]], ev[c]) for c in allc}
def tally(adj):
    s2 = collections.Counter("und" if adj[c] >= 0.05 else ("G" if A[c] < 0.5 else "E") for c in allc)
    return f"{s2['G']}-{s2['E']}-{s2['und']}"
print("sensitivity: uncorrected", tally(ps),
      "| global-Holm", tally(dict(zip(allc, holm([ps[c] for c in allc])))),
      "| global-Bonferroni", tally({c: min(1.0, ps[c]*22) for c in allc}))

# recording-precision robustness: the same runs with BOTH arms rounded to 6 decimals, and with
# both censored at the solve floor (anything below 1e-12 counted as a tie)
def rescore(tf):
    out = collections.Counter()
    for _, cells in FAMS:
        ps_, As_ = [], []
        for t, P in cells:
            g = [tf(x) for x in gal[P][t]]; e = [tf(x) for x in ev[(t, P)]]
            ps_.append(mannwhitneyu(g, e, alternative="two-sided").pvalue if len(set(g + e)) > 1 else 1.0)
            As_.append(a12_gt(g, e))
        for A, ph in zip(As_, holm(ps_)):
            out["und" if ph >= 0.05 else ("GALOP" if A < 0.5 else "EvoGP")] += 1
    return f"{out['GALOP']}-{out['EvoGP']}-{out['und']}"
print("recording robustness: both arms rounded to 6 decimals", rescore(lambda x: float(f"{x:.6f}")),
      "| both censored at 1e-12", rescore(lambda x: x if x >= 1e-12 else 0.0))
