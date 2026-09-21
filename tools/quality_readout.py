#!/usr/bin/env python3
"""Full §5.2 read-out from the CURRENT quality CSVs: 22-cell verdicts (same rules as
tools/matched_verdicts.py), per-P Nguyen wins, exact-zero medians, solve rates, the EvoGP
authors-config sensitivity arm, and the Nguyen 9/10/12 extension.
Run: python tools/quality_readout.py   (needs scipy)"""
import csv, collections, statistics as st, os, sys
from scipy.stats import mannwhitneyu
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')   # repo root, wherever the tree lives
def a12_gt(g, e): return 1 - sum((a < b) + 0.5 * (a == b) for a in g for b in e) / (len(g) * len(e))
def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0.0] * len(ps); mx = 0.0
    for rank, i in enumerate(idx):
        mx = max(mx, min(1.0, (len(ps) - rank) * ps[i])); out[i] = mx
    return out
def med(v): return st.median(v)
SOLVE = 1e-12

# ---- load Galop (regenerated) ----
gal = {1000: collections.defaultdict(list), 10000: collections.defaultdict(list), 100000: collections.defaultdict(list)}
for r in csv.DictReader(open(f'{R}/results/quality_galop_rtx.csv')): gal[int(r['P'])][r['problem']].append(float(r['test_mse']))
ev = collections.defaultdict(list)
for r in csv.DictReader(open(f'{R}/results/quality_evogp_rtx.csv')): ev[(r['problem'], int(r['P']))].append(float(r['test_mse']))

NG = [f'nguyen{i}' for i in range(1, 7)]
incomplete = [(P, t, len(gal[P][t])) for P in gal for t in gal[P] if len(gal[P][t]) != 100]
if incomplete: print('WARNING incomplete cells (P, problem, n):', incomplete)
# precision sanity: count Galop rows that are exact 0.0 (should be rare now)
print('Galop exact-0.0 rows per P:', {P: sum(v == 0.0 for t in gal[P] for v in gal[P][t]) for P in gal})

FAMS = ([(f'Nguyen@{P:g}', [(t, P) for t in NG]) for P in (1000, 10000, 100000)] +
        [('California', [('california', 10000), ('california', 100000)]), ('Feynman', [('feynman_i_9_18', 10000), ('feynman_i_9_18', 100000)])])
score = collections.Counter(); per_fam = {}; cellv = {}
print(f"\n{'family':<14}{'cell':>16}{'P':>8}{'A12':>7}{'p_holm':>10}  verdict   medG / medE")
for fname, cells in FAMS:
    ps, As = [], []
    for t, P in cells:
        g, e = gal[P][t], ev[(t, P)]
        if not g or not e: ps.append(1.0); As.append(0.5); continue
        _, pv = mannwhitneyu(g, e, alternative='two-sided'); ps.append(pv); As.append(a12_gt(g, e))
    ph = holm(ps); fam = collections.Counter()
    for (t, P), A, p in zip(cells, As, ph):
        v = 'und' if p >= 0.05 else ('GALOP' if A < 0.5 else 'EvoGP'); score[v] += 1; fam[v] += 1; cellv[(t, P)] = (v, A, p)
        g, e = gal[P][t], ev[(t, P)]
        print(f"{fname:<14}{t:>16}{P:>8}{A:>7.2f}{p:>10.1e}  {v:<8} {med(g) if g else float('nan'):.2e} / {med(e) if e else float('nan'):.2e}")
    per_fam[fname] = fam
print(f"\nAGGREGATE over 22 cells: GALOP {score['GALOP']} - {score['EvoGP']} EvoGP - {score['und']} undecided")
print('Nguyen G-E-U per P (\\qNguyenSweep):', ' / '.join(f"{per_fam[f'Nguyen@{P:g}']['GALOP']}--{per_fam[f'Nguyen@{P:g}']['EvoGP']}--{per_fam[f'Nguyen@{P:g}']['und']}" for P in (1000, 10000, 100000)))
print('Galop Nguyen wins by P:', [per_fam[f'Nguyen@{P:g}']['GALOP'] for P in (1000, 10000, 100000)])
print('California verdicts 1e4/1e5:', cellv[('california', 10000)][:2], cellv[('california', 100000)][:2])
print('Feynman A12 1e4/1e5 (\\qFeynAtwelveLo/Hi):', round(cellv[('feynman_i_9_18', 10000)][1], 2), round(cellv[('feynman_i_9_18', 100000)][1], 2))

print('\n=== exact-zero medians and solve rates (< 1e-12), Nguyen ===')
for P in (1000, 10000, 100000):
    for t in NG:
        g, e = gal[P][t], ev[(t, P)]
        if g and e:
            print(f"  {t} P={P:<6d} medG={med(g):.2e} medE={med(e):.2e} exact0 G={sum(v==0.0 for v in g)} E={sum(v==0.0 for v in e)}  solved G={sum(v<SOLVE for v in g)}% E={sum(v<SOLVE for v in e)}%")

print('\n=== sensitivity arm: EvoGP authors config (quality_sensitivity_rtx.csv) vs Galop std at P=1e4, Holm over the 8 cells ===')
ac = collections.defaultdict(list)
p = f'{R}/results/quality_sensitivity_rtx.csv'
for r in csv.DictReader(open(p)): ac[r['problem']].append(float(r['test_mse']))
cells8 = NG + ['california', 'feynman_i_9_18']; ps = []; As = []
for t in cells8:
    g, e = gal[10000][t], ac[t]; _, pv = mannwhitneyu(g, e, alternative='two-sided'); ps.append(pv); As.append(a12_gt(g, e))
for t, A, ph in zip(cells8, As, holm(ps)):
    v = 'und' if ph >= 0.05 else ('GALOP' if A < 0.5 else 'EvoGP'); par = cellv[(t, 10000)]
    print(f"  {t:>16}: authors-config A12={A:.2f} p_holm={ph:.1e} {v:<6} | parity-config A12={par[1]:.2f} {par[0]}")

print('\n=== Nguyen 9/10/12 extension (descriptive) ===')
gx = {(t, P): v for P in gal for t, v in gal[P].items()}; ex = ev
if gx:
    for t in ('nguyen9', 'nguyen10', 'nguyen12'):
        for P in (1000, 10000, 100000):
            g, e = gx[(t, P)], ex[(t, P)]
            if g and e:
                _, pv = mannwhitneyu(g, e, alternative='two-sided')
                print(f"  {t} P={P:<6d} n={len(g)}/{len(e)} A12={a12_gt(g, e):.2f} p={pv:.1e} medG={med(g):.2e} medE={med(e):.2e} solved G={sum(v<SOLVE for v in g)}% E={sum(v<SOLVE for v in e)}%")
else:
    print('  (ext CSV not regenerated yet)')

# ---------------------------------------------------------------------------------------------
# anchor quality: SRGPU / Kozax (results_h100/quality_anchors_h100.csv) and Beagle
# (results_h100/quality_beagle_h100.csv, both of its published fitness arms) next to the pair;
# "floor" = median test MSE below the solve threshold (identity-level)
print('\n=== anchor quality: median test MSE per problem and P; * = floor (median < %.0e), best = lowest median in the row ===' % SOLVE)
anc = collections.defaultdict(list)
for r in csv.DictReader(open(f'{R}/results_h100/quality_anchors_h100.csv')):
    if r['outcome'] == 'OK' and r['rep'] != 'probe':
        try: anc[(r['system'], r['problem'], int(r['P']))].append(float(r['test_mse']))
        except ValueError: pass
for r in csv.DictReader(open(f'{R}/results_h100/quality_beagle_h100.csv')):
    if r['outcome'] == 'OK':
        try: anc[('beagle_' + r['fitfunc'], r['problem'], int(r['P']))].append(float(r['test_mse']))
        except ValueError: pass
SYS = ['galop', 'evogp', 'srgpu', 'kozax', 'beagle_corr', 'beagle_std']
PROBS = NG + ['california', 'feynman_i_9_18', 'nguyen9', 'nguyen10', 'nguyen12']
floors = collections.Counter(); bests = collections.defaultdict(list)
for P in (10000, 100000):
    print(f'-- P={P}')
    print(f"  {'problem':15s} " + ' '.join(f'{s:>13s}' for s in SYS))
    for t in PROBS:
        vals = {}
        for s in SYS:
            v = gal[P][t] if s == 'galop' else ev[(t, P)] if s == 'evogp' else anc.get((s, t, P), [])
            v = [x for x in v if x == x]                     # drop NaN
            if v: vals[s] = med(v)
        if not vals: continue
        best = min(vals, key=vals.get)
        cells = []
        for s in SYS:
            if s in vals:
                m = vals[s]; flag = '*' if m < SOLVE else ' '
                if m < SOLVE: floors[(s, P)] += 1
                cells.append(f'{m:>11.1e}{flag}' + ('b' if s == best else ' '))
            else: cells.append(f'{"-":>13s}')
        bests[(best, P)].append(t)
        print(f'  {t:15s} ' + ' '.join(cells))
print('  floor medians per system (Nguyen 1-6 only):')
for s in SYS:
    for P in (10000, 100000):
        n = sum(1 for t in NG if (lambda v: bool(v) and med(v) < SOLVE)([x for x in (gal[P][t] if s == 'galop' else ev[(t, P)] if s == 'evogp' else anc.get((s, t, P), [])) if x == x]))
        print(f'    {s:12s} P={P:<7d} {n}/6')
print('  best median per problem (b above):', {f'{s}@{P}': v for (s, P), v in sorted(bests.items())})
