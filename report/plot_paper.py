#!/usr/bin/env python3
"""The paper's three figures, rendered from the shipped CSVs (fig3 speed grids + the absolute-wall table, fig5 matched quality with verdicts computed in-render, fig6 capacity).
Run: python plot_paper.py  -> figures/paper_short/   (needs numpy, matplotlib, scipy)"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "paper_short")
os.makedirs(OUT, exist_ok=True)
GAL, EVO, GRY = "#378ADD", "#BA7517", "#888888"

# ---- Fig 3 (FINAL): data-driven median heatmaps, standard config -----------
import csv
from collections import defaultdict, Counter
from statistics import median as _med
def _load(path):
    dd = defaultdict(list)
    for r in csv.DictReader(open(path)):
        try:
            dd[(r["arm"], int(r["P"]), int(r["D"]))].append(float(r["wall_s"]))
        except ValueError:
            pass   # attribution rows (e.g. TIMEBOUND_...) carry no wall time
    return dd
_R = os.path.dirname(os.path.abspath(__file__))
_loc = _load(os.path.join(_R, "..", "results", "speed_grid_rtx.csv"))
_h   = _load(os.path.join(_R, "..", "results_h100", "speed_grid_h100.csv"))   # the two re-measured P=1e3 cells carry session=redo
# H100 grid FINAL (aligned galop arms + sr=1.0 evogp arm) — no record merge
Ps = [1000, 10000, 100000, 1000000]
def _grid(dd, Dsz, stability=False):
    M, mask = [], []
    for P in Ps:
        row, mrow = [], []
        for D in Dsz:
            e, g = dd[("evogp", P, D)], dd[("galop_std", P, D)]
            me, mg = _med(e), _med(g)
            sp = max((max(v)-min(v))/_med(v) for v in (e, g))
            ok = (not stability) or sp <= 0.25          # rule: spread only
            row.append(me/mg); mrow.append(ok)
        M.append(row); mask.append(mrow)
    return M, mask
_need = all(_loc.get(("galop_std", P, D)) and _loc.get(("evogp", P, D))
            for P in Ps for D in [32, 1024, 32768]) and \
        all(_h.get(("galop_std", P, D)) and _h.get(("evogp", P, D))
            for P in Ps for D in [32, 1024, 32768, 131072])
if not _need:
    print("fig3 SKIPPED: local grid mid-regeneration (keeping last rendered fig3)")
else:
    Mloc, _ = _grid(_loc, [32, 1024, 32768])
    Mh, okh = _grid(_h, [32, 1024, 32768, 131072], stability=True)
    _bnd = _load(os.path.join(_R, "..", "results_h100", "speed_boundary_h100.csv"))
    # Layout (2026-10-12): drawn at the LNCS text width (4.8 in) so every font is its printed size;
    # the two heat maps share the top row and one colour bar, the D-extension panel takes the full
    # width below them. Fonts 8-9 pt.
    from matplotlib.ticker import LogLocator, NullFormatter
    fig = plt.figure(figsize=(4.8, 5.8), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[3, 4, 0.18], height_ratios=[1.0, 0.85], wspace=0.04, hspace=0.16)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2]); axB = fig.add_subplot(gs[1, :])
    for ax, M, ok, cols, t in ((axes[0], Mloc, None, ["32", "1,024", "32,768"], "RTX 5070 Ti"),
                               (axes[1], Mh, okh, ["32", "1,024", "32,768", "131,072"], "H100")):
        A = np.array(M)
        disp = A.copy()
        if ok is not None:
            for i in range(len(Ps)):
                for j in range(len(cols)):
                    if not ok[i][j]: disp[i][j] = np.nan
        im = ax.imshow(disp, cmap="Blues", vmin=0.8, vmax=14.5)
        for i in range(len(Ps)):
            for j in range(len(cols)):
                if ok is not None and not ok[i][j]:
                    ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, facecolor="#EEEEEE", edgecolor="white", hatch="///"))
                    ax.text(j, i, "n.m.", ha="center", va="center", fontsize=7, color="#999999")
                else:
                    v = A[i][j]
                    lab = f"{v:.2f}×" if v < 1.2 else f"{v:.1f}×"
                    ax.text(j, i, lab, ha="center", va="center",
                            color="white" if v > 11 else ("#B22222" if v < 1 else "#123"),
                            fontsize=8.5, weight="bold")
        ax.set_xticks(range(len(cols)), cols, fontsize=8)
        ax.set_yticks(range(4), ["P=10³", "P=10⁴", "P=10⁵", "P=10⁶"], fontsize=8)
        ax.set_xticks([x-0.5 for x in range(1, len(cols))], minor=True)
        ax.set_yticks([y-0.5 for y in range(1, 4)], minor=True)
        ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=2)
        ax.set_xlabel("D (fitness cases)", fontsize=8, labelpad=2)
        ax.set_title(t, fontsize=9, pad=4)
    axes[1].tick_params(axis="y", labelleft=False)     # the two maps share the P rows
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("median speedup (>1 favours Galop)", fontsize=8)
    cb.ax.tick_params(labelsize=8, length=2)
    # ---- (c) the D extension, full width
    Dseq = [32, 1024, 32768, 131072, 262144, 524288]
    STY = {1000: (0.45, "o"), 10000: (0.75, "s"), 100000: (1.0, "D")}
    for P in (1000, 10000, 100000):   # the boundary probe covers P <= 1e5; P=1e6 dropped 2026-09-16
        xs, ys, xn, yn, lo, hi = [], [], [], [], [], []
        for D in Dseq:
            src = _bnd if D in (262144, 524288) else _h
            g, e = src.get(("galop_std", P, D)), src.get(("evogp", P, D))
            if not (g and e and len(g) == 5 and len(e) == 5): continue
            r = _med(e) / _med(g)
            # same n.m. criterion as the heatmaps: spread <= 25% both arms
            if any((max(v) - min(v)) / _med(v) > 0.25 for v in (g, e)):
                xn.append(D); yn.append(r)
            else:
                xs.append(D); ys.append(r)
                lo.append(min(e) / max(g)); hi.append(max(e) / min(g))   # ratio of the extreme runs (2026-10-19): slowest EvoGP / fastest Galop and the reverse
        a, mk = STY[P]
        axB.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]],
                     fmt="none", ecolor=GAL, alpha=a, elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=2)
        axB.plot(xs, ys, marker=mk, ms=4.5, lw=1.6, color=GAL, alpha=a, zorder=3,
                 label=f"P=10{'³⁴⁵⁶'[[1000,10000,100000,1000000].index(P)]}")
        if xn:
            axB.scatter(xn, yn, marker=mk, s=24, facecolors="white",
                        edgecolors=GAL, alpha=a, zorder=3)
    axB.axhline(1.0, color="#B22222", lw=1.2, ls="--")
    axB.text(36, 1.07, "parity", color="#B22222", fontsize=8)
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xticks(Dseq, ["32", "1,024", "32,768", "131,072", "262,144", "524,288"], fontsize=8, rotation=30, ha="right", rotation_mode="anchor")
    axB.set_yticks([0.5, 1, 2, 4, 8], ["0.5×", "1×", "2×", "4×", "8×"], fontsize=8)
    axB.minorticks_off()
    axB.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    axB.xaxis.set_minor_formatter(NullFormatter())
    axB.tick_params(axis="x", which="minor", length=2, color="#999999")
    axB.tick_params(which="major", length=2.5)
    axB.set_xlabel("D (fitness cases, log axis)", fontsize=8, labelpad=2)
    axB.set_ylabel("median speedup", fontsize=8)
    axB.set_title("H100, extended D", fontsize=9, pad=4)
    axB.legend(fontsize=8, frameon=False, loc="upper right", handlelength=2.2)
    axB.grid(True, ls=":", alpha=0.3)
    # Panel letters centred under each panel; the caption carries the protocol.
    fig.canvas.draw()
    _ren = fig.canvas.get_renderer()
    for _ax, _lab in ((axes[0], "(a)"), (axes[1], "(b)"), (axB, "(c)")):
        _bb = _ax.get_position()                                                      # axes box: horizontal centre
        _tb = _ax.get_tightbbox(_ren).transformed(fig.transFigure.inverted())        # incl. tick labels and axis label: vertical anchor
        fig.text((_bb.x0 + _bb.x1) / 2, _tb.y0 - 0.008, _lab, ha="center", va="top", fontsize=9)
    fig.savefig(f"{OUT}/fig3_wholerun.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig3_wholerun_heatmaps.pdf", bbox_inches="tight"); plt.close(fig)
    # ---- Fig 4: end-to-end scaling with population on the H100 at two fitness-set sizes ------------
    # (2026-10-16) median of the five timed runs per cell with the observed min-max range shaded; a
    # no reference line: the slopes are read against the grid and stated in the text. Drawn at the LNCS
    # text width so the fonts print at 8-9 pt. Same CSV as the heat maps.
    fig4, axs = plt.subplots(1, 2, figsize=(4.8, 2.55), layout="constrained")
    for ax, D in zip(axs, (32, 32768)):
        for arm, col, name, mk in (("galop_std", GAL, "Galop", "o"), ("evogp", EVO, "EvoGP", "s")):
            mv, lo, hi = [], [], []
            for P in Ps:
                v = _h[(arm, P, D)]; mv.append(_med(v)); lo.append(min(v)); hi.append(max(v))
            ax.fill_between(Ps, lo, hi, color=col, alpha=0.22, lw=0, zorder=1)                      # observed range of the five runs
            ax.errorbar(Ps, mv, yerr=[[m - l for m, l in zip(mv, lo)], [h - m for m, h in zip(mv, hi)]],
                        fmt="none", ecolor=col, elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=2)
            ax.plot(Ps, mv, marker=mk, ms=3.6, mec="white", mew=0.5, lw=1.5, color=col, label=name, zorder=3)
        _all = [x for arm in ("galop_std", "evogp") for P in Ps for x in _h[(arm, P, D)]]
        ax.set_ylim(min(_all) * 0.55, max(_all) * 1.9); ax.set_xlim(6e2, 1.7e6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(Ps, ["10³", "10⁴", "10⁵", "10⁶"], fontsize=8); ax.minorticks_off()
        ax.tick_params(axis="y", labelsize=8); ax.tick_params(length=2.5)
        ax.set_title(f"D = {D:,}", fontsize=9, pad=4)
        ax.set_xlabel("P (population, log scale)", fontsize=8, labelpad=2)
        ax.grid(True, ls=":", alpha=0.3)
    axs[0].set_ylabel("wall-clock, 100 generations (s, log scale)", fontsize=8)
    axs[0].legend(fontsize=8, frameon=False, loc="upper left", handlelength=2.0)
    fig4.canvas.draw(); _ren4 = fig4.canvas.get_renderer()
    for _ax, _lab in zip(axs, ("(a)", "(b)")):
        _bb = _ax.get_position(); _tb = _ax.get_tightbbox(_ren4).transformed(fig4.transFigure.inverted())
        fig4.text((_bb.x0 + _bb.x1) / 2, _tb.y0 - 0.02, _lab, ha="center", va="top", fontsize=9)
    fig4.savefig(f"{OUT}/fig4_scaling.png", dpi=200, bbox_inches="tight")
    fig4.savefig(f"{OUT}/fig4_scaling.pdf", bbox_inches="tight"); plt.close(fig4)
    # Companion table: ABSOLUTE median wall-clock (s) behind the fig3 ratios, both
    # cards, Galop / EvoGP per cell (2026-09-16, reviewer ask: ratios alone are not
    # a durable reference). Same CSVs, same nmredo merge -> figures/paper_short/tab_wholerun_abs.tex
    def _fmt(v):
        # 3 significant figures below 1 s so the printed pair reproduces the figure's ratio
        return f"{v:,.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else (f"{v:.2f}" if v >= 1 else f"{v:.3g}"))
    _rows = []
    for _card, _dd, _Ds in (("RTX 5070 Ti", _loc, [32, 1024, 32768]), ("H100", _h, [32, 1024, 32768, 131072])):
        for _i, _P in enumerate(Ps):
            _cells = []
            for _D in [32, 1024, 32768, 131072]:
                if _D in _Ds:
                    _cells.append(f"{_fmt(_med(_dd[('galop_std', _P, _D)]))} / {_fmt(_med(_dd[('evogp', _P, _D)]))}")
                else:
                    _cells.append("---")
            _lead = f"\\multirow{{4}}{{*}}{{{_card}}}" if _i == 0 else ""
            _rows.append(f"{_lead} & $10^{{{3 + _i}}}$ & " + " & ".join(_cells) + " \\\\")
        _rows.append("\\hline")
    with open(f"{OUT}/tab_wholerun_abs.tex", "w") as _f:
        _f.write("% generated by plot_paper_short.py from speed_grid_rtx.csv (RTX) and\n"
                 "% speed_grid_h100.csv (H100): median of 5 timed runs, seconds.\n"
                 "% needs \\usepackage{multirow}\n"
                 "\\begin{tabular}{ll|rrrr}\n\\hline\n"
                 " & $P$ & $D{=}32$ & $D{=}1{,}024$ & $D{=}32{,}768$ & $D{=}131{,}072$ \\\\\n\\hline\n"
                 + "\n".join(_rows) + "\n\\end{tabular}\n")


# ---- Fig 5 (PRIMARY): matched-population quality — the no-stable-ordering sweep
tasks = ["Nguyen 1","Nguyen 2","Nguyen 3","Nguyen 4","Nguyen 5","Nguyen 6","California","Feynman I.9.18"]  # display names match the paper
# A12 + Holm verdicts COMPUTED from the raw CSVs (galop std arms vs evogp_refs_v2)
# under the paper's rules — never hand-transcribed.
from scipy.stats import mannwhitneyu as _mwu
def _qload(path, tcol="target"):
    d = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(_R, "..", path))):
        d[r[tcol]].append(float(r["test_mse"]))
    return d
_gq = {1000: defaultdict(list), 10000: defaultdict(list), 100000: defaultdict(list)}
for r in csv.DictReader(open(os.path.join(_R, "..", "results/quality_galop_rtx.csv"))):
    _gq[int(r["P"])][r["problem"]].append(float(r["test_mse"]))
_eq = defaultdict(list)
for r in csv.DictReader(open(os.path.join(_R, "..", "results/quality_evogp_rtx.csv"))):
    _eq[(r["problem"], int(r["P"]))].append(float(r["test_mse"]))
def _a12gt(x, y):   # A12 of (GALOP error > EvoGP error): <0.5 favours GALOP
    return 1 - sum((a < b) + 0.5*(a == b) for a in x for b in y)/(len(x)*len(y))
def _holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0]*len(ps); mx = 0
    for rank, i in enumerate(idx):
        mx = max(mx, min(1.0, (len(ps)-rank)*ps[i])); out[i] = mx
    return out
_NGK = ["nguyen1","nguyen2","nguyen3","nguyen4","nguyen5","nguyen6"]
_FAMS = ([[(t, P) for t in _NGK] for P in (1000, 10000, 100000)] +
         [[("california", 10000), ("california", 100000)],
          [("feynman_i_9_18", 10000), ("feynman_i_9_18", 100000)]])
_VER = {}
for cells in _FAMS:
    st = [(t, P, _a12gt(_gq[P][t], _eq[(t, P)]),
           _mwu(_gq[P][t], _eq[(t, P)], alternative="two-sided").pvalue) for t, P in cells]
    for (t, P, A, _), ph in zip(st, _holm([x[3] for x in st])):
        _VER[(t, P)] = (round(A, 2), "U" if ph >= 0.05 else ("G" if A < 0.5 else "E"))
_KEY = _NGK + ["california", "feynman_i_9_18"]
A = {P: [_VER.get((t, P)) for t in _KEY] for P in (1000, 10000, 100000)}
_agg = Counter(v for row in A.values() for v in row if v for v in [v[1]])
assert sum(_agg.values()) == 22, _agg
SIZ = {1000: 45, 10000: 95, 100000: 170}
VC  = {"G": GAL, "E": EVO, "U": "#AAAAAA"}
fig, ax = plt.subplots(figsize=(8.8, 4.4))
y = np.arange(len(tasks))[::-1]
ax.axvspan(0, 0.5, color=GAL, alpha=0.05); ax.axvspan(0.5, 1, color=EVO, alpha=0.05)
ax.axvline(0.5, color="black", lw=1)
for ti, yi in enumerate(y):
    pts = [(P, A[P][ti]) for P in (1000, 10000, 100000) if A[P][ti] is not None]
    xs = [a for _, (a, _) in pts]
    ax.plot(xs, [yi]*len(xs), color="#CCCCCC", lw=1.2, zorder=1)
    for P, (a, v) in pts:
        ax.scatter([a], [yi], s=SIZ[P], color=VC[v], zorder=3, edgecolor="white", linewidth=1.1)
for P, lab in ((1000, "P=10³"), (10000, "P=10⁴"), (100000, "P=10⁵")):
    ax.scatter([], [], s=SIZ[P], color="#666666", label=lab, edgecolor="white")
ax.scatter([], [], s=95, color=VC["G"], label="Galop (Holm-sig.)", edgecolor="white")
ax.scatter([], [], s=95, color=VC["E"], label="EvoGP (Holm-sig.)", edgecolor="white")
ax.scatter([], [], s=95, color=VC["U"], label="undecided", edgecolor="white")
ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False)
ax.axhline(1.5, color=GRY, ls=":", lw=1)
ax.set_yticks(y, tasks, fontsize=11); ax.tick_params(axis="x", labelsize=10.5)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.7, len(tasks)-0.4)
ax.set_xlabel("Vargha–Delaney A$_{12}$", fontsize=10.5)   # direction stated in-figure + caption (2026-09-16)
ax.text(0.03, -0.62, "← Galop favoured", fontsize=10, color=GAL, weight="bold")
ax.text(0.97, -0.62, "EvoGP favoured →", ha="right", fontsize=10, color=EVO, weight="bold")
# no title: the caption carries n, the tests and the 9-9-4 aggregate (2026-09-16)
ax.grid(True, axis="x", ls=":", alpha=0.3)
fig.savefig(f"{OUT}/fig5_matched_quality.png", dpi=170, bbox_inches="tight")
fig.savefig(f"{OUT}/fig5_matched_quality.pdf", bbox_inches="tight"); plt.close(fig)

# ---- Fig 6: the capacity pillar — every system measured to its wall ----------
# GALOP: two shipped storage layouts, two bars. Intron-removed bar = the engine's
# exact index bound 2^31-2 = 2,147,483,646: a full generation (MAX_GENERATIONS=2)
# completes there in 857 s evolve / 1,112 s process against the 1,200 s criterion
# (capacity_gen2_h100.csv, 2026-09-20). Packed bar = 1.4e9, also confirmed at gen 2.
sysn = ["Kozax", "SRGPU", "EvoGP", "Beagle", "Galop", "Galop\n(intron-removed)"]
caps = [4e4, 1.6e7, 1.6777e7, 5.36e8, 1.4e9, 2147483646]
clab = ["40.0 k", "16.0 M", "16.8 M", "536 M", "1.40 B", "2.15 B"]   # three significant figures throughout
cols6 = ["#BBBBBB", "#999999", EVO, "#6C8EBF", "#7FB3E3", GAL]
# GALOP at cap width — an ADDITIONAL, labelled arm (never a replacement of the two bars
# above): every program initialized at the 50-instruction cap (STARTING_PROGRAM_SIZE_OVERRIDE=50,
# STAGE=cap50 of capacity_h100.sbatch), rows galop_packed_cap50 / galop_eff_cap50 of
# capacity_h100.csv; ceiling = largest OK rung, the rule of every other bar. Drawn (hatched)
# only once those rows exist, so the figure is unchanged until the arm lands.
_cap50 = {}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_h100", "capacity_h100.csv")) as _fh:
    for _r in csv.DictReader(_fh):
        if _r["system"].endswith("_cap50") and _r["outcome"] == "OK":
            _cap50[_r["system"]] = max(_cap50.get(_r["system"], 0), int(_r["P"]))
def _lab3(v):   # three significant figures with k / M / B, as the labels above
    for _u, _s in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if v >= _s:
            _x = v / _s
            return (f"{_x:.0f}" if _x >= 100 else f"{_x:.1f}" if _x >= 10 else f"{_x:.2f}") + f" {_u}"
    return f"{v:.0f}"
_n_cap50 = 0
for _k, _n, _c in (("galop_packed_cap50", "Galop\n(max-length stress test)", "#7FB3E3"),
                   ("galop_eff_cap50", "Galop (intron-removed,\ncap-initialized stress test)", GAL)):
    if _k in _cap50:
        sysn.append(_n); caps.append(_cap50[_k]); clab.append(_lab3(_cap50[_k])); cols6.append(_c); _n_cap50 += 1
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.0 + 0.45 * _n_cap50), layout="constrained",
                             gridspec_kw={"width_ratios": [3, 2.5]})
b = a1.barh(sysn, caps, color=cols6, height=0.6)
for _bar in list(b)[len(b) - _n_cap50:]: _bar.set_hatch("///"); _bar.set_edgecolor("white")
a1.set_xscale("log"); a1.set_xlim(8e3, 3e10)
for r, c, l in zip(b, caps, clab):   # value labels outside every bar; the limiting mechanism is in the text, not the figure
    a1.text(c*1.12, r.get_y() + r.get_height()/2, l, va="center", fontsize=10, weight="bold")
a1.set_xlabel("maximum population completing one generation, 80 GB H100 (log)", fontsize=10.5)
a1.tick_params(axis="x", labelsize=10.5); a1.tick_params(axis="y", labelsize=11)
# (b) sustained 100-generation runs (scale_h100.csv), linear wall-clock axis. Light bar and marker =
# median of the timed runs, whisker = observed minimum to maximum. No line joins the rows: the runs
# were spread over shared nodes and the per-individual timings do not support a scaling trend. The
# layout, thread count and completion counts are in the caption.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_h100", "scale_h100.csv")) as _fh:
    _sc = list(csv.DictReader(_fh))
def _runs(system, P):
    rs = [r for r in _sc if r["system"] == system and r["P"] == str(P) and r["rep"] != "warmup"]
    return sorted(float(r["wall_s"]) for r in rs if r["outcome"] == "OK"), len(rs)
_BGL = "#6C8EBF"
_rowsb = [("Beagle\n$P=10^{8}$", _runs("beagle", 100000000), _BGL),
          ("Galop\n$P=10^{8}$", _runs("galop_eff", 100000000), GAL),
          ("Galop\n$P=10^{9}$", _runs("galop_eff", 1000000000), GAL),
          ("Galop\n$P=2^{31}{-}2$", _runs("galop_eff", 2147483646), GAL)]
for _y, (_name, (_w, _n), _col) in enumerate(_rowsb):
    _m = _med(_w)
    a2.barh(_y, _m, color=_col, alpha=0.25, height=0.52, zorder=1)
    a2.plot([min(_w), max(_w)], [_y, _y], color=_col, lw=1.8, zorder=3, solid_capstyle="butt")
    for _x in (min(_w), max(_w)): a2.plot([_x, _x], [_y - 0.13, _y + 0.13], color=_col, lw=1.8, zorder=3)
    a2.scatter([_m], [_y], s=58, color=_col, edgecolor="white", linewidth=1.0, zorder=5)
    a2.text(max(_w) + 330, _y, f"{_m:,.0f} s", va="center", fontsize=9.5, weight="bold")
a2.set_yticks(range(len(_rowsb))); a2.set_yticklabels([_r[0] for _r in _rowsb])
a2.set_xlim(0, 13200); a2.set_xticks([0, 3000, 6000, 9000, 12000])
a2.set_xticklabels(["0", "3,000", "6,000", "9,000", "12,000"])
a2.set_ylim(-0.6, 3.6)
a2.set_xlabel("wall-clock for 100 generations, H100 (s)", fontsize=10.5)
a2.tick_params(axis="x", labelsize=10.5); a2.tick_params(axis="y", labelsize=11)
for a in (a1, a2): a.grid(True, axis="x", ls=":", alpha=0.3)
# no panel titles (2026-09-16): the caption carries the protocol; letters on one baseline
fig.canvas.draw()
_bot = min(_a.get_position().y0 for _a in (a1, a2))
for _ax, _lab in ((a1, "(a)"), (a2, "(b)")):
    _bb = _ax.get_position()
    fig.text((_bb.x0 + _bb.x1) / 2, _bot - 0.13, _lab, ha="center", va="top", fontsize=10)
fig.savefig(f"{OUT}/fig6_capacity.png", dpi=170, bbox_inches="tight")
fig.savefig(f"{OUT}/fig6_capacity.pdf", bbox_inches="tight"); plt.close(fig)
