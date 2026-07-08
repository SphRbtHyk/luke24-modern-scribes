import luke_corpus as L
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys
import collections
sys.path.insert(0, os.path.dirname(__file__))
matplotlib.use("Agg")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "figure.dpi": 150,
})
BORD = "#7B1E3B"
BLUE = "#2C5F7C"
GRAY = "#4d4d4d"
OLIVE = "#6B7A3A"
SAND = "#C9A66B"
LGRAY = "#d9d9d9"
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

corpus = L.load_corpus(os.path.join(
    os.path.dirname(__file__), "..", "data/raw/"))

units, inserts, coverage, base_toks = L.collate(corpus)
core = L.core_boundary(coverage)
recs = L.classify_all(units, corpus, base_toks=base_toks,
                      inserts=inserts, upto=None)
sigs = list(corpus.keys())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. coverage
fig, ax = plt.subplots(figsize=(7.2, 3.4))
order = sigs
ncov = [coverage[s][1] + 1 for s in order]
colors = [BORD if (coverage[s][1] >= core) else SAND for s in order]
y = np.arange(len(order))
ax.barh(y, [len(base_toks)] * len(order), color=LGRAY, height=0.62, zorder=1)
ax.barh(y, ncov, color=colors, height=0.62, zorder=2)
ax.axvline(core + 1, color=GRAY, ls="--", lw=1.2, zorder=3)
ax.text(core + 4, len(order) - 0.4, f"comparable core\n(Luke 24:1–10a, {core+1} tok)",
        fontsize=8.5, color=GRAY, va="top")
ax.set_yticks(y)
ax.set_yticklabels(order)
ax.invert_yaxis()
ax.set_xlabel("tokens copied (of 406 in base witness 11)")
ax.set_title("How far each modern scribe got",
             color=BORD, fontweight="bold", loc="left")
save(fig, "fig_coverage.pdf")

# ---------------------------------------------------------------- 2. tiers donut
by_tier = collections.Counter(r["tier"] for r in recs)
labels = ["orthographic /\nphonological",
          "mechanical /\npalaeographic", "structural /\ntextual"]
keys = ["orthographic", "mechanical", "substantive"]
vals = [by_tier[k] for k in keys]
cols = [BLUE, SAND, BORD]
fig, ax = plt.subplots(figsize=(5.4, 4.2))
w, _t, at = ax.pie(vals, colors=cols, startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
                   autopct=lambda p: f"{p:.0f}%", pctdistance=0.79,
                   textprops=dict(color="white", fontweight="bold", fontsize=11))
ax.legend(w, [f"{l}  (n={v})" for l, v in zip(labels, vals)],
          loc="center", frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.08), ncol=1)
tot = sum(vals)
ax.text(0, 0, f"{tot}\nreadings", ha="center",
        va="center", fontsize=12, color=GRAY)
ax.set_title("Typology of variation (full collation)",
             color=BORD, fontweight="bold")
save(fig, "fig_tiers.pdf")

# ---------------------------------------------------------------- 3. minor cats
by_minor = collections.Counter(r["minor"] for r in recs)
name_map = {
    "single-letter-slip": "single-letter slip", "lexical-substitution": "larger substitution",
    "minor-corruption": "minor corruption", "omission": "word omission",
    "addition": "word addition", "vowel-interchange": "vowel interchange (itacism)",
    "metathesis": "metathesis", "movable-nu": "movable nu",
    "gemination": "single/double consonant", "word-division": "word division",
}
tier_col = {"orthographic": BLUE, "mechanical": SAND, "substantive": BORD}
items = by_minor.most_common()
labs = [name_map.get(k, k) for k, _ in items]
vv = [v for _, v in items]
cc = [tier_col[L.tier_of(k)] for k, _ in items]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
yy = np.arange(len(items))[::-1]
ax.barh(yy, vv, color=cc)
for y_, v in zip(yy, vv):
    ax.text(v + 1, y_, str(v), va="center", fontsize=9, color=GRAY)
ax.set_yticks(yy)
ax.set_yticklabels(labs, fontsize=9.5)
ax.set_xlabel("number of readings")
ax.set_xlim(0, max(vv) * 1.12)
ax.legend(handles=[Patch(color=tier_col[t], label=t) for t in ["orthographic", "mechanical", "substantive"]],
          frameon=False, fontsize=9, loc="lower right")
ax.set_title("Variant categories", color=BORD, fontweight="bold", loc="left")
save(fig, "fig_categories.pdf")

# ---------------------------------------------------------------- 4. per-witness
pw = collections.defaultdict(lambda: collections.Counter())
for r in recs:
    pw[r["witness"]][r["tier"]] += 1
ws = [s for s in sigs if s != L.BASE]
orth = [pw[s]["orthographic"] for s in ws]
mech = [pw[s]["mechanical"] for s in ws]
sub = [pw[s]["substantive"] for s in ws]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(ws))
ax.bar(x, orth, color=BLUE, label="orthographic")
ax.bar(x, mech, bottom=orth, color=SAND, label="mechanical")
ax.bar(x, sub, bottom=np.array(orth) +
       np.array(mech), color=BORD, label="structural")
ax.set_xticks(x)
ax.set_xticklabels(ws)
ax.set_ylabel("readings vs. base")
ax.legend(frameon=False, fontsize=9)
ax.set_title("Variation is concentrated in a few careless copies",
             color=BORD, fontweight="bold", loc="left")
save(fig, "fig_perwitness.pdf")

# ---------------------------------------------------------------- 5. heatmap
sigs2, Draw, Dnorm = L.distance_matrices(units, corpus, core)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(Dnorm, cmap="RdPu", vmin=0)
ax.set_xticks(range(len(sigs2)))
ax.set_xticklabels(sigs2)
ax.set_yticks(range(len(sigs2)))
ax.set_yticklabels(sigs2)
for i in range(len(sigs2)):
    for j in range(len(sigs2)):
        v = Dnorm[i][j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=7.5, color="white" if v > 0.28 else GRAY)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
             label="normalised variant distance")
ax.set_title("Witness distances (core block)", color=BORD, fontweight="bold")
save(fig, "fig_heatmap.pdf")

# ---------------------------------------------------------------- 6. NJ tree
nwN, children, names, root = L.neighbour_joining(sigs2, Dnorm)
# layout
leaf_order = []


def collect(n):
    if n in children:
        for c, _l in children[n]:
            collect(c)
    else:
        leaf_order.append(n)


collect(root)
ypos = {leaf: i for i, leaf in enumerate(leaf_order)}
xpos = {}


def setx(n, x0):
    if n in children:
        ys = []
        for c, l in children[n]:
            setx(c, x0 + l)
            ys.append(yof(c))
        xpos[n] = x0
    else:
        xpos[n] = x0


def yof(n):
    if n in children:
        return np.mean([yof(c) for c, _ in children[n]])
    return ypos[n]


setx(root, 0.0)
fig, ax = plt.subplots(figsize=(6.6, 4.0))
clean = {"11", "12", "21", "22", "31"}


def draw(n):
    if n in children:
        ychildren = [yof(c) for c, _ in children[n]]
        ax.plot([xpos[n], xpos[n]], [min(ychildren),
                max(ychildren)], color=GRAY, lw=1.3)
        for c, l in children[n]:
            ax.plot([xpos[n], xpos[n] + l],
                    [yof(c), yof(c)], color=GRAY, lw=1.3)
            draw(c)
    else:
        col = BLUE if n in clean else BORD
        ax.plot(xpos[n], yof(n), "o", color=col, ms=7)
        ax.text(xpos[n] + 0.006, yof(n), names[n], va="center", fontsize=11,
                color=col, fontweight="bold")


draw(root)
ax.axis("off")
ax.set_title("Neighbour-joining tree from the copied text",
             color=BORD, fontweight="bold", loc="left")
ax.text(0.0, -0.9, "blue = careful copies · red = heavily corrupted\n"
        "the tree reflects scribal care, not the intended genealogy",
        fontsize=8.5, color=GRAY)
save(fig, "fig_njtree.pdf")

print("\nAll figures written to", FIG)
