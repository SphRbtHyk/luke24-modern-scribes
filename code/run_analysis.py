import json
import luke_corpus as L
import numpy as np
import os
import sys
import collections
sys.path.insert(0, os.path.dirname(__file__))

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
corpus = L.load_corpus(DATA)

print("== witnesses ==")
for s, w in corpus.items():
    print(
        f"  {s}: {len(w['tokens']):3d} tok | {w['time']} | uncertain-markers {w['n_uncertain']}")

units, inserts, coverage, base_toks = L.collate(corpus)
core = L.core_boundary(coverage)
print(f"\nbase={L.BASE} ({len(base_toks)} tok); coverage ends (base idx):")
for s in corpus:
    print(f"  {s}: 0..{coverage[s][1]}")
print(f"core (all 8 present): 0..{core}  ends '{' '.join(base_toks[max(0, core-2):core+1])}'  "
      f"= {core+1} tokens (~{100*(core+1)/len(base_toks):.0f}% of base)")

recs = L.classify_all(units, corpus, base_toks=base_toks,
                      inserts=inserts, upto=None)
by_tier = collections.Counter(r["tier"] for r in recs)
by_minor = collections.Counter(r["minor"] for r in recs)
tot = sum(by_tier.values())
print(f"\n== full collation: {tot} non-base readings classified ==")
print("by tier:")
for t in ("orthographic", "mechanical", "substantive"):
    print(f"  {t:13s} {by_tier[t]:4d}  ({100*by_tier[t]/tot:4.1f}%)")
print("by minor category:")
for mn, v in by_minor.most_common():
    print(f"  {L.tier_of(mn):13s}/{mn:22s} {v}")
low = by_tier["orthographic"] + by_tier["mechanical"]
print(
    f"\nlow-signal (orth+mech): {low} ({100*low/tot:.1f}%) | substantive: {by_tier['substantive']} ({100*by_tier['substantive']/tot:.1f}%)")
sub_om = by_minor["omission"]
sub_int = by_tier["substantive"] - sub_om
print(
    f"   substantive split: omission {sub_om}, substitution/addition {sub_int}")

print("\nper-witness readings by tier:")
pw = collections.defaultdict(lambda: collections.Counter())
for r in recs:
    pw[r["witness"]][r["tier"]] += 1
for s in corpus:
    if s == L.BASE:
        continue
    c = pw[s]
    print(
        f"  {s}: orth {c['orthographic']:2d} | mech {c['mechanical']:2d} | subst {c['substantive']:2d} | tot {sum(c.values())}")

print("\n== diagnostic spot-checks ==")


def show(sub):
    for u in units:
        if sub in u.base:
            print(f"  base[{u.idx}]='{u.base}': " +
                  " ".join(f"{s}={'om' if u.readings[s] is None else u.readings[s]}" for s in corpus))
            return


for x in ("ουλαμμαους", "ιερουσαλημημ", "συνεπορευετο", "σταδιους"):
    show(x)

sigs, Draw, Dnorm = L.distance_matrices(units, corpus, core)
print("\n== normalised distance, core ==")
print("     " + " ".join(f"{s:>5}" for s in sigs))
for i, s in enumerate(sigs):
    print(f"{s:>4} " +
          " ".join(f"{Dnorm[i][j]:.3f}" for j in range(len(sigs))))
nwN, chN, nmN, rtN = L.neighbour_joining(sigs, Dnorm)
print("\nNJ(norm):", nwN)
dchild, dnames, droot = L.design_tree_children()
rfN, A, B = L.robinson_foulds(dchild, dnames, droot, chN, nmN, rtN, sigs)
maxrf = 2 * (len(sigs) - 3)
print(f"RF(design,NJ)={rfN} (max {maxrf})")
print("design splits:", sorted(tuple(sorted(x)) for x in A))
print("NJ splits    :", sorted(tuple(sorted(x)) for x in B))
iu = np.triu_indices(len(sigs), 1)
print(f"mean raw {Draw[iu].mean():.3f} | mean norm {Dnorm[iu].mean():.3f}")

stats = {
    "n_witnesses": len(corpus), "base": L.BASE, "n_base_tokens": len(base_toks),
    "core_end": core, "core_len": core+1, "core_pct": round(100*(core+1)/len(base_toks), 1),
    "coverage": {s: coverage[s][1] for s in corpus},
    "tokens": {s: len(corpus[s]["tokens"]) for s in corpus},
    "uncertain": {s: corpus[s]["n_uncertain"] for s in corpus},
    "total_readings": tot, "tier_counts": dict(by_tier),
    "tier_pct": {t: round(100*by_tier[t]/tot, 1) for t in by_tier},
    "minor_counts": dict(by_minor),
    "low_signal_pct": round(100*low/tot, 1),
    "substantive_pct": round(100*by_tier['substantive']/tot, 1),
    "omission_count": sub_om, "intelligent_sub_count": sub_int,
    "rf": rfN, "rf_max": maxrf,
    "mean_norm_dist": round(float(Dnorm[iu].mean()), 3),
    "mean_raw_dist": round(float(Draw[iu].mean()), 3),
    "newick_norm": nwN,
    "per_witness": {s: dict(pw[s]) for s in corpus if s != L.BASE},
}
with open(os.path.join(os.path.dirname(__file__), "..", "figures", "stats.json"), "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("\n[stats.json written]")
