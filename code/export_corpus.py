import os
import sys
import csv
import json
import shutil
import html
import luke_corpus as L
sys.path.insert(0, os.path.dirname(__file__))

HERE = os.path.dirname(__file__)
REPO = os.path.join(HERE, "..", "repo", "luke24-modern-scribes")
DATA = os.path.join(HERE, "..", "data/raw")
FIG = os.path.join(HERE, "..", "figures")


def mk(*p):
    d = os.path.join(REPO, *p)
    os.makedirs(d, exist_ok=True)
    return d


for sub in [("data", "raw"), ("data", "diplomatic"), ("derived"), ("code"), ("figures")]:
    mk(*([sub] if isinstance(sub, str) else sub))

corpus = L.load_corpus(DATA)
print("--- corpus")
print(corpus)
units, inserts, coverage, base_toks = L.collate(corpus)
core = L.core_boundary(coverage)
recs = L.classify_all(units, corpus, base_toks=base_toks,
                      inserts=inserts, upto=None)
sigs = list(corpus.keys())

# ---- raw + diplomatic text ----
for fn in os.listdir(DATA):
    shutil.copy(os.path.join(DATA, fn), os.path.join(REPO, "data", "raw", fn))
for s, w in corpus.items():
    open(os.path.join(REPO, "data", "diplomatic", f"{s}.txt"), "w", encoding="utf-8")\
        .write(" ".join(w["tokens"]) + "\n")

# ---- corpus.json ----
cj = {
    "corpus": "Luke 24 modern-scribe copies (Helsinki pilot)",
    "project": "Experimenting the Rise of Early Christianity",
    "partners": ["University of Helsinki", "Umea University", "Sorbonne University"],
    "language": "Koine Greek", "passage": "Luke 24:1-28 (road to Emmaus)",
    "script_note": "exemplar is the Bezae",
    "base_witness": L.BASE, "n_base_tokens": len(base_toks),
    "comparable_core": {"base_index_end": core, "n_tokens": core + 1,
                        "approx_passage": "Luke 24:1-10a"},
    "design_stemma": L.DESIGN_STEMMA,
    "naming_convention": "label = <parent><child-ordinal>",
    "witnesses": {
        s: {"session_time": w["time"], "n_tokens": len(w["tokens"]),
            "coverage_base_index_end": coverage[s][1],
            "uncertain_markers": w["n_uncertain"],
            "tokens": w["tokens"]}
        for s, w in corpus.items()},
    "license_data": "CC-BY-4.0", "license_code": "MIT",
}
json.dump(cj, open(os.path.join(REPO, "derived", "corpus.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

# ---- variants.csv ----
with open(os.path.join(REPO, "derived", "variants.csv"), "w", newline="", encoding="utf-8") as f:
    wtr = csv.DictWriter(
        f, fieldnames=["unit", "base", "witness", "reading", "tier", "major", "minor"])
    wtr.writeheader()
    for r in recs:
        wtr.writerow({k: r[k] for k in ["unit", "base",
                     "witness", "reading", "tier", "major", "minor"]})

# ---- alignment table (core) CSV + TSV ----


def write_align(path, sep):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=sep)
        w.writerow(["unit", "base"] + sigs)
        for u in units:
            if u.idx > core:
                break
            row = [u.idx, u.base]
            for s in sigs:
                r = u.readings[s]
                row.append("[om]" if r is None else ("" if r == "LAC" else r))
            w.writerow(row)


write_align(os.path.join(REPO, "derived", "alignment_core.csv"), ",")
write_align(os.path.join(REPO, "derived", "alignment_core.tsv"), "\t")

# ---- distances (core) ----
sigs2, Draw, Dnorm = L.distance_matrices(units, corpus, core)
with open(os.path.join(REPO, "derived", "distances_core.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([""] + sigs2)
    for i, s in enumerate(sigs2):
        w.writerow([s] + [f"{Dnorm[i][j]:.4f}" for j in range(len(sigs2))])

# ---- TEI parallel-segmentation apparatus (core block) ----


def tei_reading_groups(u):
    groups = {}
    for s in sigs:
        r = u.readings[s]
        if r == "LAC":
            continue
        key = "\u0000OM" if r is None else r
        groups.setdefault(key, []).append(s)
    return groups


lines = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<TEI xmlns="http://www.tei-c.org/ns/1.0">',
         '  <teiHeader><fileDesc>',
         '    <titleStmt><title>Luke 24 modern-scribe copies — critical apparatus (core block)</title></titleStmt>',
         '    <publicationStmt><p>Experimenting the Rise of Early Christianity. CC-BY-4.0.</p></publicationStmt>',
         '    <sourceDesc><listWit>']
for s in sigs:
    lines.append(
        f'      <witness xml:id="w{s}">Modern scribe {s} (session {corpus[s]["time"]})</witness>')
lines += ['    </listWit></sourceDesc>', '  </fileDesc></teiHeader>',
          '  <text><body><p>']
for u in units:
    if u.idx > core:
        break
    groups = tei_reading_groups(u)
    real = {k: v for k, v in groups.items() if k != "\u0000OM"}
    if len(groups) == 1 and "\u0000OM" not in groups:
        lines.append(f'    {html.escape(u.base)}')
    else:
        app = ['    <app>']
        for reading, wits in groups.items():
            witattr = " ".join(f"#w{w}" for w in wits)
            if reading == "\u0000OM":
                app.append(f'      <rdg wit="{witattr}"/>')
            else:
                app.append(
                    f'      <rdg wit="{witattr}">{html.escape(reading)}</rdg>')
        app.append('    </app>')
        lines.append("\n".join(app))
lines += ['  </p></body></text>', '</TEI>']
open(os.path.join(REPO, "derived", "apparatus_core.tei.xml"),
     "w", encoding="utf-8").write("\n".join(lines))

# ---- design stemma: Newick
open(os.path.join(REPO, "derived", "stemma_design.nwk"), "w").write(
    "(((31)21,(32)22)11,(23,24)12)O;\n")

# ---- copy code + figures ----
for f in ["luke_corpus.py", "run_analysis.py", "make_figures.py", "export_corpus.py"]:
    shutil.copy(os.path.join(HERE, f), os.path.join(REPO, "code", f))
for f in os.listdir(FIG):
    if f.endswith(".pdf"):
        shutil.copy(os.path.join(FIG, f), os.path.join(REPO, "figures", f))

print("Corpus package assembled at", REPO)
for root, _d, files in os.walk(REPO):
    for f in sorted(files):
        rel = os.path.relpath(os.path.join(root, f), REPO)
        print("  ", rel)
