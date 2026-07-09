"""
luke_corpus.py
==============
Analysis toolkit for the Helsinki "modern-scribe" Koine Greek corpus
(Gospel of Luke 24, road-to-Emmaus pericope).

Pipeline
--------
1.  parse + clean the diplomatic transcriptions produced by the checker
2.  normalise at several *phonological/orthographic* layers (Koine itacism etc.)
3.  base-text collation of every witness against a base witness
4.  build variation units and classify every non-base reading
5.  witness-to-witness distance matrix (raw vs. normalised) + neighbour joining
6.  export the corpus in several open formats (TXT, CSV/TSV, JSON, TEI-XML, Newick)
"""

from __future__ import annotations
import glob
import os
import re
import unicodedata
import itertools
import collections
from dataclasses import dataclass, field, asdict
import numpy as np

# --------------------------------------------------------------------------
# 0.  Experimental design metadata
# --------------------------------------------------------------------------

DESIGN_STEMMA = {
    "O":  ["11", "12"],  # 0 = lost archetype
    "11": ["21", "22"],
    "12": ["23", "24"],
    "21": ["31"],
    "22": ["32"],
}
WITNESSES = ["11", "12", "21", "22", "23", "24", "31", "32"]
BASE = "11"

# --------------------------------------------------------------------------
# 1.  Parsing & cleaning
# --------------------------------------------------------------------------
UNCERTAIN = set("?~")          # checker markers for illegible / uncertain
PUNCT = set("·.,*")            # ano teleia, stop, comma, asterisk marker


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def load_witness(path: str):
    raw = open(path, encoding="utf-8").read()
    lines = [l for l in raw.splitlines()]
    header = lines[0].strip()
    m = re.match(r"(\d+)\s*\(([\d.]+)\)", header)
    sig, time = (m.group(1), m.group(2)) if m else (
        os.path.basename(path)[:2], "")
    body_lines = lines[1:]
    # join hyphenated line breaks:  "απο-\nκαλυψεν" -> "αποκαλυψεν"
    joined = []
    for ln in body_lines:
        joined.append(ln.rstrip())
    text = "\n".join(joined)
    # de-hyphenate across line breaks
    text = re.sub(r"-\s*\n\s*", "", text)
    text = text.replace("\n", " ")
    # record data-quality markers before we remove them
    n_uncertain = sum(text.count(c) for c in UNCERTAIN)
    # drop marker chars and punctuation
    for c in UNCERTAIN | PUNCT:
        text = text.replace(c, " ")
    text = text.replace("(", " ").replace(")", " ")
    # NFC + lower + strip accents (there are essentially none, but be safe)
    text = unicodedata.normalize("NFC", text).lower()
    text = _strip_accents(text)
    # tokenise; drop pure-number tokens (checker's marginal line/verse numbers)
    toks = [t for t in text.split() if t and not t.isdigit()]
    return {"sig": sig, "time": time, "tokens": toks, "n_uncertain": n_uncertain}


def load_corpus(data_dir: str):
    wits = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*_checked_by_Jimi.txt"))):
        w = load_witness(path)
        wits[w["sig"]] = w
    return {s: wits[s] for s in WITNESSES if s in wits}


# --------------------------------------------------------------------------
# 2.  Normalisation layers  (Koine phonology)
# --------------------------------------------------------------------------
# Nomina sacra encountered in the transcriptions -> expanded forms (nom. sg.
# lemma-level, we only need a canonical key for comparison).
NOMINA_SACRA = {
    "ιηυ": "ιησου", "ιης": "ιησους", "χρν": "χριστον", "χρου": "χριστου",
    "θυ": "θεου", "θν": "θεον", "κυ": "κυριου", "υν": "υιον",
}


def fold_vowels(tok: str) -> str:
    """Collapse the classic Koine itacistic / vowel-interchange sets."""
    s = tok
    # order matters: handle digraphs first
    s = s.replace("ει", "ι").replace("οι", "ι").replace("υι", "ι")
    s = s.replace("αι", "ε")
    s = s.replace("η", "ι").replace("υ", "ι").replace("ω", "ο")
    # ου (=/u/) is kept distinct from ο; protect it by a placeholder
    return s


def strip_movable_nu(tok: str) -> str:
    return tok[:-1] if tok.endswith("ν") else tok


def collapse_doubles(tok: str) -> str:
    return re.sub(r"(.)\1", r"\1", tok)


def norm_key(tok: str) -> str:
    """Full normalisation used for the 'genealogical-signal' comparison:
    NS expansion + vowel folding + movable-nu + gemination."""
    t = NOMINA_SACRA.get(tok, tok)
    t = fold_vowels(t)
    t = strip_movable_nu(t)
    t = collapse_doubles(t)
    return t

# --------------------------------------------------------------------------
# 3.  Pairwise Needleman-Wunsch on token sequences (equality = norm_key match)
# --------------------------------------------------------------------------


def nw_align(a, b, key=lambda x: x, match=1, mismatch=-1, gap=-1):
    """Global alignment of token lists a (base) and b (witness).
    Returns list of (ia, ib) pairs; None marks a gap."""
    n, m = len(a), len(b)
    ka = [key(x) for x in a]
    kb = [key(x) for x in b]
    D = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        D[i][0] = i * gap
    for j in range(1, m + 1):
        D[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sc = match if ka[i-1] == kb[j-1] else mismatch
            D[i][j] = max(D[i-1][j-1] + sc, D[i-1][j] + gap, D[i][j-1] + gap)
    # traceback
    i, j, aln = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i-1][j-1] + (match if ka[i-1] == kb[j-1] else mismatch):
            aln.append((i-1, j-1))
            i -= 1
            j -= 1
        elif i > 0 and D[i][j] == D[i-1][j] + gap:
            aln.append((i-1, None))
            i -= 1
        else:
            aln.append((None, j-1))
            j -= 1
    aln.reverse()
    return aln

# --------------------------------------------------------------------------
# 4.  Collation against the base  ->  variation units
# --------------------------------------------------------------------------


@dataclass
class Unit:
    idx: int                       # base token index (-1 for pure insertions)
    base: str                      # base reading ("" if insertion)
    # sig -> str | None(=omit) | "" (=lacuna)
    readings: dict = field(default_factory=dict)


def collate(corpus):
    base_toks = corpus[BASE]["tokens"]
    # base-anchored table: for each base index, each witness's aligned tokens
    table = {i: {} for i in range(len(base_toks))}
    inserts = collections.defaultdict(
        lambda: collections.defaultdict(list))  # afterIdx -> sig -> [toks]
    coverage = {}   # sig -> (first_base_idx, last_base_idx) actually aligned
    for sig, w in corpus.items():
        aln = nw_align(base_toks, w["tokens"], key=norm_key)
        paired = [(ia, ib)
                  for (ia, ib) in aln if ia is not None and ib is not None]
        matched_ia = sorted(ia for ia, ib in paired)
        if matched_ia:
            first = matched_ia[0]
            last = matched_ia[-1]
            GAPMIN = 20                       # a jump this large = off-prefix
            for a, b in zip(matched_ia, matched_ia[1:]):
                if b - a >= GAPMIN:
                    last = a                  # cut before the spurious tail
                    break
        else:
            first, last = 0, -1
        coverage[sig] = (first, last)
        last_base_seen = -1
        for (ia, ib) in aln:
            if ia is not None and ib is not None:
                table[ia][sig] = w["tokens"][ib]
                last_base_seen = ia
            elif ia is not None and ib is None:
                # base token missing -> omission (or lacuna if past coverage)
                table[ia][sig] = None
                last_base_seen = ia
            elif ia is None and ib is not None:
                inserts[last_base_seen][sig].append(
                    w["tokens"][ib])  # extra witness token
    # build Unit list (substitution/omission units at every base index)
    units = []
    for i in range(len(base_toks)):
        readings = {}
        for sig in corpus:
            first, last = coverage[sig]
            in_cover = (first <= i <= last)
            if not in_cover:
                readings[sig] = "LAC"          # scribe simply stopped here
            elif sig in table[i] and table[i][sig] is not None:
                readings[sig] = table[i][sig]
            else:
                # true omission within covered span
                readings[sig] = None
        units.append(Unit(idx=i, base=base_toks[i], readings=readings))
    return units, inserts, coverage, base_toks


def core_boundary(coverage):
    """Last base index present in ALL witnesses = comparable core block."""
    return min(last for (_f, last) in coverage.values())


# --------------------------------------------------------------------------
# 5.  Variant classification
# --------------------------------------------------------------------------
VOWELS = set("αειηουω")


def is_anagram(a, b):
    return a != b and sorted(a) == sorted(b) and len(a) > 2


def edit_distance(a, b):
    n, m = len(a), len(b)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j-1] + 1, prev + (a[i-1] != b[j-1]))
            prev = cur
    return d[m]


# (major, minor) -> analytic tier used throughout the paper
TIER = {
    "vowel-interchange": "orthographic", "movable-nu": "orthographic",
    "gemination": "orthographic", "nomen-sacrum": "orthographic",
    "other-orthographic": "orthographic", "word-division": "orthographic",
    "metathesis": "mechanical", "single-letter-slip": "mechanical",
    "minor-corruption": "mechanical",
    "omission": "substantive", "addition": "substantive",
    "transposition": "substantive", "lexical-substitution": "substantive",
}
def tier_of(minor): return TIER.get(minor, "substantive")


def classify_reading(base_tok: str, read: str, next_base: str = None):
    """Classify one witness reading `read` against `base_tok`.
    Returns (major, minor) category strings."""
    if read is None:
        return ("substantive", "omission")
    if read == "LAC":
        return ("lacuna", "not-copied")
    if read == base_tok:
        return ("agreement", "=")
    # --- word division (scriptio continua): witness merged base[i]+base[i+1] ---
    if next_base and read == base_tok + next_base:
        return ("orthographic", "word-division")
    # --- nomina sacra ---
    if base_tok in NOMINA_SACRA or read in NOMINA_SACRA:
        if norm_key(base_tok) == norm_key(read):
            return ("orthographic", "nomen-sacrum")
    # --- metathesis (same letters, reordered) ---
    if is_anagram(base_tok, read):
        return ("mechanical", "metathesis")
    # --- vowel interchange / itacism ---
    if fold_vowels(base_tok) == fold_vowels(read):
        return ("orthographic", "vowel-interchange")
    # --- movable nu / final consonant ---
    if strip_movable_nu(fold_vowels(base_tok)) == strip_movable_nu(fold_vowels(read)):
        return ("orthographic", "movable-nu")
    # --- gemination (single/double consonant) ---
    if collapse_doubles(fold_vowels(base_tok)) == collapse_doubles(fold_vowels(read)):
        return ("orthographic", "gemination")
    # fully normalised equal -> residual orthographic
    if norm_key(base_tok) == norm_key(read):
        return ("orthographic", "other-orthographic")
    # --- substantive ---
    ed = edit_distance(base_tok, read)
    # small single-consonant slip that survives normalisation = mechanical error
    if ed == 1 and len(base_tok) > 2:
        return ("mechanical", "single-letter-slip")
    if ed <= 2 and abs(len(base_tok) - len(read)) <= 1:
        return ("mechanical", "minor-corruption")
    return ("substantive", "lexical-substitution")


def classify_all(units, corpus, base_toks=None, inserts=None, upto=None):
    """One record per (unit, witness) where the witness differs from base.
    Insertions (extra witness tokens) are added as 'addition' records."""
    recs = []
    for u in units:
        if upto is not None and u.idx > upto:
            continue
        nb = base_toks[u.idx +
                       1] if (base_toks and u.idx + 1 < len(base_toks)) else None
        for sig in corpus:
            if sig == BASE:
                continue
            r = u.readings.get(sig)
            major, minor = classify_reading(u.base, r, next_base=nb)
            if major in ("agreement", "lacuna"):
                continue
            recs.append({"unit": u.idx, "base": u.base, "witness": sig,
                         "reading": ("[om]" if r is None else r),
                         "major": major, "minor": minor, "tier": tier_of(minor)})
    if inserts:
        for after_idx, per_sig in inserts.items():
            if upto is not None and after_idx > upto:
                continue
            for sig, toks in per_sig.items():
                if sig == BASE:
                    continue
                for t in toks:
                    recs.append({"unit": after_idx, "base": "[+]", "witness": sig,
                                 "reading": t, "major": "substantive",
                                 "minor": "addition", "tier": "substantive"})
    return recs

# --------------------------------------------------------------------------
# 6.  Distance matrices + neighbour joining + Robinson-Foulds
# --------------------------------------------------------------------------


def distance_matrices(units, corpus, upto):
    sigs = list(corpus.keys())

    def dist(level):
        D = np.zeros((len(sigs), len(sigs)))
        for a, b in itertools.combinations(range(len(sigs)), 2):
            sa, sb = sigs[a], sigs[b]
            diff = comp = 0
            for u in units:
                if u.idx > upto:
                    break
                ra, rb = u.readings[sa], u.readings[sb]
                if ra == "LAC" or rb == "LAC":
                    continue
                comp += 1
                if level == "raw":
                    same = (ra == rb)
                else:  # normalised
                    ka = None if ra is None else norm_key(ra)
                    kb = None if rb is None else norm_key(rb)
                    same = (ka == kb)
                if not same:
                    diff += 1
            d = diff / comp if comp else 0.0
            D[a][b] = D[b][a] = d
        return D
    return sigs, dist("raw"), dist("norm")


def neighbour_joining(sigs, D):
    """Standard NJ. Returns Newick string + edge list for plotting."""
    nodes = list(sigs)
    D = D.astype(float).copy()
    n = len(nodes)
    active = list(range(n))
    mat = {(i, j): D[i][j] for i in range(n) for j in range(n)}
    next_id = n
    node_names = {i: sigs[i] for i in range(n)}
    children = {}      # internal node -> [(child, length), ...]
    def d(i, j): return mat[(i, j)] if i != j else 0.0
    while len(active) > 2:
        N = len(active)
        r = {i: sum(d(i, k) for k in active) for i in active}
        best, bi, bj = None, None, None
        for a in range(N):
            for b in range(a + 1, N):
                i, j = active[a], active[b]
                q = (N - 2) * d(i, j) - r[i] - r[j]
                if best is None or q < best:
                    best, bi, bj = q, i, j
        i, j = bi, bj
        dij = d(i, j)
        li = 0.5 * dij + (r[i] - r[j]) / (2 * (N - 2)) if N > 2 else 0.5 * dij
        lj = dij - li
        li, lj = max(li, 0.0), max(lj, 0.0)
        u = next_id
        next_id += 1
        children[u] = [(i, li), (j, lj)]
        node_names[u] = None
        for k in active:
            if k in (i, j):
                continue
            mat[(u, k)] = mat[(k, u)] = 0.5 * (d(i, k) + d(j, k) - dij)
        active = [x for x in active if x not in (i, j)] + [u]
    # join last two
    i, j = active
    children[next_id] = [(i, d(i, j)/2), (j, d(i, j)/2)]
    root = next_id

    def newick(node):
        if node in children:
            parts = [f"{newick(c)}:{l:.4f}" for c, l in children[node]]
            return "(" + ",".join(parts) + ")"
        return node_names[node]
    return newick(root) + ";", children, node_names, root


def _bipartitions_from_children(children, node_names, root, leaves):
    """Return set of frozenset leaf-splits (non-trivial) for an unrooted tree."""
    leafset = set(leaves)
    memo = {}

    def descend(node):
        if node in memo:
            return memo[node]
        if node not in children:
            memo[node] = {node_names[node]}
        else:
            s = set()
            for c, _l in children[node]:
                s |= descend(c)
            memo[node] = s
        return memo[node]
    descend(root)
    biparts = set()
    for node, s in memo.items():
        side = frozenset(s & leafset)
        if 1 < len(side) < len(leafset) - 1:
            biparts.add(min(side, key=lambda _: 0) and frozenset(side))
    return biparts


def robinson_foulds(childrenA, namesA, rootA, childrenB, namesB, rootB, leaves):
    A = _bipartitions_from_children(childrenA, namesA, rootA, leaves)
    B = _bipartitions_from_children(childrenB, namesB, rootB, leaves)
    # canonicalise each split by the smaller side

    def canon(bp):
        other = frozenset(leaves) - bp
        return min(bp, other, key=lambda s: (len(s), sorted(s)))
    A = {canon(x) for x in A}
    B = {canon(x) for x in B}
    return len(A ^ B), A, B


def design_tree_children(stemma=None):
    """Build children/names/root for the ground-truth stemma over the 8 witness
    leaves. 
    """
    stemma = stemma or DESIGN_STEMMA
    names, children = {}, {}

    def build(x):
        if x in stemma:                       # internal node
            star = (x, "*")
            children[star] = []
            names[star] = None
            if x in WITNESSES:                # observed internal -> pendant leaf
                names[x] = x
                children[star].append((x, 0.0))
            for c in stemma[x]:
                children[star].append((build(c), 1.0))
            return star
        names[x] = x                          # terminal leaf
        return x
    root = build("O")
    return children, names, root
