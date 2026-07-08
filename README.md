# Luke 24 Modern-Scribe Corpus (Helsinki pilot)

The first **Koine Greek** corpus of *modern* hand-copied manuscripts, produced to
supply **ground-truth data** for computational stemmatology. Eight students and
staff of the Theology Department at the University of Helsinki iteratively copied
an extract of the Gospel of Luke taken from the Bezae (**24:1–28**, the road to Emmaus).
The archetype is not included in our dataset.
As the true genealogy of the copies is known *by construction*, our dataset can be used
to score automatic stemma-reconstruction methods.

This dataset was produced as part of the **Experimenting the Rise of Early Christianity** project
(University of Helsinki · Umeå University · Sorbonne Université).

## The known ("gold") stemma

Witnesses are named according to `<parent><generation-rank>`. 

```
~~~
                O   (exemplar; lost)
               / \
             11   12
            / \    / \
          21  22 23   24
          |   |
         31   32
~~~
```

## Repository layout

```
data/raw/           original hand-checked transcriptions
data/diplomatic/    cleaned, tokenised transcriptions (one file per witness)
derived/
  corpus.json             all information (metadata + tokens + stemma)
  alignment_core.csv/.tsv collation table
  apparatus_core.tei.xml  TEI P5 critical apparatus
  variants.csv            every non-base reading, with typological classification
  distances_core.csv      pairwise normalised variant distances
  stemma_design.nwk       the known stemma in Newick format
code/                     analysis pipeline (numpy + matplotlib only)
figures/                  figures displayed at EABS (PDF)
```

## Formatting choices

Each witness is de-hyphenated, stripped of editorial markers (`?`, `~`, `·`,
marginal numbers), lower-cased and accent-folded. Witnesses are collated against
a base witness (`11`, the fullest clean copy) by Needleman–Wunsch alignment on
token sequences, with token equality defined at a *normalized* level that folds
the Koine vowel-interchange sets (itacism), movable-nu, gemination and nomina
sacra. Because every scribe copied a contiguous **prefix** and stopped when time
ran out, trailing absence is recorded as *lacuna* (not-copied), not omission; the
**comparable core** is the prefix present in all eight witnesses (Luke 24:1–10a,
130 tokens). Each non-base reading is classified into three tiers —
**orthographic/phonological**, **mechanical/palaeographic**, and
**structural/textual** — and pairwise distances feed a neighbour-joining tree
compared to the design stemma with the Robinson–Foulds metric.

## Licence

Data (`data/`, `derived/`): **CC BY 4.0**. Code (`code/`): **MIT**.

If you want to use this dataset in your scientific work, we suggest you to use the
following attribution, as well as citing this GitHub:

"Luke 24 Modern-Scribe Corpus, Sophie Robert-Hayek and Nina Nikki, 2026"