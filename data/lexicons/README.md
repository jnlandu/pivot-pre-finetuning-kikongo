# Reference lexicons

Drop lexicon files here and `python -m pivotkk audit` picks them up
automatically. They are the evidence that decides whether FLORES-200 `kon_Latn`
is Kikongo or Kituba — the marker heuristics in `audit.py` only gesture at it.

## Formats

| Extension | Shape |
|---|---|
| `.jsonl` | records with a `trg` field (what `prepare.py` writes) |
| `.tsv` | two columns, `source<TAB>target`; the target column is used |
| `.txt` | one headword per line |

The filename stem becomes the lexicon's name in the audit report, so name them
by variety: `yombe_declercq.tsv`, `kikongo_laman.tsv`, `kituba.txt`.

## Why at least two

The useful number is not raw coverage — that is confounded by lexicon size.
`differential_coverage` restricts to the forms *unique* to each lexicon and
compares coverage on that subset. With one lexicon there is nothing to compare
against, and the audit says so rather than printing a meaningless number.

So aim for **one genuine Kikongo-cluster lexicon and one Kituba lexicon**. A
corpus that is really Kituba should cover Kituba-only forms far better than
Kikongo-only forms, and the margin tells you how confidently.

## Validation

Extracted from Bagó via `scripts/extract_bago.py`, then checked against FLORES
devtest corpora of known identity:

| corpus | kikongo | kiswahili | lingala | tshiluba | best |
|---|---:|---:|---:|---:|---|
| FLORES `kon_Latn` | **0.328** | 0.085 | 0.137 | 0.116 | kikongo |
| FLORES `lin_Latn` | 0.088 | 0.044 | **0.191** | 0.070 | lingala |
| FLORES `swh_Latn` | 0.073 | **0.399** | 0.069 | 0.069 | kiswahili |
| FLORES `lua_Latn` | 0.116 | 0.079 | 0.090 | **0.296** | tshiluba |
| SMOL `en_kg` | **0.391** | 0.129 | 0.191 | 0.157 | kikongo |

Every control lands on its own label. Always run the controls: a broken lexicon
or a bad slice shows up immediately as a control picking the wrong language.

Two caveats. **GATITOS is not independent of SMOL** — same Google project, likely
the same translators — so SMOL scoring well against it is close to circular; use
Bagó or Laman for anything load-bearing. And **Bagó's Kikongo column is a
labelled claim by its authors**, so corroborate with a second, independently
compiled dictionary (Laman 1936) before publishing.

## Where to get them## Where to get them

See [`../../RESOURCES.md`](../../RESOURCES.md) for sources, licences, and what
each one does and does not permit. Short version: Laman (1936) is free, verified,
and public domain; the modern Kiyombe lexicon is in copyright and needs either a
purchase or the author's permission before anything derived from it is
published.

**Nothing in this directory is committed.** `data/` is gitignored, partly for
size and partly because some of these sources cannot be redistributed.
