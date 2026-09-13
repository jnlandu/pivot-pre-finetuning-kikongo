# Pivot Pre-finetuning for Low-Resource MT: English → Kikongo

## Current experiment: reviewed SMOL Lingala

Start with [configs/smol/README.md](configs/smol/README.md). Kikongo is now split
by whole documents and duplicate-linked groups: 2,011 train / 153 dev / 302 test.
The first pivot experiment uses 2,916 English–Lingala pairs from the dataset
fully reviewed by the project user, after refreshing held-out exclusions.
The older mined-data experiment description below is retained as background;
its data counts and training instructions are superseded by the SMOL runbook.
Full GPU runs have not yet been performed.

A replication and extension of [Pivot Pre-finetuning for Low Resource MT: A Case
Study in Kikamba](https://openreview.net/pdf?id=PaHmtktx86H) (ICLR Tiny Papers),
moved from Kenya to the Congo Basin.

## The original result

Take a pretrained multilingual seq2seq model that has never seen the target
language. Before finetuning it on your handful of target-language pairs,
**pre-finetune it on a related, better-resourced language**. The Kikamba study
reports ~40% chrF over a non-finetuned baseline, gains appearing at 50k pivot
pairs and plateauing by 100k, and — the interesting claim — *linguistic
relatedness beating pivot data quality*: noisy Kiswahili and Kinyarwanda pivots
outperformed cleaner French.

## Why Kikongo makes this a sharper test

The French control is weak in the Kenyan setting: French is simply an unrelated
high-resource language there. In the DRC it is the **official language**, the
language of schooling, and the source of heavy lexical borrowing. If relatedness
still beats French here, the claim survives a much harder control. If it does
not, that is equally worth knowing.

Kikongo also sits in a dense neighbourhood, which lets us vary relatedness and
resource level nearly independently:

| Pivot | en– pairs | Relation to Kikongo (H10) |
|---|---:|---|
| Lingala | 2,910,515 | Bantu C30, DRC lingua franca |
| Luba-Kasai | 292,212 | Bantu L30 |
| Congo-Swahili | 271,892 | Bantu G40, DRC contact variety |
| Swahili | 23,513,175 | Bantu G40, mid-resource control |
| Kimbundu / Umbundu | 196k / 298k | Angola, H20 / R10 |
| Sango | 3,607,548 | **Non-Bantu** creole, regionally adjacent |
| French | ~1,000,000 | High-resource *and* the official language |

Sango is the sharpest instrument here: it is geographically and functionally
close but genetically unrelated, so it separates "related" from merely "nearby"
in a way the Kikamba study could not.

## The finding: FLORES+ dropped a correctly-labelled language

**FLORES+ — the maintained successor to FLORES-200 — dropped `kon_Latn` and
kept only `ktu_Latn` (Kituba).** But `kon_Latn` is Kikongo.

Tested against the Kikongo, Lingala, Kiswahili and Tshiluba columns of the Bagó
six-language dictionary (Sene Mongaba et al. 2021), via `make audit`:

| corpus | kikongo | kiswahili | lingala | tshiluba | best |
|---|---:|---:|---:|---:|---|
| FLORES `kon_Latn` devtest | **0.328** | 0.085 | 0.137 | 0.116 | kikongo |
| FLORES `lin_Latn` devtest | 0.088 | 0.044 | **0.191** | 0.070 | lingala |
| FLORES `swh_Latn` devtest | 0.073 | **0.399** | 0.069 | 0.069 | kiswahili |
| FLORES `lua_Latn` devtest | 0.116 | 0.079 | 0.090 | **0.296** | tshiluba |
| SMOL `en_kg` | **0.391** | 0.129 | 0.191 | 0.157 | kikongo |

Three independent controls land on their own labels, and `kon_Latn` matches
Kikongo at 2.4x its next-best. A Kikongo speaker on this project confirms it.

So the problem is not mislabelled data. It is that **the maintained benchmark
has removed a real language.** Anyone evaluating on FLORES+ today cannot
evaluate Kikongo at all — H10, several million speakers — and the only surviving
Kikongo eval set is the legacy FLORES-200 release.

### A methodological note worth keeping

An earlier version of this repo reached the opposite conclusion from a hand-built
list of "diagnostic" grammatical markers, which scored FLORES `kon_Latn` at 73%
Kituba TAM particles. That marker list was wrong: the forms it treated as
Kituba-specific are shared across the whole Kikongo cluster.

Hand-built morphological markers written without a speaker are not evidence.
Lexicon discrimination against a published dictionary, with controls in
languages of known identity, is. `audit.py` now does the latter and reports the
controls alongside every result, so a broken method shows up as a control
landing on the wrong language.

## Data

| Role | Source | Size |
|---|---|---|
| Target gold train | Google SMOL (SmolSent + SmolDoc), `en_kg` | 2,016 pairs |
| Target dev / test | held out from the same | 150 / 300 |
| Target lexicon | GATITOS `en_kg` | 5,231 entries |
| Eval | FLORES-200 `kon_Latn` dev / devtest | 997 / 1,012 |
| Pivots | NLLB-v1 & MT560 mined bitext (see table above) | 10k / 50k / 100k ladders |

Two details that matter:

- The mined dumps arrive **sorted by LASER margin score descending**. A naive
  `head(n)` silently confounds the size ladder with a quality ladder, so
  `sample_ladder` strides across the score range by default (`--sampling
  stratified`).
- Mined corpora repeat one English sentence against many target sentences.
  `clean_pairs` deduplicates on the source side as well as the pair.

## Setup

```bash
make venv
make data      # ~10 min, CPU + network. Do this on your laptop.
make audit     # answer the kon_Latn question before spending money
```

`make data` writes `data/processed/`, which is small. **rsync that to the GPU
box rather than rebuilding it there** — you are paying by the hour, and the
pivot downloads are multi-GB.

## Running experiments

```bash
make smoke                                     # ~2 min, verifies the loop
python -m pivotkk train configs/experiments/lin_50k.yaml
python -m pivotkk eval runs/lin_50k
make table                                     # collect every run
```

Stage A checkpoints are **content-addressed and cached** in
`runs/_stage_a_cache/`, keyed by everything stage A depends on and nothing else.
Re-running the sweep after a crash costs nothing for what already finished, and
stage-B hyperparameter changes reuse stage A for free. This is the single
biggest cost lever in the repo.

## Notes on the GPU

`scripts/provision_gpu.sh` bootstraps a fresh Ubuntu GPU VM (Scaleway L4/L40S, or
a GCP `g2`/`a2` instance) and verifies the GPU before you install anything else.

- **Never run mT5 in fp16** — it overflows and you get NaN losses. `bf16: true`
  is set in every config; keep it, and pick a GPU that supports bf16 (L4, L40S,
  A100, H100 all do; a T4 does not).
- Rough cost: mT5-base, 100k pivot pairs, 1 epoch, batch 16 ≈ 1–1.5 h on an L4.
  The full 21-config grid is not worth buying up front. `scripts/run_sweep.sh`
  runs a 10-config core sweep, ordered cheapest-first so a partial run still
  gives you a usable table.
- Start the sweep with the control (`00_no_pivot`). If it does not produce a
  near-zero score, something is wrong with the pipeline, not with the science.

## Layout

```
configs/experiments/   21 configs: control, 6 pivots x 3 sizes, 2 ablations
src/pivotkk/
  sources.py           verified dataset registry (checked live 2026-08-28)
  prepare.py           download, clean, dedup, build ladders
  audit.py             is kon_Latn actually Kikongo?
  train.py             stage A pivot pre-finetune -> stage B target finetune
  evaluate.py          chrF++ / BLEU / spBLEU on both test sets
scripts/
  provision_gpu.sh     fresh-VM bootstrap
  run_sweep.sh         resumable core sweep
```

## Reference resources

[`RESOURCES.md`](RESOURCES.md) records the Kikongo and Yombe dictionaries and
grammars that feed the language-identity audit -- what is free, what is in
copyright, and what each licence allows. Laman's 1,183-page Kikongo-French
dictionary (1936) is public domain and verified downloadable.

## Where this goes next

[`DIRECTIONS.md`](DIRECTIONS.md) covers how to strengthen this experiment, two
things measured and ruled out, and five alternative directions -- including the
argument for making the language-identity audit the primary paper.

## Open decisions

- **Base model.** `google/mt5-base` is the faithful choice: mT5 has seen neither
  Kikongo nor any of the pivots, so pivot pre-finetuning genuinely teaches a new
  language. NLLB-200 already claims `kon_Latn`, which makes it a useful topline
  but a contaminated base.
- **Kituba as a pivot.** Linguistically the nearest neighbour, and FLORES+ has a
  clean `ktu_Latn` eval set. But if `kon_Latn` is itself Kituba, the experiment
  is circular. Resolve the audit first.
- **Marker validation.** The `audit.py` marker list needs a speaker's review
  before any of this is publishable.
