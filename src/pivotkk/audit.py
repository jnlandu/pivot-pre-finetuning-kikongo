"""Is `kon_Latn` actually Kikongo?

FLORES+ (the maintained successor to FLORES-200) dropped kon_Latn and kept only
ktu_Latn, Kituba -- a creolised trade language descended from Kikongo but not
mutually identical with it. Every Kikongo MT number in the literature is
computed on a test set whose language label the benchmark's own maintainers
subsequently withdrew.

This module gathers the evidence rather than assuming the answer:
  1. overlap   -- how much do the differently-labelled corpora literally share?
  2. markers   -- frequency of diagnostic Kituba vs Kikongo grammatical forms
  3. lexicons  -- which reference lexicon does each corpus match best?
  4. domain    -- how much of each corpus is religious boilerplate?

Step 3 is the one that can settle the question, and it needs reference lexicons
for genuine Kikongo-cluster varieties. See `data/lexicons/README.md` and
`RESOURCES.md` for where those come from and what their licences allow.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from . import sources as S
from .prepare import PROC, clean_pairs, load_pivot_frame

DATA = Path(__file__).resolve().parents[2] / "data"
LEXICON_DIR = DATA / "lexicons"

# Very short forms match everything and carry no dialect signal.
MIN_FORM_LEN = 3

# Diagnostic forms. Kituba lost most of the Kikongo noun-class agreement and
# uses invariant preverbal TAM particles; Kikongo proper keeps richer morphology.
# These are indicative, not decisive -- read the flagged sentences yourself, and
# prefer the lexicon evidence below once you have real reference lexicons.
MARKERS = {
    "kituba_tam": [r"\bke\b", r"\bta\b", r"\bme\b", r"\bvandaka\b", r"\bsalaka\b"],
    "kituba_lex": [r"\bmunu\b", r"\bnge\b", r"\bbeto\b", r"\bsambu\b", r"\bmutindu\b"],
    "kikongo_lex": [r"\bmono\b", r"\bngeye\b", r"\byeto\b", r"\bkadi\b", r"\bmpila\b"],
    "religious": [
        r"\bYezu\b", r"\bYehowa\b", r"\bNzambi\b", r"\bBiblia\b",
        r"\bKristu\b", r"\bmpeve\b",
    ],
}


def marker_rates(texts: list[str]) -> dict[str, float]:
    """Share of sentences containing at least one marker from each group."""
    n = max(len(texts), 1)
    rates = {}
    for group, patterns in MARKERS.items():
        rx = re.compile("|".join(patterns), re.IGNORECASE)
        rates[group] = round(sum(bool(rx.search(t)) for t in texts) / n, 4)
    return rates


def overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    """Sentence-level overlap between two corpora, on both sides."""
    out = {}
    for side in ("src", "trg"):
        sa, sb = set(a[side].str.strip()), set(b[side].str.strip())
        union = len(sa | sb) or 1
        out[f"{side}_jaccard"] = round(len(sa & sb) / union, 4)
        out[f"{side}_shared"] = len(sa & sb)
    return out


# --------------------------------------------------------------------------- #
# reference lexicons
# --------------------------------------------------------------------------- #
def load_lexicon(path: Path) -> set[str]:
    """Read a reference lexicon into a set of target-language word forms.

    Three formats, chosen by extension:
      .jsonl  records with a `trg` field (the shape prepare.py writes)
      .tsv    two columns, source<TAB>target; target is taken
      .txt    one headword per line
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
        col = "trg" if "trg" in df.columns else df.columns[-1]
        raw = df[col].astype(str).tolist()
    elif path.suffix in (".tsv", ".csv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, header=None, dtype=str).fillna("")
        raw = df[df.columns[-1]].tolist()
    elif path.suffix == ".txt":
        raw = path.read_text(encoding="utf-8").splitlines()
    else:
        raise ValueError(f"unsupported lexicon format: {path.suffix}")

    return {
        w
        for entry in raw
        for w in re.findall(r"\w+", str(entry).lower())
        if len(w) >= MIN_FORM_LEN
    }


def discover_lexicons(directory: Path | None = None) -> dict[str, set[str]]:
    """Load every reference lexicon on disk, plus GATITOS if prepare.py has run."""
    directory = Path(directory) if directory else LEXICON_DIR
    lexicons: dict[str, set[str]] = {}

    gatitos = PROC / "target" / "smol_lexicon.jsonl"
    if gatitos.exists():
        lexicons["gatitos_kg"] = load_lexicon(gatitos)

    if directory.exists():
        for path in sorted(directory.iterdir()):
            if path.suffix in (".jsonl", ".tsv", ".csv", ".txt") and path.stem != "README":
                try:
                    lexicons[path.stem] = load_lexicon(path)
                except Exception as exc:
                    print(f"  ! could not read {path.name}: {exc}")
    return lexicons


def corpus_vocab(texts: list[str]) -> set[str]:
    return {
        w
        for t in texts
        for w in re.findall(r"\w+", str(t).lower())
        if len(w) >= MIN_FORM_LEN
    }


def lexicon_coverage(texts: list[str], lexicon) -> float:
    """Fraction of a lexicon's forms attested in a corpus.

    Accepts either a DataFrame with a `trg` column or a pre-loaded set of forms,
    so older callers keep working.
    """
    if isinstance(lexicon, pd.DataFrame):
        forms = {
            w
            for t in lexicon["trg"]
            for w in re.findall(r"\w+", str(t).lower())
            if len(w) >= MIN_FORM_LEN
        }
    else:
        forms = set(lexicon)
    if not forms:
        return 0.0
    vocab = corpus_vocab(texts)
    return round(len(forms & vocab) / len(forms), 4)


def differential_coverage(
    texts: list[str], lexicons: dict[str, set[str]]
) -> dict[str, object]:
    """Which reference lexicon does this corpus match best?

    Raw coverage is confounded by lexicon size, so the discriminative figure is
    what matters: for each pair of lexicons, restrict to the forms *unique* to
    one of them and compare coverage on that subset only. A corpus that is
    really variety A should cover A-only forms far better than B-only forms.
    """
    vocab = corpus_vocab(texts)
    out: dict[str, object] = {
        "raw": {
            name: {
                "n_forms": len(forms),
                "coverage": round(len(forms & vocab) / len(forms), 4) if forms else 0.0,
            }
            for name, forms in lexicons.items()
        }
    }

    discriminative: dict[str, dict] = {}
    names = sorted(lexicons)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            only_a = lexicons[a] - lexicons[b]
            only_b = lexicons[b] - lexicons[a]
            if len(only_a) < 20 or len(only_b) < 20:
                continue  # too few distinguishing forms to say anything
            cov_a = len(only_a & vocab) / len(only_a)
            cov_b = len(only_b & vocab) / len(only_b)
            discriminative[f"{a}_vs_{b}"] = {
                f"{a}_only_forms": len(only_a),
                f"{b}_only_forms": len(only_b),
                f"{a}_only_coverage": round(cov_a, 4),
                f"{b}_only_coverage": round(cov_b, 4),
                "leans": a if cov_a > cov_b else b,
                "margin": round(abs(cov_a - cov_b), 4),
            }
    out["discriminative"] = discriminative
    return out


def bootstrap_identity(
    texts: list[str],
    lexicons: dict[str, set[str]],
    n_boot: int = 1000,
    seed: int = 13,
) -> dict[str, object]:
    """How stable is the winning language, if the dictionary had been different?

    Coverage is computed against one compiler's word list. Resampling the
    lexicon forms with replacement asks the question a reviewer will: would a
    differently-sampled dictionary of the same size have picked the same
    language? Reports the win rate for the argmax and a 95% interval on its
    margin over the runner-up.
    """
    import random

    rng = random.Random(seed)
    vocab = corpus_vocab(texts)
    pools = {name: sorted(forms) for name, forms in lexicons.items() if forms}
    if len(pools) < 2:
        return {}

    wins: Counter = Counter()
    margins: list[float] = []
    for _ in range(n_boot):
        cov = {}
        for name, pool in pools.items():
            draw = [pool[rng.randrange(len(pool))] for _ in range(len(pool))]
            cov[name] = sum(f in vocab for f in draw) / len(draw)
        ordered = sorted(cov.items(), key=lambda kv: kv[1], reverse=True)
        wins[ordered[0][0]] += 1
        margins.append(ordered[0][1] - ordered[1][1])

    margins.sort()
    winner, n_wins = wins.most_common(1)[0]
    return {
        "winner": winner,
        "win_rate": round(n_wins / n_boot, 4),
        "margin_mean": round(sum(margins) / len(margins), 4),
        "margin_ci95": [
            round(margins[int(0.025 * len(margins))], 4),
            round(margins[int(0.975 * len(margins))], 4),
        ],
        "n_boot": n_boot,
    }


# --------------------------------------------------------------------------- #
def main(
    sample: int = 50_000,
    corpora: list[str] | None = None,
    lexicon_dir: Path | None = None,
) -> dict:
    corpora = corpora or ["kon_mined", "kon_mt560", "kik_mt560", "lin"]

    lexicons = discover_lexicons(lexicon_dir)
    if lexicons:
        print("[audit] reference lexicons:")
        for name, forms in lexicons.items():
            print(f"  {name:24s} {len(forms):6,} forms")
    if len(lexicons) < 2:
        print(
            "[audit] NOTE: fewer than two reference lexicons, so the "
            "discriminative comparison cannot run. See RESOURCES.md for where "
            "to get Kikongo-cluster lexicons."
        )

    frames: dict[str, pd.DataFrame] = {}
    report: dict[str, dict] = {}

    for key in corpora:
        df = clean_pairs(load_pivot_frame(key)).head(sample)
        frames[key] = df
        texts = df["trg"].tolist()
        report[key] = {
            "hf_id": S.pivot(key).hf_id,
            "n_sampled": len(df),
            "markers": marker_rates(texts),
            "lexicons": differential_coverage(texts, lexicons),
        }
        print(f"[audit] {key}: markers={report[key]['markers']}")
        for pair, verdict in report[key]["lexicons"]["discriminative"].items():
            print(f"         {pair}: leans {verdict['leans']} (margin {verdict['margin']})")

    # SMOL is the professionally-translated reference point for what the `kg`
    # label is supposed to mean.
    smol_path = PROC / "target" / "smol_train.jsonl"
    if smol_path.exists():
        smol = pd.read_json(smol_path, lines=True)
        texts = smol["trg"].tolist()
        report["smol_kg"] = {
            "hf_id": "google/smol",
            "n_sampled": len(smol),
            "markers": marker_rates(texts),
            "lexicons": differential_coverage(texts, lexicons),
        }
        print(f"[audit] smol_kg: markers={report['smol_kg']['markers']}")

    # FLORES-200 kon_Latn is the eval set whose label is actually in dispute.
    flores = PROC / "eval" / "flores200_devtest.jsonl"
    if flores.exists():
        df = pd.read_json(flores, lines=True)
        texts = df["trg"].tolist()
        report["flores200_kon"] = {
            "hf_id": "facebook/flores (kon_Latn devtest)",
            "n_sampled": len(df),
            "markers": marker_rates(texts),
            "lexicons": differential_coverage(texts, lexicons),
        }
        print(f"[audit] flores200_kon: markers={report['flores200_kon']['markers']}")

    report["overlap"] = {}
    keys = list(frames)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            report["overlap"][f"{a}|{b}"] = overlap(frames[a], frames[b])
            print(f"[audit] overlap {a}|{b}: {report['overlap'][f'{a}|{b}']}")

    out = PROC / "audit_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[audit] written -> {out}")
    return report
