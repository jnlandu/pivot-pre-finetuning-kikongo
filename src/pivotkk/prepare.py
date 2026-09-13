"""Build every dataset the experiments need, from raw download to train-ready jsonl.

Run this on your laptop before renting a GPU. Everything here is CPU + network
bound, and the processed output is small enough to rsync to the VM.
"""
from __future__ import annotations

import io
import json
import random
import tarfile
import urllib.request
import unicodedata
from pathlib import Path

import pandas as pd

from . import sources as S

DATA = Path(__file__).resolve().parents[2] / "data"
RAW, PROC = DATA / "raw", DATA / "processed"


# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #
def clean_pairs(
    df: pd.DataFrame,
    *,
    min_chars: int = 5,
    max_chars: int = 500,
    max_ratio: float = 3.0,
) -> pd.DataFrame:
    """Drop empties, near-duplicates, and pairs with implausible length ratios."""
    df = df.dropna(subset=["src", "trg"]).copy()
    df["src"] = df["src"].astype(str).str.strip()
    df["trg"] = df["trg"].astype(str).str.strip()

    ok = (
        df["src"].str.len().between(min_chars, max_chars)
        & df["trg"].str.len().between(min_chars, max_chars)
    )
    df = df[ok]

    # Length-ratio filter: a 5x length gap is nearly always a misalignment.
    ratio = df["src"].str.len() / df["trg"].str.len().clip(lower=1)
    df = df[(ratio <= max_ratio) & (ratio >= 1 / max_ratio)]

    # Deduplicate on the source side too: mined corpora repeat one English
    # sentence against many target sentences, which teaches the model noise.
    df = df.drop_duplicates(subset=["src", "trg"])
    df = df.drop_duplicates(subset=["src"])
    return df.reset_index(drop=True)


def sample_ladder(
    df: pd.DataFrame, n: int, *, how: str = "stratified", seed: int = 13
) -> pd.DataFrame:
    """Take n rows from a similarity-sorted mined corpus.

    The michsethowusu dumps arrive sorted by LASER margin score descending, so a
    naive head(n) buys you the shortest, most formulaic sentences -- mostly
    Watchtower boilerplate -- and silently confounds the data-size ladder with a
    quality ladder. 'stratified' spreads the sample across the score range so
    10k/50k/100k differ in quantity only.
    """
    if n >= len(df):
        return df.copy()
    if how == "top":
        return df.head(n).copy()
    if how == "random":
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    if how == "stratified":
        step = len(df) / n
        idx = [int(i * step) for i in range(n)]
        return df.iloc[idx].reset_index(drop=True)
    raise ValueError(f"unknown sampling strategy {how!r}")


# --------------------------------------------------------------------------- #
# target gold data (Google SMOL)
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def split_target_pairs(rows: list[dict], seed: int = 13) -> dict[str, pd.DataFrame]:
    """Keep documents and duplicate-linked documents in one partition.

    Link before cleaning: even a duplicate later removed must not connect
    training and held-out documents. Match normalized source and target text.
    """
    parents = {r["group_id"]: r["group_id"] for r in rows}

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    seen = {}
    for r in rows:
        for side in ("src", "trg"):
            key = (side, normalize_text(r[side]))
            if not key[1]:
                continue
            if key in seen:
                parents[root(r["group_id"])] = root(seen[key])
            seen[key] = r["group_id"]
    frame = clean_pairs(pd.DataFrame(rows))
    frame["group_id"] = frame.group_id.map(root)
    frame = frame.loc[~frame.src.map(normalize_text).duplicated()].copy()
    groups = list(frame.groupby("group_id", sort=True))
    rng = random.Random(seed)
    rng.shuffle(groups)
    # Preserve approximately the document/standalone mixture in each split.
    buckets = {}
    for _, group in groups:
        kind = "smoldoc" if "smoldoc" in set(group.origin) else "smolsent"
        buckets.setdefault(kind, []).append(group)
    parts = {k: [] for k in ("train", "dev", "test")}
    n_test = min(300, len(frame) // 4)
    for bucket in buckets.values():
        total = sum(len(g) for g in bucket)
        targets = {"test": round(total * n_test / len(frame)),
                   "dev": round(total * (n_test // 2) / len(frame))}
        counts = {"test": 0, "dev": 0}
        for group in bucket:
            dest = "train"
            for name in ("test", "dev"):
                if abs(counts[name] + len(group) - targets[name]) < abs(counts[name] - targets[name]):
                    dest = name
                    counts[name] += len(group)
                    break
            parts[dest].append(group)
    return {k: pd.concat(v).sample(frac=1, random_state=seed).reset_index(drop=True)
            if v else frame.iloc[:0].copy() for k, v in parts.items()}


def _fetch_jsonl(url: str, dest: Path) -> list[dict]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dest)
    with dest.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_target_gold(seed: int = 13) -> dict[str, pd.DataFrame]:
    """SmolSent + SmolDoc -> sentence pairs; GATITOS kept separate as a lexicon.

    SmolDoc stores parallel *lists* (srcs/trgs) per document. They are aligned
    index-wise; documents where the lists disagree in length are dropped rather
    than guessed at.
    """
    print("[target] Google SMOL en->kg")
    rows: list[dict] = []

    sent = _fetch_jsonl(S.SMOL_FILES["smolsent"][0], RAW / "smol" / "smolsent_en_kg.jsonl")
    rows += [{"src": r["src"], "trg": r["trg"], "origin": "smolsent",
              "sentence_id": r.get("id", i), "document_id": None,
              "group_id": f"sent:{r.get('id', i)}"} for i, r in enumerate(sent)]

    doc = _fetch_jsonl(S.SMOL_FILES["smoldoc"][0], RAW / "smol" / "smoldoc_en_kg.jsonl")
    dropped = 0
    for r in doc:
        srcs, trgs = r.get("srcs", []), r.get("trgs", [])
        if len(srcs) != len(trgs):
            dropped += 1
            continue
        rows += [{"src": s, "trg": t, "origin": "smoldoc", "document_id": r["id"],
                  "sentence_index": i, "group_id": f"doc:{r['id']}"}
                 for i, (s, t) in enumerate(zip(srcs, trgs))]
    if dropped:
        print(f"  dropped {dropped} smoldoc docs with mismatched src/trg lengths")

    splits = split_target_pairs(rows, seed)
    print(f"  {sum(len(v) for v in splits.values())} clean sentence pairs")

    lex_raw = _fetch_jsonl(S.SMOL_FILES["gatitos"][0], RAW / "smol" / "gatitos_en_kg.jsonl")
    lex = pd.DataFrame(
        [
            {"src": r["src"], "trg": t, "origin": "gatitos"}
            for r in lex_raw
            for t in r.get("trgs", [])
        ]
    ).drop_duplicates()
    print(f"  {len(lex)} GATITOS lexicon entries")

    # Held-out SMOL test set. This is the second evaluation axis: it is
    # professionally translated and provenance-independent from FLORES-200, so
    # agreement (or not) between the two is itself a result.
    splits["lexicon"] = lex
    for name, frame in splits.items():
        _write(frame, PROC / "target" / f"smol_{name}.jsonl")
    return splits


# --------------------------------------------------------------------------- #
# evaluation (FLORES-200 + FLORES+)
# --------------------------------------------------------------------------- #
def build_flores200() -> dict[str, pd.DataFrame]:
    """Pull the legacy FLORES-200 tarball -- the only source still carrying kon_Latn."""
    print("[eval] FLORES-200")
    tgz = RAW / "flores200_dataset.tar.gz"
    tgz.parent.mkdir(parents=True, exist_ok=True)
    if not tgz.exists():
        print(f"  downloading {S.FLORES200_TARBALL} (~25 MB)")
        urllib.request.urlretrieve(S.FLORES200_TARBALL, tgz)

    wanted = {
        f"{split}/{lang}.{split}": (split, lang)
        for split in ("dev", "devtest")
        for lang in S.FLORES200_LANGS
    }
    got: dict[tuple[str, str], list[str]] = {}
    with tarfile.open(tgz) as tar:
        for member in tar.getmembers():
            for suffix, key in wanted.items():
                if member.name.endswith(suffix):
                    fh = tar.extractfile(member)
                    if fh is None:
                        continue
                    text = io.TextIOWrapper(fh, encoding="utf-8").read()
                    got[key] = text.splitlines()

    out: dict[str, pd.DataFrame] = {}
    for split in ("dev", "devtest"):
        eng = got.get((split, "eng_Latn"))
        kon = got.get((split, "kon_Latn"))
        if not eng or not kon:
            print(f"  WARNING: missing {split} for eng/kon -- skipping")
            continue
        frame = pd.DataFrame({"src": eng, "trg": kon})
        out[split] = frame
        _write(frame, PROC / "eval" / f"flores200_{split}.jsonl")
        print(f"  {split}: {len(frame)} pairs")
    return out


# --------------------------------------------------------------------------- #
# pivot ladders
# --------------------------------------------------------------------------- #
def load_pivot_frame(key: str) -> pd.DataFrame:
    """Load a pivot corpus into a uniform src/trg/similarity frame."""
    from datasets import load_dataset

    src = S.pivot(key)
    print(f"[pivot] {key}: {src.hf_id} ({src.rows:,} rows upstream)")

    if key in S.OPUS100_PIVOTS:
        ds = load_dataset(src.hf_id, S.OPUS100_PIVOTS[key], split="train")
        df = pd.DataFrame(
            {
                "src": [r["en"] for r in ds["translation"]],
                "trg": [r["fr"] for r in ds["translation"]],
            }
        )
        df["similarity"] = float("nan")
        return df

    ds = load_dataset(src.hf_id, split="train")
    cols = ds.column_names
    # Schema is similarity | English | <TargetLanguageName>; the third column's
    # name differs per corpus, so detect it instead of hardcoding.
    trg_col = next(c for c in cols if c not in ("similarity", "English"))
    df = ds.to_pandas().rename(columns={"English": "src", trg_col: "trg"})
    return df[["src", "trg"] + (["similarity"] if "similarity" in cols else [])]


def build_pivot(key: str, sizes: list[int], *, how: str = "stratified") -> None:
    df = clean_pairs(load_pivot_frame(key))
    print(f"  {len(df):,} pairs after cleaning")
    for n in sizes:
        subset = sample_ladder(df, n, how=how)
        path = PROC / "pivot" / f"{key}_{n}.jsonl"
        _write(subset[["src", "trg"]], path)
        print(f"  -> {path.name}: {len(subset):,}")


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def main(pivots: list[str], sizes: list[int], how: str = "stratified") -> None:
    build_target_gold()
    build_flores200()
    for key in pivots:
        build_pivot(key, sizes, how=how)
    print(f"\nprocessed data in {PROC}")
