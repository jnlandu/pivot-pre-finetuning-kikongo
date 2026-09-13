#!/usr/bin/env python3
"""OCR Laman's Dictionnaire kikongo-français (1936) into a Kikongo word list.

Laman is 1,276 pages of image scan with no text layer, free from KAOWARSOM and
public domain (Laman d. 1944). It is the independently-compiled second reference
the audit needs: Bago's Kikongo column is its own authors' labelled claim, and a
1936 dictionary by a different compiler corroborates it from outside.

Pipeline: rasterise in batches -> tesseract (French model; the script is Latin
and French is the metalanguage) -> subtract French vocabulary -> frequency
filter. The result is a bag of Kikongo forms, which is all `audit.py` consumes;
we do not attempt to parse entry structure.

Usage:
    python scripts/ocr_laman.py laman.pdf --out data/lexicons/laman_kikongo.txt
    python scripts/ocr_laman.py laman.pdf --pages 100-140     # trial run first
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

MIN_FORM_LEN = 3

# Laman's headwords are `headword, abbrev. definition ...`, one entry per
# paragraph, with continuation lines in pure French. Anchoring on the leading
# token before the first comma is far more precise than filtering French out of
# the whole page.
HEADWORD = re.compile(r"^\s{0,3}([^\W\d_][\w'-]{1,30}),")


def fold(form: str) -> str:
    """Laman orthography -> FLORES orthography.

    Strips the length/tone diacritics (kala) and the morpheme hyphens
    (ka-lumba) that Laman uses and modern written Kikongo does not.
    """
    form = form.replace("-", "").replace("'", "")
    decomposed = unicodedata.normalize("NFD", form)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
# Kikongo orthography in Laman: no c/j/q/r/x, and these letters in OCR output
# are near-always scanning artefacts or stray French.
IMPLAUSIBLE = re.compile(r"[cjqrxâêîôûàèùç]")


def require(tool: str) -> None:
    if not shutil.which(tool):
        sys.exit(f"{tool} not found. brew install tesseract poppler")


def french_vocabulary(repo: Path) -> set[str]:
    """Everything we can cheaply establish as French, to subtract from the OCR."""
    forms: set[str] = set()

    # A full French wordlist is the single biggest lever on precision: Laman's
    # definitions are dictionary-register French (emousse, festivite, bagatelle)
    # that a corpus-derived vocabulary never covers. Fetch once:
    #   curl -sL https://raw.githubusercontent.com/words/an-array-of-french-words/master/index.json \
    #     -o data/raw/french_words.json
    wordlist = repo / "data" / "raw" / "french_words.json"
    if wordlist.exists():
        import json

        forms |= {fold(w) for w in json.loads(wordlist.read_text())}
    else:
        print("  ! data/raw/french_words.json missing -- precision will suffer")

    bago_fr = repo / "data" / "lexicons" / "bago_french.txt"
    if bago_fr.exists():
        forms |= {fold(w) for w in bago_fr.read_text().split() if w.strip()}

    flores = repo / "data" / "raw" / "flores200_dataset.tar.gz"
    if flores.exists():
        import io
        import tarfile

        with tarfile.open(flores) as tar:
            for member in tar.getmembers():
                if member.name.endswith(("dev/fra_Latn.dev", "devtest/fra_Latn.devtest")):
                    fh = tar.extractfile(member)
                    if fh is None:
                        continue
                    text = io.TextIOWrapper(fh, encoding="utf-8").read()
                    forms |= {fold(w) for w in re.findall(r"[^\W\d_]+", text)}
    return forms


def ocr_pages(pdf: Path, first: int, last: int, dpi: int, jobs: int) -> str:
    """Rasterise and OCR one batch, cleaning up images as we go."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-gray", "-png",
             "-f", str(first), "-l", str(last), str(pdf), str(tmpdir / "p")],
            check=True, capture_output=True,
        )
        images = sorted(tmpdir.glob("p*.png"))
        if not images:
            return ""
        # tesseract one process per image, `jobs` at a time.
        cmd = (
            f"printf '%s\\n' {' '.join(f'{i}' for i in map(str, images))} | "
            f"xargs -P {jobs} -I{{}} sh -c 'tesseract \"$1\" \"$1\" -l fra --psm 3 "
            f"quiet 2>/dev/null' _ {{}}"
        )
        subprocess.run(cmd, shell=True, capture_output=True)
        return "\n".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in sorted(tmpdir.glob("p*.txt"))
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/lexicons/laman_kikongo.txt"))
    ap.add_argument("--pages", default=None, help="e.g. 100-140; default all")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=1,
                    help="headwords are mostly unique; 1 is right here")
    ap.add_argument("--def-ratio", type=float, default=2.0,
                    help="drop forms seen in definition position more than "
                         "this multiple of their headword count")
    ap.add_argument("--keep-implausible", action="store_true",
                    help="skip the c/j/q/r/x orthography filter")
    ap.add_argument("--raw-text", type=Path, default=None,
                    help="also dump the raw OCR text here")
    args = ap.parse_args()

    require("pdftoppm")
    require("tesseract")

    n_pages = int(subprocess.run(
        ["pdfinfo", str(args.pdf)], capture_output=True, text=True
    ).stdout.split("Pages:")[1].split()[0])
    if args.pages:
        first, last = (int(x) for x in args.pages.split("-"))
    else:
        first, last = 1, n_pages
    print(f"[laman] {args.pdf.name}: {n_pages} pages, OCR-ing {first}-{last}")

    repo = Path(__file__).resolve().parents[1]
    french = french_vocabulary(repo)
    print(f"[laman] French vocabulary to subtract: {len(french):,} forms")

    # Two counters, because the book supervises itself. A form appearing at
    # line-start-before-comma is a headword candidate; everything after the
    # first comma is definition text, which is French. French words leak into
    # headword position when a wrapped definition line happens to start with
    # `word,` -- but those same words appear in definition position far more
    # often. The ratio separates them without needing an external wordlist.
    counter: Counter = Counter()
    definition: Counter = Counter()
    chunks: list[str] = []
    for start in range(first, last + 1, args.batch):
        stop = min(start + args.batch - 1, last)
        text = ocr_pages(args.pdf, start, stop, args.dpi, args.jobs)
        chunks.append(text)
        for line in text.splitlines():
            m = HEADWORD.match(line)
            if m:
                word = fold(m.group(1))
                if len(word) >= MIN_FORM_LEN and word.isalpha():
                    counter[word] += 1
            head, _, tail = line.partition(",")
            for word in re.findall(r"[^\W\d_]+", tail):
                folded = fold(word)
                if len(folded) >= MIN_FORM_LEN:
                    definition[folded] += 1
        done = stop - first + 1
        print(f"  pages {start}-{stop}  ({done}/{last - first + 1})  "
              f"{len(counter):,} distinct forms", flush=True)

    if args.raw_text:
        args.raw_text.write_text("\n".join(chunks), encoding="utf-8")

    kept = {w for w, c in counter.items() if c >= args.min_count}
    print(f"[laman] {len(counter):,} raw -> {len(kept):,} above min-count")
    kept = {
        w for w in kept
        if definition.get(w, 0) <= args.def_ratio * counter[w]
    }
    print(f"[laman] {len(kept):,} after definition-position ratio filter")
    kept -= french
    print(f"[laman] {len(kept):,} after subtracting French")
    if not args.keep_implausible:
        kept = {w for w in kept if not IMPLAUSIBLE.search(w)}
        print(f"[laman] {len(kept):,} after orthography filter")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sorted(kept)) + "\n", encoding="utf-8")
    print(f"[laman] written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
