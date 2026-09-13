#!/usr/bin/env python3
"""Extract per-language word lists from the Bagó six-language dictionary.

Sene Mongaba, Ngandu, Miteo, Manguanda & Sene Mortoni (2021), *Bagó -
Kimbangula - Kamusi - Nkongamyaku*, a French / English / Lingala / Kikongo /
Kiswahili / Tshiluba dictionary. archive.org/details/bago-sene-mongaba

The audit needs a *bag of word forms per language*, not aligned entries, which
is a far easier target than full row alignment: the OCR wraps entries across
lines unpredictably, but column x-positions are stable. We slice by the
character offsets of the column headers, which appear on 302 of 408 pages.

Usage:
    python scripts/extract_bago.py bago_text.pdf --out data/lexicons

Licence note: Bagó is a 2021 copyrighted work with no licence stated on
archive.org. Word lists derived from it are fine as local audit evidence;
redistributing them needs the authors' permission. See RESOURCES.md.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

LANGS = ["Français", "English", "Lingala", "Kikongo", "Kiswahili", "Tshiluba"]
# OCR mangles the cedilla and drops accents; accept the common variants.
HEADER_ALIASES = {
    "Français": ["Français", "Francais", "Frangais", "Frangais"],
    "English": ["English"],
    "Lingala": ["Lingala"],
    "Kikongo": ["Kikongo"],
    "Kiswahili": ["Kiswahili"],
    "Tshiluba": ["Tshiluba"],
}
MIN_FORM_LEN = 3


def pdf_to_layout_text(pdf: Path) -> str:
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            capture_output=True, check=True,
        )
    except FileNotFoundError:
        sys.exit("pdftotext not found -- install poppler (brew install poppler)")
    return out.stdout.decode("utf-8", errors="replace")


def find_header(line: str) -> dict[str, int] | None:
    """Return {language: char offset} if this line is a column header."""
    found: dict[str, int] = {}
    for lang, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            idx = line.find(alias)
            if idx >= 0:
                found[lang] = idx
                break
    # Need the two anchors that bracket Kikongo, or slicing is guesswork.
    if "Lingala" in found and "Kikongo" in found and len(found) >= 4:
        return found
    return None


def column_ranges(header: dict[str, int], width: int) -> dict[str, tuple[int, int]]:
    ordered = sorted(header.items(), key=lambda kv: kv[1])
    ranges: dict[str, tuple[int, int]] = {}
    for i, (lang, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else width
        ranges[lang] = (start, end)
    return ranges


def extract(text: str) -> dict[str, Counter]:
    """Walk the pages, slicing each line by the most recent header's offsets."""
    forms: dict[str, Counter] = {lang: Counter() for lang in LANGS}
    ranges: dict[str, tuple[int, int]] | None = None
    pages = text.split("\f")
    pages_with_header = 0

    for page in pages:
        lines = page.splitlines()
        page_ranges = None
        for i, line in enumerate(lines):
            header = find_header(line)
            if header:
                page_ranges = column_ranges(header, max(len(x) for x in lines) + 40)
                pages_with_header += 1
                ranges = page_ranges
                body = lines[i + 1 :]
                break
        else:
            body = lines  # no header on this page; reuse the previous one

        if ranges is None:
            continue

        for line in body:
            if not line.strip():
                continue
            padded = line.ljust(max(e for _, e in ranges.values()))
            for lang, (start, end) in ranges.items():
                cell = padded[start:end].strip()
                if not cell:
                    continue
                for word in re.findall(r"[^\W\d_]+", cell.lower()):
                    if len(word) >= MIN_FORM_LEN:
                        forms[lang][word] += 1

    print(f"  pages: {len(pages)}, with header: {pages_with_header}")
    return forms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path, help="bago sene mongaba_text.pdf")
    ap.add_argument("--out", type=Path, default=Path("data/lexicons"))
    ap.add_argument(
        "--min-count", type=int, default=2,
        help="drop forms seen fewer than this many times (OCR noise)",
    )
    args = ap.parse_args()

    print(f"[bago] reading {args.pdf}")
    forms = extract(pdf_to_layout_text(args.pdf))

    args.out.mkdir(parents=True, exist_ok=True)
    for lang, counter in forms.items():
        kept = sorted(w for w, c in counter.items() if c >= args.min_count)
        if not kept:
            continue
        slug = {
            "Français": "bago_french", "English": "bago_english",
            "Lingala": "bago_lingala", "Kikongo": "bago_kikongo",
            "Kiswahili": "bago_kiswahili", "Tshiluba": "bago_tshiluba",
        }[lang]
        path = args.out / f"{slug}.txt"
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"  {lang:11s} {len(counter):6,} raw -> {len(kept):6,} kept  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
