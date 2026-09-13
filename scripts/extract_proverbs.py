#!/usr/bin/env python3
"""Extract a parallel proverb dataset from Kunzika's *Dicionário de Provérbios Kikongo*.

Emanuel Kunzika (2008), *Dicionário de Provérbios Kikongo: traduzidos e
explicados em português, francês e inglês*, Editorial Nzila, Luanda. 362 scanned
pages, no text layer, so everything here starts from OCR.

The book supervises its own parse. Its "COMO LER ESTE DICIONARIO" page (printed
p.27) states the format outright: proverbs are entered in Kikongo in alphabetical
order; the Portuguese, French and English translations follow in that order in
bold; explanations, commentary and equivalent adages follow each translation in
normal characters. Every entry in the book obeys it.

Bold is exactly what OCR throws away, so we recover the structure from three
properties that survive rasterisation:

  1. Headwords are in capitals and terminate with a full stop, so they anchor the
     entry boundaries even when they wrap over three lines.
  2. The three translation blocks always appear in the order pt -> fr -> en, so
     labelling lines by language is a *monotonic* segmentation, not a free
     classification. `segment_languages` solves it with a small DP, which lets an
     individual misread line be outvoted by its neighbours instead of splitting a
     block in two.
  3. Within a block the translation is the first sentence and the commentary is
     everything after it, so `split_translation` only has to find the first line
     that closes a sentence.

Portuguese and French fight over the OCR model -- `por` reads "são/mão/difícil"
correctly but turns "même/très" into "méme/três", and `fra+eng` does the reverse.
So the book is OCR'd twice and each language is taken from the pass that reads it
best. The two passes are parsed independently and aligned on their headwords.

Usage:
    python scripts/extract_proverbs.py references/proverbe-kikongo.pdf
    python scripts/extract_proverbs.py references/proverbe-kikongo.pdf --pages 31-60

Licence note: Kunzika 2008 is an in-copyright work. The extracted dataset is
local research evidence; redistributing it needs the publisher's permission.
See RESOURCES.md.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tarfile
import unicodedata
from collections import Counter
from pathlib import Path

# Printed page number = PDF page number - 2 throughout the scan.
PAGE_OFFSET = 2
# Front matter (title, sponsors, prefaces in four languages, introductions,
# the how-to-read page) ends at PDF p.30; the first proverb opens p.31.
FIRST_PROVERB_PAGE = 31
# The last proverb closes on PDF p.358; p.359 opens the bibliography.
LAST_PROVERB_PAGE = 358

LANGS = ["pt", "fr", "en"]
LANG_NAMES = {"pt": "Portuguese", "fr": "French", "en": "English"}

# Hand-picked function words, used only when FLORES is not on disk. The real
# scorer is trained from FLORES-200 below; this is the degraded fallback.
FALLBACK_MARKERS: dict[str, set[str]] = {
    "pt": {"não", "são", "está", "é", "este", "esta", "provérbio", "dos", "das",
           "um", "uma", "ao", "pelo", "para", "com", "porque", "também", "muito",
           "quem", "mas", "seu", "sua", "ou", "mais", "todos", "quando", "sobre"},
    "fr": {"les", "des", "une", "dans", "est", "ce", "cette", "proverbe", "qui",
           "pour", "sur", "ne", "pas", "plus", "avec", "ses", "elle", "nous",
           "vous", "tout", "quand", "comme", "mais", "aussi", "leur", "sont"},
    "en": {"the", "of", "and", "to", "is", "that", "who", "this", "saying",
           "when", "in", "for", "with", "his", "her", "they", "not", "are",
           "be", "will", "as", "by", "from", "one", "said", "which", "there"},
}
# Log-probability floor for a token the training text never showed. Low enough
# that an unknown word is evidence against a language, high enough that one
# proper noun cannot outvote a line of ordinary function words.
UNKNOWN_LOGP = -12.0
# The book gives all three translations for every proverb, so a segmentation
# that leaves a language empty is claiming the book broke its own format. It is
# usually wrong -- a short block of quotations scores weakly and gets absorbed
# by a neighbour. Charging for an empty block in log-odds makes the DP prefer
# the layout the book promises, without forbidding the rare entry that really
# is missing one.
EMPTY_BLOCK_PENALTY = 20.0

# A headword is a run of capitals closing with a full stop. Entry 2 of the book
# ends "... (KYANDI) cf. prov.1)." so a lowercase cross-reference tail has to be
# tolerated; we test the uppercase *ratio* rather than `str.isupper`.
MIN_HEADWORD_UPPER_RATIO = 0.75
MIN_HEADWORD_LETTERS = 8
SENTENCE_END = re.compile(r"[.!?]\s*[»\"”’')\]]*\s*$")
# "Yovo" is Kikongo for "or": the book uses it to record an alternate wording of
# the proverb just given, on its own line and in capitals. It reads as a second
# headword, and treating it as one splits the entry in two -- the real headword
# keeps no translations and the variant inherits all of them.
YOVO = re.compile(r"^\s*yovo\b\s*[:;.,-]?\s*", re.IGNORECASE)
# OCR merges a short line into the one below it often enough to matter. A
# headword that has swallowed the translation under it shows up as a run of
# lowercase words, which capitals never produce.
MERGED_BODY_RUN = 3
# Scanner artefacts: the book's gutter shows up as stray bars and dots in the
# left margin on most pages, and tesseract emits them as their own short lines.
MARGIN_NOISE = re.compile(r"^[\s|{}\[\]<>\\/_~^`'\".,:;!*+=-]*$")
LEADING_BAR = re.compile(r"^\s*[|{}\[\]<>]\s*")
PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")


def require(tool: str) -> None:
    if not shutil.which(tool):
        sys.exit(f"{tool} not found. brew install tesseract poppler")


# --------------------------------------------------------------------------- OCR


def ocr_book(pdf: Path, cache: Path, langs: str, first: int, last: int,
             dpi: int, jobs: int, batch: int) -> dict[int, str]:
    """OCR pages `first..last` into `cache`, one text file per page.

    Cached per page, so a re-run after a crash costs only the missing pages and
    a second language pass over an already-rasterised book is the only real
    expense. Images are deleted per batch; the full book at 300dpi is ~9GB.
    """
    cache.mkdir(parents=True, exist_ok=True)
    missing = [p for p in range(first, last + 1)
               if not (cache / f"p-{p:03d}.txt").exists()]
    if missing:
        print(f"[ocr:{langs}] {len(missing)} pages to OCR", flush=True)
    for start in range(first, last + 1, batch):
        stop = min(start + batch - 1, last)
        if all((cache / f"p-{p:03d}.txt").exists() for p in range(start, stop + 1)):
            continue
        images = cache / "_img"
        images.mkdir(exist_ok=True)
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-gray", "-png",
             "-f", str(start), "-l", str(stop), str(pdf), str(images / "p")],
            check=True, capture_output=True,
        )
        pngs = sorted(images.glob("p*.png"))
        cmd = (
            f"printf '%s\\n' {' '.join(str(p) for p in pngs)} | "
            f"xargs -P {jobs} -I{{}} sh -c "
            f"'tesseract \"$1\" \"${{1%.png}}\" -l {langs} --psm 6 quiet 2>/dev/null' _ {{}}"
        )
        subprocess.run(cmd, shell=True, capture_output=True)
        for txt in images.glob("p*.txt"):
            page = int(re.search(r"(\d+)$", txt.stem).group(1))
            (cache / f"p-{page:03d}.txt").write_text(
                txt.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        shutil.rmtree(images)
        print(f"  pages {start}-{stop}", flush=True)

    pages: dict[int, str] = {}
    for p in range(first, last + 1):
        f = cache / f"p-{p:03d}.txt"
        if f.exists():
            pages[p] = f.read_text(encoding="utf-8", errors="replace")
    return pages


# ------------------------------------------------------------------------ clean


def clean_page(text: str) -> list[str]:
    """Drop scanner artefacts and page numbers; return content lines."""
    out: list[str] = []
    for raw in text.splitlines():
        line = LEADING_BAR.sub("", raw).strip()
        if not line or MARGIN_NOISE.match(line) or PAGE_NUMBER.match(line):
            continue
        out.append(line)
    return out


def dehyphenate(lines: list[str]) -> list[str]:
    """Rejoin words the typesetter broke across a line ("confli-" + "tos")."""
    out: list[str] = []
    for line in lines:
        if out and out[-1].endswith("-") and line[:1].islower():
            out[-1] = out[-1][:-1] + line
        else:
            out.append(line)
    return out


# ---------------------------------------------------------------------- entries


def is_headword(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if len(letters) < MIN_HEADWORD_LETTERS:
        return False
    upper = sum(c.isupper() for c in letters) / len(letters)
    return upper >= MIN_HEADWORD_UPPER_RATIO


def cut_headword(line: str) -> tuple[str, str]:
    """Separate a headword from translation text OCR merged onto its line.

    Capitals never run to three lowercase words in a row, so the first such run
    marks where the headword stopped. The walk-back hands any trailing stray --
    a gutter bar, or the bare "E" left when "É" lost its accent -- to the body,
    where it belongs.
    """
    tokens = line.split()
    run = 0
    for i, token in enumerate(tokens):
        if any(c.islower() for c in token):
            run += 1
            if run >= MERGED_BODY_RUN:
                cut = i - run + 1
                while cut > 0 and len(tokens[cut - 1].strip(".,;:!?»«()|'’-")) <= 1:
                    cut -= 1
                return " ".join(tokens[:cut]), LEADING_BAR.sub("", " ".join(tokens[cut:]))
        else:
            run = 0
    return line, ""


def split_entries(pages: dict[int, str]) -> list[dict]:
    """Walk the book as one stream and cut it at every headword.

    Entries routinely straddle a page break, so the pages are concatenated
    before splitting rather than parsed one at a time. Each line carries the PDF
    page it came from, which gives the entry its printed page number.
    """
    stream: list[tuple[str, int]] = []
    for page in sorted(pages):
        for line in dehyphenate(clean_page(pages[page])):
            stream.append((line, page))

    entries: list[dict] = []
    current: dict | None = None
    head_buf: list[str] = []

    def open_entry(head: str, tail: str, page: int) -> None:
        """Start an entry -- or, for a `Yovo:` line, record a variant wording."""
        nonlocal current
        variant = YOVO.match(head)
        if variant and current is not None:
            current["variant"] = head[variant.end():]
        else:
            current = {"kikongo": head, "variant": "", "page": page, "body": []}
            entries.append(current)
        if tail and current is not None:
            current["body"].append(tail)

    for line, page in stream:
        if head_buf:
            # A headword that wrapped: keep taking capitalised lines until one
            # closes the sentence.
            closing = SENTENCE_END.search(line) or not is_headword(line)
            head, tail = cut_headword(line)
            head_buf.append(head)
            if closing:
                open_entry(" ".join(h for h in head_buf if h), tail, page)
                head_buf = []
            continue
        if not is_headword(line):
            if current is not None:
                current["body"].append(line)
            continue
        head, tail = cut_headword(line)
        if SENTENCE_END.search(head) or tail:
            open_entry(head, tail, page)
        else:
            head_buf = [head]
    if head_buf:
        open_entry(" ".join(head_buf), "", stream[-1][1])
    return merge_bodyless(entries)


def merge_bodyless(entries: list[dict]) -> list[dict]:
    """Fold an entry with no body at all into the one that follows it.

    Long headwords run to three or four printed lines, and a clause inside one
    can close with a full stop ("... KU VONDI LONGO LWAME KO.") that looks
    exactly like the end of the headword. The split lands mid-proverb and leaves
    the first half with nothing under it. Since the book translates every
    proverb, a body-less entry is never a real entry, and rejoining it to the
    next one restores the headword the typesetter actually wrapped.
    """
    merged: list[dict] = []
    for entry in entries:
        if merged and not merged[-1]["body"] and not merged[-1]["variant"]:
            orphan = merged.pop()
            entry["kikongo"] = f"{orphan['kikongo']} {entry['kikongo']}"
            entry["page"] = orphan["page"]
        merged.append(entry)
    return merged


# --------------------------------------------------------------------- language


class LanguageModel:
    """Unigram naive Bayes over Portuguese, French and English.

    Trained on the FLORES-200 dev sets, which give ~1k sentences of ordinary
    prose per language. That is far better evidence than a hand-written marker
    list: it weights every function word by how often it actually occurs, so a
    line carries a verdict even when it holds none of the words a human would
    have thought to list. It also costs nothing to trust -- FLORES is already in
    `data/raw` for the evaluation sets, and the three metalanguages of this book
    are all high-resource there.
    """

    def __init__(self, counts: dict[str, Counter]):
        self.logp: dict[str, dict[str, float]] = {}
        for lang, counter in counts.items():
            total = sum(counter.values()) + len(counter)
            self.logp[lang] = {
                w: math.log((c + 1) / total) for w, c in counter.items()
            }

    @classmethod
    def from_flores(cls, tarball: Path) -> "LanguageModel | None":
        if not tarball.exists():
            return None
        wanted = {"por_Latn": "pt", "fra_Latn": "fr", "eng_Latn": "en"}
        counts = {lang: Counter() for lang in LANGS}
        with tarfile.open(tarball) as tar:
            for member in tar.getmembers():
                stem = Path(member.name).name.split(".")[0]
                if stem not in wanted or not member.name.endswith((".dev", ".devtest")):
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                text = io.TextIOWrapper(fh, encoding="utf-8").read()
                counts[wanted[stem]].update(tokenise(text))
        if not all(counts.values()):
            return None
        return cls(counts)

    @classmethod
    def from_markers(cls) -> "LanguageModel":
        """Fallback: treat each marker as if seen once in its own language."""
        return cls({lang: Counter(words) for lang, words in FALLBACK_MARKERS.items()})

    def score(self, line: str) -> dict[str, float]:
        tokens = tokenise(line)
        if not tokens:
            return {lang: 0.0 for lang in LANGS}
        return {
            lang: sum(self.logp[lang].get(t, UNKNOWN_LOGP) for t in tokens)
            for lang in LANGS
        }


def tokenise(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower())


def segment_languages(body: list[str], model: LanguageModel) -> dict[str, list[str]]:
    """Cut the body into pt -> fr -> en blocks, in that order.

    The order is guaranteed by the book, which turns language ID from a
    per-line decision into a choice of two split points. Maximising the total
    score over both split points lets a line that scores wrong on its own -- a
    bare quotation, a Bible reference, a line of proper nouns -- be carried by
    the block around it. A per-line argmax would cut such a line out as its own
    block and lose the text.
    """
    n = len(body)
    if n == 0:
        return {lang: [] for lang in LANGS}
    scores = [model.score(line) for line in body]
    # prefix[lang][k] = score of body[:k] labelled `lang`
    prefix = {lang: [0.0] * (n + 1) for lang in LANGS}
    for lang in LANGS:
        for k, s in enumerate(scores):
            prefix[lang][k + 1] = prefix[lang][k] + s[lang]

    best, best_cut = None, (0, 0)
    for i in range(n + 1):           # end of pt
        for j in range(i, n + 1):    # end of fr
            total = (prefix["pt"][i]
                     + prefix["fr"][j] - prefix["fr"][i]
                     + prefix["en"][n] - prefix["en"][j])
            total -= EMPTY_BLOCK_PENALTY * ((i == 0) + (j == i) + (j == n))
            if best is None or total > best:
                best, best_cut = total, (i, j)
    i, j = best_cut
    return {"pt": body[:i], "fr": body[i:j], "en": body[j:]}


def split_translation(block: list[str]) -> tuple[str, str]:
    """First sentence is the translation; the rest is explanation and adages."""
    if not block:
        return "", ""
    for k, line in enumerate(block):
        if SENTENCE_END.search(line):
            return " ".join(block[:k + 1]), " ".join(block[k + 1:])
    return " ".join(block), ""


# ---------------------------------------------------------------------- parsing


def parse_pass(pages: dict[int, str], model: LanguageModel) -> list[dict]:
    records = []
    for entry in split_entries(pages):
        blocks = segment_languages(entry["body"], model)
        rec = {
            "kikongo": normalise_headword(entry["kikongo"]),
            "kikongo_variant": normalise_headword(entry["variant"]),
            "page": entry["page"] - PAGE_OFFSET,
        }
        for lang in LANGS:
            translation, explanation = split_translation(blocks[lang])
            rec[lang] = tidy(translation)
            rec[f"{lang}_explanation"] = tidy(explanation)
        records.append(rec)
    return records


def tidy(text: str) -> str:
    """Collapse OCR whitespace and drop the gutter marks that survived cleaning."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[|`'’\"~*_.,:;\-\s]+", "", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def normalise_headword(text: str) -> str:
    text = tidy(text)
    return text.rstrip(" .") + "." if text else text


def key(headword: str) -> str:
    """Alignment key: fold case and drop everything OCR gets wrong."""
    folded = unicodedata.normalize("NFD", headword.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", folded)


def align(primary: list[dict], secondary: list[dict]) -> tuple[list[dict], int]:
    """Attach the `por`-pass Portuguese to the `fra+eng`-pass entries.

    The two passes read the same images and almost always find the same entries,
    but a headword misread in one pass can shift the sequence, so we match on
    the folded headword instead of trusting the index. Duplicate headwords are
    matched in order of appearance.
    """
    buckets: dict[str, list[dict]] = {}
    for rec in secondary:
        buckets.setdefault(key(rec["kikongo"]), []).append(rec)

    def take(k: str) -> dict | None:
        pool = buckets.get(k)
        if pool:
            rec = pool.pop(0)
            if not pool:
                del buckets[k]
            return rec
        return None

    matched = 0
    leftover: list[dict] = []
    for rec in primary:
        other = take(key(rec["kikongo"]))
        if other is None:
            leftover.append(rec)
            continue
        rec["pt"] = other["pt"]
        rec["pt_explanation"] = other["pt_explanation"]
        matched += 1

    # Second round for the entries whose headword the two passes read
    # differently -- almost always a terminal I/L confusion in capitals
    # (MENGI/MENGL) or a space dropped inside a word. Close-matching only the
    # residue keeps a near-miss from stealing a pair that matched exactly.
    for rec in leftover:
        k = key(rec["kikongo"])
        hit = difflib.get_close_matches(k, list(buckets), n=1, cutoff=0.9)
        if not hit:
            continue
        other = take(hit[0])
        if other is None:
            continue
        rec["pt"] = other["pt"]
        rec["pt_explanation"] = other["pt_explanation"]
        matched += 1
    return primary, matched


# ------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/proverbs"))
    ap.add_argument("--cache", type=Path, default=Path("data/raw/proverbs_ocr"))
    ap.add_argument("--pages", default=None, help="e.g. 31-60; default all proverbs")
    ap.add_argument("--flores", type=Path,
                    default=Path("data/raw/flores200_dataset.tar.gz"),
                    help="FLORES-200 tarball, used to train the pt/fr/en split")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()

    require("pdftoppm")
    require("tesseract")

    n_pages = int(subprocess.run(
        ["pdfinfo", str(args.pdf)], capture_output=True, text=True
    ).stdout.split("Pages:")[1].split()[0])
    if args.pages:
        first, last = (int(x) for x in args.pages.split("-"))
    else:
        first, last = FIRST_PROVERB_PAGE, min(LAST_PROVERB_PAGE, n_pages)
    print(f"[proverbs] {args.pdf.name}: {n_pages} pages, parsing {first}-{last}")

    # Two passes: `fra+eng` reads French and English (and the capitalised
    # headwords) correctly, `por` reads Portuguese correctly. Neither reads all
    # three -- see the module docstring.
    fe = ocr_book(args.pdf, args.cache / "fra_eng", "fra+eng",
                  first, last, args.dpi, args.jobs, args.batch)
    pt = ocr_book(args.pdf, args.cache / "por", "por",
                  first, last, args.dpi, args.jobs, args.batch)

    model = LanguageModel.from_flores(args.flores)
    if model is None:
        print(f"  ! {args.flores} missing -- falling back to hand-written markers; "
              f"run `make data` for a better language split")
        model = LanguageModel.from_markers()
    else:
        print(f"[proverbs] language model trained on FLORES-200 "
              f"({args.flores.name})")

    records = parse_pass(fe, model)
    print(f"[proverbs] {len(records)} entries from the fra+eng pass")
    pt_records = parse_pass(pt, model)
    print(f"[proverbs] {len(pt_records)} entries from the por pass")
    records, matched = align(records, pt_records)
    print(f"[proverbs] {matched}/{len(records)} entries matched across passes")

    for i, rec in enumerate(records, 1):
        rec["id"] = f"kunzika-{i:04d}"

    complete = [r for r in records if all(r[l] for l in LANGS)]
    print(f"[proverbs] {len(complete)}/{len(records)} entries have all three "
          f"translations")
    for lang in LANGS:
        have = sum(1 for r in records if r[lang])
        expl = sum(1 for r in records if r[f"{lang}_explanation"])
        print(f"    {LANG_NAMES[lang]:<11} {have:>4} translations, "
              f"{expl:>4} explanations")

    args.out.mkdir(parents=True, exist_ok=True)
    fields = (["id", "kikongo", "kikongo_variant"]
              + [f for lang in LANGS for f in (lang, f"{lang}_explanation")]
              + ["page"])
    with (args.out / "proverbs.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps({k: rec[k] for k in fields}, ensure_ascii=False) + "\n")
    with (args.out / "proverbs.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec[k] for k in fields})
    print(f"[proverbs] wrote {args.out}/proverbs.jsonl and proverbs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
