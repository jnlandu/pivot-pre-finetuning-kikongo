"""Verified data source registry.

Every identifier here was checked against the live HF / fbaipublicfiles APIs on
2026-08-28. Row counts are the full upstream sizes before any filtering.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PivotSource:
    """An English->pivot mined bitext corpus on the Hub."""

    key: str
    hf_id: str
    rows: int
    iso: str
    family: str
    note: str = ""


# ---------------------------------------------------------------------------
# Pivot candidates. All are michsethowusu re-publications of NLLB-v1 (OPUS) or
# MT560 mined bitext, sharing the schema: similarity(float) | English | <Lang>.
# The target-language column name varies per corpus, so loaders auto-detect it
# rather than hardcoding -- see prepare.load_pivot_frame.
# ---------------------------------------------------------------------------
PIVOTS: dict[str, PivotSource] = {
    "kon_mined": PivotSource(
        "kon_mined", "michsethowusu/english-kongo_sentence-pairs", 2_228_449,
        "kon", "Bantu H10",
        "NLLB kon_Latn. Same provenance as the FLORES-200 eval set -- using this "
        "as a pivot for a Kikongo target is leakage, not transfer. Audit only.",
    ),
    "kon_mt560": PivotSource(
        "kon_mt560", "michsethowusu/english-kongo_sentence-pairs_mt560", 206_234,
        "kon", "Bantu H10", "MT560 'Kongo' label.",
    ),
    "kik_mt560": PivotSource(
        "kik_mt560", "michsethowusu/english-kikongo_sentence-pairs_mt560", 169_875,
        "kng", "Bantu H10",
        "MT560 'Kikongo' label -- distinct from kon_mt560 upstream. Overlap is "
        "an open question the audit answers.",
    ),
    "lin": PivotSource(
        "lin", "michsethowusu/english-lingala_sentence-pairs", 2_910_515,
        "lin", "Bantu C30", "Nearest well-resourced DRC neighbour.",
    ),
    "lua": PivotSource(
        "lua", "michsethowusu/english-luba-kasai_sentence-pairs_mt560", 292_212,
        "lua", "Bantu L30", "Tshiluba.",
    ),
    "swc": PivotSource(
        "swc", "michsethowusu/english-congo-swahili_sentence-pairs_mt560", 271_892,
        "swc", "Bantu G40", "Congolese Swahili -- the in-country contact variety.",
    ),
    "swh": PivotSource(
        "swh", "michsethowusu/english-swahili_sentence-pairs", 23_513_175,
        "swh", "Bantu G40", "Coastal Swahili. Mid-resource Bantu control.",
    ),
    "umb": PivotSource(
        "umb", "michsethowusu/english-umbundu_sentence-pairs", 298_095,
        "umb", "Bantu R10", "Angola. Geographically adjacent, genetically distant.",
    ),
    "kmb": PivotSource(
        "kmb", "michsethowusu/english-kimbundu_sentence-pairs", 196_248,
        "kmb", "Bantu H20", "Angola. Closest non-H10 relative in the registry.",
    ),
    "sag": PivotSource(
        "sag", "michsethowusu/english-sango_sentence-pairs", 3_607_548,
        "sag", "Ubangian creole",
        "Non-Bantu regional lingua franca. Separates 'related' from 'nearby'.",
    ),
    "fra": PivotSource(
        "fra", "Helsinki-NLP/opus-100", 1_000_000,
        "fra", "Indo-European",
        "High-resource control AND the DRC official language -- carries more "
        "weight here than French did in the Kikamba study.",
    ),
}

# opus-100 does not share the michsethowusu schema; handled as a special case.
OPUS100_PIVOTS = {"fra": "en-fr"}

# ---------------------------------------------------------------------------
# Target-language gold data. Google SMOL, professionally translated, served as
# raw jsonl (the datasets-server index is unavailable for this repo).
# ---------------------------------------------------------------------------
SMOL_BASE = "https://huggingface.co/datasets/google/smol/raw/main"
SMOL_FILES = {
    # name -> (path, records, schema)
    "smolsent": (f"{SMOL_BASE}/smolsent/en_kg.jsonl", 863, "src/trg"),
    "smoldoc": (f"{SMOL_BASE}/smoldoc/en_kg.jsonl", 130, "srcs/trgs"),
    "gatitos": (f"{SMOL_BASE}/gatitos/en_kg.jsonl", 3_998, "src/trgs"),
}

# ---------------------------------------------------------------------------
# Evaluation. FLORES+ dropped kon_Latn and kept only ktu_Latn (Kituba), so the
# Kikongo eval has to come from the legacy FLORES-200 tarball. That deletion is
# itself a finding -- see audit.py and README.
# ---------------------------------------------------------------------------
FLORES200_TARBALL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES200_LANGS = ["kon_Latn", "eng_Latn", "lin_Latn", "swh_Latn", "fra_Latn"]
FLORES_PLUS_HF = "openlanguagedata/flores_plus"
FLORES_PLUS_LANGS = ["ktu_Latn", "lin_Latn", "eng_Latn"]

TARGET_ISO = "kon"
TARGET_NAME = "Kikongo"


def pivot(key: str) -> PivotSource:
    if key not in PIVOTS:
        raise KeyError(f"unknown pivot {key!r}; known: {sorted(PIVOTS)}")
    return PIVOTS[key]
