"""Checks for the Kunzika proverb extractor. No PDF, no OCR, no network.

Each test pins one of the OCR failures that actually showed up in the 362-page
scan, so a regression here means the dataset silently loses entries again.
"""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "extract_proverbs",
    Path(__file__).parents[1] / "scripts" / "extract_proverbs.py",
)
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


@pytest.fixture(scope="module")
def model():
    """FLORES if it is on disk, hand-written markers otherwise.

    The assertions below hold either way -- they are about block *order* and
    boundaries, not about the fine-grained scores FLORES buys.
    """
    flores = Path(__file__).parents[1] / "data" / "raw" / "flores200_dataset.tar.gz"
    return E.LanguageModel.from_flores(flores) or E.LanguageModel.from_markers()


def test_headword_detection_tolerates_lowercase_cross_reference():
    # Entry 2 of the book: capitals with a lowercase "cf." tail.
    assert E.is_headword("A ENGI AVWANDANGA VA KYANDU (KYANDI) cf. prov.1).")
    assert E.is_headword("ANSIDIDI MBOLONGO YA TIYA VA MOKO.")
    assert not E.is_headword("Puseram-lhe uma beringela quente na mão.")
    assert not E.is_headword("A.")


def test_cut_headword_recovers_a_swallowed_translation():
    # OCR merges the short headword line into the translation under it.
    head, tail = E.cut_headword(
        "FU KYA NKELE, KU MPUTU KIATUKA, E um defeito da fabrica da espingarda.")
    assert head == "FU KYA NKELE, KU MPUTU KIATUKA,"
    # The stray "E" is the accent-stripped "É" of the translation, not headword.
    assert tail.startswith("E um defeito")


def test_cut_headword_keeps_an_inline_yovo_variant_whole():
    # "yovo" (Kikongo "or") joins two wordings inside one headword; the capitals
    # resume straight after it, so it must not read as the start of the body.
    line = "KANA TOTA NTUMBU MU NZILA, yovo: KANA BONGOLOLA KIMA MU NZILA."
    head, tail = E.cut_headword(line)
    assert head == line
    assert tail == ""


def test_dehyphenate_rejoins_a_broken_word():
    assert E.dehyphenate(["as fontes de confli-", "tos entre irmãos."]) == [
        "as fontes de conflitos entre irmãos."
    ]
    # A hyphen before a capital is a dash, not a break.
    assert len(E.dehyphenate(["proverbe écossais -", "Larousse."])) == 2


def test_yovo_line_becomes_a_variant_not_a_second_entry():
    # p.51: the alternate wording sits on its own line, in capitals, and used to
    # be parsed as an entry of its own -- stealing every translation from the
    # proverb above it.
    pages = {31: "\n".join([
        "DYA KWAKU, NWA KWAKU, KANSI FULU KYA LEKA IKIBADIDI.",
        "Yovo: DYA, NWA, KANSI FULU KYE LEKA IKIBADIDI.",
        "Pode comer, pode beber, mas o lugar onde dormir importa muito.",
    ])}
    entries = E.split_entries(pages)
    assert len(entries) == 1
    assert entries[0]["kikongo"].startswith("DYA KWAKU")
    assert entries[0]["variant"].startswith("DYA, NWA")
    assert entries[0]["body"] == [
        "Pode comer, pode beber, mas o lugar onde dormir importa muito."]


def test_bodyless_entry_is_folded_into_the_next_one():
    # A full stop inside a long wrapped headword splits it mid-proverb. The
    # first half has no translations under it, which no real entry ever has.
    pages = {31: "\n".join([
        "E KANDA KANDA, KU VONDI LONGO LWAME KO.",
        "E NKENTO, NKENTO, KUVONDI KO UNGUDI WAME.",
        "Familia, familia, não destrua o meu matrimonio.",
    ])}
    entries = E.split_entries(pages)
    assert len(entries) == 1
    assert entries[0]["kikongo"].startswith("E KANDA KANDA")
    assert "NKENTO" in entries[0]["kikongo"]


def test_segment_languages_keeps_the_pt_fr_en_order(model):
    body = [
        "As lágrimas têm um conduto lacrimal, o muco tem também o seu.",
        "Este provérbio é utilizado quando se trata de dois assuntos diferentes.",
        "Les larmes ont un conduit lacrymal, la mucosité a aussi le sien.",
        "Ce proverbe est utilisé lorsqu'il s'agit de deux problèmes différents.",
        "The tears have the tear duct and the mucus has its also.",
        "This saying is used when two matters cannot be tackled at the same time.",
    ]
    blocks = E.segment_languages(body, model)
    assert blocks["pt"] == body[0:2]
    assert blocks["fr"] == body[2:4]
    assert blocks["en"] == body[4:6]


def test_segment_languages_does_not_starve_a_short_block(model):
    # A two-line Portuguese block carrying almost no function words used to be
    # absorbed into the French one, losing the Portuguese translation entirely.
    body = [
        "A escarradeira enche-se de boca à boca.",
        "«A união faz a força».",
        "Le crachoir se remplit de bouche en bouche.",
        "«L'union fait la force».",
        "The spittoon is filled from mouth to mouth.",
    ]
    blocks = E.segment_languages(body, model)
    assert blocks["pt"] and blocks["fr"] and blocks["en"]
    assert blocks["pt"][0].startswith("A escarradeira")
    assert blocks["fr"][0].startswith("Le crachoir")


def test_split_translation_takes_the_first_sentence_only():
    block = [
        "Puseram-lhe uma beringela quente na mão.",
        "Este provérbio é o mesmo que «colocar uma batata quente na mão».",
    ]
    translation, explanation = E.split_translation(block)
    assert translation == block[0]
    assert explanation == block[1]


def test_split_translation_follows_a_wrapped_sentence_to_its_end():
    block = [
        "Les fils d'une même mère qui travaillent dans un même champ, doivent",
        "éviter des conflits possibles en délimitant leurs proprietés respecti-",
        "ves.",
        "«Soyez frères dans la vie commune».",
    ]
    translation, explanation = E.split_translation(block)
    assert translation.endswith("respecti- ves.")
    assert explanation == block[3]


def test_align_matches_headwords_the_two_passes_read_differently():
    # Terminal I/L confusion in capitals is the usual disagreement.
    primary = [{"kikongo": "DIVENGE NKWENO AVO KU VENGE DIOKO MENO NGANI.",
                "pt": "", "pt_explanation": ""}]
    secondary = [{"kikongo": "DIVENGE NKWENO AVO KU VENGE DIOKO MENO NGANL.",
                  "pt": "boa leitura", "pt_explanation": "explicação"}]
    merged, matched = E.align(primary, secondary)
    assert matched == 1
    assert merged[0]["pt"] == "boa leitura"


def test_align_prefers_an_exact_match_over_a_near_one():
    primary = [{"kikongo": "MBWA A.", "pt": "", "pt_explanation": ""},
               {"kikongo": "MBWA B.", "pt": "", "pt_explanation": ""}]
    secondary = [{"kikongo": "MBWA B.", "pt": "bee", "pt_explanation": ""},
                 {"kikongo": "MBWA A.", "pt": "aye", "pt_explanation": ""}]
    merged, matched = E.align(primary, secondary)
    assert matched == 2
    assert merged[0]["pt"] == "aye" and merged[1]["pt"] == "bee"


def test_tidy_strips_gutter_marks_but_keeps_quotation():
    assert E.tidy("| ' Os antepassados cumpriram os seus deveres .") == (
        "Os antepassados cumpriram os seus deveres.")
    assert E.tidy("«A união faz a força».").startswith("«")
