"""Fast checks that need no GPU and no model download."""
from pathlib import Path

import pandas as pd
import pytest

from pivotkk import sources as S
from pivotkk.evaluate import score
from pivotkk.prepare import clean_pairs, sample_ladder
from pivotkk.train import Config

CONFIGS = sorted((Path(__file__).parents[1] / "configs" / "experiments").glob("*.yaml")) + sorted((Path(__file__).parents[1] / "configs" / "smol").glob("*.yaml"))


def test_clean_pairs_drops_junk():
    df = pd.DataFrame(
        {
            "src": ["a good english sentence", "a good english sentence", "hi", "x" * 40],
            "trg": ["mbote ya kitoko mpenza", "different target text here", "mbote", "y"],
        }
    )
    out = clean_pairs(df)
    # too short, bad ratio, and the duplicated source all go
    assert len(out) == 1
    assert out.iloc[0]["src"] == "a good english sentence"


def test_sample_ladder_is_not_just_head():
    df = pd.DataFrame({"src": [f"s{i}" for i in range(1000)], "trg": [f"t{i}" for i in range(1000)]})
    strat = sample_ladder(df, 10, how="stratified")
    assert len(strat) == 10
    # stratified must span the score-sorted range, unlike head()
    assert strat["src"].tolist() != df.head(10)["src"].tolist()
    assert sample_ladder(df, 5000, how="stratified").shape[0] == 1000


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_config_loads_with_known_pivot(path):
    cfg = Config.load(path)
    assert cfg.bf16 is True, "mT5 must not run in fp16"
    if cfg.pivot is not None:
        assert cfg.pivot in S.PIVOTS


def test_stage_a_cache_key_ignores_stage_b_settings(tmp_path):
    a = Config.load(CONFIGS[0])
    a.pivot, a.pivot_size = "lin", 50_000
    b = Config.load(CONFIGS[0])
    b.pivot, b.pivot_size = "lin", 50_000
    data = tmp_path / 'pivot.jsonl'
    data.write_text('{"src":"hello","trg":"mbote"}\n')
    a.pivot_data = b.pivot_data = str(data)
    b.target_lr, b.target_epochs, b.use_lexicon = 9e-9, 3.0, True
    assert a.stage_a_key() == b.stage_a_key()

    b.pivot_size = 10_000
    assert a.stage_a_key() != b.stage_a_key()


def test_stage_a_cache_tracks_data_and_training_settings(tmp_path):
    from dataclasses import replace
    data = tmp_path / 'pivot.jsonl'
    data.write_text('first dataset')
    cfg = Config(pivot='lin', pivot_data=str(data))
    key = cfg.stage_a_key()
    assert replace(cfg, target_seed=99, seed=99).stage_a_key() == key
    for change in ({'pivot_seed': 99}, {'batch_size': 1}, {'grad_accum': 9}, {'bf16': False}, {'model_revision': 'different'}):
        assert replace(cfg, **change).stage_a_key() != key
    data.write_text('changed dataset')
    assert cfg.stage_a_key() != key


def test_target_split_keeps_duplicate_linked_documents_together():
    from pivotkk.prepare import split_target_pairs, normalize_text
    rows = []
    for doc in range(40):
        for i in range(8):
            # Docs 0 and 1 share a source sentence and must share a group.
            src = 'shared duplicate sentence' if doc < 2 and i == 0 else f'English document {doc} sentence {i}'
            rows.append(dict(src=src, trg=f'Target document {doc} sentence {i}',
                             origin='smoldoc', document_id=str(doc), group_id=f'doc:{doc}'))
    split = split_target_pairs(rows)
    again = split_target_pairs(rows)
    assignments = {}
    for name, frame in split.items():
        assert not frame.empty
        assert frame.equals(again[name])
        for doc in frame.document_id:
            assert doc not in assignments or assignments[doc] == name
            assignments[doc] = name
    assert assignments['0'] == assignments['1']
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        for side in ['src', 'trg', 'group_id']:
            assert not set(split[a][side].map(normalize_text)) & set(split[b][side].map(normalize_text))


def test_score_returns_chrf():
    m = score(["mbote na yo"], ["mbote na yo"])
    assert m["chrf++"] == pytest.approx(100.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# lexicon-based audit
# --------------------------------------------------------------------------- #
def test_load_lexicon_handles_three_formats(tmp_path):
    from pivotkk.audit import load_lexicon

    (tmp_path / "a.txt").write_text("mono\nngeye\nkadi\nyo\n", encoding="utf-8")
    assert load_lexicon(tmp_path / "a.txt") == {"mono", "ngeye", "kadi"}  # "yo" too short

    (tmp_path / "b.tsv").write_text("I\tmono\nyou\tngeye\n", encoding="utf-8")
    assert load_lexicon(tmp_path / "b.tsv") == {"mono", "ngeye"}

    pd.DataFrame({"src": ["I"], "trg": ["mono mu"]}).to_json(
        tmp_path / "c.jsonl", orient="records", lines=True
    )
    assert load_lexicon(tmp_path / "c.jsonl") == {"mono"}


def test_lexicon_coverage_accepts_frame_or_set():
    from pivotkk.audit import lexicon_coverage

    texts = ["mono kele na nzo", "ngeye mpila"]
    assert lexicon_coverage(texts, {"mono", "ngeye", "absent"}) == pytest.approx(2 / 3, abs=1e-4)
    frame = pd.DataFrame({"src": ["I", "you"], "trg": ["mono", "ngeye"]})
    assert lexicon_coverage(texts, frame) == 1.0


def test_differential_coverage_picks_the_right_variety():
    from pivotkk.audit import differential_coverage

    # Two disjoint reference lexicons, 25 forms each (above the 20-form floor).
    kikongo = {f"kik{i:03d}" for i in range(25)}
    kituba = {f"ktu{i:03d}" for i in range(25)}
    # A corpus written almost entirely in the "kituba" forms.
    texts = [" ".join(sorted(kituba))] + ["kik001"]

    out = differential_coverage(texts, {"kikongo": kikongo, "kituba": kituba})
    verdict = out["discriminative"]["kikongo_vs_kituba"]
    assert verdict["leans"] == "kituba"
    assert verdict["kituba_only_coverage"] == 1.0
    assert verdict["margin"] > 0.9
    assert out["raw"]["kituba"]["n_forms"] == 25


def test_differential_coverage_stays_silent_without_enough_forms():
    from pivotkk.audit import differential_coverage

    out = differential_coverage(["mono"], {"a": {"mono"}, "b": {"munu"}})
    # fewer than 20 distinguishing forms each -- refuse to draw a conclusion
    assert out["discriminative"] == {}


def test_discover_lexicons_skips_readme(tmp_path):
    from pivotkk.audit import discover_lexicons

    (tmp_path / "README.md").write_text("not a lexicon", encoding="utf-8")
    (tmp_path / "yombe.txt").write_text("mbote\nkitoko\n", encoding="utf-8")
    found = discover_lexicons(tmp_path)
    assert "yombe" in found
    assert "README" not in found
