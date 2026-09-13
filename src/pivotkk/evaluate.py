"""Evaluate on both Kikongo test sets and report chrF++ / spBLEU.

Two eval sets, deliberately:
  flores200  -- shares provenance with the NLLB-mined training data
  smol       -- professionally translated, provenance-independent

A model that scores well on one and badly on the other is telling you the two
corpora are not the same language variety. Always report both.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import sources as S

ROOT = Path(__file__).resolve().parents[2]
DATA, RUNS = ROOT / "data", ROOT / "runs"

EVAL_SETS = {
    "flores200": DATA / "processed" / "eval" / "flores200_devtest.jsonl",
    "smol": DATA / "processed" / "target" / "smol_test.jsonl",
}


def translate(
    model, tokenizer, sentences: list[str], *, batch_size: int = 32,
    max_new_tokens: int = 128, num_beams: int = 4, device: str = "cuda",
) -> list[str]:
    import torch

    prefix = f"translate English to {S.TARGET_NAME}: "
    out: list[str] = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = [prefix + s for s in sentences[i : i + batch_size]]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            ).to(device)
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, num_beams=num_beams
            )
            out += tokenizer.batch_decode(gen, skip_special_tokens=True)
            print(f"  {min(i + batch_size, len(sentences))}/{len(sentences)}", end="\r")
    print()
    return out


def score(hyps: list[str], refs: list[str]) -> dict[str, float]:
    import sacrebleu

    chrf = sacrebleu.CHRF(word_order=2)  # chrF++
    refs_wrapped = [refs]
    metrics = {
        "chrf++": chrf.corpus_score(hyps, refs_wrapped).score,
        "bleu": sacrebleu.BLEU().corpus_score(hyps, refs_wrapped).score,
    }
    try:
        # spBLEU is the FLORES-200 standard; needs the flores200 spm tokenizer.
        metrics["spbleu"] = sacrebleu.BLEU(
            tokenize="flores200"
        ).corpus_score(hyps, refs_wrapped).score
    except Exception as exc:  # tokenizer unavailable offline
        print(f"  (spBLEU skipped: {exc})")
    return metrics


def main(
    checkpoint: str, *, limit: int | None = None, device: str | None = None,
    save: bool = True,
) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] {checkpoint} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device).eval()

    results: dict[str, dict] = {}
    for name, path in EVAL_SETS.items():
        if not path.exists():
            print(f"  {name}: {path} missing, skipping")
            continue
        df = pd.read_json(path, lines=True)
        if limit:
            df = df.head(limit)
        print(f"[eval] {name}: {len(df)} pairs")
        hyps = translate(
            model, tokenizer, df["src"].tolist(), device=device
        )
        results[name] = score(hyps, df["trg"].tolist())
        print(f"  {name}: {results[name]}")
        if save:
            out = Path(checkpoint) / f"predictions_{name}.jsonl"
            pd.DataFrame(
                {"src": df["src"], "ref": df["trg"], "hyp": hyps}
            ).to_json(out, orient="records", lines=True, force_ascii=False)

    if save:
        (Path(checkpoint) / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def collect(runs_dir: Path = RUNS) -> pd.DataFrame:
    """Gather every run's metrics.json into one comparison table."""
    rows = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        run = metrics_path.parent
        row: dict = {"run": run.name}
        cfg_path = run / "config.yaml"
        if cfg_path.exists():
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text())
            row |= {"pivot": cfg.get("pivot"), "pivot_size": cfg.get("pivot_size")}
        for eval_name, metrics in json.loads(metrics_path.read_text()).items():
            for metric, value in metrics.items():
                row[f"{eval_name}_{metric}"] = round(value, 2)
        rows.append(row)
    return pd.DataFrame(rows)
