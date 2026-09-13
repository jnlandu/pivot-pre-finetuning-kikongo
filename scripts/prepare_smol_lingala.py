"""Prepare downloaded SMOL Lingala, excluding Kikongo held-out documents.

Run: .venv/bin/python scripts/prepare_smol_lingala.py
Downloads must exist in data/raw/smol; existing target splits are unchanged.
"""
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from pivotkk.prepare import clean_pairs

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/smol'
OUT = ROOT / 'data/processed/pivot'


def read(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def write(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def main():
    sent_path = RAW / 'smolsent_en_ln.jsonl'
    doc_path = RAW / 'smoldoc_en_ln.jsonl'
    kg_path = RAW / 'smoldoc_en_kg.jsonl'
    sent, docs, kg_docs = map(read, (sent_path, doc_path, kg_path))
    eval_paths = [ROOT / f'data/processed/target/smol_{s}.jsonl' for s in ('dev', 'test')]
    eval_paths += [ROOT / f'data/processed/eval/flores200_{s}.jsonl' for s in ('dev', 'devtest')]
    eval_sources = {p.stem: {norm(r['src']) for r in read(p)} for p in eval_paths}
    held = set().union(*eval_sources.values())
    # Recover document membership lost in the current sentence-level target split.
    blocked_docs = [d for d in kg_docs if any(norm(s) in held for s in d['srcs'])]
    blocked_ids = {d['id'] for d in blocked_docs}
    blocked_sources = held | {norm(s) for d in blocked_docs for s in d['srcs']}
    rows = []
    excluded_docs = mismatched_docs = 0
    for d in docs:
        if d['id'] in blocked_ids or any(norm(s) in blocked_sources for s in d['srcs']):
            excluded_docs += 1
            continue
        if len(d['srcs']) != len(d['trgs']):
            mismatched_docs += 1
            continue
        rows.extend({'src': s, 'trg': t, 'origin': 'smoldoc', 'document_id': d['id'], 'sentence_index': i}
                    for i, (s, t) in enumerate(zip(d['srcs'], d['trgs'])))
    rows.extend({'src': r['src'], 'trg': r['trg'], 'origin': 'smolsent', 'document_id': None,
                 'sentence_id': r.get('id')} for r in sent if norm(r['src']) not in blocked_sources)
    frame = clean_pairs(pd.DataFrame(rows))
    frame = frame.loc[~frame.src.map(norm).duplicated()]
    # Roundtrip converts nullable pandas fields into JSON null.
    result = [json.loads(line) for line in frame.to_json(orient='records', lines=True, force_ascii=False).splitlines()]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'lin_smol.jsonl'
    write(path, result)
    reviewed_path = ROOT / 'data/processed/snapshots/before_document_split/lin_smol.jsonl'
    reviewed_keys = {(r['src'], r['trg']) for r in read(reviewed_path)} if reviewed_path.exists() else set()
    reviewed = [r for r in result if (r['src'], r['trg']) in reviewed_keys]
    pending = [r for r in result if (r['src'], r['trg']) not in reviewed_keys]
    write(OUT / 'lin_smol_reviewed.jsonl', reviewed)
    write(OUT / 'lin_smol_pending_review.jsonl', pending)
    sources = {norm(r['src']) for r in result}
    report = {
        'source': 'https://huggingface.co/datasets/google/smol',
        'input_sentences': len(sent), 'input_documents': len(docs),
        'input_document_sentence_pairs': sum(len(d['srcs']) for d in docs),
        'blocked_kikongo_documents': len(blocked_docs),
        'excluded_lingala_documents': excluded_docs, 'mismatched_documents': mismatched_docs,
        'output_pairs': len(result), 'origins': dict(Counter(r['origin'] for r in result)),
        'previously_reviewed_pairs_retained': len(reviewed),
        'newly_eligible_pairs_needing_review': len(pending),
        'normalized_source_duplicates': len(result) - len(sources),
        'evaluation_source_overlap': {k: len(sources & v) for k, v in eval_sources.items()},
        'blocked_document_source_overlap': len(sources & blocked_sources),
        'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [sent_path, doc_path, kg_path, *eval_paths]},
        'output_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'limitations': ['User approved all pairs in the archived 3053-pair dataset; newly eligible pairs are not covered by that review.',
                       'No systematic domain classification performed.',
                       'Semantic near-duplicates are not detected.',
                       'Rebuild this dataset if target evaluation splits change.'],
    }
    assert result and len(sources) == len(result) and not (sources & blocked_sources)
    (OUT / 'lin_smol.audit.json').write_text(json.dumps(report, indent=2) + '\n')
    sample = random.Random(13).sample(result, min(100, len(result)))
    write(OUT / 'lin_smol.review.jsonl', [{**r, 'alignment_ok': None, 'lingala_ok': None, 'domain': None, 'notes': ''} for r in sample])
    print(json.dumps({k:v for k,v in report.items() if 'sha256' not in k}, indent=2))


if __name__ == '__main__':
    main()
