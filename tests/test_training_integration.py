"""Exercise real checkpoint saving/reloading with a tiny random local model."""
import json
from pathlib import Path

import pytest


def test_two_stage_training_selects_checkpoint_and_reuses_cache(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, MT5Config, MT5ForConditionalGeneration
    from pivotkk import train as T

    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setenv('HF_DATASETS_CACHE', str(tmp_path / 'datasets_cache'))
    monkeypatch.setattr(T, 'ROOT', tmp_path)
    monkeypatch.setattr(T, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(T, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(T, 'STAGE_A_CACHE', tmp_path / 'cache')
    vocab = {'<pad>': 0, '</s>': 1, '<unk>': 2, 'hello': 3, 'mbote': 4, 'world': 5}
    raw = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='<pad>', eos_token='</s>', unk_token='<unk>')
    base = tmp_path / 'base'
    model = MT5ForConditionalGeneration(MT5Config(vocab_size=6, d_model=16, d_ff=32, num_layers=1,
                                              num_decoder_layers=1, num_heads=2, d_kv=8,
                                              decoder_start_token_id=0, eos_token_id=1, pad_token_id=0))
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    row = json.dumps({'src': 'hello world', 'trg': 'mbote'}) + '\n'
    for s in ['train', 'dev', 'test']:
        p = T.DATA / 'processed/target' / f'smol_{s}.jsonl'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(row * 4)
    pivot = tmp_path / 'pivot.jsonl'
    pivot.write_text(row * 4)
    cfg = T.Config(name='tiny', base_model=str(base), pivot='lin', pivot_data=str(pivot), pivot_size=4,
                   batch_size=2, grad_accum=1, bf16=False, smoke=True, target_epochs=2)
    a = T.run_stage_a(cfg, tokenizer)
    assert (a / 'complete.json').exists()
    marker = (a / 'complete.json').stat().st_mtime_ns
    assert T.run_stage_a(cfg, tokenizer) == a
    assert (a / 'complete.json').stat().st_mtime_ns == marker
    b = T.run_stage_b(cfg, tokenizer, a)
    meta = json.loads((b / 'complete.json').read_text())
    assert meta['best_checkpoint'] and meta['best_eval_loss'] is not None
    loaded = MT5ForConditionalGeneration.from_pretrained(b)
    assert loaded.generate(**tokenizer('hello', return_tensors='pt'), max_new_tokens=3).shape[0] == 1
    assert (b / 'trainer_state.json').exists()
    cfg.pivot, cfg.name = None, 'tiny_baseline'
    baseline = T.run_stage_b(cfg, tokenizer, base)
    assert (baseline / 'complete.json').exists()
    cfg.pivot, cfg.name = 'lin', 'tiny'
    with pytest.raises(FileExistsError):
        T.run_stage_b(cfg, tokenizer, a)
