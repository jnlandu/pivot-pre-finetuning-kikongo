"""Two-stage pivot pre-finetuning.

Stage A: finetune the base model on English->pivot bitext.
Stage B: continue from A and finetune on the tiny English->Kikongo gold set.

Stage A checkpoints are content-addressed and reused across experiments, because
stage A is where the GPU hours go and you are paying for them by the hour.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import inspect
from importlib.metadata import version, PackageNotFoundError
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd
import yaml

from . import sources as S

ROOT = Path(__file__).resolve().parents[2]
DATA, RUNS = ROOT / "data", ROOT / "runs"
STAGE_A_CACHE = RUNS / "_stage_a_cache"

LANG_NAMES = {
    "kon_mined": "Kikongo", "kon_mt560": "Kikongo", "kik_mt560": "Kikongo",
    "lin": "Lingala", "lua": "Tshiluba", "swc": "Congolese Swahili",
    "swh": "Swahili", "umb": "Umbundu", "kmb": "Kimbundu",
    "sag": "Sango", "fra": "French",
}


@dataclass
class Config:
    name: str = "baseline"
    base_model: str = "google/mt5-base"
    pivot: str | None = None
    pivot_data: str | None = None
    model_revision: str = "main"
    pivot_size: int = 50_000
    pivot_epochs: float = 1.0
    pivot_lr: float = 5e-4
    target_epochs: float = 20.0
    target_lr: float = 3e-4
    use_lexicon: bool = False
    batch_size: int = 16
    grad_accum: int = 2
    max_source_len: int = 128
    max_target_len: int = 128
    bf16: bool = True
    seed: int = 13
    pivot_seed: int = 13
    target_seed: int | None = None
    early_stopping_patience: int = 3
    smoke: bool = False

    def pivot_path(self) -> Path:
        return ROOT / self.pivot_data if self.pivot_data else DATA / "processed" / "pivot" / f"{self.pivot}_{self.pivot_size}.jsonl"

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def stage_a_key(self) -> str:
        """Content hash of everything stage A depends on -- and nothing else."""
        payload = {
            "base_model": self.base_model, "pivot": self.pivot,
            "pivot_size": self.pivot_size, "pivot_epochs": self.pivot_epochs,
            "pivot_lr": self.pivot_lr, "max_source_len": self.max_source_len,
            "max_target_len": self.max_target_len, "seed": self.pivot_seed,
            "batch_size": self.batch_size, "grad_accum": self.grad_accum,
            "bf16": self.bf16, "revision": self.model_revision,
            "data_sha256": file_hash(self.pivot_path()),
            "trainer_sha256": file_hash(Path(__file__)),
            "packages": package_versions(),
            "smoke": self.smoke,
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return f"{self.pivot}_{self.pivot_size}_{hashlib.sha1(blob).hexdigest()[:10]}"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions() -> dict:
    result = {}
    for name in ("torch", "transformers", "datasets", "accelerate"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `make data` first")
    return pd.read_json(path, lines=True)


def _to_dataset(df: pd.DataFrame, target_lang: str, tokenizer, cfg: Config):
    from datasets import Dataset

    prefix = f"translate English to {target_lang}: "

    def tok(batch):
        model_inputs = tokenizer(
            [prefix + s for s in batch["src"]],
            max_length=cfg.max_source_len, truncation=True,
        )
        labels = tokenizer(
            text_target=batch["trg"], max_length=cfg.max_target_len, truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = Dataset.from_pandas(df[["src", "trg"]].reset_index(drop=True))
    return ds.map(tok, batched=True, remove_columns=["src", "trg"])


def _trainer(model, tokenizer, train_ds, eval_ds, out_dir: Path, *, lr, epochs, cfg, seed):
    from transformers import (
        DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback,
    )

    warmup = {"warmup_ratio": 0.03} if "warmup_ratio" in inspect.signature(Seq2SeqTrainingArguments).parameters else {"warmup_steps": 0.03}
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        learning_rate=lr,
        num_train_epochs=epochs,
        max_steps=2 if cfg.smoke else -1,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        # mT5 overflows in fp16; bf16 is the only safe half precision here.
        bf16=cfg.bf16,
        fp16=False,
        **warmup,
        optim="adamw_torch",
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch" if eval_ds is not None else "no",
        save_total_limit=2,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False if eval_ds is not None else None,
        report_to=[],
        seed=seed,
        data_seed=seed,
        predict_with_generate=False,
    )
    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)]
        if eval_ds is not None else [],
    )


def run_stage_a(cfg: Config, tokenizer) -> Path:
    """Pivot pre-finetuning. Returns a path to the resulting model."""
    from transformers import AutoModelForSeq2SeqLM, set_seed

    if not cfg.pivot:
        return Path(cfg.base_model)

    cache_dir = STAGE_A_CACHE / cfg.stage_a_key()
    if (cache_dir / "complete.json").exists():
        print(f"[stage A] cache hit -> {cache_dir}")
        return cache_dir

    src = S.pivot(cfg.pivot)
    lang = LANG_NAMES.get(cfg.pivot, cfg.pivot)
    path = cfg.pivot_path()
    df = _read_jsonl(path)
    if len(df) != cfg.pivot_size:
        raise ValueError(f"Expected {cfg.pivot_size} pivot pairs, found {len(df)} in {path}")
    if cfg.smoke:
        df = df.head(200)
    print(f"[stage A] {cfg.pivot} ({src.family}) en->{lang}, {len(df):,} pairs")

    set_seed(cfg.pivot_seed)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model, revision=cfg.model_revision)
    trainer = _trainer(
        model, tokenizer, _to_dataset(df, lang, tokenizer, cfg), None,
        cache_dir / "_work", lr=cfg.pivot_lr, epochs=cfg.pivot_epochs, cfg=cfg, seed=cfg.pivot_seed,
    )
    trainer.train()

    cache_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(cache_dir)
    tokenizer.save_pretrained(cache_dir)
    (cache_dir / "stage_a.json").write_text(json.dumps(asdict(cfg), indent=2))
    shutil.rmtree(cache_dir / "_work", ignore_errors=True)
    (cache_dir / "complete.json").write_text(json.dumps({"key": cfg.stage_a_key()}))
    print(f"[stage A] saved -> {cache_dir}")
    return cache_dir


def run_stage_b(cfg: Config, tokenizer, init_from: Path) -> Path:
    """Finetune on the tiny Kikongo gold set."""
    from transformers import AutoModelForSeq2SeqLM, set_seed

    train = _read_jsonl(DATA / "processed" / "target" / "smol_train.jsonl")
    dev = _read_jsonl(DATA / "processed" / "target" / "smol_dev.jsonl")
    if cfg.use_lexicon:
        lex = _read_jsonl(DATA / "processed" / "target" / "smol_lexicon.jsonl")
        train = pd.concat([train, lex[["src", "trg"]]], ignore_index=True)
        print(f"[stage B] +{len(lex):,} GATITOS lexicon entries")
    if cfg.smoke:
        train, dev = train.head(64), dev.head(16)
    print(f"[stage B] en->Kikongo, {len(train):,} pairs, init from {init_from}")

    out = RUNS / cfg.name
    if (out / "complete.json").exists():
        raise FileExistsError(f"Completed run exists: {out}; choose a new experiment name")
    target_seed = cfg.target_seed if cfg.target_seed is not None else cfg.seed
    set_seed(target_seed)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(init_from), **({"revision": cfg.model_revision} if not cfg.pivot else {}))
    trainer = _trainer(
        model, tokenizer,
        _to_dataset(train, S.TARGET_NAME, tokenizer, cfg),
        _to_dataset(dev, S.TARGET_NAME, tokenizer, cfg),
        out / "_work", lr=cfg.target_lr, epochs=cfg.target_epochs, cfg=cfg, seed=target_seed,
    )
    trainer.train()

    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))
    metadata = {"best_checkpoint": trainer.state.best_model_checkpoint,
                "best_eval_loss": trainer.state.best_metric,
                "epochs_completed": trainer.state.epoch,
                "stage_a": str(init_from), "target_seed": target_seed,
                "data_sha256": {f"smol_{s}.jsonl": file_hash(DATA / "processed/target" / f"smol_{s}.jsonl")
                                for s in ("train", "dev", "test")},
                "packages": package_versions()}
    trainer.state.save_to_json(str(out / "trainer_state.json"))
    shutil.rmtree(out / "_work", ignore_errors=True)
    (out / "complete.json").write_text(json.dumps(metadata, indent=2))
    print(f"[stage B] saved -> {out}")
    return out


def main(config_path: str | Path, smoke: bool = False) -> Path:
    from transformers import AutoTokenizer

    cfg = Config.load(config_path)
    cfg.smoke = cfg.smoke or smoke
    if cfg.smoke:
        cfg.name += "_smoke"
    import torch
    if cfg.bf16 and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise RuntimeError("This configuration requires a CUDA GPU with bf16 support. Use bf16: false for a small CPU test.")
    # Resolve the mutable Hub branch before computing a cache key.
    if not Path(cfg.base_model).exists():
        from huggingface_hub import model_info
        cfg.model_revision = model_info(cfg.base_model, revision=cfg.model_revision).sha
    print(f"=== {cfg.name} === pivot={cfg.pivot or 'NONE'} size={cfg.pivot_size}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, revision=cfg.model_revision)
    init = run_stage_a(cfg, tokenizer)
    return run_stage_b(cfg, tokenizer, init)
