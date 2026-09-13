# First SMOL experiment

Compare direct English–Kikongo fine-tuning with English–Lingala pre-fine-tuning
followed by the same Kikongo fine-tuning. Three matched target seeds: 13, 17, 23.
Stage A uses seed 13 and one epoch; its completed checkpoint is shared across
all three pivot runs. This measures Stage B variability, not Stage A variability.

Data after document-level splitting:

- Kikongo: 2,011 train / 153 validation / 302 test pairs.
- Lingala: 2,916 pairs from the 3,053-pair dataset fully reviewed by the user.
  The other 137 reviewed pairs now overlap held-out Kikongo documents/sentences.
- Another 1,318 Lingala pairs became eligible but are pending review and are
  excluded from these configs.

Stage B runs up to 20 epochs, selects the lowest validation loss, and stops
after three evaluations without improvement. Each run saves the selected model,
training history, data hashes, resolved model revision and completion metadata.
Test scores are never used for checkpoint selection. Stage A cache identity
includes data, training settings, implementation and installed library versions.

The old data and review provenance are in
`data/processed/snapshots/before_document_split/`.
Audits are in `data/processed/target/split_audit.json` and
`data/processed/pivot/lin_smol.audit.json`.
Rebuild pivot exclusions and update counts/configs if target splits change.

On a CUDA GPU with bf16 support, after installing project requirements:

```bash
.venv/bin/python -m pivotkk train configs/smol/lin_smol_seed13.yaml --smoke
.venv/bin/python -m pivotkk eval runs/lin_smol_seed13_smoke --limit 20
bash scripts/run_smol_sweep.sh
```

Smoke mode uses two optimizer steps per stage and a separate output/cache.
A tiny random local T5 integration test verifies training and checkpoint handling;
it does not establish mT5-base quality or GPU memory requirements.
Completed run directories are protected against overwriting; choose a new run
name for reruns. The sweep stops if it encounters a completed run.
