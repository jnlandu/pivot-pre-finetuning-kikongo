#!/usr/bin/env bash
# Baseline and reviewed SMOL Lingala, three matched Stage B seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
"$PY" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "A CUDA GPU with bf16 support is required"'
for seed in 13 17 23; do
  for condition in no_pivot lin_smol; do
    exp="${condition}_seed${seed}"
    "$PY" -m pivotkk train "configs/smol/${exp}.yaml"
    "$PY" -m pivotkk eval "runs/${exp}"
  done
done
"$PY" -m pivotkk table
