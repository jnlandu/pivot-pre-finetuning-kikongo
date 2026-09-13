#!/usr/bin/env bash
# Bootstrap a fresh Ubuntu GPU VM (Scaleway L4/L40S, or a GCP g2/a2 instance).
#
# Assumes the NVIDIA driver is already present -- both providers' deep-learning
# images ship one. Verify with nvidia-smi before paying for anything else.
set -euo pipefail

nvidia-smi || { echo "no GPU visible -- wrong image?"; exit 1; }

sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git rsync

python3 -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/pip install -q -e .

.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}, {p.total_memory / 1e9:.0f} GB, bf16={torch.cuda.is_bf16_supported()}")
PY

echo
echo "Next:"
echo "  1. rsync data/processed/ from your laptop (do NOT rebuild it here)"
echo "  2. make smoke      # ~2 min, catches config errors before they cost money"
echo "  3. make sweep"
