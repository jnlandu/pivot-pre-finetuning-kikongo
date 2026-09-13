#!/usr/bin/env bash
# Core sweep. Ordered cheapest-first so a partial run still yields a result.
#
# Stage A checkpoints are content-addressed and reused, so re-running this after
# a crash costs nothing for the experiments that already finished.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

EXPERIMENTS=${EXPERIMENTS:-"
00_no_pivot
lin_10k lin_50k lin_100k
swc_10k swc_50k swc_100k
fra_10k fra_50k fra_100k
"}

for exp in $EXPERIMENTS; do
  cfg="configs/experiments/${exp}.yaml"
  [ -f "$cfg" ] || { echo "!! missing $cfg"; continue; }

  if [ -f "runs/${exp}/metrics.json" ]; then
    echo "== $exp already scored, skipping"
    continue
  fi

  echo "== $exp"
  $PY -m pivotkk train "$cfg"
  $PY -m pivotkk eval "runs/${exp}"
done

$PY -m pivotkk table | tee runs/summary.txt
