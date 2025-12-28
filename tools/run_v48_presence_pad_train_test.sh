#!/usr/bin/env bash
set -euo pipefail

# V48: presence + pad/EOS öğretimi (A çözümü) train + reproducible test

ROOT="/home/alki/projects/clrernetlm"
PY="/home/alki/miniconda3/envs/clrernet/bin/python"

cd "$ROOT"

WORK_DIR="work_dirs/v48_overfit32_presence_pad"
TEST_WORK_DIR="work_dirs/v48_test100_overfit32_presence_pad_repro"

mkdir -p "$WORK_DIR"

echo "== TRAIN =="
$PY tools/train_lanelm_v4_fixed.py \
  --list-path dataset/list/train_2k.txt \
  --overfit-size 32 \
  --epochs 100 \
  --num-pseudo-points 0 \
  --presence-weight 1.0 \
  --pad-loss-weight 1.0 \
  --ss-max-prob 0.2 \
  --ar-rollout-max-weight 0.05 \
  --ar-rollout-min-weight 0.02 \
  --x-embedding-scale 1.0 \
  --lane-embedding-boost 1.0 \
  --work-dir "$WORK_DIR" \
  --device cuda | tee "$WORK_DIR/train.log"

echo "== TEST (reproducible, no stale predictions) =="
$PY tools/test_lanelm_runner.py \
  configs/lanelm/lanelm_v4_culane_test_v48_overfit32_presence_pad.py \
  --work-dir "$TEST_WORK_DIR" \
  --clean-preds \
  --seed 0 \
  --no-parallel-metric

echo "Done."


