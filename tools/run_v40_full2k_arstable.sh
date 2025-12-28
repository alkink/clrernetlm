#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/alki/projects/clrernetlm"
PY="/home/alki/miniconda3/envs/clrernet/bin/python"

LIST_PATH="${1:-dataset/list/train_gt_2k_subset.txt}"
EPOCHS="${2:-100}"
DEVICE="${3:-cuda}"

cd "$ROOT"

TRAIN_WD="work_dirs/v40_train2k_full_arstable"
TEST_WD="work_dirs/v40_test100_full2k_arstable"
CFG="configs/lanelm/lanelm_v4_culane_test_v40_full2k_arstable.py"

mkdir -p "$TRAIN_WD" "$TEST_WD"

echo "==================== TRAIN V40 full-2k AR-stable ===================="
$PY tools/train_lanelm_v4_fixed.py \
  --list-path "$LIST_PATH" \
  --overfit-size 0 \
  --epochs "$EPOCHS" \
  --num-pseudo-points 0 \
  --x-embedding-scale 1.0 \
  --lane-embedding-boost 1.0 \
  --ss-max-prob 0.2 \
  --ar-rollout-max-weight 0.05 \
  --ar-rollout-min-weight 0.02 \
  --presence-weight 0.0 \
  --pad-loss-weight 1.0 \
  --work-dir "$TRAIN_WD" \
  --device "$DEVICE" | tee "$TRAIN_WD/train.log"

echo "==================== TEST V40 test_100 ===================="
$PY tools/test_lanelm_runner.py "$CFG" --work-dir "$TEST_WD" | tee "$TEST_WD/test.log"

echo "[V40] DONE. Metrics: $TEST_WD/*/*.json"


