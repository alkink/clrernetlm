#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/alki/projects/clrernetlm"
PY="/home/alki/miniconda3/envs/clrernet/bin/python"

LIST_PATH="${1:-dataset/list/train_gt_2k_subset.txt}"
EPOCHS="${2:-50}"
DEVICE="${3:-cuda}"
PAD_W="${4:-0.1}"

cd "$ROOT"

TRAIN_WD="work_dirs/v39_train2k_full_pad01"
TEST_WD="work_dirs/v39_test100_full2k_pad01"
CFG="configs/lanelm/lanelm_v4_culane_test_v39_full2k_pad01.py"

mkdir -p "$TRAIN_WD" "$TEST_WD"

echo "==================== TRAIN full-2k (overfit-size=0) ===================="
$PY tools/train_lanelm_v4_fixed.py \
  --list-path "$LIST_PATH" \
  --overfit-size 0 \
  --epochs "$EPOCHS" \
  --num-pseudo-points 0 \
  --x-embedding-scale 1.0 \
  --lane-embedding-boost 1.0 \
  --ss-max-prob 0.0 \
  --ar-rollout-max-weight 0.0 \
  --ar-rollout-min-weight 0.0 \
  --presence-weight 0.0 \
  --pad-loss-weight "$PAD_W" \
  --work-dir "$TRAIN_WD" \
  --device "$DEVICE" | tee "$TRAIN_WD/train.log"

echo "==================== TEST test_100 ===================="
$PY tools/test_lanelm_runner.py "$CFG" --work-dir "$TEST_WD" | tee "$TEST_WD/test.log"

echo "[V39] DONE. Check $TEST_WD/*json for metrics."


