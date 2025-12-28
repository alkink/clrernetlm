zx#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/alki/projects/clrernetlm"
PY="/home/alki/miniconda3/envs/clrernet/bin/python"

LIST_PATH="${1:-dataset/list/train_gt_2k_subset.txt}"
EPOCHS="${2:-200}"
DEVICE="${3:-cuda}"
PAD_W="${4:-0.1}"

cd "$ROOT"

echo "[V38] list_path=$LIST_PATH epochs=$EPOCHS device=$DEVICE pad_loss_weight=$PAD_W"

train_one () {
  local N="$1"
  local WD="work_dirs/v38_train2k_overfit${N}_pad01"
  mkdir -p "$WD"
  echo "==================== TRAIN overfit-size=$N -> $WD ===================="
  $PY tools/train_lanelm_v4_fixed.py \
    --list-path "$LIST_PATH" \
    --overfit-size "$N" \
    --epochs "$EPOCHS" \
    --num-pseudo-points 0 \
    --x-embedding-scale 1.0 \
    --lane-embedding-boost 1.0 \
    --ss-max-prob 0.0 \
    --ar-rollout-max-weight 0.0 \
    --ar-rollout-min-weight 0.0 \
    --presence-weight 0.0 \
    --pad-loss-weight "$PAD_W" \
    --work-dir "$WD" \
    --device "$DEVICE" | tee "$WD/train.log"
}

test_one () {
  local N="$1"
  local CFG="configs/lanelm/lanelm_v4_culane_test_v38_overfit${N}_2k_pad01.py"
  local WD="work_dirs/v38_test100_overfit${N}_pad01"
  mkdir -p "$WD"
  echo "==================== TEST overfit-size=$N -> $WD ===================="
  $PY tools/test_lanelm_runner.py "$CFG" --work-dir "$WD" | tee "$WD/test.log"
}

for N in 1 4 8; do
  train_one "$N"
  test_one "$N"
done

echo "[V38] DONE. Check work_dirs/v38_test100_overfit*_pad01/*json for metrics."


