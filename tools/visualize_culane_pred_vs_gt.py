#!/usr/bin/env python3
"""
Visualize CULane predictions (.lines.txt) vs GT (.lines.txt) on original images.

This is meant for debugging metric drops / inconsistencies:
  - Reads `list_path` (e.g. dataset/list/test_100.txt)
  - Loads image from data_root/<sub_img_name>
  - Loads GT from data_root/<sub_img_name>.lines.txt (if exists)
  - Loads prediction from pred_dir/<sub_img_name>.lines.txt (if exists)
  - Overlays:
      GT: green
      PRED: red
  - Saves to out_dir preserving subfolders
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def parse_lines_txt(path: Path) -> List[np.ndarray]:
    """Return list of lanes; each lane is (N,2) float32 in original pixel coords."""
    if not path.exists():
        return []
    txt = path.read_text().strip()
    if not txt:
        return []
    lanes = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        vals = list(map(float, parts))
        pts = np.array([(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)], dtype=np.float32)
        if len(pts) >= 2:
            lanes.append(pts)
    return lanes


def draw_lanes(img_bgr: np.ndarray, lanes: List[np.ndarray], color: Tuple[int, int, int], thickness: int) -> None:
    h, w = img_bgr.shape[:2]
    for pts in lanes:
        for k in range(len(pts) - 1):
            x1, y1 = int(round(pts[k][0])), int(round(pts[k][1]))
            x2, y2 = int(round(pts[k + 1][0])), int(round(pts[k + 1][1]))
            if 0 <= x1 < w and 0 <= x2 < w and 0 <= y1 < h and 0 <= y2 < h:
                cv2.line(img_bgr, (x1, y1), (x2, y2), color, thickness)


def main():
    p = argparse.ArgumentParser(description="Visualize CULane pred vs GT on original images")
    p.add_argument("--data-root", default="dataset", help="CULane root containing images and GT .lines.txt")
    p.add_argument("--list-path", required=True, help="List file (e.g. dataset/list/test_100.txt)")
    p.add_argument("--pred-dir", required=True, help="Prediction directory containing *.lines.txt (same relative layout)")
    p.add_argument("--out-dir", required=True, help="Output directory for overlay images")
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--thickness", type=int, default=2)
    p.add_argument("--no-gt", action="store_true", help="Do not draw GT")
    p.add_argument("--no-pred", action="store_true", help="Do not draw predictions")
    args = p.parse_args()

    data_root = Path(args.data_root)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [ln.strip() for ln in Path(args.list_path).read_text().splitlines() if ln.strip()]
    items = []
    for ln in lines:
        sub = ln.split()[0]
        if sub.startswith("/"):
            sub = sub[1:]
        items.append(sub)

    end = min(len(items), int(args.start_idx) + int(args.max_samples))
    sel = items[int(args.start_idx) : end]
    print(f"total_list={len(items)} start={args.start_idx} end={end} saving={len(sel)}")

    saved = 0
    for sub in sel:
        img_path = data_root / sub
        if not img_path.exists():
            print(f"[WARN] missing image: {img_path}")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] failed to read: {img_path}")
            continue

        gt_path = img_path.with_suffix(".lines.txt")
        pred_path = (pred_dir / sub).with_suffix(".lines.txt")

        if not args.no_gt:
            gt_lanes = parse_lines_txt(gt_path)
            draw_lanes(img, gt_lanes, (0, 255, 0), int(args.thickness))
        if not args.no_pred:
            pred_lanes = parse_lines_txt(pred_path)
            draw_lanes(img, pred_lanes, (0, 0, 255), int(args.thickness))

        out_path = (out_dir / sub).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), img)
        if not ok:
            print(f"[WARN] failed to write: {out_path}")
            continue
        saved += 1

    print(f"Saved {saved} overlays to: {out_dir}")


if __name__ == "__main__":
    main()


