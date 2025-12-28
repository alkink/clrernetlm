#!/usr/bin/env python3
"""
GT Round-Trip IoU Debug (Model-free)

Amaç:
  Tokenizer + coord transform + CULane metric hattını model olmadan test etmek.

Akış (her örnek için):
  Dataset'in resized GT lane'leri -> encode -> decode -> coords_to_lane_normalized -> CULane format
  ve original GT (.lines.txt) ile culane_metric üzerinden karşılaştırma.

Önemli:
  "IoU ~ 1.0 olmalı" beklentisi pratikte garanti değil; çünkü:
  - GT noktaları seyrek, biz fixed T=40 y-sample yapıyoruz
  - Metric kendi interpolation + lane width çizimini yapıyor
  - GT'de görüntü dışı x'ler bulunabiliyor; tokenizer clamp/skip uyguluyor
Bu script bir "pipeline sanity" ve üst sınır ölçümüdür.
"""

import argparse
import os
from typing import List, Tuple

import numpy as np

from configs.clrernet.culane.dataset_culane_clrernet import compose_cfg, crop_bbox, img_scale
from libs.datasets import CulaneDataset
from libs.datasets.metrics.culane_metric import load_culane_img_data, culane_metric
from libs.models.detectors.lanelm_detector import coords_to_lane_normalized
from libs.models.lanelm import LaneTokenizer, LaneTokenizerConfig


clean_pipeline = [
    dict(type="Compose", params=compose_cfg),
    dict(
        type="Crop",
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]


def _prediction_string_from_lanes(lanes, ori_w=1640, ori_h=590, y_step=2) -> str:
    """CULaneMetric.get_prediction_string ile aynı mantık, dosyaya yazmadan."""
    ys = np.arange(0, ori_h, y_step) / ori_h
    out = []
    for lane in lanes:
        lane_min_y = lane.min_y - 0.05
        lane_max_y = lane.max_y + 0.05
        ys_in_range = ys[(ys >= lane_min_y) & (ys <= lane_max_y)]
        if len(ys_in_range) < 2:
            continue
        xs = lane(ys_in_range)
        valid_mask = (xs >= 0) & (xs < 1)
        xs = xs * ori_w
        lane_xs = xs[valid_mask]
        lane_ys = ys_in_range[valid_mask] * ori_h
        if len(lane_xs) < 2:
            continue
        lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
        lane_str = " ".join(
            ["{:.5f} {:.5f}".format(x, y) for x, y in zip(lane_xs, lane_ys)]
        )
        if lane_str:
            out.append(lane_str)
    return "\n".join(out) if out else ""


def _parse_pred_string(pred_string: str) -> List[List[Tuple[float, float]]]:
    pred_data: List[List[Tuple[float, float]]] = []
    for line in pred_string.split("\n"):
        if not line.strip():
            continue
        coords = line.split()
        lane = [
            (float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords), 2)
        ]
        pred_data.append(lane)
    return pred_data


def main():
    p = argparse.ArgumentParser(description="GT round-trip IoU debug (model-free)")
    p.add_argument("--list", dest="list_path", required=True, help="e.g. dataset/list/test_100.txt")
    p.add_argument("--data-root", default="dataset")
    p.add_argument("--num-samples", type=int, default=10, help="GT içeren örnek sayısı")
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--smooth", action="store_true", help="Tokenizer decode smoothing aç (ablation)")
    args = p.parse_args()

    tokenizer = LaneTokenizer(
        LaneTokenizerConfig(
            img_w=800,
            img_h=320,
            num_steps=40,
            nbins_x=800,
            x_mode="absolute",
        )
    )

    dataset = CulaneDataset(
        data_root=args.data_root,
        data_list=args.list_path,
        pipeline=[dict(type="albumentation", pipelines=clean_pipeline)],
        diff_file=None,
        test_mode=False,  # gt_points lazım
    )

    if args.start_idx >= len(dataset):
        raise SystemExit(f"start-idx {args.start_idx} >= dataset size {len(dataset)}")

    iou_thresholds = [0.1, 0.5, 0.75]
    tp = [0, 0, 0]
    fp = [0, 0, 0]
    n_gts = 0
    n_used = 0

    idx = args.start_idx
    while idx < len(dataset) and n_used < args.num_samples:
        sample = dataset[idx]
        sub_img_name = sample.get("sub_img_name", f"sample_{idx}")
        gt_points_resized = sample.get("gt_points", [])

        gt_file = os.path.join(args.data_root, sub_img_name.replace(".jpg", ".lines.txt"))
        if not os.path.exists(gt_file):
            print(f"[SKIP] missing GT file: {gt_file}")
            idx += 1
            continue

        anno = load_culane_img_data(gt_file)
        if len(anno) == 0:
            print(f"[SKIP] no GT lanes in file: {sub_img_name}")
            idx += 1
            continue

        lanes_pred = []
        for lane_pts in gt_points_resized[:4]:
            if len(lane_pts) < 2:
                continue
            pts = np.array(lane_pts, dtype=np.float32).reshape(-1, 2)
            x_tokens, y_tokens = tokenizer.encode_single_lane(pts)
            coords_resized = tokenizer.decode_single_lane(
                x_tokens, y_tokens, smooth=bool(args.smooth)
            )
            lane = coords_to_lane_normalized(
                coords_resized=coords_resized,
                tokenizer_cfg=tokenizer.cfg,
                crop_bbox=tuple(crop_bbox),
                img_w=800,
                img_h=320,
                ori_img_w=1640,
                ori_img_h=590,
            )
            if lane is not None and lane.points is not None and lane.points.shape[0] >= 2:
                lanes_pred.append(lane)

        pred_str = _prediction_string_from_lanes(lanes_pred)
        pred = _parse_pred_string(pred_str)

        res = culane_metric(
            pred=pred,
            anno=anno,
            cat="test_all",
            width=30,
            iou_thresholds=iou_thresholds,
            img_shape=(590, 1640, 3),
        )

        n_gts += int(res["n_gt"])
        for k in range(len(iou_thresholds)):
            hits = res["hits"][k]
            tp[k] += int(np.sum(hits))
            fp[k] += int(len(hits) - np.sum(hits))

        n_used += 1
        print(
            f"[{idx}] {sub_img_name} | GT={len(anno)} pred={len(pred)} "
            f"TP@0.5={int(np.sum(res['hits'][1]))}/{len(pred)}"
        )

        idx += 1

    print("\n=== GT ROUND-TRIP SUMMARY ===")
    print(f"samples_used={n_used}  total_gt_lanes={n_gts}  smooth={bool(args.smooth)}")
    if n_used == 0 or n_gts == 0:
        print("GT içeren örnek bulunamadı (bu aralık noline olabilir). start-idx artırıp tekrar dene.")
        return

    eps = 1e-8
    for k, thr in enumerate(iou_thresholds):
        prec = tp[k] / (tp[k] + fp[k] + eps)
        rec = tp[k] / (n_gts + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        print(f"IoU={thr}: TP={tp[k]} FP={fp[k]} Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")


if __name__ == "__main__":
    main()


