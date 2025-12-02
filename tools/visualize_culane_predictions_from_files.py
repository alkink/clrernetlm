#!/usr/bin/env python3
"""
CULane prediction görselleştirme script'i.

- Girdi: CULane formatında prediction dosyaları (.lines.txt)
  (ör: work_dirs/lanelm_v4_test_fixed_full/predictions/xxx/yyy.lines.txt)
- GT: dataset içindeki orijinal .lines.txt dosyaları
- Çıktı: Crop+resize sonrası görüntü üzerinde
  - YEŞİL: GT şeritler
  - KIRMIZI/MAVİ/MAGENTA/CYAN: Tahmin şeritleri
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from libs.datasets.metrics.culane_metric import load_culane_img_data
from configs.clrernet.culane.dataset_culane_clrernet import (
    compose_cfg,
    crop_bbox,
    img_scale,
)
from libs.datasets import CulaneDataset


def build_clean_dataset(data_root: str, data_list: str):
    """Test pipeline ile CULaneDataset oluştur."""
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
    dataset = CulaneDataset(
        data_root=data_root,
        data_list=data_list,
        pipeline=[dict(type="albumentation", pipelines=clean_pipeline)],
        diff_file=None,
        test_mode=True,
    )
    return dataset


def to_resized_coords(lanes, crop_bbox, ori_w=1640.0, ori_h=590.0, img_w=800.0, img_h=320.0):
    """Orijinal CULane koordinatlarını crop+resize sonrası (800x320) uzaya çevir."""
    x_min, y_min, x_max, y_max = crop_bbox
    crop_h = float(y_max - y_min)  # 320

    resized_lanes = []
    for lane in lanes:
        if not lane or len(lane) < 2:
            continue
        pts_orig = np.array(lane, dtype=np.float32).reshape(-1, 2)

        # Crop: sadece y'den y_min çıkar
        pts_cropped = pts_orig.copy()
        pts_cropped[:, 1] = pts_cropped[:, 1] - y_min

        # Resize: X: 1640 -> 800, Y: 320 -> 320
        pts_resized = pts_cropped.copy()
        pts_resized[:, 0] = pts_resized[:, 0] / ori_w * img_w
        pts_resized[:, 1] = pts_resized[:, 1] / crop_h * img_h

        # Görüntü sınırları içinde kalan noktaları al
        valid_mask = (
            (pts_resized[:, 0] >= 0)
            & (pts_resized[:, 0] < img_w)
            & (pts_resized[:, 1] >= 0)
            & (pts_resized[:, 1] < img_h)
        )
        pts_resized = pts_resized[valid_mask]
        if len(pts_resized) >= 2:
            resized_lanes.append(pts_resized)
    return resized_lanes


def visualize_from_files(pred_root, data_root, data_list, save_dir, max_samples=50):
    os.makedirs(save_dir, exist_ok=True)

    dataset = build_clean_dataset(data_root, data_list)
    print(f"Dataset size: {len(dataset)}")

    # data_list içindeki satırlar üzerinden gidiyoruz
    with open(data_list, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    total = len(lines)
    num = min(max_samples, total)
    print(f"Visualizing first {num} samples from list...")

    for idx in range(num):
        rel_path = lines[idx].split()[0]  # CULane list: 'driver_xx/.../xxx.jpg ...'
        # Leading slash'ı kaldır (CULane format: /driver_xx/...)
        sub_img_name = rel_path.lstrip('/')  # dataset bu path'i kullanıyor

        img_path = os.path.join(data_root, sub_img_name)
        pred_path = os.path.join(pred_root, sub_img_name.replace(".jpg", ".lines.txt"))
        gt_path = os.path.join(data_root, sub_img_name.replace(".jpg", ".lines.txt"))

        if not os.path.exists(img_path):
            print(f"[WARN] Image not found: {img_path}, skipping")
            continue

        # Prediction ve GT dosyalarını yükle
        pred_lanes = load_culane_img_data(pred_path) if os.path.exists(pred_path) else []
        gt_lanes = load_culane_img_data(gt_path) if os.path.exists(gt_path) else []

        if not pred_lanes and not gt_lanes:
            print(f"[INFO] No pred/GT lanes for {sub_img_name}, skipping")
            continue

        # Görüntüyü yükle ve pipeline ile aynı crop+resize uygula
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        # Crop
        x_min, y_min, x_max, y_max = crop_bbox
        img_cropped = img[y_min:y_max, x_min:x_max]
        # Resize
        img_resized = cv2.resize(img_cropped, (int(img_scale[0]), int(img_scale[1])))

        img_vis = img_resized.copy()

        # Pred ve GT'yi resized uzaya taşı
        pred_resized = to_resized_coords(pred_lanes, crop_bbox)
        gt_resized = to_resized_coords(gt_lanes, crop_bbox)

        # Tahminleri çiz (RENKLER)
        pred_colors = [
            (0, 0, 255),    # kırmızı
            (255, 0, 0),    # mavi
            (255, 0, 255),  # magenta
            (0, 255, 255),  # sarı-cyan
        ]
        for l_idx, lane_pts in enumerate(pred_resized):
            color = pred_colors[l_idx % len(pred_colors)]
            for i in range(len(lane_pts) - 1):
                p1 = (int(lane_pts[i][0]), int(lane_pts[i][1]))
                p2 = (int(lane_pts[i + 1][0]), int(lane_pts[i + 1][1]))
                cv2.line(img_vis, p1, p2, color, 2)

        # GT'yi çiz (YEŞİL)
        for lane_pts in gt_resized:
            for i in range(len(lane_pts) - 1):
                p1 = (int(lane_pts[i][0]), int(lane_pts[i][1]))
                p2 = (int(lane_pts[i + 1][0]), int(lane_pts[i + 1][1]))
                cv2.line(img_vis, p1, p2, (0, 255, 0), 3)

        # Metin
        cv2.putText(
            img_vis,
            "GREEN=GT, COLORS=Pred",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img_vis,
            sub_img_name,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Kaydet
        safe_name = sub_img_name.replace("/", "_").replace("\\", "_")
        out_path = os.path.join(save_dir, f"{idx:05d}_{safe_name}.jpg")
        cv2.imwrite(out_path, img_vis)

        if (idx + 1) % 20 == 0:
            print(f"  Saved {idx + 1}/{num} visualizations...")

    print(f"✓ Visualizations saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CULane predictions from .lines.txt files vs GT"
    )
    parser.add_argument(
        "--pred-root",
        required=True,
        help="Prediction root directory (e.g., work_dirs/.../predictions)",
    )
    parser.add_argument(
        "--data-root",
        default="dataset",
        help="CULane dataset root (contains driver_xx/ folders)",
    )
    parser.add_argument(
        "--data-list",
        default="dataset/list/test.txt",
        help="CULane list file used for test (default: test.txt)",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/vis_culane_preds",
        help="Output directory for visualization images",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of samples to visualize",
    )

    args = parser.parse_args()

    visualize_from_files(
        pred_root=args.pred_root,
        data_root=args.data_root,
        data_list=args.data_list,
        save_dir=args.out_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()


