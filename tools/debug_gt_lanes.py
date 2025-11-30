"""Debug GT lane coordinates in training data."""
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from libs.datasets import CulaneDataset
from configs.clrernet.culane.dataset_culane_clrernet import (
    train_al_pipeline,
    crop_bbox,
    img_scale,
)

# Build minimal pipeline: just crop+resize, no augmentation
val_pipeline = [
    dict(type="albumentation", pipelines=[
        dict(type="Compose", params=dict(bboxes=False, keypoints=True, masks=True)),
        dict(
            type="Crop",
            x_min=crop_bbox[0],
            x_max=crop_bbox[2],
            y_min=crop_bbox[1],
            y_max=crop_bbox[3],
            p=1,
        ),
        dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
    ])
]

dataset = CulaneDataset(
    data_root="dataset",
    data_list="dataset/list/train_gt.txt",
    pipeline=val_pipeline,
    test_mode=False,
)

print(f"Dataset size: {len(dataset)}")

# Get first sample with lanes
for idx in range(min(10, len(dataset))):
    data = dataset[idx]

    gt_points = data.get("gt_points", [])
    if len(gt_points) > 0:
        print(f"\n{'='*80}")
        print(f"SAMPLE {idx}: {len(gt_points)} lanes found")
        print(f"{'='*80}")

        img = data["img"]
        print(f"\nImage shape: {img.shape} (H, W, C)")

        for lane_idx, lane in enumerate(gt_points[:3]):  # First 3 lanes
            lane = np.array(lane).reshape(-1, 2)
            print(f"\nLane {lane_idx}:")
            print(f"  Points: {len(lane)}")
            print(f"  X range: [{lane[:, 0].min():.1f}, {lane[:, 0].max():.1f}]")
            print(f"  Y range: [{lane[:, 1].min():.1f}, {lane[:, 1].max():.1f}]")
            print(f"  First 5 points:")
            for i in range(min(5, len(lane))):
                print(f"    ({lane[i, 0]:.1f}, {lane[i, 1]:.1f})")

            # Check if lane is diagonal/vertical (not horizontal)
            y_span = lane[:, 1].max() - lane[:, 1].min()
            x_span = lane[:, 0].max() - lane[:, 0].min()
            print(f"  Span: X={x_span:.1f}, Y={y_span:.1f}")

            if y_span < 50:
                print(f"  ⚠️  WARNING: Lane is too horizontal! Y span = {y_span:.1f}")
            else:
                print(f"  ✅ Lane looks vertical/diagonal")

        # Visualize first sample
        if idx == 0:
            vis_img = img.copy()
            for lane in gt_points:
                lane = np.array(lane).reshape(-1, 2).astype(np.int32)
                cv2.polylines(vis_img, [lane], False, (0, 255, 0), 2)

            out_path = "work_dirs/debug_gt_lanes.png"
            cv2.imwrite(out_path, vis_img)
            print(f"\n✅ Saved visualization: {out_path}")

        break  # Only check first sample with lanes

print("\n" + "="*80)
print("CROP BBOX INFO:")
print("="*80)
print(f"Original image: 1640x590")
print(f"Crop bbox: {crop_bbox} (x_min, y_min, x_max, y_max)")
print(f"Crop region: X=[{crop_bbox[0]}, {crop_bbox[2]}], Y=[{crop_bbox[1]}, {crop_bbox[3]}]")
print(f"Crop size: {crop_bbox[2]-crop_bbox[0]}x{crop_bbox[3]-crop_bbox[1]}")
print(f"Resized to: {img_scale}")
