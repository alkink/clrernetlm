#!/usr/bin/env python3
"""
Analyze GT lane counts in training vs test datasets.
This helps identify if the model is learning to always predict 4 lanes.
"""
import argparse
from collections import Counter
from libs.datasets import CulaneDataset
from configs.clrernet.culane.dataset_culane_clrernet import (
    compose_cfg, crop_bbox, img_scale
)

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


def analyze_gt_lane_counts(data_root, list_path, dataset_name):
    """Analyze GT lane counts."""
    print(f"\n{'='*80}")
    print(f"GT LANE COUNTS ANALYSIS: {dataset_name}")
    print(f"{'='*80}\n")
    
    dataset = CulaneDataset(
        data_root=data_root,
        data_list=list_path,
        pipeline=[dict(type="albumentation", pipelines=clean_pipeline)],
        diff_file=None,
        test_mode=False,
    )
    
    lane_counts = []
    for i, sample in enumerate(dataset):
        gt_points = sample['gt_points']
        valid_lanes = [l for l in gt_points if len(l) >= 4]
        lane_counts.append(len(valid_lanes))
    
    counter = Counter(lane_counts)
    
    print(f"Total samples: {len(dataset)}")
    print(f"\nLane count distribution:")
    for count in sorted(counter.keys()):
        num_samples = counter[count]
        percentage = (num_samples / len(dataset)) * 100
        print(f"  {count} lanes: {num_samples} samples ({percentage:.1f}%)")
    
    print(f"\nStatistics:")
    print(f"  Mean: {sum(lane_counts) / len(lane_counts):.2f}")
    print(f"  Min: {min(lane_counts)}")
    print(f"  Max: {max(lane_counts)}")
    
    # Check if always 4 lanes
    if counter.get(4, 0) == len(dataset):
        print(f"\n  ⚠️  WARNING: ALL samples have exactly 4 lanes!")
        print(f"  This might cause the model to always predict 4 lanes!")
    elif counter.get(4, 0) / len(dataset) > 0.8:
        print(f"\n  ⚠️  WARNING: {counter.get(4, 0) / len(dataset) * 100:.1f}% of samples have 4 lanes!")
        print(f"  This might bias the model to always predict 4 lanes!")
    
    return counter


def main():
    parser = argparse.ArgumentParser(description="Analyze GT lane counts")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--train-list", default="dataset/list/train_100.txt")
    parser.add_argument("--test-list", default="dataset/list/test_100.txt")
    args = parser.parse_args()
    
    train_counter = analyze_gt_lane_counts(args.data_root, args.train_list, "TRAINING")
    test_counter = analyze_gt_lane_counts(args.data_root, args.test_list, "TEST")
    
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}\n")
    
    print("Training vs Test lane count distribution:")
    all_counts = set(train_counter.keys()) | set(test_counter.keys())
    for count in sorted(all_counts):
        train_num = train_counter.get(count, 0)
        test_num = test_counter.get(count, 0)
        print(f"  {count} lanes: Train={train_num}, Test={test_num}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()








