#!/usr/bin/env python3
"""
Visualize predictions from saved .lines.txt files vs ground truth.
Reads prediction files from work_dirs and visualizes them on original images.
"""
import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from libs.datasets.metrics.culane_metric import load_culane_img_data


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize predictions from saved files')
    parser.add_argument(
        '--pred-dir',
        type=str,
        default='work_dirs/lanelm_v4_test_valid_gt_full/predictions',
        help='Directory containing prediction .lines.txt files'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='dataset',
        help='CULane dataset root directory'
    )
    parser.add_argument(
        '--data-list',
        type=str,
        default='dataset/list/test_valid_gt.txt',
        help='Test data list file'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='work_dirs/lanelm_v4_test_valid_gt_full/visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50,
        help='Maximum number of samples to visualize'
    )
    parser.add_argument(
        '--gt-dir',
        type=str,
        default=None,
        help='Ground truth directory (if None, uses same directory as images)'
    )
    return parser.parse_args()


def load_image_paths(data_list_path, data_root):
    """Load image paths from data list file."""
    image_paths = []
    with open(data_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Remove leading '/' if present
            if line.startswith('/'):
                line = line[1:]
            img_path = os.path.join(data_root, line)
            if os.path.exists(img_path):
                image_paths.append((line, img_path))
    return image_paths


def draw_lanes_on_image(img, lanes, color, thickness=3):
    """Draw lanes on image."""
    for lane in lanes:
        if len(lane) < 2:
            continue
        pts = np.array(lane, dtype=np.int32)
        # Draw as polyline
        for i in range(len(pts) - 1):
            cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), color, thickness)
    return img


def visualize_sample(
    img_path,
    pred_path,
    gt_path,
    out_path,
    pred_color=(0, 0, 255),  # Red (BGR)
    gt_color=(0, 255, 0),     # Green (BGR)
):
    """Visualize a single sample: prediction vs ground truth."""
    # Load original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load image {img_path}")
        return False
    
    # Load predictions
    pred_lanes = []
    if pred_path.exists():
        pred_lanes = load_culane_img_data(str(pred_path))
    
    # Load ground truth
    gt_lanes = []
    if gt_path.exists():
        gt_lanes = load_culane_img_data(str(gt_path))
    
    # Draw predictions (red)
    if pred_lanes:
        img = draw_lanes_on_image(img, pred_lanes, pred_color, thickness=3)
    
    # Draw ground truth (green)
    if gt_lanes:
        img = draw_lanes_on_image(img, gt_lanes, gt_color, thickness=2)
    
    # Add text overlay
    text_y = 30
    cv2.putText(img, f"Pred: {len(pred_lanes)} lanes", (10, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
    cv2.putText(img, f"GT: {len(gt_lanes)} lanes", (10, text_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gt_color, 2)
    
    # Save visualization
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def main():
    args = parse_args()
    
    # Load image paths
    print(f"Loading image paths from {args.data_list}...")
    image_paths = load_image_paths(args.data_list, args.data_root)
    print(f"Found {len(image_paths)} images")
    
    # Limit number of samples
    if args.max_samples > 0:
        image_paths = image_paths[:args.max_samples]
    
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize each sample
    success_count = 0
    for img_rel_path, img_abs_path in tqdm(image_paths, desc="Visualizing"):
        # Prediction file path
        pred_path = pred_dir / Path(img_rel_path).with_suffix('.lines.txt')
        
        # GT file path - same directory as image (CULane format)
        if args.gt_dir:
            gt_path = Path(args.gt_dir) / Path(img_rel_path).with_suffix('.lines.txt')
        else:
            # GT is in the same directory as the image
            gt_path = Path(img_abs_path).with_suffix('.lines.txt')
        
        # Output path
        out_path = out_dir / Path(img_rel_path).with_suffix('.png')
        
        if visualize_sample(img_abs_path, pred_path, gt_path, out_path):
            success_count += 1
    
    print(f"\n✓ Successfully visualized {success_count}/{len(image_paths)} samples")
    print(f"✓ Visualizations saved to: {out_dir}")


if __name__ == '__main__':
    main()

