#!/usr/bin/env python3
"""
Step-by-step IoU calculation debug.
This script loads actual GT and prediction files and calculates IoU step by step.
NO ASSUMPTIONS - ONLY EVIDENCE.
"""
import argparse
import os
import numpy as np
import cv2
from pathlib import Path

from libs.datasets.metrics.culane_metric import load_culane_img_data
from libs.utils.visualizer import draw_lane
from libs.utils.lane_utils import interp

def debug_iou_step_by_step(gt_path, pred_path, sample_name, save_dir):
    """Debug IoU calculation step by step with actual data."""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"STEP-BY-STEP IoU DEBUG: {sample_name}")
    print(f"{'='*80}\n")
    
    # ========== STEP 1: Load GT and Prediction ==========
    print("STEP 1: Loading GT and Prediction files")
    print("-" * 80)
    
    if not os.path.exists(gt_path):
        print(f"ERROR: GT file not found: {gt_path}")
        return
    
    if not os.path.exists(pred_path):
        print(f"ERROR: Prediction file not found: {pred_path}")
        return
    
    gt_data = load_culane_img_data(gt_path)
    pred_data = load_culane_img_data(pred_path)
    
    print(f"GT file: {gt_path}")
    print(f"  GT lanes: {len(gt_data)}")
    for i, lane in enumerate(gt_data[:4]):
        if len(lane) >= 2:
            lane_arr = np.array(lane)
            xs = lane_arr[:, 0]
            ys = lane_arr[:, 1]
            print(f"  GT Lane {i}: {len(lane)} points")
            print(f"    X: [{xs.min():.1f}, {xs.max():.1f}] (min={xs.min():.1f}, max={xs.max():.1f})")
            print(f"    Y: [{ys.min():.1f}, {ys.max():.1f}] (min={ys.min():.1f}, max={ys.max():.1f})")
            if xs.min() < 0:
                print(f"    ⚠️  NEGATIVE X: {xs.min():.1f}")
            if xs.max() >= 1640:
                print(f"    ⚠️  X > 1640: {xs.max():.1f}")
    
    print(f"\nPrediction file: {pred_path}")
    print(f"  Pred lanes: {len(pred_data)}")
    for i, lane in enumerate(pred_data[:4]):
        if len(lane) >= 2:
            lane_arr = np.array(lane)
            xs = lane_arr[:, 0]
            ys = lane_arr[:, 1]
            print(f"  Pred Lane {i}: {len(lane)} points")
            print(f"    X: [{xs.min():.1f}, {xs.max():.1f}] (min={xs.min():.1f}, max={xs.max():.1f})")
            print(f"    Y: [{ys.min():.1f}, {ys.max():.1f}] (min={ys.min():.1f}, max={ys.max():.1f})")
            if xs.min() < 0:
                print(f"    ⚠️  NEGATIVE X: {xs.min():.1f}")
            if xs.max() >= 1640:
                print(f"    ⚠️  X > 1640: {xs.max():.1f}")
    
    # ========== STEP 2: Interpolate ==========
    print(f"\nSTEP 2: Interpolating lanes (as CULaneMetric does)")
    print("-" * 80)
    
    interp_gt = []
    interp_pred = []
    
    for i, lane in enumerate(gt_data[:4]):
        if len(lane) >= 2:
            interp_lane = interp(lane, n=5)
            interp_gt.append(interp_lane)
            print(f"  GT Lane {i}: {len(lane)} → {len(interp_lane)} points (interpolated)")
    
    for i, lane in enumerate(pred_data[:4]):
        if len(lane) >= 2:
            interp_lane = interp(lane, n=5)
            interp_pred.append(interp_lane)
            print(f"  Pred Lane {i}: {len(lane)} → {len(interp_lane)} points (interpolated)")
    
    if len(interp_gt) == 0 or len(interp_pred) == 0:
        print("ERROR: No lanes to compare")
        return
    
    # ========== STEP 3: Draw lanes (as discrete_cross_iou does) ==========
    print(f"\nSTEP 3: Drawing lanes (as discrete_cross_iou does)")
    print("-" * 80)
    
    img_shape = (590, 1640, 3)
    width = 30
    
    # Draw GT lanes
    gt_masks = []
    for i, lane in enumerate(interp_gt):
        mask = draw_lane(lane, img=None, img_shape=img_shape, width=width) > 0
        gt_masks.append(mask)
        pixel_count = mask.sum()
        print(f"  GT Lane {i} mask: {pixel_count} pixels")
        if pixel_count == 0:
            print(f"    ⚠️  WARNING: GT Lane {i} has 0 pixels!")
    
    # Draw Pred lanes
    pred_masks = []
    for i, lane in enumerate(interp_pred):
        mask = draw_lane(lane, img=None, img_shape=img_shape, width=width) > 0
        pred_masks.append(mask)
        pixel_count = mask.sum()
        print(f"  Pred Lane {i} mask: {pixel_count} pixels")
        if pixel_count == 0:
            print(f"    ⚠️  WARNING: Pred Lane {i} has 0 pixels!")
    
    # ========== STEP 4: Calculate IoU (as discrete_cross_iou does) ==========
    print(f"\nSTEP 4: Calculating IoU (as discrete_cross_iou does)")
    print("-" * 80)
    
    ious = np.zeros((len(pred_masks), len(gt_masks)))
    
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            intersection = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()
            
            if union == 0:
                iou = 0.0
            else:
                iou = intersection / union
            
            ious[i, j] = iou
            
            print(f"  Pred Lane {i} vs GT Lane {j}:")
            print(f"    Intersection: {intersection} pixels")
            print(f"    Union: {union} pixels")
            print(f"    IoU: {iou:.6f}")
            
            if iou == 0.0:
                print(f"    ⚠️  IoU = 0.0!")
                # Debug why IoU is 0
                pred_pixels = pred_mask.sum()
                gt_pixels = gt_mask.sum()
                print(f"      Pred pixels: {pred_pixels}, GT pixels: {gt_pixels}")
                if pred_pixels == 0:
                    print(f"      → Pred has 0 pixels!")
                if gt_pixels == 0:
                    print(f"      → GT has 0 pixels!")
                if pred_pixels > 0 and gt_pixels > 0:
                    # Check if they overlap at all
                    overlap_check = (pred_mask & gt_mask).any()
                    print(f"      → Any overlap: {overlap_check}")
                    if not overlap_check:
                        print(f"      → NO OVERLAP - lanes are in different locations!")
    
    # ========== STEP 5: Visualize masks ==========
    print(f"\nSTEP 5: Visualizing masks")
    print("-" * 80)
    
    # Create visualization
    vis_img = np.zeros((590, 1640, 3), dtype=np.uint8)
    
    # Draw GT masks (RED) - use proper indexing
    for i, mask in enumerate(gt_masks):
        vis_img[mask] = [0, 0, 255]  # BGR: Red
    
    # Draw Pred masks (GREEN) - use proper indexing  
    for i, mask in enumerate(pred_masks):
        vis_img[mask] = [0, 255, 0]  # BGR: Green
    
    # Overlap = YELLOW (cyan)
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            overlap = pred_mask & gt_mask
            vis_img[overlap] = [255, 255, 0]  # BGR: Cyan (yellow)
    
    save_path = os.path.join(save_dir, f"iou_debug_{sample_name.replace('/', '_')}.jpg")
    cv2.imwrite(save_path, vis_img)
    print(f"  Saved visualization to: {save_path}")
    print(f"  Colors: RED=GT, GREEN=Pred, YELLOW=Overlap")
    
    # ========== STEP 6: Summary ==========
    print(f"\nSTEP 6: Summary")
    print("-" * 80)
    print(f"IoU Matrix:")
    print(ious)
    print(f"\nMax IoU per GT lane:")
    for j in range(len(gt_masks)):
        max_iou = ious[:, j].max()
        best_pred = ious[:, j].argmax()
        print(f"  GT Lane {j}: Max IoU = {max_iou:.6f} (with Pred Lane {best_pred})")
    
    print(f"\nMax IoU per Pred lane:")
    for i in range(len(pred_masks)):
        max_iou = ious[i, :].max()
        best_gt = ious[i, :].argmax()
        print(f"  Pred Lane {i}: Max IoU = {max_iou:.6f} (with GT Lane {best_gt})")
    
    # Check if any IoU > 0.5
    iou_05_count = (ious > 0.5).sum()
    print(f"\nIoU > 0.5: {iou_05_count} pairs")
    if iou_05_count == 0:
        print(f"  ⚠️  NO IoU > 0.5 - This explains F1=0.0000 at IoU 0.5!")
    
    print(f"\n{'='*80}")
    print(f"DEBUG COMPLETE")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Step-by-step IoU debug")
    parser.add_argument("--gt-file", required=True, help="GT .lines.txt file path")
    parser.add_argument("--pred-file", required=True, help="Prediction .lines.txt file path")
    parser.add_argument("--sample-name", required=True, help="Sample name for output")
    parser.add_argument("--save-dir", default="work_dirs/debug_iou_step_by_step")
    args = parser.parse_args()
    
    debug_iou_step_by_step(
        gt_path=args.gt_file,
        pred_path=args.pred_file,
        sample_name=args.sample_name,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()

