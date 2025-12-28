#!/usr/bin/env python3
"""
Debug real prediction vs GT to find the root cause.
Check if predictions are in wrong location or wrong format.
"""
import argparse
import os
import numpy as np
import cv2
from pathlib import Path

from libs.datasets.metrics.culane_metric import load_culane_img_data
from libs.utils.visualizer import draw_lane
from libs.utils.lane_utils import interp

def debug_real_prediction_vs_gt(gt_path, pred_path, img_path, sample_name, save_dir):
    """Debug real prediction vs GT with actual image."""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"REAL PREDICTION vs GT DEBUG: {sample_name}")
    print(f"{'='*80}\n")
    
    # Load GT and Prediction
    gt_data = load_culane_img_data(gt_path)
    pred_data = load_culane_img_data(pred_path)
    
    # Load image
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"ERROR: Could not load image: {img_path}")
            return
        print(f"Image shape: {img.shape}")
    else:
        print(f"WARNING: Image not found: {img_path}")
        img = np.zeros((590, 1640, 3), dtype=np.uint8)
    
    print(f"\nGT lanes: {len(gt_data)}")
    for i, lane in enumerate(gt_data[:4]):
        if len(lane) >= 2:
            lane_arr = np.array(lane)
            xs = lane_arr[:, 0]
            ys = lane_arr[:, 1]
            print(f"  GT Lane {i}: {len(lane)} points")
            print(f"    X: [{xs.min():.1f}, {xs.max():.1f}]")
            print(f"    Y: [{ys.min():.1f}, {ys.max():.1f}]")
            if xs.min() < 0:
                print(f"    ⚠️  NEGATIVE X: {xs.min():.1f}")
            if xs.max() >= 1640:
                print(f"    ⚠️  X > 1640: {xs.max():.1f}")
    
    print(f"\nPred lanes: {len(pred_data)}")
    for i, lane in enumerate(pred_data[:4]):
        if len(lane) >= 2:
            lane_arr = np.array(lane)
            xs = lane_arr[:, 0]
            ys = lane_arr[:, 1]
            print(f"  Pred Lane {i}: {len(lane)} points")
            print(f"    X: [{xs.min():.1f}, {xs.max():.1f}]")
            print(f"    Y: [{ys.min():.1f}, {ys.max():.1f}]")
    
    # Visualize on image
    vis_img = img.copy()
    
    # Draw GT (GREEN)
    for i, lane in enumerate(gt_data[:4]):
        if len(lane) >= 2:
            lane_arr = np.array(lane, dtype=np.int32)
            # Clip to image bounds
            lane_arr[:, 0] = np.clip(lane_arr[:, 0], 0, img.shape[1] - 1)
            lane_arr[:, 1] = np.clip(lane_arr[:, 1], 0, img.shape[0] - 1)
            cv2.polylines(vis_img, [lane_arr], isClosed=False, color=(0, 255, 0), thickness=3)
    
    # Draw Pred (RED)
    for i, lane in enumerate(pred_data[:4]):
        if len(lane) >= 2:
            lane_arr = np.array(lane, dtype=np.int32)
            # Clip to image bounds
            lane_arr[:, 0] = np.clip(lane_arr[:, 0], 0, img.shape[1] - 1)
            lane_arr[:, 1] = np.clip(lane_arr[:, 1], 0, img.shape[0] - 1)
            cv2.polylines(vis_img, [lane_arr], isClosed=False, color=(0, 0, 255), thickness=2)
    
    # Calculate IoU
    print(f"\n--- IoU Calculation ---")
    interp_gt = np.array([interp(lane, n=5) for lane in gt_data], dtype=object)
    interp_pred = np.array([interp(lane, n=5) for lane in pred_data], dtype=object)
    
    img_shape = (590, 1640, 3)
    width = 30
    
    gt_masks = [draw_lane(lane, img=None, img_shape=img_shape, width=width) > 0 for lane in interp_gt]
    pred_masks = [draw_lane(lane, img=None, img_shape=img_shape, width=width) > 0 for lane in interp_pred]
    
    print(f"GT masks: {len(gt_masks)}, Pred masks: {len(pred_masks)}")
    
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            intersection = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()
            iou = intersection / union if union > 0 else 0.0
            print(f"  Pred {i} vs GT {j}: IoU = {iou:.6f}")
            if iou > 0.1:
                print(f"    ✅ IoU > 0.1")
            if iou > 0.5:
                print(f"    ✅✅ IoU > 0.5")
    
    # Save visualization
    save_path = os.path.join(save_dir, f"debug_{sample_name.replace('/', '_')}.jpg")
    cv2.imwrite(save_path, vis_img)
    print(f"\nSaved visualization to: {save_path}")
    print(f"  GREEN = GT, RED = Prediction")
    
    # Check if predictions are in completely wrong location
    print(f"\n--- Location Analysis ---")
    if len(gt_data) > 0 and len(pred_data) > 0:
        gt_center_x = np.array([np.array(lane)[:, 0].mean() for lane in gt_data if len(lane) >= 2])
        pred_center_x = np.array([np.array(lane)[:, 0].mean() for lane in pred_data if len(lane) >= 2])
        
        print(f"GT center X: {gt_center_x}")
        print(f"Pred center X: {pred_center_x}")
        
        if len(gt_center_x) > 0 and len(pred_center_x) > 0:
            avg_gt_x = gt_center_x.mean()
            avg_pred_x = pred_center_x.mean()
            print(f"Average GT X: {avg_gt_x:.1f}")
            print(f"Average Pred X: {avg_pred_x:.1f}")
            print(f"Difference: {abs(avg_gt_x - avg_pred_x):.1f} px")
            
            if abs(avg_gt_x - avg_pred_x) > 200:
                print(f"  ⚠️  Predictions are in WRONG LOCATION!")
                print(f"  Average difference: {abs(avg_gt_x - avg_pred_x):.1f} px")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Debug real prediction vs GT")
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--img-file", required=True)
    parser.add_argument("--sample-name", required=True)
    parser.add_argument("--save-dir", default="work_dirs/debug_real_prediction_vs_gt")
    args = parser.parse_args()
    
    debug_real_prediction_vs_gt(
        gt_path=args.gt_file,
        pred_path=args.pred_file,
        img_path=args.img_file,
        sample_name=args.sample_name,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()








