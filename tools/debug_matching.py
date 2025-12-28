#!/usr/bin/env python3
"""
Debug linear_sum_assignment matching in CULaneMetric.
This helps identify if lane matching is the issue.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from libs.datasets.metrics.culane_metric import load_culane_img_data, interp, discrete_cross_iou

def debug_matching(gt_path, pred_path):
    """Debug lane matching step by step."""
    print(f"\n{'='*80}")
    print(f"LANE MATCHING DEBUG")
    print(f"{'='*80}\n")
    
    # Load GT and Prediction
    gt_data = load_culane_img_data(gt_path)
    pred_data = load_culane_img_data(pred_path)
    
    print(f"GT lanes: {len(gt_data)}, Pred lanes: {len(pred_data)}\n")
    
    # Interpolate (as CULaneMetric does)
    interp_gt = np.array([interp(lane, n=5) for lane in gt_data], dtype=object)
    interp_pred = np.array([interp(lane, n=5) for lane in pred_data], dtype=object)
    
    # Calculate IoU matrix (as culane_metric does)
    img_shape = (590, 1640, 3)
    width = 30
    ious = discrete_cross_iou(interp_pred, interp_gt, width=width, img_shape=img_shape)
    
    print("IoU Matrix:")
    print(ious)
    print()
    
    # Linear sum assignment (as culane_metric does)
    row_ind, col_ind = linear_sum_assignment(1 - ious)
    
    print("Linear Sum Assignment:")
    print(f"  Row indices (pred): {row_ind}")
    print(f"  Col indices (GT): {col_ind}")
    print()
    
    # Calculate matched IoUs
    pred_ious = np.zeros(len(pred_data))
    pred_ious[row_ind] = ious[row_ind, col_ind]
    
    print("Matched IoUs:")
    for i, (pred_idx, gt_idx) in enumerate(zip(row_ind, col_ind)):
        iou_val = ious[pred_idx, gt_idx]
        pred_ious[pred_idx] = iou_val
        print(f"  Pred Lane {pred_idx} ↔ GT Lane {gt_idx}: IoU = {iou_val:.6f}")
    
    print()
    print("All Pred IoUs (after matching):")
    for i, iou_val in enumerate(pred_ious):
        print(f"  Pred Lane {i}: IoU = {iou_val:.6f}")
    
    # Check hits at different thresholds
    print()
    print("Hits at different IoU thresholds:")
    for thr in [0.1, 0.5, 0.75]:
        hits = pred_ious > thr
        hit_count = hits.sum()
        print(f"  IoU > {thr}: {hit_count}/{len(pred_data)} predictions hit")
        if hit_count > 0:
            hit_indices = np.where(hits)[0]
            print(f"    Hit predictions: {hit_indices}")
            for idx in hit_indices:
                matched_gt = col_ind[np.where(row_ind == idx)[0][0]] if idx in row_ind else None
                print(f"      Pred {idx} (IoU={pred_ious[idx]:.6f}) matched with GT {matched_gt}")
    
    # Analyze why IoU is low
    print()
    print("IoU Analysis:")
    for i, (pred_idx, gt_idx) in enumerate(zip(row_ind, col_ind)):
        iou_val = ious[pred_idx, gt_idx]
        if iou_val < 0.5:
            print(f"  Pred {pred_idx} vs GT {gt_idx}: IoU={iou_val:.6f} < 0.5")
            # Get original coordinates for analysis
            pred_lane = pred_data[pred_idx]
            gt_lane = gt_data[gt_idx]
            pred_arr = np.array(pred_lane)
            gt_arr = np.array(gt_lane)
            
            print(f"    Pred X: [{pred_arr[:, 0].min():.1f}, {pred_arr[:, 0].max():.1f}], Y: [{pred_arr[:, 1].min():.1f}, {pred_arr[:, 1].max():.1f}]")
            print(f"    GT X: [{gt_arr[:, 0].min():.1f}, {gt_arr[:, 0].max():.1f}], Y: [{gt_arr[:, 1].min():.1f}, {gt_arr[:, 1].max():.1f}]")
            
            # Calculate center distance
            pred_center_x = (pred_arr[:, 0].min() + pred_arr[:, 0].max()) / 2
            pred_center_y = (pred_arr[:, 1].min() + pred_arr[:, 1].max()) / 2
            gt_center_x = (gt_arr[:, 0].min() + gt_arr[:, 0].max()) / 2
            gt_center_y = (gt_arr[:, 1].min() + gt_arr[:, 1].max()) / 2
            
            center_dist = np.sqrt((pred_center_x - gt_center_x)**2 + (pred_center_y - gt_center_y)**2)
            print(f"    Center distance: {center_dist:.1f} px")
            
            # Check X/Y overlap
            pred_x_min, pred_x_max = pred_arr[:, 0].min(), pred_arr[:, 0].max()
            gt_x_min, gt_x_max = gt_arr[:, 0].min(), gt_arr[:, 0].max()
            pred_y_min, pred_y_max = pred_arr[:, 1].min(), pred_arr[:, 1].max()
            gt_y_min, gt_y_max = gt_arr[:, 1].min(), gt_arr[:, 1].max()
            
            x_overlap = not (pred_x_max < gt_x_min or pred_x_min > gt_x_max)
            y_overlap = not (pred_y_max < gt_y_min or pred_y_min > gt_y_max)
            
            print(f"    X overlap: {x_overlap}, Y overlap: {y_overlap}")
            
            if x_overlap and y_overlap:
                # Calculate overlap percentage
                x_overlap_size = min(pred_x_max, gt_x_max) - max(pred_x_min, gt_x_min)
                y_overlap_size = min(pred_y_max, gt_y_max) - max(pred_y_min, gt_y_min)
                pred_x_size = pred_x_max - pred_x_min
                pred_y_size = pred_y_max - pred_y_min
                gt_x_size = gt_x_max - gt_x_min
                gt_y_size = gt_y_max - gt_y_min
                
                x_overlap_pct = (x_overlap_size / max(pred_x_size, gt_x_size)) * 100
                y_overlap_pct = (y_overlap_size / max(pred_y_size, gt_y_size)) * 100
                
                print(f"    X overlap: {x_overlap_size:.1f} px ({x_overlap_pct:.1f}% of max range)")
                print(f"    Y overlap: {y_overlap_size:.1f} px ({y_overlap_pct:.1f}% of max range)")
                
                if x_overlap_pct < 50 or y_overlap_pct < 50:
                    print(f"    ⚠️  Low overlap percentage - lanes are partially overlapping but not well aligned")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python debug_matching.py <gt_file> <pred_file>")
        sys.exit(1)
    
    debug_matching(sys.argv[1], sys.argv[2])








