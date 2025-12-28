#!/usr/bin/env python3
"""
Debug script to test draw_lane function and IoU calculation.
This helps identify if image bounds clipping is causing the issue.
"""
import numpy as np
import cv2
from libs.utils.visualizer import draw_lane
from libs.utils.lane_utils import interp
from libs.datasets.metrics.culane_metric import discrete_cross_iou

# Test 1: draw_lane with bounds-outside values
print("=== Test 1: draw_lane with bounds-outside values ===")
img_shape = (590, 1640, 3)

# GT lane with negative X
gt_lane_neg_x = np.array([
    [-14.1, 510.0],
    [100.0, 400.0],
    [732.8, 290.0],
], dtype=np.float32)

# GT lane with X > 1640
gt_lane_large_x = np.array([
    [862.6, 440.0],
    [1200.0, 350.0],
    [1650.4, 290.0],
], dtype=np.float32)

# Prediction lane (all in bounds)
pred_lane = np.array([
    [88.1, 482.0],
    [400.0, 350.0],
    [723.7, 290.0],
], dtype=np.float32)

# Draw lanes
img_gt_neg = draw_lane(gt_lane_neg_x, img=None, img_shape=img_shape, width=30)
img_gt_large = draw_lane(gt_lane_large_x, img=None, img_shape=img_shape, width=30)
img_pred = draw_lane(pred_lane, img=None, img_shape=img_shape, width=30)

print(f"GT (neg X) pixels: {img_gt_neg.sum()}")
print(f"GT (large X) pixels: {img_gt_large.sum()}")
print(f"Pred pixels: {img_pred.sum()}")

# Test 2: IoU calculation
print("\n=== Test 2: IoU calculation ===")

# Interpolate lanes (as CULaneMetric does)
interp_gt_neg = interp(gt_lane_neg_x.tolist(), n=5)
interp_gt_large = interp(gt_lane_large_x.tolist(), n=5)
interp_pred = interp(pred_lane.tolist(), n=5)

print(f"Interpolated GT (neg X): {len(interp_gt_neg)} points")
print(f"Interpolated GT (large X): {len(interp_gt_large)} points")
print(f"Interpolated Pred: {len(interp_pred)} points")

# Calculate IoU
ious_neg = discrete_cross_iou(
    np.array([interp_pred], dtype=object),
    np.array([interp_gt_neg], dtype=object),
    width=30,
    img_shape=img_shape,
)

ious_large = discrete_cross_iou(
    np.array([interp_pred], dtype=object),
    np.array([interp_gt_large], dtype=object),
    width=30,
    img_shape=img_shape,
)

print(f"IoU (GT neg X vs Pred): {ious_neg[0, 0]:.4f}")
print(f"IoU (GT large X vs Pred): {ious_large[0, 0]:.4f}")

# Test 3: GT vs Prediction from debug log
print("\n=== Test 3: Real GT vs Prediction from debug log ===")

# GT Lane 0 from debug log
gt_lane_0 = np.array([
    [-14.1, 510.0],
    [100.0, 450.0],
    [300.0, 400.0],
    [500.0, 350.0],
    [732.8, 290.0],
], dtype=np.float32)

# Pred String Lane 0 from debug log (first and last few points)
# We'll use interpolated points from prediction string
pred_lane_0_points = []
# From debug: X[88.1, 723.7], Y[290.0, 482.0], 97 points
# Create a simple approximation
pred_lane_0 = np.array([
    [88.1, 482.0],
    [200.0, 400.0],
    [400.0, 350.0],
    [600.0, 320.0],
    [723.7, 290.0],
], dtype=np.float32)

print(f"GT Lane 0: {len(gt_lane_0)} points, X[{gt_lane_0[:, 0].min():.1f}, {gt_lane_0[:, 0].max():.1f}], Y[{gt_lane_0[:, 1].min():.1f}, {gt_lane_0[:, 1].max():.1f}]")
print(f"Pred Lane 0: {len(pred_lane_0)} points, X[{pred_lane_0[:, 0].min():.1f}, {pred_lane_0[:, 0].max():.1f}], Y[{pred_lane_0[:, 1].min():.1f}, {pred_lane_0[:, 1].max():.1f}]")

# Interpolate
interp_gt_0 = interp(gt_lane_0.tolist(), n=5)
interp_pred_0 = interp(pred_lane_0.tolist(), n=5)

# Calculate IoU
ious_0 = discrete_cross_iou(
    np.array([interp_pred_0], dtype=object),
    np.array([interp_gt_0], dtype=object),
    width=30,
    img_shape=img_shape,
)

print(f"IoU (GT Lane 0 vs Pred Lane 0): {ious_0[0, 0]:.4f}")

# Test 4: Check if clipping affects IoU
print("\n=== Test 4: Clipping effect on IoU ===")

# Clip GT to image bounds
gt_lane_0_clipped = gt_lane_0.copy()
gt_lane_0_clipped[:, 0] = np.clip(gt_lane_0_clipped[:, 0], 0, 1639)
gt_lane_0_clipped[:, 1] = np.clip(gt_lane_0_clipped[:, 1], 0, 589)

print(f"GT Lane 0 (clipped): X[{gt_lane_0_clipped[:, 0].min():.1f}, {gt_lane_0_clipped[:, 0].max():.1f}], Y[{gt_lane_0_clipped[:, 1].min():.1f}, {gt_lane_0_clipped[:, 1].max():.1f}]")

# Interpolate clipped GT
interp_gt_0_clipped = interp(gt_lane_0_clipped.tolist(), n=5)

# Calculate IoU with clipped GT
ious_0_clipped = discrete_cross_iou(
    np.array([interp_pred_0], dtype=object),
    np.array([interp_gt_0_clipped], dtype=object),
    width=30,
    img_shape=img_shape,
)

print(f"IoU (GT Lane 0 clipped vs Pred Lane 0): {ious_0_clipped[0, 0]:.4f}")
print(f"IoU difference: {abs(ious_0[0, 0] - ious_0_clipped[0, 0]):.4f}")

print("\n=== Test complete ===")








