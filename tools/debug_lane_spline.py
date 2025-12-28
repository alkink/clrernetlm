#!/usr/bin/env python3
"""
Debug script to test Lane class spline interpolation in normalized space.
This helps identify if spline interpolation is the root cause.
"""
import numpy as np
from libs.utils.lane_utils import Lane

# Test 1: Normalized space'de Lane oluştur (Y artarak)
print("=== Test 1: Normalized space, Y artarak (ascending) ===")
points_norm_asc = np.array([
    [0.1, 0.5],  # X=0.1, Y=0.5
    [0.2, 0.6],  # X=0.2, Y=0.6
    [0.3, 0.7],  # X=0.3, Y=0.7
    [0.4, 0.8],  # X=0.4, Y=0.8
], dtype=np.float32)

lane_asc = Lane(points_norm_asc)
print(f"Points (Y ascending): {points_norm_asc}")
print(f"min_y: {lane_asc.min_y:.4f}, max_y: {lane_asc.max_y:.4f}")

# Test Y değerleri (normalized, artarak)
test_ys = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
xs = lane_asc(test_ys)
print(f"Test Ys: {test_ys}")
print(f"Interpolated Xs: {xs}")
print(f"Expected Xs: [0.1, 0.2, 0.3, 0.4]")
print(f"Difference: {np.abs(xs - np.array([0.1, 0.2, 0.3, 0.4]))}")

# Test 2: get_prediction_string mantığı
print("\n=== Test 2: get_prediction_string mantığı ===")
ori_h = 590
y_step = 2
ys = np.arange(0, ori_h, y_step) / ori_h  # Normalized Y (0-1, artarak)
print(f"Y values (first 10): {ys[:10]}")
print(f"Y values (last 10): {ys[-10:]}")

# Lane'in Y range'i içindeki Y değerlerini filtrele
lane_min_y = lane_asc.min_y
lane_max_y = lane_asc.max_y
ys_in_range = ys[(ys >= lane_min_y) & (ys <= lane_max_y)]
print(f"Lane Y range: [{lane_min_y:.4f}, {lane_max_y:.4f}]")
print(f"Ys in range (first 10): {ys_in_range[:10]}")
print(f"Ys in range (last 10): {ys_in_range[-10:]}")

# Interpolate
xs_norm = lane_asc(ys_in_range)
print(f"Interpolated Xs (normalized, first 10): {xs_norm[:10]}")
print(f"Interpolated Xs (normalized, last 10): {xs_norm[-10:]}")

# Filter invalid values
valid_mask = (xs_norm >= 0) & (xs_norm < 1)
xs_norm_valid = xs_norm[valid_mask]
ys_in_range_valid = ys_in_range[valid_mask]
print(f"Valid points: {len(xs_norm_valid)}/{len(xs_norm)}")

# Convert to original space
ori_w = 1640
xs_orig = xs_norm_valid * ori_w
ys_orig = ys_in_range_valid * ori_h
print(f"Original Xs (first 10): {xs_orig[:10]}")
print(f"Original Ys (first 10): {ys_orig[:10]}")

# Reverse (bottom-to-top format)
xs_orig_rev = xs_orig[::-1]
ys_orig_rev = ys_orig[::-1]
print(f"Reversed Xs (first 10): {xs_orig_rev[:10]}")
print(f"Reversed Ys (first 10): {ys_orig_rev[:10]}")

# Test 3: Debug log'dan gerçek değerler
print("\n=== Test 3: Debug log'dan gerçek değerler ===")
# Lane 0: Normalized X[0.0610, 0.4355], Y[0.4983, 0.8102]
points_real = np.array([
    [0.0610, 0.4983],
    [0.4355, 0.8102],
], dtype=np.float32)

# Daha fazla point ekle (interpolate etmek için)
y_min = 0.4983
y_max = 0.8102
y_mid = (y_min + y_max) / 2
x_min = 0.0610
x_max = 0.4355
x_mid = (x_min + x_max) / 2

points_real_extended = np.array([
    [x_min, y_min],
    [x_mid, y_mid],
    [x_max, y_max],
], dtype=np.float32)

lane_real = Lane(points_real_extended)
print(f"Real points: {points_real_extended}")
print(f"min_y: {lane_real.min_y:.4f}, max_y: {lane_real.max_y:.4f}")

# Test Y değerleri
test_ys_real = np.array([0.4983, 0.6, 0.7, 0.8102], dtype=np.float32)
xs_real = lane_real(test_ys_real)
print(f"Test Ys: {test_ys_real}")
print(f"Interpolated Xs: {xs_real}")
print(f"Expected Xs (approx): [{x_min:.4f}, {x_mid:.4f}, {x_mid:.4f}, {x_max:.4f}]")

# get_prediction_string mantığı
ys_real = np.arange(0, ori_h, y_step) / ori_h
ys_in_range_real = ys_real[(ys_real >= lane_real.min_y) & (ys_real <= lane_real.max_y)]
print(f"Ys in range: {len(ys_in_range_real)} points")
if len(ys_in_range_real) > 0:
    xs_real_interp = lane_real(ys_in_range_real)
    valid_mask_real = (xs_real_interp >= 0) & (xs_real_interp < 1)
    xs_real_valid = xs_real_interp[valid_mask_real]
    ys_real_valid = ys_in_range_real[valid_mask_real]
    print(f"Valid points after interpolation: {len(xs_real_valid)}/{len(ys_in_range_real)}")
    if len(xs_real_valid) > 0:
        print(f"First 5 Xs (normalized): {xs_real_valid[:5]}")
        print(f"First 5 Ys (normalized): {ys_real_valid[:5]}")
        print(f"Last 5 Xs (normalized): {xs_real_valid[-5:]}")
        print(f"Last 5 Ys (normalized): {ys_real_valid[-5:]}")

print("\n=== Test complete ===")








