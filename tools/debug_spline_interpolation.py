#!/usr/bin/env python3
"""
Debug spline interpolation in Lane class.
This helps identify if spline interpolation is causing zigzag.
"""
import numpy as np
from libs.utils.lane_utils import Lane
from libs.models.lanelm import LaneTokenizer, LaneTokenizerConfig
from libs.models.detectors.lanelm_detector import coords_to_lane_normalized
from configs.clrernet.culane.dataset_culane_clrernet import crop_bbox

def debug_spline_interpolation():
    """Debug spline interpolation step by step."""
    print(f"\n{'='*80}")
    print(f"SPLINE INTERPOLATION DEBUG")
    print(f"{'='*80}\n")
    
    # Real test prediction coordinates (from file)
    pred_file = "work_dirs/lanelm_v4_test_fixed_100/predictions/driver_100_30frame/05251517_0433.MP4/02970.lines.txt"
    
    with open(pred_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        print("No predictions in file")
        return
    
    # Parse first lane
    lane0_str = lines[0].strip()
    coords = lane0_str.split()
    xs = [float(coords[i]) for i in range(0, len(coords), 2)]
    ys = [float(coords[i+1]) for i in range(0, len(coords), 2)]
    
    print(f"Test prediction file: {pred_file}")
    print(f"Lane 0: {len(xs)} points")
    print(f"X range: [{min(xs):.1f}, {max(xs):.1f}]")
    print(f"Y range: [{min(ys):.1f}, {max(ys):.1f}]")
    print(f"\nFirst 20 X values:")
    print(f"  {xs[:20]}")
    
    # Calculate zigzag metric
    if len(xs) > 1:
        diffs = np.diff(xs)
        zigzag = np.std(diffs)
        print(f"\nZigzag metric (std of X diffs): {zigzag:.4f}")
        print(f"First 20 diffs:")
        print(f"  {diffs[:20]}")
    
    # Now simulate what happens in test path
    print(f"\n--- Simulating Test Path ---")
    
    # Step 1: Decode tokens (as detector does)
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=800,
        img_h=320,
        num_steps=40,
        nbins_x=200,
        x_mode='absolute',
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    
    # We need to reverse engineer: what tokens produced these coordinates?
    # Actually, let's check what coords_to_lane_normalized produces
    
    # Create a simple test: smooth coords → Lane → get_prediction_string
    print(f"\n--- Test: Smooth coords → Lane → get_prediction_string ---")
    
    # Simulate smooth coords (from decode_single_lane with smooth=True)
    # Use a simple smooth line
    y_smooth = np.linspace(290, 510, 50)
    x_smooth = 100 + 2 * y_smooth  # Simple linear relationship
    
    print(f"Smooth input: {len(x_smooth)} points")
    print(f"X range: [{x_smooth.min():.1f}, {x_smooth.max():.1f}]")
    print(f"Y range: [{y_smooth.min():.1f}, {y_smooth.max():.1f}]")
    
    # Convert to normalized (as coords_to_lane_normalized does)
    coords_resized = np.stack([x_smooth, y_smooth], axis=1)
    lane = coords_to_lane_normalized(
        coords_resized=coords_resized,
        tokenizer_cfg=tokenizer_cfg,
        crop_bbox=crop_bbox,
        img_w=800,
        img_h=320,
        ori_img_w=1640,
        ori_img_h=590,
    )
    
    if lane is None:
        print("ERROR: Lane is None")
        return
    
    print(f"\nLane points (normalized): {len(lane.points)} points")
    print(f"X range (normalized): [{lane.points[:, 0].min():.4f}, {lane.points[:, 0].max():.4f}]")
    print(f"Y range (normalized): [{lane.points[:, 1].min():.4f}, {lane.points[:, 1].max():.4f}]")
    
    # Now simulate get_prediction_string
    ori_h = 590
    ori_w = 1640
    y_step = 2
    ys = np.arange(0, ori_h, y_step) / ori_h
    
    lane_min_y = lane.min_y - 0.05
    lane_max_y = lane.max_y + 0.05
    ys_in_range = ys[(ys >= lane_min_y) & (ys <= lane_max_y)]
    
    print(f"\nget_prediction_string simulation:")
    print(f"  Ys in range: {len(ys_in_range)} points")
    print(f"  Y range: [{ys_in_range.min():.4f}, {ys_in_range.max():.4f}]")
    
    # Interpolate using Lane spline
    xs_interp = lane(ys_in_range)
    valid_mask = (xs_interp >= 0) & (xs_interp < 1)
    xs_interp = xs_interp[valid_mask] * ori_w
    ys_interp = ys_in_range[valid_mask] * ori_h
    
    print(f"  Interpolated: {len(xs_interp)} points")
    print(f"  X range: [{xs_interp.min():.1f}, {xs_interp.max():.1f}]")
    print(f"  Y range: [{ys_interp.min():.1f}, {ys_interp.max():.1f}]")
    
    # Calculate zigzag
    if len(xs_interp) > 1:
        diffs_interp = np.diff(xs_interp)
        zigzag_interp = np.std(diffs_interp)
        print(f"  Zigzag metric (std of X diffs): {zigzag_interp:.4f}")
        print(f"  First 20 diffs:")
        print(f"    {diffs_interp[:20]}")
    
    # Compare with original
    print(f"\n--- Comparison ---")
    print(f"Original smooth input zigzag: {np.std(np.diff(x_smooth)):.4f}")
    print(f"After spline interpolation zigzag: {zigzag_interp:.4f}")
    print(f"Ratio: {zigzag_interp / np.std(np.diff(x_smooth)):.4f}x")
    
    if zigzag_interp > np.std(np.diff(x_smooth)) * 1.5:
        print(f"  ⚠️  Spline interpolation INCREASES zigzag!")
    else:
        print(f"  ✅ Spline interpolation preserves smoothness")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    debug_spline_interpolation()








