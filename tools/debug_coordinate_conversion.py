#!/usr/bin/env python3
"""
Debug coordinate conversion step by step.
Check if coords_to_lane_normalized is correct.
"""
import numpy as np
from libs.models.detectors.lanelm_detector import coords_to_lane_normalized
from libs.models.lanelm import LaneTokenizerConfig
from configs.clrernet.culane.dataset_culane_clrernet import crop_bbox

def debug_coordinate_conversion():
    """Debug coordinate conversion."""
    print(f"\n{'='*80}")
    print(f"COORDINATE CONVERSION DEBUG")
    print(f"{'='*80}\n")
    
    # Test case: Known resized coordinates
    # Resized space: (800, 320)
    # Original space: (1640, 590)
    # Crop bbox: (0, 270, 1640, 590)
    
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=800,
        img_h=320,
        num_steps=40,
        nbins_x=200,
        x_mode='absolute',
    )
    
    crop_bbox_test = crop_bbox  # (0, 270, 1640, 590)
    img_w = 800
    img_h = 320
    ori_img_w = 1640
    ori_img_h = 590
    
    print(f"Crop bbox: {crop_bbox_test}")
    print(f"Resized: {img_w}x{img_h}")
    print(f"Original: {ori_img_w}x{ori_img_h}\n")
    
    # Test 1: Resized center point
    print("--- Test 1: Resized center point (400, 160) ---")
    coords_resized = np.array([[400.0, 160.0]], dtype=np.float32)
    print(f"Resized coords: X={coords_resized[0, 0]}, Y={coords_resized[0, 1]}")
    
    lane = coords_to_lane_normalized(
        coords_resized=coords_resized,
        tokenizer_cfg=tokenizer_cfg,
        crop_bbox=crop_bbox_test,
        img_w=img_w,
        img_h=img_h,
        ori_img_w=ori_img_w,
        ori_img_h=ori_img_h,
    )
    
    if lane is not None:
        print(f"Normalized coords: X={lane.points[0, 0]:.6f}, Y={lane.points[0, 1]:.6f}")
        
        # Convert back to original
        x_orig_calc = lane.points[0, 0] * ori_img_w
        y_orig_calc = lane.points[0, 1] * ori_img_h
        print(f"Original coords (calculated): X={x_orig_calc:.1f}, Y={y_orig_calc:.1f}")
        
        # Expected original
        x_scale = ori_img_w / img_w  # 1640 / 800 = 2.05
        y_scale = (crop_bbox_test[3] - crop_bbox_test[1]) / img_h  # 320 / 320 = 1.0
        x_orig_expected = coords_resized[0, 0] * x_scale  # 400 * 2.05 = 820
        y_orig_expected = coords_resized[0, 1] * y_scale + crop_bbox_test[1]  # 160 * 1.0 + 270 = 430
        print(f"Original coords (expected): X={x_orig_expected:.1f}, Y={y_orig_expected:.1f}")
        
        if abs(x_orig_calc - x_orig_expected) < 1.0 and abs(y_orig_calc - y_orig_expected) < 1.0:
            print("  ✅ Conversion is CORRECT!")
        else:
            print(f"  ⚠️  Conversion ERROR! Diff: X={abs(x_orig_calc - x_orig_expected):.1f}, Y={abs(y_orig_calc - y_orig_expected):.1f}")
    
    # Test 2: Resized left edge
    print("\n--- Test 2: Resized left edge (0, 160) ---")
    coords_resized = np.array([[0.0, 160.0]], dtype=np.float32)
    print(f"Resized coords: X={coords_resized[0, 0]}, Y={coords_resized[0, 1]}")
    
    lane = coords_to_lane_normalized(
        coords_resized=coords_resized,
        tokenizer_cfg=tokenizer_cfg,
        crop_bbox=crop_bbox_test,
        img_w=img_w,
        img_h=img_h,
        ori_img_w=ori_img_w,
        ori_img_h=ori_img_h,
    )
    
    if lane is not None:
        x_orig_calc = lane.points[0, 0] * ori_img_w
        y_orig_calc = lane.points[0, 1] * ori_img_h
        print(f"Original coords (calculated): X={x_orig_calc:.1f}, Y={y_orig_calc:.1f}")
        
        x_orig_expected = 0.0 * x_scale  # 0
        y_orig_expected = 160.0 * y_scale + crop_bbox_test[1]  # 160 + 270 = 430
        print(f"Original coords (expected): X={x_orig_expected:.1f}, Y={y_orig_expected:.1f}")
    
    # Test 3: Resized right edge
    print("\n--- Test 3: Resized right edge (799, 160) ---")
    coords_resized = np.array([[799.0, 160.0]], dtype=np.float32)
    print(f"Resized coords: X={coords_resized[0, 0]}, Y={coords_resized[0, 1]}")
    
    lane = coords_to_lane_normalized(
        coords_resized=coords_resized,
        tokenizer_cfg=tokenizer_cfg,
        crop_bbox=crop_bbox_test,
        img_w=img_w,
        img_h=img_h,
        ori_img_w=ori_img_w,
        ori_img_h=ori_img_h,
    )
    
    if lane is not None:
        x_orig_calc = lane.points[0, 0] * ori_img_w
        y_orig_calc = lane.points[0, 1] * ori_img_h
        print(f"Original coords (calculated): X={x_orig_calc:.1f}, Y={y_orig_calc:.1f}")
        
        x_orig_expected = 799.0 * x_scale  # 799 * 2.05 = 1637.95
        y_orig_expected = 160.0 * y_scale + crop_bbox_test[1]  # 160 + 270 = 430
        print(f"Original coords (expected): X={x_orig_expected:.1f}, Y={y_orig_expected:.1f}")
        
        if abs(x_orig_calc - x_orig_expected) < 1.0:
            print("  ✅ Conversion is CORRECT!")
        else:
            print(f"  ⚠️  Conversion ERROR! Diff: X={abs(x_orig_calc - x_orig_expected):.1f}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    debug_coordinate_conversion()








