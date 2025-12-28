#!/usr/bin/env python3
"""
Debug script to compare training visualization vs test inference on the same image.
This helps identify normalization/coordinate conversion issues.
"""
import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path

from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig
from libs.datasets import CulaneDataset
from libs.datasets.metrics.culane_metric import load_culane_img_data
from libs.models.detectors.lanelm_detector import autoregressive_decode, coords_to_lane_normalized
from tools.train_lanelm_culane_v3 import build_frozen_clrernet_backbone
from tools.train_lanelm_v4_fixed import extract_p5_feat, visual_first_decode
from configs.clrernet.culane.dataset_culane_clrernet import (
    compose_cfg, crop_bbox, img_scale
)

# Clean pipeline (matches training)
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


def debug_single_image(
    lanelm_ckpt,
    config_path,
    backbone_ckpt,
    data_root,
    list_path,
    sample_idx,
    device,
    save_dir
):
    """Compare training visualization vs test inference on the same image."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading LaneLM checkpoint: {lanelm_ckpt}")
    ckpt = torch.load(lanelm_ckpt, map_location=device)
    cfg = ckpt['config']
    
    # Build models
    print("Building models...")
    clrernet = build_frozen_clrernet_backbone(config_path, backbone_ckpt, device)
    
    lanelm = LaneLMModel(
        nbins_x=cfg['nbins_x'],
        max_y_tokens=cfg['num_points'] + 1,
        embed_dim=cfg['embed_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        max_seq_len=80,
        visual_in_channels=(64,),  # P5-only
    )
    lanelm.load_state_dict(ckpt['model_state_dict'])
    lanelm.to(device).eval()
    
    # Build tokenizer
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=800,
        img_h=320,
        num_steps=40,
        nbins_x=cfg['nbins_x'],
        x_mode='absolute',
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    
    # Load dataset
    print(f"Loading dataset from {list_path}...")
    dataset = CulaneDataset(
        data_root=data_root,
        data_list=list_path,
        pipeline=[dict(type="albumentation", pipelines=clean_pipeline)],
        diff_file=None,
        test_mode=False,  # Need GT for comparison
    )
    
    if sample_idx >= len(dataset):
        print(f"Error: sample_idx {sample_idx} >= dataset size {len(dataset)}")
        return
    
    sample = dataset[sample_idx]
    sub_img_name = sample.get('sub_img_name', f'sample_{sample_idx}')
    print(f"\n=== Debugging sample: {sub_img_name} ===")
    
    # Get image tensor
    img_tensor = torch.from_numpy(sample['img']).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Extract features (P5-only, matches training)
    with torch.no_grad():
        feats = extract_p5_feat(clrernet, img_tensor)
        visual_tokens = lanelm.encode_visual_tokens(feats)
    
    # ========== 1. TRAINING VISUALIZATION PATH ==========
    print("\n--- Training Visualization Path ---")
    all_preds_train = visual_first_decode(lanelm, visual_tokens[:1], tokenizer, device, max_lanes=4)
    
    train_coords = []
    for l_idx, (x_tokens, y_tokens) in enumerate(all_preds_train):
        coords = tokenizer.decode_single_lane(x_tokens, y_tokens, smooth=True)
        train_coords.append(coords)
        print(f"  Lane {l_idx}: {len(coords)} points in resized space (800x320)")
        if len(coords) > 0:
            print(f"    X range: [{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}]")
            print(f"    Y range: [{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")
    
    # ========== 2. TEST INFERENCE PATH ==========
    print("\n--- Test Inference Path ---")
    x_tokens_all, y_tokens_all = autoregressive_decode(
        lanelm_model=lanelm,
        visual_tokens=visual_tokens,
        tokenizer_cfg=tokenizer.cfg,
        max_lanes=4,
        temperature=0.0,
    )
    
    test_coords_resized = []
    test_coords_normalized = []
    for l in range(x_tokens_all.shape[1]):
        x_tok = x_tokens_all[0, l].numpy()
        y_tok = y_tokens_all[0, l].numpy()
        
        # Decode to resized space (same as training)
        coords_resized = tokenizer.decode_single_lane(x_tok, y_tok, smooth=True)
        test_coords_resized.append(coords_resized)
        
        # Convert to normalized space (test path)
        lane = coords_to_lane_normalized(
            coords_resized=coords_resized,
            tokenizer_cfg=tokenizer.cfg,
            crop_bbox=crop_bbox,
            img_w=800,
            img_h=320,
            ori_img_w=1640,
            ori_img_h=590,
        )
        
        if lane is not None and lane.points is not None:
            test_coords_normalized.append(lane.points)
            print(f"  Lane {l}: {len(coords_resized)} points resized, {len(lane.points)} points normalized")
            if len(coords_resized) > 0:
                print(f"    Resized X: [{coords_resized[:, 0].min():.1f}, {coords_resized[:, 0].max():.1f}]")
                print(f"    Resized Y: [{coords_resized[:, 1].min():.1f}, {coords_resized[:, 1].max():.1f}]")
            if len(lane.points) > 0:
                print(f"    Normalized X: [{lane.points[:, 0].min():.4f}, {lane.points[:, 0].max():.4f}]")
                print(f"    Normalized Y: [{lane.points[:, 1].min():.4f}, {lane.points[:, 1].max():.4f}]")
        else:
            test_coords_normalized.append(None)
            print(f"  Lane {l}: Failed to convert to normalized (lane is None)")
    
    # ========== 3. COMPARISON ==========
    print("\n--- Comparison: Training vs Test (Resized Space) ---")
    for l_idx in range(min(len(train_coords), len(test_coords_resized))):
        train_c = train_coords[l_idx]
        test_c = test_coords_resized[l_idx]
        
        if len(train_c) == 0 and len(test_c) == 0:
            print(f"  Lane {l_idx}: Both empty")
            continue
        elif len(train_c) == 0:
            print(f"  Lane {l_idx}: Training empty, Test has {len(test_c)} points")
            continue
        elif len(test_c) == 0:
            print(f"  Lane {l_idx}: Training has {len(train_c)} points, Test empty")
            continue
        
        # Compare point counts
        print(f"  Lane {l_idx}:")
        print(f"    Training: {len(train_c)} points")
        print(f"    Test: {len(test_c)} points")
        
        # Compare coordinate ranges
        train_x_range = [train_c[:, 0].min(), train_c[:, 0].max()]
        train_y_range = [train_c[:, 1].min(), train_c[:, 1].max()]
        test_x_range = [test_c[:, 0].min(), test_c[:, 0].max()]
        test_y_range = [test_c[:, 1].min(), test_c[:, 1].max()]
        
        print(f"    Training X: [{train_x_range[0]:.1f}, {train_x_range[1]:.1f}]")
        print(f"    Test X: [{test_x_range[0]:.1f}, {test_x_range[1]:.1f}]")
        print(f"    X diff: [{abs(train_x_range[0] - test_x_range[0]):.1f}, {abs(train_x_range[1] - test_x_range[1]):.1f}]")
        
        print(f"    Training Y: [{train_y_range[0]:.1f}, {train_y_range[1]:.1f}]")
        print(f"    Test Y: [{test_y_range[0]:.1f}, {test_y_range[1]:.1f}]")
        print(f"    Y diff: [{abs(train_y_range[0] - test_y_range[0]):.1f}, {abs(train_y_range[1] - test_y_range[1]):.1f}]")
        
        # Try to match points (if same Y values)
        if len(train_c) > 0 and len(test_c) > 0:
            # Find common Y range
            common_y_min = max(train_c[:, 1].min(), test_c[:, 1].min())
            common_y_max = min(train_c[:, 1].max(), test_c[:, 1].max())
            
            if common_y_max > common_y_min:
                # Sample points at same Y values
                sample_ys = np.linspace(common_y_min, common_y_max, 10)
                train_xs = []
                test_xs = []
                
                for sy in sample_ys:
                    # Find closest points in training
                    train_dists = np.abs(train_c[:, 1] - sy)
                    train_idx = np.argmin(train_dists)
                    if train_dists[train_idx] < 5.0:  # Within 5 pixels
                        train_xs.append(train_c[train_idx, 0])
                    
                    # Find closest points in test
                    test_dists = np.abs(test_c[:, 1] - sy)
                    test_idx = np.argmin(test_dists)
                    if test_dists[test_idx] < 5.0:  # Within 5 pixels
                        test_xs.append(test_c[test_idx, 0])
                
                if len(train_xs) > 0 and len(test_xs) > 0:
                    min_len = min(len(train_xs), len(test_xs))
                    train_xs = train_xs[:min_len]
                    test_xs = test_xs[:min_len]
                    x_diffs = np.abs(np.array(train_xs) - np.array(test_xs))
                    mean_diff = np.mean(x_diffs)
                    max_diff = np.max(x_diffs)
                    print(f"    Mean X difference (at same Y): {mean_diff:.2f} px")
                    print(f"    Max X difference: {max_diff:.2f} px")
                    if mean_diff > 10.0:
                        print(f"    ⚠️  WARNING: Large X difference! (>10px)")
    
    # ========== 4. VISUALIZATION ==========
    print("\n--- Creating visualizations ---")
    img_vis = (img_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    
    # Draw GT (GREEN)
    if 'gt_points' in sample and sample['gt_points']:
        for lane in sample['gt_points'][:4]:
            if lane and len(lane) >= 2:
                pts = np.array(lane, dtype=np.float32).reshape(-1, 2)
                for k in range(len(pts) - 1):
                    p1 = (int(pts[k][0]), int(pts[k][1]))
                    p2 = (int(pts[k+1][0]), int(pts[k+1][1]))
                    if 0 <= p1[0] < 800 and 0 <= p2[0] < 800 and 0 <= p1[1] < 320 and 0 <= p2[1] < 320:
                        cv2.line(img_vis, p1, p2, (0, 255, 0), 3)
    
    # Draw Training predictions (RED)
    colors_train = [(0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
    for l_idx, coords in enumerate(train_coords):
        if len(coords) >= 2:
            for k in range(len(coords) - 1):
                p1 = (int(coords[k][0]), int(coords[k][1]))
                p2 = (int(coords[k+1][0]), int(coords[k+1][1]))
                if 0 <= p1[0] < 800 and 0 <= p2[0] < 800:
                    cv2.line(img_vis, p1, p2, colors_train[l_idx % 4], 2)
    
    # Draw Test predictions (BLUE) - resized space
    for l_idx, coords in enumerate(test_coords_resized):
        if coords is not None and len(coords) >= 2:
            for k in range(len(coords) - 1):
                p1 = (int(coords[k][0]), int(coords[k][1]))
                p2 = (int(coords[k+1][0]), int(coords[k+1][1]))
                if 0 <= p1[0] < 800 and 0 <= p2[0] < 800:
                    cv2.line(img_vis, p1, p2, (255, 255, 0), 1)  # Yellow for test
    
    # Add text
    cv2.putText(img_vis, "GREEN=GT, RED=Training, YELLOW=Test", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img_vis, sub_img_name, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save
    save_path = os.path.join(save_dir, f"debug_sample_{sample_idx}.jpg")
    cv2.imwrite(save_path, img_vis)
    print(f"  Saved visualization to: {save_path}")
    
    # ========== 5. NORMALIZATION TEST ==========
    print("\n--- Normalization Test ---")
    print(f"Crop bbox: {crop_bbox}")
    print(f"Image size: 800x320 (resized), 1640x590 (original)")
    
    # Test a known point
    test_x_resized = 400.0  # Middle of resized width
    test_y_resized = 160.0  # Middle of resized height
    
    x_scale = 1640.0 / 800.0
    y_scale = (crop_bbox[3] - crop_bbox[1]) / 320.0
    
    x_orig = test_x_resized * x_scale
    y_orig = test_y_resized * y_scale + crop_bbox[1]
    
    x_norm = x_orig / 1640.0
    y_norm = y_orig / 590.0
    
    print(f"  Test point (resized): X={test_x_resized:.1f}, Y={test_y_resized:.1f}")
    print(f"  → Original: X={x_orig:.1f}, Y={y_orig:.1f}")
    print(f"  → Normalized: X={x_norm:.4f}, Y={y_norm:.4f}")
    print(f"  Scale factors: X={x_scale:.4f}, Y={y_scale:.4f}")
    
    # Check if normalized coordinates are in valid range
    if x_norm < 0.0 or x_norm >= 1.0 or y_norm < 0.0 or y_norm >= 1.0:
        print(f"  ⚠️  WARNING: Normalized coordinates out of [0, 1) range!")
    else:
        print(f"  ✓ Normalized coordinates in valid range")
    
    print("\n=== Debug complete ===")


def main():
    parser = argparse.ArgumentParser(description="Debug training vs test mismatch")
    parser.add_argument("--lanelm-ckpt", required=True, help="LaneLM checkpoint path")
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--list-path", default="dataset/list/train_100.txt")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to debug")
    parser.add_argument("--save-dir", default="work_dirs/debug_training_test_mismatch")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    debug_single_image(
        lanelm_ckpt=args.lanelm_ckpt,
        config_path=args.config,
        backbone_ckpt=args.checkpoint,
        data_root=args.data_root,
        list_path=args.list_path,
        sample_idx=args.sample_idx,
        device=device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()








