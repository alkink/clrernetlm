#!/usr/bin/env python3
"""Debug script to find why predictions are empty.

This script:
1. Loads a single test image
2. Runs inference through LaneLMDetector
3. Logs every step of the prediction pipeline
4. Identifies where lanes are being filtered out
"""

import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig
from libs.models.detectors.lanelm_detector import autoregressive_decode, coords_to_lane_normalized
from libs.datasets.culane_dataset import CulaneDataset
from tools.train_lanelm_culane_v3 import LaneLMHyperParams, build_frozen_clrernet_backbone
from tools.train_lanelm_v4_fixed import extract_p5_feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="work_dirs/lanelm_v4_fixed/lanelm_v4_best.pth")
    parser.add_argument("--config", type=str, default="configs/clrernet/clrernet_culane_dla34.py")
    parser.add_argument("--checkpoint", type=str, default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    
    # Build tokenizer
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=800,
        img_h=320,
        num_steps=40,
        nbins_x=200,
        x_mode="absolute",
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    
    # Build model
    hparams = LaneLMHyperParams(
        nbins_x=200,
        num_points=40,
        embed_dim=256,
        num_layers=4,
        max_lanes=4,
    )
    
    visual_in_channels = (64,)  # P5 Only
    lanelm = LaneLMModel(
        nbins_x=hparams.nbins_x,
        max_y_tokens=hparams.num_points + 1,
        embed_dim=hparams.embed_dim,
        num_layers=hparams.num_layers,
        num_heads=8,
        ffn_dim=512,
        max_seq_len=80,
        visual_in_channels=visual_in_channels,
    ).to(device)
    
    lanelm.load_state_dict(state_dict, strict=False)
    lanelm.eval()
    
    # Load CLRerNet
    clrernet = build_frozen_clrernet_backbone(args.config, args.checkpoint, device)
    
    # Load dataset
    dataset = CulaneDataset(
        data_root="dataset",
        data_list="dataset/list/test_100.txt",
        test_mode=True,
    )
    
    # Get sample
    sample = dataset[args.sample_idx]
    img = sample["inputs"]
    gt_points = sample["gt_points"]
    filename = sample.get("metainfo", {}).get("sub_img_name", f"sample_{args.sample_idx}")
    
    print(f"\n{'='*80}")
    print(f"DEBUGGING: {filename}")
    print(f"{'='*80}")
    print(f"GT lanes: {len(gt_points)}")
    for i, lane in enumerate(gt_points):
        print(f"  Lane {i}: {len(lane)} points")
    
    # Prepare image
    if isinstance(img, torch.Tensor):
        img_tensor = img.unsqueeze(0).to(device)
    else:
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    
    if img_tensor.dtype == torch.uint8:
        img_tensor = img_tensor.float() / 255.0
    
    print(f"\nImage shape: {img_tensor.shape}")
    print(f"Image range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    
    # Extract features
    feats = extract_p5_feat(clrernet, img_tensor)
    print(f"\nFeatures shape: {[f.shape for f in feats]}")
    
    visual_tokens = lanelm.encode_visual_tokens(feats)
    print(f"Visual tokens shape: {visual_tokens.shape}")
    
    # Decode
    print(f"\n{'='*80}")
    print("AUTOREGRESSIVE DECODE")
    print(f"{'='*80}")
    
    x_tokens_all, y_tokens_all = autoregressive_decode(
        lanelm_model=lanelm.to(device),
        visual_tokens=visual_tokens,
        tokenizer_cfg=tokenizer_cfg,
        max_lanes=hparams.max_lanes,
        temperature=0.0,
        use_presence_filter=False,  # Disabled for debugging
        presence_threshold=0.5,
    )
    
    print(f"x_tokens_all shape: {x_tokens_all.shape}")
    print(f"y_tokens_all shape: {y_tokens_all.shape}")
    
    # Analyze tokens
    x_tok = x_tokens_all[0].numpy()  # (max_lanes, T)
    y_tok = y_tokens_all[0].numpy()  # (max_lanes, T)
    
    print(f"\n{'='*80}")
    print("TOKEN ANALYSIS")
    print(f"{'='*80}")
    
    pad_token_x = tokenizer_cfg.pad_token_x
    print(f"Pad token X: {pad_token_x}")
    
    for l in range(x_tok.shape[0]):
        x_tokens_lane = x_tok[l]
        y_tokens_lane = y_tok[l]
        
        # Count valid tokens
        non_pad_mask = (x_tokens_lane != pad_token_x) & (x_tokens_lane != 0)
        valid_count = non_pad_mask.sum()
        
        print(f"\nLane {l}:")
        print(f"  Valid tokens: {valid_count}/{len(x_tokens_lane)}")
        print(f"  X tokens range: [{x_tokens_lane.min()}, {x_tokens_lane.max()}]")
        print(f"  Y tokens range: [{y_tokens_lane.min()}, {y_tokens_lane.max()}]")
        print(f"  X tokens (first 10): {x_tokens_lane[:10]}")
        
        if valid_count >= 2:
            # Try decoding
            print(f"  → Decoding lane {l}...")
            coords_resized = tokenizer.decode_single_lane(x_tokens_lane, y_tokens_lane, smooth=True)
            print(f"  → Decoded coords shape: {coords_resized.shape}")
            
            if coords_resized.shape[0] >= 2:
                print(f"  → Coords range: X[{coords_resized[:, 0].min():.1f}, {coords_resized[:, 0].max():.1f}], "
                      f"Y[{coords_resized[:, 1].min():.1f}, {coords_resized[:, 1].max():.1f}]")
                
                # Try converting to Lane
                lane = coords_to_lane_normalized(
                    coords_resized=coords_resized,
                    tokenizer_cfg=tokenizer_cfg,
                    crop_bbox=(0, 270, 1640, 590),
                    img_w=800,
                    img_h=320,
                    ori_img_w=1640,
                    ori_img_h=590,
                )
                
                if lane is not None:
                    print(f"  → Lane created successfully! Points: {lane.points.shape[0]}")
                    print(f"  → Lane Y range: [{lane.min_y:.4f}, {lane.max_y:.4f}]")
                else:
                    print(f"  → ❌ Lane creation FAILED (coords_to_lane_normalized returned None)")
            else:
                print(f"  → ❌ Decoded coords < 2 points")
        else:
            print(f"  → ❌ Skipped: valid_count < 2")
    
    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    lanes_pred = []
    for l in range(x_tok.shape[0]):
        x_tokens_lane = x_tok[l]
        pad_token_x = tokenizer_cfg.pad_token_x
        non_pad_mask = (x_tokens_lane != pad_token_x) & (x_tokens_lane != 0)
        
        if non_pad_mask.sum() < 2:
            continue
        
        coords_resized = tokenizer.decode_single_lane(x_tok[l], y_tok[l], smooth=True)
        
        if coords_resized.shape[0] < 2:
            continue
        
        lane = coords_to_lane_normalized(
            coords_resized=coords_resized,
            tokenizer_cfg=tokenizer_cfg,
            crop_bbox=(0, 270, 1640, 590),
            img_w=800,
            img_h=320,
            ori_img_w=1640,
            ori_img_h=590,
        )
        if lane is not None and lane.points is not None and lane.points.shape[0] >= 2:
            lanes_pred.append(lane)
    
    print(f"Final lanes_pred count: {len(lanes_pred)}")
    print(f"GT lanes count: {len(gt_points)}")
    
    if len(lanes_pred) == 0:
        print("\n❌ PROBLEM FOUND: No lanes passed all filters!")
        print("   Check the token analysis above to see where lanes are being filtered out.")


if __name__ == "__main__":
    main()

