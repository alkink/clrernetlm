#!/usr/bin/env python3
"""
Compare training visualization tokens vs test inference tokens.
This helps identify why training looks smooth but test is zigzag.
"""
import argparse
import os
import numpy as np
import torch
from pathlib import Path

from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig
from tools.train_lanelm_culane_v3 import build_frozen_clrernet_backbone
from tools.train_lanelm_v4_fixed import extract_full_fpn_feats, visual_first_decode
from libs.models.detectors.lanelm_detector import autoregressive_decode
from libs.datasets import CulaneDataset
from configs.clrernet.culane.dataset_culane_clrernet import (
    compose_cfg, crop_bbox, img_scale
)

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


def compare_tokens(
    lanelm_ckpt,
    config_path,
    backbone_ckpt,
    data_root,
    list_path,
    sample_idx,
    device,
    save_dir
):
    """Compare training vs test tokens."""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"TRAINING vs TEST TOKEN COMPARISON")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    ckpt = torch.load(lanelm_ckpt, map_location=device)
    cfg = ckpt['config']
    
    # Build models
    clrernet = build_frozen_clrernet_backbone(config_path, backbone_ckpt, device)
    
    # V4 FIX: checkpoint ile aynı visual token yolu (Full FPN) kullanılmalı
    lanelm = LaneLMModel(
        nbins_x=cfg['nbins_x'],
        max_y_tokens=cfg['num_points'] + 1,
        embed_dim=cfg['embed_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        max_seq_len=80,
        visual_in_channels=(64, 64, 64),
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
    dataset = CulaneDataset(
        data_root=data_root,
        data_list=list_path,
        pipeline=[dict(type="albumentation", pipelines=clean_pipeline)],
        diff_file=None,
        test_mode=False,
    )
    
    if sample_idx >= len(dataset):
        print(f"Error: sample_idx {sample_idx} >= dataset size {len(dataset)}")
        return
    
    sample = dataset[sample_idx]
    sub_img_name = sample.get('sub_img_name', f'sample_{sample_idx}')
    print(f"Sample: {sub_img_name}\n")
    
    # ========== TRAINING PATH (visual_first_decode) ==========
    print("--- TRAINING PATH (visual_first_decode) ---")
    img_tensor = torch.from_numpy(sample['img']).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        feats = extract_full_fpn_feats(clrernet, img_tensor)
        visual_tokens = lanelm.encode_visual_tokens(feats)
        
        # Training path: visual_first_decode
        all_preds_train = visual_first_decode(
            lanelm, visual_tokens, tokenizer, device, max_lanes=4
        )
        # visual_first_decode returns list of (x_tokens, y_tokens) numpy arrays
        # Convert to tensor format to match test
        x_tokens_train_list = [torch.from_numpy(pred[0]) for pred in all_preds_train]
        y_tokens_train_list = [torch.from_numpy(pred[1]) for pred in all_preds_train]
        # Stack to match test format: (1, max_lanes, T)
        x_tokens_train = torch.stack(x_tokens_train_list, dim=0).unsqueeze(0)  # (1, max_lanes, T)
        y_tokens_train = torch.stack(y_tokens_train_list, dim=0).unsqueeze(0)  # (1, max_lanes, T)
    
    print(f"Training tokens shape: {x_tokens_train.shape}")
    print(f"Training Lane 0 tokens (first 20):")
    train_tokens_l0 = x_tokens_train[0, 0].cpu().numpy()
    print(f"  X tokens: {train_tokens_l0[:20]}")
    
    # Decode with smoothing (as training visualization does)
    train_coords = tokenizer.decode_single_lane(
        train_tokens_l0, 
        y_tokens_train[0, 0].cpu().numpy(), 
        smooth=True
    )
    print(f"Training coords (first 10 points):")
    print(f"  X: {train_coords[:10, 0]}")
    print(f"  Y: {train_coords[:10, 1]}")
    
    # Calculate zigzag metric (variance of differences)
    if len(train_coords) > 1:
        train_diffs = np.diff(train_coords[:, 0])
        train_zigzag = np.std(train_diffs)
        print(f"Training zigzag metric (std of X diffs): {train_zigzag:.4f}")
    
    # ========== TEST PATH (autoregressive_decode) ==========
    print("\n--- TEST PATH (autoregressive_decode) ---")
    
    with torch.no_grad():
        # Test path: autoregressive_decode (same as LaneLMDetector)
        # NOTE: Presence head bu checkpoint'te çoğu zaman eğitilmediği için (presence_weight=0),
        # presence_filter açıkken lane'ler rastgele elenip reorder olur.
        # Train vs test token karşılaştırması için presence_filter KAPALI olmalı.
        x_tokens_test, y_tokens_test = autoregressive_decode(
            lanelm_model=lanelm,
            visual_tokens=visual_tokens,
            tokenizer_cfg=tokenizer.cfg,
            max_lanes=4,
            temperature=0.0,
            use_presence_filter=False,
            presence_threshold=0.5,
        )
    
    print(f"Test tokens shape: {x_tokens_test.shape}")
    print(f"Test Lane 0 tokens (first 20):")
    test_tokens_l0 = x_tokens_test[0, 0].cpu().numpy()
    print(f"  X tokens: {test_tokens_l0[:20]}")
    
    # Decode with smoothing (as test does)
    test_coords = tokenizer.decode_single_lane(
        test_tokens_l0,
        y_tokens_test[0, 0].cpu().numpy(),
        smooth=True
    )
    print(f"Test coords (first 10 points):")
    print(f"  X: {test_coords[:10, 0]}")
    print(f"  Y: {test_coords[:10, 1]}")
    
    # Calculate zigzag metric
    if len(test_coords) > 1:
        test_diffs = np.diff(test_coords[:, 0])
        test_zigzag = np.std(test_diffs)
        print(f"Test zigzag metric (std of X diffs): {test_zigzag:.4f}")
    
    # ========== COMPARISON ==========
    print("\n--- COMPARISON ---")
    
    # Compare tokens
    token_diff = np.abs(train_tokens_l0 - test_tokens_l0)
    print(f"Token differences (first 20):")
    print(f"  {token_diff[:20]}")
    print(f"  Max diff: {token_diff.max()}, Mean diff: {token_diff.mean():.4f}")
    
    if token_diff.max() == 0:
        print("  ✅ Tokens are IDENTICAL!")
    else:
        print(f"  ⚠️  Tokens are DIFFERENT! Max diff: {token_diff.max()}")
        # Find where they differ
        diff_indices = np.where(token_diff > 0)[0]
        print(f"  Different at indices: {diff_indices[:20]}")
    
    # Compare coords
    if len(train_coords) > 0 and len(test_coords) > 0:
        min_len = min(len(train_coords), len(test_coords))
        coord_diff = np.abs(train_coords[:min_len, 0] - test_coords[:min_len, 0])
        print(f"\nCoordinate differences (first 20):")
        print(f"  {coord_diff[:20]}")
        print(f"  Max diff: {coord_diff.max():.4f} px, Mean diff: {coord_diff.mean():.4f} px")
        
        if coord_diff.max() < 1.0:
            print("  ✅ Coordinates are nearly identical!")
        else:
            print(f"  ⚠️  Coordinates differ! Max diff: {coord_diff.max():.4f} px")
    
    # Compare zigzag
    if len(train_coords) > 1 and len(test_coords) > 1:
        print(f"\nZigzag comparison:")
        print(f"  Training: {train_zigzag:.4f}")
        print(f"  Test: {test_zigzag:.4f}")
        print(f"  Ratio: {test_zigzag / train_zigzag:.4f}x")
        
        if test_zigzag > train_zigzag * 1.5:
            print(f"  ⚠️  Test is MORE zigzag than training!")
        elif test_zigzag < train_zigzag * 0.5:
            print(f"  ⚠️  Test is LESS zigzag than training!")
        else:
            print(f"  ✅ Zigzag levels are similar")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare training vs test tokens")
    parser.add_argument("--lanelm-ckpt", required=True)
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--list-path", default="dataset/list/train_100.txt")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--save-dir", default="work_dirs/debug_training_vs_test_tokens")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    compare_tokens(
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

