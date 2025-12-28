#!/usr/bin/env python3
"""
Deep architectural analysis: Compare training vs test inference path step by step.
This helps identify if there's an architectural issue causing generalization failure.
"""
import argparse
import os
import numpy as np
import torch
from pathlib import Path

from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig
from tools.train_lanelm_culane_v3 import build_frozen_clrernet_backbone
from tools.train_lanelm_v4_fixed import extract_p5_feat, visual_first_decode
from libs.models.detectors.lanelm_detector import autoregressive_decode, coords_to_lane_normalized
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


def debug_architectural_issue(
    lanelm_ckpt,
    config_path,
    backbone_ckpt,
    data_root,
    train_list_path,
    test_list_path,
    device,
    save_dir
):
    """Deep architectural analysis."""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"ARCHITECTURAL ANALYSIS: Training vs Test")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    ckpt = torch.load(lanelm_ckpt, map_location=device)
    cfg = ckpt['config']
    
    # Build models
    clrernet = build_frozen_clrernet_backbone(config_path, backbone_ckpt, device)
    
    lanelm = LaneLMModel(
        nbins_x=cfg['nbins_x'],
        max_y_tokens=cfg['num_points'] + 1,
        embed_dim=cfg['embed_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        max_seq_len=80,
        visual_in_channels=(64,),
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
    
    # Load datasets
    train_dataset = CulaneDataset(
        data_root=data_root,
        data_list=train_list_path,
        pipeline=[dict(type="albumentation", pipelines=clean_pipeline)],
        diff_file=None,
        test_mode=False,
    )
    
    test_dataset = CulaneDataset(
        data_root=data_root,
        data_list=test_list_path,
        pipeline=[dict(type="albumentation", pipelines=clean_pipeline)],
        diff_file=None,
        test_mode=False,  # Load GT for comparison
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}\n")
    
    # ========== ANALYZE MULTIPLE SAMPLES ==========
    train_samples_to_check = [0, 10, 20]
    test_samples_to_check = [0, 10, 20, 50]
    
    print("="*80)
    print("TRAINING SAMPLES ANALYSIS")
    print("="*80)
    
    train_results = []
    for idx in train_samples_to_check:
        if idx >= len(train_dataset):
            continue
        
        sample = train_dataset[idx]
        sub_img_name = sample.get('sub_img_name', f'train_{idx}')
        
        # Predict
        img_tensor = torch.from_numpy(sample['img']).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            feats = extract_p5_feat(clrernet, img_tensor)
            visual_tokens = lanelm.encode_visual_tokens(feats)
            
            x_tokens_all, y_tokens_all = autoregressive_decode(
                lanelm_model=lanelm,
                visual_tokens=visual_tokens,
                tokenizer_cfg=tokenizer.cfg,
                max_lanes=4,
                temperature=0.0,
            )
        
        # Get GT
        gt_points = sample['gt_points']
        gt_lane_count = len([l for l in gt_points if len(l) >= 4])
        
        # Decode predictions
        pred_coords_list = []
        for l in range(x_tokens_all.shape[1]):
            x_tok = x_tokens_all[0, l].numpy()
            y_tok = y_tokens_all[0, l].numpy()
            coords = tokenizer.decode_single_lane(x_tok, y_tok, smooth=True)
            if len(coords) >= 2:
                pred_coords_list.append(coords)
        
        print(f"\nTrain Sample {idx}: {sub_img_name}")
        print(f"  GT lanes: {gt_lane_count}")
        print(f"  Pred lanes: {len(pred_coords_list)}")
        
        if len(pred_coords_list) > 0:
            first_pred = pred_coords_list[0]
            print(f"  Pred Lane 0: X[{first_pred[:, 0].min():.1f}, {first_pred[:, 0].max():.1f}], Y[{first_pred[:, 1].min():.1f}, {first_pred[:, 1].max():.1f}]")
        
        train_results.append({
            'idx': idx,
            'gt_count': gt_lane_count,
            'pred_count': len(pred_coords_list),
        })
    
    print("\n" + "="*80)
    print("TEST SAMPLES ANALYSIS")
    print("="*80)
    
    test_results = []
    for idx in test_samples_to_check:
        if idx >= len(test_dataset):
            continue
        
        sample = test_dataset[idx]
        sub_img_name = sample.get('sub_img_name', f'test_{idx}')
        
        # Predict
        img_tensor = torch.from_numpy(sample['img']).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            feats = extract_p5_feat(clrernet, img_tensor)
            visual_tokens = lanelm.encode_visual_tokens(feats)
            
            x_tokens_all, y_tokens_all = autoregressive_decode(
                lanelm_model=lanelm,
                visual_tokens=visual_tokens,
                tokenizer_cfg=tokenizer.cfg,
                max_lanes=4,
                temperature=0.0,
            )
        
        # Get GT
        gt_points = sample['gt_points']
        gt_lane_count = len([l for l in gt_points if len(l) >= 4])
        
        # Decode predictions
        pred_coords_list = []
        for l in range(x_tokens_all.shape[1]):
            x_tok = x_tokens_all[0, l].numpy()
            y_tok = y_tokens_all[0, l].numpy()
            coords = tokenizer.decode_single_lane(x_tok, y_tok, smooth=True)
            if len(coords) >= 2:
                pred_coords_list.append(coords)
        
        print(f"\nTest Sample {idx}: {sub_img_name}")
        print(f"  GT lanes: {gt_lane_count}")
        print(f"  Pred lanes: {len(pred_coords_list)}")
        
        if len(pred_coords_list) > 0:
            first_pred = pred_coords_list[0]
            print(f"  Pred Lane 0: X[{first_pred[:, 0].min():.1f}, {first_pred[:, 0].max():.1f}], Y[{first_pred[:, 1].min():.1f}, {first_pred[:, 1].max():.1f}]")
        
        test_results.append({
            'idx': idx,
            'gt_count': gt_lane_count,
            'pred_count': len(pred_coords_list),
        })
    
    # ========== COMPARE VISUAL TOKENS ==========
    print("\n" + "="*80)
    print("VISUAL TOKENS COMPARISON")
    print("="*80)
    
    train_sample = train_dataset[0]
    test_sample = test_dataset[0] if len(test_dataset) > 0 else None
    
    train_img = torch.from_numpy(train_sample['img']).permute(2, 0, 1).float() / 255.0
    train_img = train_img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        train_feats = extract_p5_feat(clrernet, train_img)
        train_vis_tokens = lanelm.encode_visual_tokens(train_feats)
        
        print(f"Train visual tokens:")
        print(f"  Shape: {train_vis_tokens.shape}")
        print(f"  Mean: {train_vis_tokens.mean():.6f}")
        print(f"  Std: {train_vis_tokens.std():.6f}")
        print(f"  Min: {train_vis_tokens.min():.6f}, Max: {train_vis_tokens.max():.6f}")
    
    if test_sample is not None:
        test_img = torch.from_numpy(test_sample['img']).permute(2, 0, 1).float() / 255.0
        test_img = test_img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            test_feats = extract_p5_feat(clrernet, test_img)
            test_vis_tokens = lanelm.encode_visual_tokens(test_feats)
            
            print(f"\nTest visual tokens:")
            print(f"  Shape: {test_vis_tokens.shape}")
            print(f"  Mean: {test_vis_tokens.mean():.6f}")
            print(f"  Std: {test_vis_tokens.std():.6f}")
            print(f"  Min: {test_vis_tokens.min():.6f}, Max: {test_vis_tokens.max():.6f}")
            
            # Compare
            diff = (train_vis_tokens - test_vis_tokens).abs()
            print(f"\nDifference:")
            print(f"  Mean diff: {diff.mean():.6f}")
            print(f"  Max diff: {diff.max():.6f}")
    
    # ========== CHECK MODEL CONFIDENCE ==========
    print("\n" + "="*80)
    print("MODEL CONFIDENCE ANALYSIS")
    print("="*80)
    
    # Check logits distribution
    train_img = torch.from_numpy(train_dataset[0]['img']).permute(2, 0, 1).float() / 255.0
    train_img = train_img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        feats = extract_p5_feat(clrernet, train_img)
        visual_tokens = lanelm.encode_visual_tokens(feats)
        
        # Get logits for first lane, first timestep
        B = visual_tokens.shape[0]
        T = tokenizer.cfg.num_steps
        y_fixed = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        lane_ids = torch.zeros(B, dtype=torch.long, device=device)
        x_in = torch.zeros(B, T, dtype=torch.long, device=device)
        
        logits_x, _ = lanelm(visual_tokens, x_in, y_fixed, lane_indices=lane_ids)
        
        # First timestep logits
        first_logits = logits_x[0, 0, :].cpu().numpy()
        probs = torch.softmax(logits_x[0, 0, :], dim=0).cpu().numpy()
        
        print(f"First timestep logits (train sample):")
        print(f"  Shape: {first_logits.shape}")
        print(f"  Mean: {first_logits.mean():.6f}")
        print(f"  Std: {first_logits.std():.6f}")
        print(f"  Max prob: {probs.max():.6f}")
        print(f"  Entropy: {-np.sum(probs * np.log(probs + 1e-10)):.6f}")
        print(f"  Top-5 tokens: {np.argsort(probs)[-5:][::-1]}")
        print(f"  Top-5 probs: {np.sort(probs)[-5:][::-1]}")
    
    if test_sample is not None:
        test_img = torch.from_numpy(test_sample['img']).permute(2, 0, 1).float() / 255.0
        test_img = test_img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            feats = extract_p5_feat(clrernet, test_img)
            visual_tokens = lanelm.encode_visual_tokens(feats)
            
            logits_x, _ = lanelm(visual_tokens, x_in, y_fixed, lane_indices=lane_ids)
            
            first_logits = logits_x[0, 0, :].cpu().numpy()
            probs = torch.softmax(logits_x[0, 0, :], dim=0).cpu().numpy()
            
            print(f"\nFirst timestep logits (test sample):")
            print(f"  Mean: {first_logits.mean():.6f}")
            print(f"  Std: {first_logits.std():.6f}")
            print(f"  Max prob: {probs.max():.6f}")
            print(f"  Entropy: {-np.sum(probs * np.log(probs + 1e-10)):.6f}")
            print(f"  Top-5 tokens: {np.argsort(probs)[-5:][::-1]}")
            print(f"  Top-5 probs: {np.sort(probs)[-5:][::-1]}")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Architectural analysis")
    parser.add_argument("--lanelm-ckpt", required=True)
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--train-list", default="dataset/list/train_100.txt")
    parser.add_argument("--test-list", default="dataset/list/test_100.txt")
    parser.add_argument("--save-dir", default="work_dirs/debug_architectural")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    debug_architectural_issue(
        lanelm_ckpt=args.lanelm_ckpt,
        config_path=args.config,
        backbone_ckpt=args.checkpoint,
        data_root=args.data_root,
        train_list_path=args.train_list,
        test_list_path=args.test_list,
        device=device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()








