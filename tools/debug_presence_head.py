#!/usr/bin/env python3
"""Debug script to analyze presence head behavior and filtering.

This script:
1. Loads a trained LaneLM model
2. Runs inference on a few test images
3. Logs presence logits for each lane
4. Tests different presence thresholds
5. Visualizes predictions with/without presence filtering
"""

import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig
from libs.models.detectors.lanelm_detector import autoregressive_decode
from libs.datasets.culane_dataset import CulaneDataset
from libs.utils.visualizer import draw_lane
from tools.train_lanelm_culane_v3 import LaneLMHyperParams


def load_model_and_data(ckpt_path, config_path, checkpoint_path, device):
    """Load model, tokenizer, and a sample dataset."""
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    
    # Extract config from checkpoint if available
    if "config" in ckpt:
        hparams_dict = ckpt["config"]
        hparams = LaneLMHyperParams(**hparams_dict)
    else:
        # Default config
        hparams = LaneLMHyperParams(
            nbins_x=200,
            num_points=40,
            embed_dim=256,
            num_layers=4,
            max_lanes=4,
        )
    
    # Build tokenizer
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=hparams.img_w,
        img_h=hparams.img_h,
        num_steps=hparams.num_points,
        nbins_x=200,
        x_mode="absolute",
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    
    # Build model
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
    
    # Load weights
    lanelm.load_state_dict(state_dict, strict=False)
    lanelm.eval()
    
    # Load CLRerNet backbone (frozen)
    from tools.train_lanelm_v4_fixed import build_frozen_clrernet_backbone, extract_p5_feat
    clrernet = build_frozen_clrernet_backbone(config_path, checkpoint_path, device)
    
    # Load dataset
    dataset = CulaneDataset(
        data_root="dataset",
        data_list="dataset/list/test_100.txt",
        test_mode=True,
    )
    
    return lanelm, tokenizer, tokenizer_cfg, clrernet, dataset, hparams


def analyze_presence_logits(lanelm, visual_tokens, tokenizer_cfg, max_lanes, device):
    """Analyze presence logits for all lanes without filtering."""
    B, _, _ = visual_tokens.shape
    T = tokenizer_cfg.num_steps
    pad_token_x = tokenizer_cfg.pad_token_x
    
    all_presence_logits = []
    all_presence_probs = []
    all_x_tokens = []
    
    for lane_idx in range(max_lanes):
        y_fixed = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        x_out = torch.zeros(B, T, dtype=torch.long, device=device)
        lane_indices = torch.full((B,), lane_idx, dtype=torch.long, device=device)
        
        # Decode full sequence
        for t in range(T):
            x_in = torch.zeros_like(x_out)
            if t > 0:
                x_in[:, 1:t+1] = x_out[:, :t]
                x_in[:, 0] = x_out[:, 0]
            
            # Get presence logits on final step
            if t == T - 1:
                logits_x, _, presence_logits = lanelm(
                    visual_tokens, x_in, y_fixed, lane_indices=lane_indices,
                    return_presence=True,
                )
            else:
                logits_x, _ = lanelm(
                    visual_tokens, x_in, y_fixed, lane_indices=lane_indices,
                )
            
            pred_x = torch.argmax(logits_x[:, t, :], dim=-1)
            pred_x = pred_x.clamp(0, lanelm.nbins_x - 1)
            x_out[:, t] = pred_x
        
        # Store presence info
        if presence_logits is not None:
            presence_probs = torch.sigmoid(presence_logits).squeeze(-1)  # (B,)
            all_presence_logits.append(presence_logits.cpu().numpy())
            all_presence_probs.append(presence_probs.cpu().numpy())
        else:
            all_presence_logits.append(None)
            all_presence_probs.append(None)
        
        all_x_tokens.append(x_out.cpu().numpy())
    
    return all_presence_logits, all_presence_probs, all_x_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="work_dirs/lanelm_v4_fixed/lanelm_v4_best.pth")
    parser.add_argument("--config", type=str, default="configs/clrernet/clrernet_culane_dla34.py")
    parser.add_argument("--checkpoint", type=str, default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5, 0.7])
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and data
    print("Loading model and data...")
    lanelm, tokenizer, tokenizer_cfg, clrernet, dataset, hparams = load_model_and_data(
        args.ckpt, args.config, args.checkpoint, device
    )
    
    print(f"\n{'='*80}")
    print("PRESENCE HEAD DEBUG ANALYSIS")
    print(f"{'='*80}\n")
    
    # Analyze first N samples
    num_samples = min(args.num_samples, len(dataset))
    all_presence_stats = []
    
    for sample_idx in range(num_samples):
        sample = dataset[sample_idx]
        img = sample["inputs"]
        gt_points = sample["gt_points"]
        filename = sample.get("metainfo", {}).get("sub_img_name", f"sample_{sample_idx}")
        
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx + 1}/{num_samples}: {filename}")
        print(f"{'='*80}")
        print(f"GT lanes: {len(gt_points)}")
        
        # Prepare image
        if isinstance(img, torch.Tensor):
            img_tensor = img.unsqueeze(0).to(device)
        else:
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
        
        if img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float() / 255.0
        
        # Extract features
        from tools.train_lanelm_v4_fixed import extract_p5_feat
        feats = extract_p5_feat(clrernet, img_tensor)
        visual_tokens = lanelm.encode_visual_tokens(feats)
        
        # Analyze presence logits
        presence_logits_list, presence_probs_list, x_tokens_list = analyze_presence_logits(
            lanelm, visual_tokens, tokenizer_cfg, hparams.max_lanes, device
        )
        
        # Print presence statistics
        print(f"\nPresence Logits and Probabilities:")
        print(f"{'Lane':<6} {'Logit':<12} {'Prob':<12} {'Valid Tokens':<15} {'Pass (0.5)':<12}")
        print("-" * 70)
        
        for lane_idx in range(hparams.max_lanes):
            if presence_logits_list[lane_idx] is not None:
                logit_val = presence_logits_list[lane_idx][0, 0]
                prob_val = presence_probs_list[lane_idx][0]
                x_tokens = x_tokens_list[lane_idx][0]
                valid_tokens = (x_tokens != tokenizer_cfg.pad_token_x).sum()
                passes = "YES" if prob_val > 0.5 else "NO"
                
                print(f"{lane_idx:<6} {logit_val:>12.4f} {prob_val:>12.4f} {valid_tokens:>15} {passes:>12}")
            else:
                print(f"{lane_idx:<6} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'N/A':<12}")
        
        # Test different thresholds
        print(f"\nThreshold Analysis:")
        print(f"{'Threshold':<12} {'Lanes Passed':<15} {'Total Valid Tokens':<20}")
        print("-" * 50)
        
        for threshold in args.thresholds:
            passed_lanes = 0
            total_valid = 0
            for lane_idx in range(hparams.max_lanes):
                if presence_probs_list[lane_idx] is not None:
                    prob = presence_probs_list[lane_idx][0]
                    if prob > threshold:
                        passed_lanes += 1
                        x_tokens = x_tokens_list[lane_idx][0]
                        valid = (x_tokens != tokenizer_cfg.pad_token_x).sum()
                        total_valid += valid
            
            print(f"{threshold:<12.2f} {passed_lanes:<15} {total_valid:<20}")
        
        # Store stats
        all_presence_stats.append({
            "filename": filename,
            "gt_lanes": len(gt_points),
            "presence_probs": [p[0] if p is not None else 0.0 for p in presence_probs_list],
            "valid_tokens": [(x != tokenizer_cfg.pad_token_x).sum() for x in x_tokens_list],
        })
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    all_probs = []
    for stats in all_presence_stats:
        all_probs.extend(stats["presence_probs"])
    
    if all_probs:
        print(f"Presence Probabilities:")
        print(f"  Mean: {np.mean(all_probs):.4f}")
        print(f"  Std:  {np.std(all_probs):.4f}")
        print(f"  Min:  {np.min(all_probs):.4f}")
        print(f"  Max:  {np.max(all_probs):.4f}")
        print(f"  Median: {np.median(all_probs):.4f}")
        print(f"\nLanes passing threshold=0.5: {sum(1 for p in all_probs if p > 0.5)}/{len(all_probs)}")
        print(f"Lanes passing threshold=0.3: {sum(1 for p in all_probs if p > 0.3)}/{len(all_probs)}")
        print(f"Lanes passing threshold=0.1: {sum(1 for p in all_probs if p > 0.1)}/{len(all_probs)}")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}")
    
    if all_probs:
        mean_prob = np.mean(all_probs)
        if mean_prob < 0.3:
            print("⚠️  WARNING: Mean presence probability is very low!")
            print("   - Presence head may not be trained properly")
            print("   - Consider lowering threshold to 0.1-0.3")
            print("   - Or disable presence filtering temporarily")
        elif mean_prob < 0.5:
            print("⚠️  WARNING: Mean presence probability is below 0.5")
            print("   - Consider lowering threshold to 0.3-0.4")
        else:
            print("✓ Presence probabilities look reasonable")
            print("   - Threshold 0.5 should work, but 0.3-0.4 might be safer")


if __name__ == "__main__":
    main()

