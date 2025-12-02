#!/usr/bin/env python3
"""
Quick visualization of test predictions vs GT.
Uses the same logic as training visualization.
"""
import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig
from libs.datasets import CulaneDataset
from libs.datasets.metrics.culane_metric import load_culane_img_data
from libs.models.detectors.lanelm_detector import autoregressive_decode
from tools.train_lanelm_culane_v3 import build_frozen_clrernet_backbone
from configs.clrernet.culane.dataset_culane_clrernet import (
    compose_cfg, crop_bbox, img_scale
)

# Clean pipeline (no augmentation for test) - matches train_lanelm_v4_fixed.py
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


def extract_p5_feat(model, imgs: torch.Tensor):
    """Extract P5 Only (highest level, lowest resolution) for reduced noise."""
    with torch.no_grad():
        feats = model.extract_feat(imgs)  # Returns [P3, P4, P5]
        if isinstance(feats, (tuple, list)):
            feats = (feats[-1],)  # P5 only
    return feats


def visualize_test_samples(
    model, clrernet_model, dataset, tokenizer, device, 
    save_dir, max_samples=10, max_lanes=4
):
    """Visualize predictions vs ground truth for test samples."""
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter samples with GT
    data_root = dataset.img_prefix if hasattr(dataset, 'img_prefix') else 'dataset'
    valid_samples = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        sub_img_name = sample.get('sub_img_name', '')
        
        # Load GT manually from .lines.txt file
        gt_path = os.path.join(data_root, sub_img_name.replace('.jpg', '.lines.txt'))
        if os.path.exists(gt_path):
            gt_data = load_culane_img_data(gt_path)
            if gt_data and len(gt_data) > 0:
                has_valid_lane = False
                for lane in gt_data:
                    if lane and len(lane) >= 2:
                        has_valid_lane = True
                        break
                if has_valid_lane:
                    valid_samples.append((idx, gt_data))
    
    print(f"Found {len(valid_samples)} samples with valid GT out of {len(dataset)} total")
    
    num_samples = min(max_samples, len(valid_samples))
    print(f"Visualizing {num_samples} samples...")
    
    try:
        import cv2
    except ImportError:
        print("ERROR: cv2 not found. Please install opencv-python:")
        print("  conda activate clrernet")
        print("  pip install opencv-python")
        return
    
    for sample_idx_idx, (sample_idx, gt_data) in enumerate(valid_samples[:num_samples]):
        sample = dataset[sample_idx]
        sample['gt_points'] = gt_data
        img_tensor = torch.from_numpy(sample['img']).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        sub_img_name = sample.get('sub_img_name', f'sample_{sample_idx}')
        
        with torch.no_grad():
            # Extract features (P5 only, matches training)
            feats = extract_p5_feat(clrernet_model, img_tensor)
            visual_tokens = model.encode_visual_tokens(feats)
            
            # Use EXACT same decode logic as test script
            x_tokens_all, y_tokens_all = autoregressive_decode(
                lanelm_model=model,
                visual_tokens=visual_tokens,
                tokenizer_cfg=tokenizer.cfg,
                max_lanes=max_lanes,
                temperature=0.0,
            )
            
            # Process predictions exactly like test script
            x_tok = x_tokens_all[0].numpy()  # Shape: (max_lanes, T)
            y_tok = y_tokens_all[0].numpy()  # Shape: (max_lanes, T)
            
            # Visualize image
            img_vis = (img_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
            
            # Draw predictions (colored) - using same decode as test script
            colors = [(0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]  # BGR format
            pred_count = 0
            for l in range(x_tok.shape[0]):
                coords_resized = tokenizer.decode_single_lane(x_tok[l], y_tok[l], smooth=True)
                if coords_resized.shape[0] >= 2:
                    pred_count += 1
                    for k in range(len(coords_resized) - 1):
                        p1 = (int(coords_resized[k][0]), int(coords_resized[k][1]))
                        p2 = (int(coords_resized[k+1][0]), int(coords_resized[k+1][1]))
                        if 0 <= p1[0] < 800 and 0 <= p2[0] < 800 and 0 <= p1[1] < 320 and 0 <= p2[1] < 320:
                            cv2.line(img_vis, p1, p2, colors[l % 4], 2)
            
            # Draw GT (GREEN)
            gt_count = 0
            if 'gt_points' in sample and sample['gt_points']:
                for lane in sample['gt_points'][:max_lanes]:
                    if not lane or len(lane) < 2:
                        continue
                    
                    # Convert GT from original to resized coordinates
                    pts_orig = np.array(lane, dtype=np.float32).reshape(-1, 2)
                    
                    # Apply crop and resize transformation (matches pipeline)
                    x_min, y_min, x_max, y_max = crop_bbox
                    crop_h = float(y_max - y_min)  # 320
                    ori_w = 1640.0
                    
                    # Crop: subtract y_min from y coordinates
                    pts_cropped = pts_orig.copy()
                    pts_cropped[:, 1] = pts_cropped[:, 1] - y_min
                    
                    # Resize: scale to (800, 320)
                    pts_resized = pts_cropped.copy()
                    pts_resized[:, 0] = pts_resized[:, 0] / ori_w * 800.0  # X: 1640 -> 800
                    pts_resized[:, 1] = pts_resized[:, 1] / crop_h * 320.0  # Y: 320 -> 320 (no change after crop)
                    
                    # Filter points within resized image bounds
                    valid_mask = (pts_resized[:, 0] >= 0) & (pts_resized[:, 0] < 800) & \
                                 (pts_resized[:, 1] >= 0) & (pts_resized[:, 1] < 320)
                    pts_resized = pts_resized[valid_mask]
                    
                    if len(pts_resized) >= 2:
                        gt_count += 1
                        for k in range(len(pts_resized) - 1):
                            p1 = (int(pts_resized[k][0]), int(pts_resized[k][1]))
                            p2 = (int(pts_resized[k+1][0]), int(pts_resized[k+1][1]))
                            if 0 <= p1[0] < 800 and 0 <= p2[0] < 800 and 0 <= p1[1] < 320 and 0 <= p2[1] < 320:
                                cv2.line(img_vis, p1, p2, (0, 255, 0), 3)  # GREEN, thicker line for GT
            
            # Add text info
            cv2.putText(img_vis, f"GREEN=GT ({gt_count} lanes)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img_vis, f"RED/BLUE/MAGENTA/CYAN=Pred ({pred_count} lanes)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(img_vis, sub_img_name, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save visualization
            safe_name = sub_img_name.replace('/', '_').replace('\\', '_')
            save_path = os.path.join(save_dir, f"{sample_idx_idx:04d}_{safe_name}.jpg")
            cv2.imwrite(save_path, img_vis)
            
            if (sample_idx_idx + 1) % 5 == 0:
                print(f"  Processed {sample_idx_idx + 1}/{num_samples} samples...")
    
    print(f"✓ Visualizations saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Quick visualize LaneLM test predictions")
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--lanelm-ckpt", default="work_dirs/lanelm_v4_fixed/lanelm_v4_best.pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--data-list", default="dataset/list/train_100.txt", help="Use train_100.txt for simpler images")
    parser.add_argument("--work-dir", default="work_dirs/lanelm_v4_test_vis_quick")
    parser.add_argument("--max-samples", type=int, default=10, help="Max number of samples to visualize")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Load LaneLM checkpoint to get config
    print("Loading LaneLM checkpoint...")
    ckpt = torch.load(args.lanelm_ckpt, map_location=device)
    cfg = ckpt['config']
    
    # Build models
    print("Building models...")
    clrernet = build_frozen_clrernet_backbone(args.config, args.checkpoint, device)
    
    model = LaneLMModel(
        nbins_x=cfg['nbins_x'],
        max_y_tokens=cfg['num_points'] + 1,
        embed_dim=cfg['embed_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        ffn_dim=cfg['ffn_dim'],
        max_seq_len=80,
        visual_in_channels=(64,),  # P5 Only
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    
    # Build tokenizer
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=800,
        img_h=320,
        num_steps=40,
        nbins_x=cfg['nbins_x'],
        x_mode='absolute',
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    
    # Build dataset
    print("Building dataset...")
    dataset = CulaneDataset(
        data_root=args.data_root,
        data_list=args.data_list,
        pipeline=[dict(type="albumentation", pipelines=clean_pipeline)],
        diff_file=None,
        test_mode=True,
    )
    
    # Visualize
    visualize_test_samples(
        model, clrernet, dataset, tokenizer, device,
        args.work_dir, args.max_samples, max_lanes=4
    )


if __name__ == "__main__":
    main()

