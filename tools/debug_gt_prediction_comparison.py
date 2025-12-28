#!/usr/bin/env python3
"""
Debug script to compare GT and prediction loading/format in CULaneMetric.
This helps identify if GT and prediction are in the same coordinate space.
"""
import argparse
import os
import numpy as np
import torch
from pathlib import Path

from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig
from libs.datasets.metrics.culane_metric import load_culane_img_data
from libs.models.detectors.lanelm_detector import autoregressive_decode, coords_to_lane_normalized
from tools.train_lanelm_culane_v3 import build_frozen_clrernet_backbone
from tools.train_lanelm_v4_fixed import extract_p5_feat
from configs.clrernet.culane.dataset_culane_clrernet import (
    compose_cfg, crop_bbox, img_scale
)

# Clean pipeline
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


def debug_gt_prediction(
    lanelm_ckpt,
    config_path,
    backbone_ckpt,
    data_root,
    list_path,
    sample_idx,
    device,
    save_dir
):
    """Compare GT and prediction loading/format."""
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
    
    # Load dataset
    from libs.datasets import CulaneDataset
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
    print(f"\n=== Debugging GT vs Prediction: {sub_img_name} ===\n")
    
    # ========== 1. GT LOADING ==========
    print("--- 1. GT Loading (from .lines.txt file) ---")
    gt_path = os.path.join(data_root, sub_img_name.replace('.jpg', '.lines.txt'))
    if not os.path.exists(gt_path):
        print(f"  ERROR: GT file not found: {gt_path}")
        return
    
    gt_data = load_culane_img_data(gt_path)
    print(f"  GT file: {gt_path}")
    print(f"  GT lanes: {len(gt_data)}")
    
    for i, lane in enumerate(gt_data[:4]):
        if len(lane) >= 2:
            lane_arr = np.array(lane)
            xs = lane_arr[:, 0]
            ys = lane_arr[:, 1]
            print(f"  GT Lane {i}: {len(lane)} points")
            print(f"    X range (original): [{xs.min():.1f}, {xs.max():.1f}]")
            print(f"    Y range (original): [{ys.min():.1f}, {ys.max():.1f}]")
            print(f"    First point: ({xs[0]:.1f}, {ys[0]:.1f})")
            print(f"    Last point: ({xs[-1]:.1f}, {ys[-1]:.1f})")
    
    # ========== 2. PREDICTION ==========
    print("\n--- 2. Prediction (from model) ---")
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
    
    # Convert to normalized Lane objects (same as test)
    lanes_pred = []
    for l in range(x_tokens_all.shape[1]):
        x_tok = x_tokens_all[0, l].numpy()
        y_tok = y_tokens_all[0, l].numpy()
        
        coords_resized = tokenizer.decode_single_lane(x_tok, y_tok, smooth=True)
        lane = coords_to_lane_normalized(
            coords_resized=coords_resized,
            tokenizer_cfg=tokenizer.cfg,
            crop_bbox=crop_bbox,
            img_w=800,
            img_h=320,
            ori_img_w=1640,
            ori_img_h=590,
        )
        
        if lane is not None and lane.points is not None and len(lane.points) >= 2:
            lanes_pred.append(lane)
            print(f"  Pred Lane {l}: {len(lane.points)} points (normalized)")
            print(f"    X range (normalized): [{lane.points[:, 0].min():.4f}, {lane.points[:, 0].max():.4f}]")
            print(f"    Y range (normalized): [{lane.points[:, 1].min():.4f}, {lane.points[:, 1].max():.4f}]")
            print(f"    min_y: {lane.min_y:.4f}, max_y: {lane.max_y:.4f}")
    
    # ========== 3. PREDICTION STRING ==========
    print("\n--- 3. Prediction String (get_prediction_string) ---")
    from libs.datasets.metrics.culane_metric import CULaneMetric
    
    # Create a dummy CULaneMetric to use get_prediction_string
    class DummyCULaneMetric:
        def __init__(self):
            self.ori_w = 1640
            self.ori_h = 590
            self.y_step = 2
        
        def get_prediction_string(self, lanes):
            ys = np.arange(0, self.ori_h, self.y_step) / self.ori_h
            out = []
            for lane in lanes:
                lane_min_y = lane.min_y
                lane_max_y = lane.max_y
                ys_in_range = ys[(ys >= lane_min_y) & (ys <= lane_max_y)]
                
                if len(ys_in_range) < 2:
                    continue
                
                xs = lane(ys_in_range)
                valid_mask = (xs >= 0) & (xs < 1)
                xs = xs * self.ori_w
                lane_xs = xs[valid_mask]
                lane_ys = ys_in_range[valid_mask] * self.ori_h
                
                if len(lane_xs) < 2:
                    continue
                
                lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
                
                lane_str = " ".join(
                    ["{:.5f} {:.5f}".format(x, y) for x, y in zip(lane_xs, lane_ys)]
                )
                if lane_str != "":
                    out.append(lane_str)
            return "\n".join(out) if len(out) > 0 else ""
    
    dummy_metric = DummyCULaneMetric()
    pred_string = dummy_metric.get_prediction_string(lanes_pred)
    
    print(f"  Prediction string length: {len(pred_string)} chars")
    pred_lines = pred_string.split('\n')
    print(f"  Prediction lanes: {len(pred_lines)}")
    
    for i, line in enumerate(pred_lines[:4]):
        coords = line.split()
        xs = [float(coords[j]) for j in range(0, len(coords), 2)]
        ys = [float(coords[j+1]) for j in range(0, len(coords), 2)]
        print(f"  Pred String Lane {i}: {len(xs)} points")
        print(f"    X range (original): [{min(xs):.1f}, {max(xs):.1f}]")
        print(f"    Y range (original): [{min(ys):.1f}, {max(ys):.1f}]")
        print(f"    First point: ({xs[0]:.1f}, {ys[0]:.1f})")
        print(f"    Last point: ({xs[-1]:.1f}, {ys[-1]:.1f})")
    
    # ========== 4. COMPARISON ==========
    print("\n--- 4. GT vs Prediction Comparison (Original Space) ---")
    
    for i in range(min(len(gt_data), len(pred_lines))):
        gt_lane = gt_data[i]
        pred_line = pred_lines[i] if i < len(pred_lines) else None
        
        if pred_line is None:
            print(f"  Lane {i}: GT has {len(gt_lane)} points, Prediction MISSING")
            continue
        
        gt_arr = np.array(gt_lane)
        pred_coords = pred_line.split()
        pred_xs = [float(pred_coords[j]) for j in range(0, len(pred_coords), 2)]
        pred_ys = [float(pred_coords[j+1]) for j in range(0, len(pred_coords), 2)]
        
        print(f"  Lane {i}:")
        print(f"    GT: {len(gt_lane)} points, X[{gt_arr[:, 0].min():.1f}, {gt_arr[:, 0].max():.1f}], Y[{gt_arr[:, 1].min():.1f}, {gt_arr[:, 1].max():.1f}]")
        print(f"    Pred: {len(pred_xs)} points, X[{min(pred_xs):.1f}, {max(pred_xs):.1f}], Y[{min(pred_ys):.1f}, {max(pred_ys):.1f}]")
        
        # Check if ranges overlap
        gt_x_min, gt_x_max = gt_arr[:, 0].min(), gt_arr[:, 0].max()
        pred_x_min, pred_x_max = min(pred_xs), max(pred_xs)
        gt_y_min, gt_y_max = gt_arr[:, 1].min(), gt_arr[:, 1].max()
        pred_y_min, pred_y_max = min(pred_ys), max(pred_ys)
        
        x_overlap = not (pred_x_max < gt_x_min or pred_x_min > gt_x_max)
        y_overlap = not (pred_y_max < gt_y_min or pred_y_min > gt_y_max)
        
        print(f"    X overlap: {x_overlap}, Y overlap: {y_overlap}")
        
        if not x_overlap or not y_overlap:
            print(f"    ⚠️  WARNING: No overlap in {'X' if not x_overlap else ''}{' and ' if not x_overlap and not y_overlap else ''}{'Y' if not y_overlap else ''}!")
        
        # Calculate mean distance (if we can match points)
        if x_overlap and y_overlap:
            # Simple distance calculation (not perfect but gives an idea)
            gt_center_x = (gt_x_min + gt_x_max) / 2
            gt_center_y = (gt_y_min + gt_y_max) / 2
            pred_center_x = (pred_x_min + pred_x_max) / 2
            pred_center_y = (pred_y_min + pred_y_max) / 2
            
            center_dist = np.sqrt((gt_center_x - pred_center_x)**2 + (gt_center_y - pred_center_y)**2)
            print(f"    Center distance: {center_dist:.1f} px")
    
    print("\n=== Debug complete ===")


def main():
    parser = argparse.ArgumentParser(description="Debug GT vs Prediction comparison")
    parser.add_argument("--lanelm-ckpt", required=True, help="LaneLM checkpoint path")
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--list-path", default="dataset/list/train_100.txt")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to debug")
    parser.add_argument("--save-dir", default="work_dirs/debug_gt_prediction")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    debug_gt_prediction(
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

