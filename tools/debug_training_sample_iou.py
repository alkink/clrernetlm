#!/usr/bin/env python3
"""
Debug: Calculate IoU for a training sample using the same evaluation as test.
This helps identify if the problem is in training or test evaluation.
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
from libs.datasets.metrics.culane_metric import load_culane_img_data, culane_metric
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


def debug_training_sample_iou(
    lanelm_ckpt,
    config_path,
    backbone_ckpt,
    data_root,
    list_path,
    sample_idx,
    device,
    smooth: bool = False,
):
    """Calculate IoU for a training sample using test evaluation."""
    print(f"\n{'='*80}")
    print(f"TRAINING SAMPLE IoU DEBUG")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    ckpt = torch.load(lanelm_ckpt, map_location=device)
    cfg = ckpt['config']
    sd = ckpt.get('model_state_dict', {})

    # Infer how many visual levels were used from checkpoint weights
    # (P5-only => 1 level, Full FPN => 3 levels)
    n_levels = 1
    try:
        if "visual_encoder.level_embed.weight" in sd:
            n_levels = int(sd["visual_encoder.level_embed.weight"].shape[0])
    except Exception:
        n_levels = 1
    visual_in_channels = tuple([64] * n_levels)
    
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
        visual_in_channels=visual_in_channels,
    )
    lanelm.load_state_dict(sd, strict=True)
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
    
    # Get GT from dataset (resized space)
    gt_points_resized = sample['gt_points']
    print(f"GT lanes (resized space): {len(gt_points_resized)}")
    for i, lane in enumerate(gt_points_resized[:4]):
        if len(lane) >= 2:
            pts = np.array(lane, dtype=np.float32).reshape(-1, 2)
            print(f"  GT Lane {i}: {len(pts)} points")
            print(f"    X: [{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}]")
            print(f"    Y: [{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}]")

    # ========== Round-trip sanity: GT -> tokens -> dequantize -> compare to spline ==========
    # Model-independent: isolates tokenizer quantization/dequantization error.
    try:
        print("\n--- Tokenizer Round-Trip Sanity (GT lane 0) ---")
        if len(gt_points_resized) > 0 and len(gt_points_resized[0]) >= 2:
            pts0 = np.array(gt_points_resized[0], dtype=np.float32).reshape(-1, 2)
            x_toks0, y_toks0 = tokenizer.encode_single_lane(pts0)
            sample_ys = tokenizer._compute_sample_ys()
            spline = tokenizer._fit_spline(pts0)
            if spline is not None:
                xs_cont = spline(sample_ys)
                valid = (x_toks0 != tokenizer.cfg.pad_token_x) & (y_toks0 < tokenizer.T)
                if valid.any():
                    decoded_x = (x_toks0.astype(np.float32) / max(1, tokenizer.cfg.nbins_x - 1)) * (tokenizer.cfg.img_w - 1)
                    abs_err = np.abs(decoded_x[valid] - xs_cont[valid])
                    print(
                        f"valid_steps={int(valid.sum())}/{tokenizer.T}  "
                        f"mean_abs_err={abs_err.mean():.3f}px  max_abs_err={abs_err.max():.3f}px"
                    )
                else:
                    print("No valid steps after encoding (all padding).")
    except Exception as e:
        print(f"[WARN] Round-trip sanity failed: {e}")
    
    # Get GT from file (original space)
    gt_file = os.path.join(data_root, sub_img_name.replace('.jpg', '.lines.txt'))
    if os.path.exists(gt_file):
        gt_data_original = load_culane_img_data(gt_file)
        print(f"\nGT lanes (original space): {len(gt_data_original)}")
        for i, lane in enumerate(gt_data_original[:4]):
            if len(lane) >= 2:
                lane_arr = np.array(lane)
                print(f"  GT Lane {i}: {len(lane)} points")
                print(f"    X: [{lane_arr[:, 0].min():.1f}, {lane_arr[:, 0].max():.1f}]")
                print(f"    Y: [{lane_arr[:, 1].min():.1f}, {lane_arr[:, 1].max():.1f}]")
    
    # Predict using model
    img_tensor = torch.from_numpy(sample['img']).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Match checkpoint visual tokenization: P5-only vs Full FPN
        if n_levels == 1:
            feats = extract_p5_feat(clrernet, img_tensor)
        else:
            from tools.train_lanelm_v4_fixed import extract_full_fpn_feats
            feats = extract_full_fpn_feats(clrernet, img_tensor)
        visual_tokens = lanelm.encode_visual_tokens(feats)
        
        # Test path: autoregressive_decode
        x_tokens_all, y_tokens_all = autoregressive_decode(
            lanelm_model=lanelm,
            visual_tokens=visual_tokens,
            tokenizer_cfg=tokenizer.cfg,
            max_lanes=4,
            temperature=0.0,
        )
    
    # Convert to Lane objects (as test does)
    lanes_pred = []
    for l in range(x_tokens_all.shape[1]):
        x_tok = x_tokens_all[0, l].numpy()
        y_tok = y_tokens_all[0, l].numpy()
        
        coords_resized = tokenizer.decode_single_lane(x_tok, y_tok, smooth=smooth)
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
    
    print(f"\nPredicted lanes: {len(lanes_pred)}")
    
    # Convert lanes to prediction format (as CULaneMetric does)
    from libs.datasets.metrics.culane_metric import CULaneMetric
    class DummyMetric:
        def __init__(self):
            self.ori_w = 1640
            self.ori_h = 590
            self.y_step = 2
        
        def get_prediction_string(self, lanes):
            ys = np.arange(0, self.ori_h, self.y_step) / self.ori_h
            out = []
            for lane in lanes:
                lane_min_y = lane.min_y - 0.05
                lane_max_y = lane.max_y + 0.05
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
    
    dummy_metric = DummyMetric()
    pred_string = dummy_metric.get_prediction_string(lanes_pred)
    
    # Parse prediction string
    pred_data = []
    for line in pred_string.split('\n'):
        if line.strip():
            coords = line.split()
            lane = [(float(coords[i]), float(coords[i+1])) for i in range(0, len(coords), 2)]
            pred_data.append(lane)
    
    print(f"Prediction lanes (from string): {len(pred_data)}")
    
    # Calculate IoU using culane_metric
    if len(gt_data_original) > 0 and len(pred_data) > 0:
        results = culane_metric(
            pred=pred_data,
            anno=gt_data_original,
            cat='test',
            width=30,
            iou_thresholds=[0.1, 0.5, 0.75],
            img_shape=(590, 1640, 3),
        )
        
        print(f"\n--- IoU Results ---")
        print(f"GT lanes: {results['n_gt']}")
        print(f"Hits @ 0.1: {results['hits'][0].sum()}/{len(pred_data)}")
        print(f"Hits @ 0.5: {results['hits'][1].sum()}/{len(pred_data)}")
        print(f"Hits @ 0.75: {results['hits'][2].sum()}/{len(pred_data)}")
        
        if results['hits'][1].sum() == 0:
            print(f"  ⚠️  NO hits @ IoU 0.5!")
            print(f"  This explains why F1@0.5 is so low!")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Debug training sample IoU")
    parser.add_argument("--lanelm-ckpt", required=True)
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--list-path", default="dataset/list/train_100.txt")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Enable tokenizer smoothing in decode (ablation). Default: off.",
    )
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    debug_training_sample_iou(
        lanelm_ckpt=args.lanelm_ckpt,
        config_path=args.config,
        backbone_ckpt=args.checkpoint,
        data_root=args.data_root,
        list_path=args.list_path,
        sample_idx=args.sample_idx,
        device=device,
        smooth=bool(args.smooth),
    )


if __name__ == "__main__":
    main()








