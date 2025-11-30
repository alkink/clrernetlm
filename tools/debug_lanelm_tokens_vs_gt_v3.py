"""
LaneLM Token vs GT Debug Script (v3).

Purpose:
  Compare predicted tokens vs GT tokens numerically to calculate Mean Absolute Error (MAE) in pixels.
  This verifies if the low loss (0.26) actually corresponds to high geometric accuracy.
  
  Checks:
  - Token accuracy (Class match).
  - Pixel distance error (MAE).
  - Visualizes GT (Green) vs Pred (Red) on the same image.
"""

import argparse
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from libs.datasets import CulaneDataset
from libs.models.lanelm import LaneTokenizer, LaneTokenizerConfig
from tools.train_lanelm_culane_v3 import (
    LaneLMHyperParams,
    build_frozen_clrernet_backbone,
    build_lanelm_model_v3,
)
from configs.clrernet.culane.dataset_culane_clrernet import (
    compose_cfg,
    crop_bbox,
    img_scale,
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

def extract_p5_feat(model, imgs: torch.Tensor):
    with torch.no_grad():
        feats = model.extract_feat(imgs)
        p5 = feats[-1]
    return [p5]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--lanelm-checkpoint", required=True, help="Path to trained LaneLM .pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--list-path", default="dataset/list/train_100.txt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Dataset (No Aug)
    dataset = CulaneDataset(
        data_root=args.data_root,
        data_list=args.list_path,
        pipeline=[dict(type="albumentation", pipelines=clean_pipeline)],
        test_mode=False
    )
    
    # Load Models
    clrernet = build_frozen_clrernet_backbone(args.config, args.checkpoint, device)
    
    hparams = LaneLMHyperParams(
        nbins_x=300, # Total vocab
        num_points=40,
        embed_dim=256,
        num_layers=4,
        max_lanes=4
    )
    lanelm = build_lanelm_model_v3(hparams, visual_in_channels=(64,)).to(device)
    
    # Load Checkpoint
    ckpt = torch.load(args.lanelm_checkpoint, map_location=device)
    lanelm.load_state_dict(ckpt["model_state_dict"])
    lanelm.eval()
    print(f"Loaded LaneLM from {args.lanelm_checkpoint} (Epoch {ckpt.get('epoch', '?')})")
    
    # Tokenizer
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=hparams.img_w, img_h=hparams.img_h,
        num_steps=hparams.num_points, 
        nbins_x=200, 
        x_mode="relative_disjoint", 
        max_abs_dx=32
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    bos_token_ids = [296, 297, 298, 299]
    
    # Debug Loop
    total_mae = 0
    count = 0
    
    indices = np.random.choice(len(dataset), min(len(dataset), args.num_samples), replace=False)
    
    for idx in indices:
        data = dataset[idx]
        img = data["img"].unsqueeze(0).to(device) # (1, 3, H, W)
        gt_lanes = data["lanes"] # List of List of points
        
        # Filter valid lanes
        gt_lanes = [l for l in gt_lanes if len(l) >= 4][:4]
        if not gt_lanes: continue
            
        # Encode Image
        feats = extract_p5_feat(clrernet, img)
        visual_tokens = lanelm.encode_visual_tokens(feats)
        
        # Decode for each lane
        pred_lanes = []
        
        for l_idx in range(4): # Max 4 lanes
            # Prepare Inputs
            T = tokenizer.cfg.num_steps
            pad_token_x = tokenizer.cfg.pad_token_x
            boundary_idx = tokenizer.cfg.nbins_x
            
            lane_id_tensor = torch.tensor([l_idx], device=device)
            bos_id = bos_token_ids[l_idx]
            
            y_tokens = torch.arange(T, device=device).unsqueeze(0)
            x_tokens = torch.full((1, T), pad_token_x, dtype=torch.long, device=device)
            
            # Autoregressive Generation
            curr_x = torch.full((1, T), pad_token_x, dtype=torch.long, device=device)
            curr_y = y_tokens.clone()
            curr_x[:, 0] = bos_id
            
            for t in range(T):
                logits_x, _ = lanelm(visual_tokens, curr_x, curr_y, lane_indices=lane_id_tensor)
                step_logits = logits_x[:, t, :]
                
                # Masking (Same as training)
                # We removed masking in training, so maybe raw argmax is fair?
                # BUT let's apply basic boundary check
                # t=0: Absolute only (idx >= 200 is ignored)
                # if t == 0: step_logits[:, boundary_idx:] = -1e9
                # else: step_logits[:, 1:boundary_idx] = -1e9 
                
                # Mask BOS
                for bid in bos_token_ids:
                    step_logits[:, bid] = -float('inf')
                
                next_token = torch.argmax(step_logits, dim=-1)
                x_tokens[:, t] = next_token
                if t + 1 < T:
                    curr_x[:, t+1] = next_token
            
            # Decode Prediction
            pred_x_np = x_tokens[0].cpu().numpy()
            pred_y_np = y_tokens[0].cpu().numpy()
            pred_coords = tokenizer.decode_single_lane(pred_x_np, pred_y_np)
            pred_lanes.append(pred_coords)
            
            # Compare with GT if available for this index
            if l_idx < len(gt_lanes):
                gt_pts = np.array(gt_lanes[l_idx]).reshape(-1, 2)
                gt_x_tokens, gt_y_tokens = tokenizer.encode_single_lane(gt_pts)
                gt_coords_decoded = tokenizer.decode_single_lane(gt_x_tokens, gt_y_tokens) # Re-decoded GT to match grid
                
                # Calculate MAE on decoded coordinates (Apple to Apple)
                # We need to interpolate or match rows. Since Y is fixed steps, we can match directly?
                # Tokenizer produces fixed Y steps.
                
                min_len = min(len(pred_coords), len(gt_coords_decoded))
                if min_len > 0:
                    diff = np.abs(pred_coords[:min_len, 0] - gt_coords_decoded[:min_len, 0])
                    mae = np.mean(diff)
                    total_mae += mae
                    count += 1
                    print(f"Img {idx} Lane {l_idx}: MAE = {mae:.2f} px")
                    print(f"  GT Tokens (First 5): {gt_x_tokens[:5]}")
                    print(f"  Pr Tokens (First 5): {pred_x_np[:5]}")
        
        # Visualization
        img_vis = (img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        
        # Draw GT (Green)
        for l_pts in gt_lanes:
            l_pts = np.array(l_pts).reshape(-1, 2).astype(int)
            for k in range(len(l_pts) - 1):
                cv2.line(img_vis, tuple(l_pts[k]), tuple(l_pts[k+1]), (0, 255, 0), 2)
        
        # Draw Pred (Red)
        for l_pts in pred_lanes:
            if len(l_pts) < 2: continue
            l_pts = l_pts.astype(int)
            for k in range(len(l_pts) - 1):
                cv2.line(img_vis, tuple(l_pts[k]), tuple(l_pts[k+1]), (0, 0, 255), 2)
                
        os.makedirs("work_dirs/debug_vis", exist_ok=True)
        cv2.imwrite(f"work_dirs/debug_vis/debug_img{idx}.jpg", img_vis)

    if count > 0:
        print(f"\nOverall Mean Pixel Error (MAE): {total_mae / count:.2f} pixels")
    else:
        print("No valid comparisons made.")

if __name__ == "__main__":
    main()
