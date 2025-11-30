"""
LaneLM Prediction Debugger & Error Analysis - FIX (Oracle Y).

Fix:
  - Previously, we fed dummy Y tokens (0..39) to the model.
  - Since we trained with 'no-y-loss', the model relies on INPUT Y to know which row to predict X for.
  - Now, we feed Ground Truth Y tokens (Oracle) to properly evaluate X accuracy.
"""

import argparse
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from scipy.spatial.distance import cdist

from libs.datasets import CulaneDataset
from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig
from tools.train_lanelm_culane_v3 import (
    LaneLMHyperParams,
    collate_lanelm_batch_v3,
    build_frozen_clrernet_backbone,
    build_lanelm_model_v3,
)
from configs.clrernet.culane.dataset_culane_clrernet import (
    compose_cfg,
    crop_bbox,
    img_scale,
)

def extract_p5_feat(model, imgs: torch.Tensor):
    with torch.no_grad():
        feats = model.extract_feat(imgs)
        p5 = feats[-1]
    return [p5]

def autoregressive_decode_oracle_y(model, visual_tokens, tokenizer, device, bos_token_ids, gt_y_tokens_list):
    """
    Decodes X tokens given the Ground Truth Y tokens (Oracle Y).
    """
    model.eval()
    B = visual_tokens.shape[0] # Should be 1 for this script
    nbins_x = tokenizer.cfg.nbins_x
    
    decoded_lanes = []
    
    with torch.no_grad():
        for b in range(B):
            batch_lanes = []
            # We assume 4 lanes max, and gt_y_tokens_list has the GT Ys for this image
            # gt_y_tokens_list is a list of numpy arrays or tensors for each lane in the image
            
            for l_idx, bos_id in enumerate(bos_token_ids):
                # Get the GT Y tokens for this specific lane index
                # Note: If GT has fewer lanes than 4, we might not have GT Y for lane 3.
                # In that case, we can't use Oracle Y effectively for missing lanes.
                # We'll try to use the GT Y if available, else use dummy (which will fail, but that's expected for non-existent lanes)
                
                if l_idx < len(gt_y_tokens_list[b]):
                    y_seq = torch.from_numpy(gt_y_tokens_list[b][l_idx]).to(device).unsqueeze(0) # (1, T)
                else:
                    # No GT for this lane index, skip or use dummy
                    # If we skip, we output empty
                    batch_lanes.append(np.array([]))
                    continue

                T = y_seq.shape[1]
                
                # Prepare inputs
                curr_vis = visual_tokens[b:b+1]
                lane_idx_tensor = torch.tensor([l_idx], device=device)
                
                # Start with BOS
                x_in = torch.zeros((1, T), dtype=torch.long, device=device)
                x_in[0, 0] = bos_id
                
                # Use GT Y as input (Teacher Forcing for Y)
                y_in = y_seq.clone()
                
                # Greedy decoding for X
                for t in range(T):
                    logits_x, _ = model(curr_vis, x_in, y_in, lane_indices=lane_idx_tensor)
                    
                    step_logits = logits_x[:, t, :]
                    
                    # Masking
                    if t == 0:
                        step_logits[:, 0] = -float('inf')
                        step_logits[:, nbins_x:] = -float('inf')
                    else:
                        step_logits[:, 0] = -float('inf')
                        step_logits[:, 1:nbins_x] = -float('inf')
                    
                    for bid in bos_token_ids:
                        step_logits[:, bid] = -float('inf')
                        
                    pred_token = torch.argmax(step_logits, dim=-1)
                    
                    if t + 1 < T:
                        x_in[0, t+1] = pred_token
                
                # Decode
                x_tokens = x_in[0].cpu().numpy()
                y_tokens = y_in[0].cpu().numpy()
                
                valid_mask = (x_tokens != 0) & (x_tokens != tokenizer.cfg.pad_token_x)
                if valid_mask.sum() > 2:
                    points = tokenizer.decode_single_lane(x_tokens, y_tokens)
                    batch_lanes.append(points)
                else:
                    batch_lanes.append(np.array([]))
            decoded_lanes.append(batch_lanes)
    return decoded_lanes

def calculate_lane_error(pred_points, gt_points):
    if len(pred_points) < 2 or len(gt_points) < 2:
        return 999.0
    dists = cdist(pred_points, gt_points)
    min_dists = dists.min(axis=1)
    return np.mean(min_dists)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--checkpoint", default="work_dirs/lanelm_100_imgs/lanelm_100_best.pth")
    parser.add_argument("--backbone-ckpt", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--list-path", default="dataset/list/train_100.txt")
    parser.add_argument("--save-dir", default="work_dirs/lanelm_100_imgs/debug_vis_oracle")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    nbins_x = 200
    max_abs_dx = 32
    bos_token_ids = [296, 297, 298, 299]
    
    hparams = LaneLMHyperParams(
        nbins_x=300, num_points=40, embed_dim=256, num_layers=4, max_lanes=4
    )
    
    print(f"Loading model from {args.checkpoint}...")
    clrernet = build_frozen_clrernet_backbone(args.config, args.backbone_ckpt, device)
    lanelm = build_lanelm_model_v3(hparams, visual_in_channels=(64,)).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    lanelm.load_state_dict(ckpt["model_state_dict"])
    lanelm.eval()
    
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=800, img_h=320, num_steps=40,
        nbins_x=nbins_x, x_mode="relative_disjoint", max_abs_dx=max_abs_dx
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    
    clean_pipeline = [
        dict(type="Compose", params=compose_cfg),
        dict(type="Crop", x_min=crop_bbox[0], x_max=crop_bbox[2], y_min=crop_bbox[1], y_max=crop_bbox[3], p=1),
        dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
    ]
    pipeline = [dict(type="albumentation", pipelines=clean_pipeline)]
    dataset = CulaneDataset(args.data_root, args.list_path, pipeline=pipeline, test_mode=False)
    subset = Subset(dataset, list(range(min(len(dataset), 20))))
    dataloader = DataLoader(subset, batch_size=1, collate_fn=collate_lanelm_batch_v3)
    
    print("Starting Oracle-Y Analysis...")
    total_error = 0.0
    lane_counts = 0
    
    for i, batch in enumerate(dataloader):
        img_tensor = batch["inputs"].to(device)
        
        feats = extract_p5_feat(clrernet, img_tensor)
        visual_tokens = lanelm.encode_visual_tokens(feats)
        
        # Prepare GT for Oracle Y
        gt_lanes_raw = batch["gt_points"][0]
        gt_y_tokens_list = [] # List of Y tokens for each lane
        gt_lanes_processed = []
        
        for raw_lane in gt_lanes_raw:
            if len(raw_lane) < 4: continue
            pts = np.array(raw_lane).reshape(-1, 2)
            x_tok, y_tok = tokenizer.encode_single_lane(pts)
            
            gt_y_tokens_list.append(y_tok) # Save Y for Oracle
            
            decoded_gt = tokenizer.decode_single_lane(x_tok, y_tok)
            gt_lanes_processed.append(decoded_gt)
            
        # Inference with Oracle Y
        # Note: We wrap gt_y_tokens_list in a list because batch size is 1
        pred_lanes_batch = autoregressive_decode_oracle_y(
            lanelm, visual_tokens, tokenizer, device, bos_token_ids, [gt_y_tokens_list]
        )
        pred_lanes = pred_lanes_batch[0]
        
        # Visualization
        img_vis = (img_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        
        for lane in gt_lanes_processed:
            for k in range(len(lane) - 1):
                cv2.line(img_vis, (int(lane[k][0]), int(lane[k][1])), 
                         (int(lane[k+1][0]), int(lane[k+1][1])), (0, 255, 0), 2)
                         
        img_error = 0.0
        matched_lanes = 0
        
        # Compare mapped lanes
        # pred_lanes has index matching gt_y_tokens_list order
        # gt_lanes_processed has same order
        
        for idx, lane in enumerate(pred_lanes):
            if len(lane) == 0: continue
            if idx >= len(gt_lanes_processed): continue # Should not happen
            
            gt_lane = gt_lanes_processed[idx]
            
            # Draw Pred
            for k in range(len(lane) - 1):
                cv2.line(img_vis, (int(lane[k][0]), int(lane[k][1])), 
                         (int(lane[k+1][0]), int(lane[k+1][1])), (0, 0, 255), 2)
            
            # Error
            dist = calculate_lane_error(lane, gt_lane)
            if dist < 900:
                img_error += dist
                matched_lanes += 1
                cv2.putText(img_vis, f"{dist:.1f}", (int(lane[len(lane)//2][0]), int(lane[len(lane)//2][1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if matched_lanes > 0:
            avg_img_error = img_error / matched_lanes
            total_error += img_error
            lane_counts += matched_lanes
            print(f"Img {i}: Avg Error = {avg_img_error:.2f} px")
        else:
            print(f"Img {i}: No lanes matched.")

        cv2.imwrite(os.path.join(args.save_dir, f"debug_oracle_{i:03d}.jpg"), img_vis)

    if lane_counts > 0:
        print(f"\nOverall Mean Pixel Error (Oracle Y): {total_error / lane_counts:.2f} px")

if __name__ == "__main__":
    main()
