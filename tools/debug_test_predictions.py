"""
Debug script to visualize LaneLM predictions vs ground truth.
Shows exactly what the model is producing and compares with GT.
"""

import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path

from libs.datasets import CulaneDataset
from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig

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

# Clean test pipeline
test_pipeline = [
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


def autoregressive_decode(
    lanelm_model, visual_tokens, num_points, nbins_x, max_lanes, bos_token_ids
):
    device = visual_tokens.device
    B = visual_tokens.shape[0]
    T = num_points
    pad_token_x = 0
    
    all_x = []
    all_y = []

    for lane_idx in range(max_lanes):
        lane_id_tensor = torch.full((B,), lane_idx, dtype=torch.long, device=device)
        current_bos = bos_token_ids[lane_idx]
        
        x_out = torch.full((B, T), pad_token_x, dtype=torch.long, device=device)
        y_out = torch.full((B, T), 0, dtype=torch.long, device=device)
        
        x_in = torch.full((B, T), pad_token_x, dtype=torch.long, device=device)
        x_in[:, 0] = current_bos
        y_in = torch.full((B, T), 0, dtype=torch.long, device=device)
        
        for t in range(T):
            logits_x, logits_y = lanelm_model(
                visual_tokens, x_tokens=x_in, y_tokens=y_in, lane_indices=lane_id_tensor
            )
            
            step_logits_x = logits_x[:, t, :]
            for bid in bos_token_ids:
                step_logits_x[:, bid] = -float('inf')
            pred_x = torch.argmax(step_logits_x, dim=-1)
            
            step_logits_y = logits_y[:, t, :]
            pred_y = torch.argmax(step_logits_y, dim=-1)
            
            x_out[:, t] = pred_x
            y_out[:, t] = pred_y
            
            if t + 1 < T:
                x_in[:, t+1] = pred_x
                y_in[:, t+1] = pred_y
        
        all_x.append(x_out.cpu())
        all_y.append(y_out.cpu())

    return torch.stack(all_x, dim=1), torch.stack(all_y, dim=1)


def load_gt_lanes(gt_path):
    """Load ground truth lanes from .lines.txt file."""
    lanes = []
    if not os.path.exists(gt_path):
        return lanes
    
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            points = []
            for i in range(0, len(parts), 2):
                x = float(parts[i])
                y = float(parts[i+1])
                points.append((x, y))
            if len(points) >= 2:
                lanes.append(points)
    return lanes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--backbone-checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--checkpoint", default="work_dirs/lanelm_2k_subset/lanelm_2k_best.pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--list-path", default="dataset/list/test_debug_50.txt")
    parser.add_argument("--work-dir", default="work_dirs/lanelm_debug_vis")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Model config
    nbins_x = 200
    max_abs_dx = 32
    total_vocab_size = 300
    bos_token_ids = [296, 297, 298, 299]
    max_lanes = 4

    hparams = LaneLMHyperParams(
        nbins_x=total_vocab_size,
        num_points=40,
        embed_dim=256,
        num_layers=4,
        max_lanes=max_lanes,
    )
    
    # Tokenizer
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=hparams.img_w,
        img_h=hparams.img_h,
        num_steps=hparams.num_points,
        nbins_x=nbins_x,
        x_mode="relative_disjoint",
        max_abs_dx=max_abs_dx
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    
    # Load models
    print("Loading backbone...")
    clrernet = build_frozen_clrernet_backbone(args.config, args.backbone_checkpoint, device)
    
    print("Loading LaneLM...")
    lanelm = build_lanelm_model_v3(hparams, visual_in_channels=(64,)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        lanelm.load_state_dict(ckpt["model_state_dict"])
    else:
        lanelm.load_state_dict(ckpt)
    lanelm.eval()
    
    # Load dataset
    pipeline = [dict(type="albumentation", pipelines=test_pipeline)]
    dataset = CulaneDataset(
        data_root=args.data_root,
        data_list=args.list_path,
        pipeline=pipeline,
        diff_file=None,
        test_mode=True,
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Process samples
    num_samples = min(args.num_samples, len(dataset))
    
    for idx in range(num_samples):
        sample = dataset[idx]
        img_resized = sample["img"]  # (H, W, 3) numpy, 0-255
        sub_img_name = sample.get("sub_img_name", sample.get("filename", f"img_{idx}"))
        
        # Load original image for visualization
        orig_img_path = os.path.join(args.data_root, sub_img_name)
        if os.path.exists(orig_img_path):
            orig_img = cv2.imread(orig_img_path)
        else:
            # Use resized image scaled up
            orig_img = cv2.resize(img_resized, (1640, 590))
        
        # Load ground truth
        gt_path = orig_img_path.replace('.jpg', '.lines.txt')
        gt_lanes = load_gt_lanes(gt_path)
        
        # Prepare input
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            feats = extract_p5_feat(clrernet, img_tensor)
            visual_tokens = lanelm.encode_visual_tokens(feats)
            
            x_tokens_all, y_tokens_all = autoregressive_decode(
                lanelm, visual_tokens, tokenizer.cfg.num_steps,
                lanelm.nbins_x, max_lanes, bos_token_ids
            )
        
        x_tokens_np = x_tokens_all[0].numpy()  # (max_lanes, T)
        y_tokens_np = y_tokens_all[0].numpy()
        
        # Create visualization
        vis_img = orig_img.copy()
        
        # Draw ground truth (GREEN)
        for lane in gt_lanes:
            pts = np.array(lane, dtype=np.int32)
            for i in range(len(pts) - 1):
                cv2.line(vis_img, tuple(pts[i]), tuple(pts[i+1]), (0, 255, 0), 3)
        
        # Draw predictions (RED)
        pred_lanes_resized = []  # For resized image visualization
        colors = [(0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]  # Different colors per lane
        
        for lane_idx in range(max_lanes):
            xt = x_tokens_np[lane_idx]
            yt = y_tokens_np[lane_idx]
            
            # Decode to resized coordinates
            coords_resized = tokenizer.decode_single_lane(xt, yt)
            
            if coords_resized.shape[0] >= 2:
                pred_lanes_resized.append((lane_idx, coords_resized))
                
                # Scale to original image
                scale_x = 1640 / 800
                scale_y = 320 / 320  # Same height after crop
                crop_y = 270
                
                lane_pts = []
                for pt in coords_resized:
                    x = pt[0] * scale_x
                    y = pt[1] * scale_y + crop_y
                    if 0 <= x <= 1640 and crop_y <= y <= 590:
                        lane_pts.append((int(x), int(y)))
                
                if len(lane_pts) >= 2:
                    for i in range(len(lane_pts) - 1):
                        cv2.line(vis_img, lane_pts[i], lane_pts[i+1], colors[lane_idx], 2)
        
        # Add text info
        cv2.putText(vis_img, "GREEN=GT, RED/BLUE/MAGENTA/CYAN=Pred", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_img, f"GT lanes: {len(gt_lanes)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_img, f"Pred lanes: {len(pred_lanes_resized)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Save visualization
        save_name = sub_img_name.replace('/', '_').replace('.jpg', '_debug.jpg')
        save_path = os.path.join(args.work_dir, save_name)
        cv2.imwrite(save_path, vis_img)
        
        # Print token analysis
        print(f"\n=== {sub_img_name} ===")
        print(f"GT lanes: {len(gt_lanes)}")
        for i, lane in enumerate(gt_lanes):
            print(f"  GT Lane {i}: X range [{min(p[0] for p in lane):.1f}, {max(p[0] for p in lane):.1f}]")
        
        print(f"Pred lanes: {len(pred_lanes_resized)}")
        for lane_idx, coords in pred_lanes_resized:
            print(f"  Pred Lane {lane_idx}: X tokens unique: {len(set(x_tokens_np[lane_idx]))}")
            print(f"    First 5 X tokens: {x_tokens_np[lane_idx][:5]}")
            print(f"    Resized X range: [{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}]")
    
    print(f"\nVisualizations saved to {args.work_dir}/")


if __name__ == "__main__":
    main()


