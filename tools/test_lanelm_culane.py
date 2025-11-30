"""
LaneLM CULane Test Script.

Purpose:
  Evaluate LaneLM model on CULane test set using official metrics.
  Generates lane predictions, saves them as .lines.txt files,
  and computes F1/Precision/Recall scores.

Usage:
  python tools/test_lanelm_culane.py --checkpoint work_dirs/lanelm_2k_subset/lanelm_2k_best.pth
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader

from libs.datasets import CulaneDataset
from libs.datasets.metrics.culane_metric import eval_predictions
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

# Clean test pipeline (no augmentations)
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
    """Extract P5 feature from frozen CLRerNet backbone."""
    with torch.no_grad():
        feats = model.extract_feat(imgs)
        p5 = feats[-1]
    return [p5]


def collate_test_batch(batch):
    """Collate function for test dataloader."""
    imgs = torch.stack([torch.from_numpy(item["img"]).permute(2, 0, 1).float() / 255.0 for item in batch])
    sub_img_names = [item.get("sub_img_name", item.get("filename", "unknown")) for item in batch]
    return {"inputs": imgs, "sub_img_names": sub_img_names}


def autoregressive_decode(
    lanelm_model,
    visual_tokens,
    num_points,
    nbins_x,
    max_lanes,
    bos_token_ids,
):
    """Autoregressive decoding for LaneLM inference."""
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
                visual_tokens, 
                x_in, 
                y_in, 
                None,  # visual_padding_mask
                lane_id_tensor  # lane_indices
            )
            
            # Greedy X (mask BOS tokens)
            step_logits_x = logits_x[:, t, :]
            for bid in bos_token_ids:
                step_logits_x[:, bid] = -float('inf')
            pred_x = torch.argmax(step_logits_x, dim=-1)
            
            # Greedy Y
            step_logits_y = logits_y[:, t, :]
            pred_y = torch.argmax(step_logits_y, dim=-1)
            
            # Clamp to valid ranges
            pred_x = pred_x.clamp(0, nbins_x - 1)
            pred_y = pred_y.clamp(0, T)  # max_y_tokens = T + 1
            
            x_out[:, t] = pred_x
            y_out[:, t] = pred_y
            
            if t + 1 < T:
                x_in[:, t+1] = pred_x
                y_in[:, t+1] = pred_y
        
        all_x.append(x_out.cpu())
        all_y.append(y_out.cpu())

    return torch.stack(all_x, dim=1), torch.stack(all_y, dim=1)


def hallucination_removal(lane_points, width=800):
    """Apply Hallucination Removal (HR) from LaneLM paper (Alg. 1)."""
    if len(lane_points) <= 10:
        return lane_points
    
    points = np.array(lane_points)
    x = points[:, 0]
    
    # Calculate absolute differences between adjacent x-coordinates
    diff = np.abs(x[1:] - x[:-1])
    
    # Threshold: 2 * 85th percentile of diffs
    theta = 2 * np.percentile(diff, 85)
    
    # Find first point where diff exceeds theta
    # "p <- argmin(diff > theta)" means the first index where condition is true
    exceeds = np.where(diff > theta)[0]
    if len(exceeds) > 0:
        p = exceeds[0]
        # Truncate lane at p+1 (keep up to p, inclusive of the jump start?)
        # Paper says: "points with offsets ... exceeding ... along with their subsequent points, will be filtered out."
        # So we keep up to p.
        return lane_points[:p+1]
        
    return lane_points


def tokens_to_culane_format(x_tokens, y_tokens, tokenizer, ori_img_h=590, ori_img_w=1640, crop_y=270, smooth=True):
    """Convert LaneLM tokens to CULane format (list of (x, y) tuples).
    
    The model operates on resized images (800x320) after cropping.
    We need to:
    1. Decode tokens to resized coordinates (0-800, 0-320)
    2. Scale X: 800 -> 1640 (original width)
    3. Scale Y: 320 -> 320 (crop height is same) then add crop_y offset
    """
    # Decode tokens to resized image coordinates
    coords_resized = tokenizer.decode_single_lane(x_tokens, y_tokens, smooth=smooth)
    
    if coords_resized.shape[0] < 2:
        return []
    
    # Scale factors
    # Model trained on 800x320 crop, original is 1640x590 with crop at y=270
    scale_x = ori_img_w / tokenizer.cfg.img_w  # 1640 / 800 = 2.05
    crop_height = ori_img_h - crop_y  # 590 - 270 = 320
    scale_y = crop_height / tokenizer.cfg.img_h  # 320 / 320 = 1.0
    
    lane_points = []
    for i in range(coords_resized.shape[0]):
        x = coords_resized[i, 0] * scale_x
        y = coords_resized[i, 1] * scale_y + crop_y  # Add crop offset
        
        # Clip to valid image bounds
        if x < 0 or x > ori_img_w:
            continue  # Skip invalid points
        if y < crop_y or y > ori_img_h:
            continue  # Skip invalid points
            
        lane_points.append((x, y))
    
    return lane_points


def get_prediction_string(lanes):
    """Convert lanes to CULane prediction string format."""
    out_lines = []
    for lane in lanes:
        if len(lane) < 2:
            continue
        lane_str = " ".join([f"{x:.5f} {y:.5f}" for x, y in lane])
        out_lines.append(lane_str)
    return "\n".join(out_lines)


def main():
    parser = argparse.ArgumentParser(description="Test LaneLM on CULane dataset")
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py",
                        help="CLRerNet config file for backbone")
    parser.add_argument("--backbone-checkpoint", default="clrernet_culane_dla34_ema.pth",
                        help="CLRerNet checkpoint for frozen backbone")
    parser.add_argument("--checkpoint", required=True,
                        help="LaneLM checkpoint to test")
    parser.add_argument("--data-root", default="dataset/culane",
                        help="CULane dataset root")
    parser.add_argument("--list-path", default="dataset/culane/list/test.txt",
                        help="Test list file")
    parser.add_argument("--work-dir", default="work_dirs/lanelm_test",
                        help="Output directory for predictions and results")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Test batch size")
    parser.add_argument("--device", default="cuda",
                        help="Device to use")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    pred_dir = os.path.join(args.work_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # Model hyperparameters (must match training)
    nbins_x = 200 
    max_abs_dx = 32 
    total_vocab_size = 300
    # V4 uses lane_indices instead of BOS tokens, so use 0 (padding) as start
    bos_token_ids = [0, 0, 0, 0]  # No explicit BOS for absolute tokenization
    max_lanes = 4

    hparams = LaneLMHyperParams(
        nbins_x=total_vocab_size, 
        num_points=40,
        embed_dim=256,
        num_layers=4,
        max_lanes=max_lanes, 
    )

    # Build tokenizer
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=hparams.img_w, 
        img_h=hparams.img_h,
        num_steps=hparams.num_points, 
        nbins_x=nbins_x,
        x_mode="relative_disjoint", 
        max_abs_dx=max_abs_dx
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    
    # Build models
    print("Loading CLRerNet backbone...")
    clrernet = build_frozen_clrernet_backbone(args.config, args.backbone_checkpoint, device)
    
    print("Loading LaneLM model...")
    lanelm = build_lanelm_model_v3(hparams, visual_in_channels=(64,)).to(device)
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Try to load config from checkpoint
    if "config" in ckpt:
        print("Found config in checkpoint, overriding defaults...")
        saved_cfg = ckpt["config"]
        # Update hparams from saved config
        if "nbins_x" in saved_cfg: nbins_x = saved_cfg["nbins_x"]
        if "num_points" in saved_cfg: hparams.num_points = saved_cfg["num_points"]
        if "embed_dim" in saved_cfg: hparams.embed_dim = saved_cfg["embed_dim"]
        if "num_layers" in saved_cfg: hparams.num_layers = saved_cfg["num_layers"]
        if "max_lanes" in saved_cfg: max_lanes = saved_cfg["max_lanes"]
        
        # Update hparams object
        hparams.nbins_x = nbins_x
        hparams.max_lanes = max_lanes
        
        print(f"Loaded config: nbins_x={nbins_x}, num_points={hparams.num_points}, embed_dim={hparams.embed_dim}")
    else:
        print("WARNING: No config found in checkpoint, using hardcoded defaults (nbins_x=200).")
        print("If your model was trained with nbins_x=800, this will cause zigzagging!")

    # Re-build tokenizer with potentially updated config
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=hparams.img_w, 
        img_h=hparams.img_h,
        num_steps=hparams.num_points, 
        nbins_x=nbins_x,
        x_mode="relative_disjoint" if nbins_x < 300 else "absolute", # Heuristic: small nbins usually means relative
        max_abs_dx=max_abs_dx
    )
    # Override x_mode if it was absolute in training (usually nbins_x=800 is absolute)
    if nbins_x == 800:
        tokenizer_cfg.x_mode = "absolute"
        
    tokenizer = LaneTokenizer(tokenizer_cfg)

    # Re-build model with updated hparams
    # Check if checkpoint has Full FPN (v4) or single channel (v3)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if "visual_encoder.proj_per_level.1.weight" in state_dict:
        # V4 model with Full FPN
        print("Detected V4 model (Full FPN)")
        visual_in_channels = (64, 64, 64)
    else:
        # V3 model with single channel
        visual_in_channels = (64,)
    
    # Get max_seq_len from checkpoint
    pos_emb_shape = state_dict.get("keypoint_embed.pos_embedding.weight", None)
    max_seq_len = pos_emb_shape.shape[0] if pos_emb_shape is not None else hparams.num_points
    print(f"Using max_seq_len={max_seq_len}")
    
    max_y_tokens = hparams.num_points + 1
    lanelm = LaneLMModel(
        nbins_x=nbins_x,
        max_y_tokens=max_y_tokens,
        embed_dim=hparams.embed_dim,
        num_layers=hparams.num_layers,
        num_heads=hparams.num_heads,
        ffn_dim=hparams.ffn_dim,
        max_seq_len=max_seq_len,
        visual_in_channels=visual_in_channels,
    ).to(device)

    if "model_state_dict" in ckpt:
        lanelm.load_state_dict(ckpt["model_state_dict"])
    else:
        lanelm.load_state_dict(ckpt)
    lanelm.eval()
    print(f"Loaded LaneLM checkpoint from {args.checkpoint}")
    
    # Build test dataloader
    print("Loading test dataset...")
    pipeline = [dict(type="albumentation", pipelines=test_pipeline)]
    test_dataset = CulaneDataset(
        data_root=args.data_root,
        data_list=args.list_path,
        pipeline=pipeline,
        diff_file=None,
        test_mode=True,
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_test_batch,
    )
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            imgs = batch["inputs"].to(device)
            sub_img_names = batch["sub_img_names"]

            # Extract visual features
            if len(visual_in_channels) == 3:
                # Full FPN (V4)
                with torch.no_grad():
                    feats = clrernet.extract_feat(imgs)
            else:
                # P5 only (V3)
                feats = extract_p5_feat(clrernet, imgs)
            visual_tokens = lanelm.encode_visual_tokens(feats)

            # Decode lanes
            x_tokens_all, y_tokens_all = autoregressive_decode(
                lanelm_model=lanelm,
                visual_tokens=visual_tokens,
                num_points=tokenizer.cfg.num_steps,
                nbins_x=lanelm.nbins_x,
                max_lanes=max_lanes,
                bos_token_ids=bos_token_ids,
            )

            x_tokens_np = x_tokens_all.numpy()
            y_tokens_np = y_tokens_all.numpy()
            
            # Save predictions
            for i, sub_name in enumerate(sub_img_names):
                lanes = []
                for lane_idx in range(max_lanes):
                    xt = x_tokens_np[i, lane_idx]
                    yt = y_tokens_np[i, lane_idx]
                    # Enable smoothing
                    lane_pts = tokens_to_culane_format(xt, yt, tokenizer, smooth=True)
                    
                    # Apply Hallucination Removal
                    lane_pts = hallucination_removal(lane_pts)
                    
                    if len(lane_pts) >= 2:
                        lanes.append(lane_pts)
                
                # Save to prediction file
                dst_path = Path(pred_dir) / Path(sub_name).with_suffix(".lines.txt")
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(str(dst_path), "w") as f:
                    output = get_prediction_string(lanes)
                    if len(output) > 0:
                        print(output, file=f)
    
    # Evaluate predictions
    print("\nEvaluating predictions...")
    categories_dir = os.path.join(args.data_root, "list/test_split")
    
    results = eval_predictions(
        pred_dir=pred_dir,
        anno_dir=args.data_root,
        list_path=args.list_path,
        categories_dir=categories_dir,
        iou_thresholds=[0.1, 0.5, 0.75],
        width=30,
        use_parallel=True,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("LaneLM CULane Evaluation Results")
    print("="*60)
    
    for iou_thr in [0.1, 0.5, 0.75]:
        print(f"\nIoU Threshold = {iou_thr}:")
        if f"F1_{iou_thr}" in results:
            print(f"  F1:        {results[f'F1_{iou_thr}']:.4f}")
            print(f"  Precision: {results[f'Precision{iou_thr}']:.4f}")
            print(f"  Recall:    {results[f'Recall{iou_thr}']:.4f}")
    
    # Save results to file
    results_file = os.path.join(args.work_dir, "results.txt")
    with open(results_file, "w") as f:
        f.write("LaneLM CULane Evaluation Results\n")
        f.write("="*60 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
