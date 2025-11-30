"""
LaneLM v4 Training Script - FIXED VERSION

Key Fixes from Debug Analysis:
1. Full FPN (P3+P4+P5) instead of P5-only → More visual tokens with spatial detail
2. 2D Positional Embedding → Preserve spatial structure
3. Absolute Tokenization → Simpler, easier to learn (no relative delta confusion)
4. Single image overfit first → Verify model can learn before scaling

This script addresses the POSTERIOR COLLAPSE issue where cross-attention
was uniform (0.998 uniformity score) and model ignored visual information.
"""

import argparse
import os
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from libs.datasets import CulaneDataset
from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig
from tools.train_lanelm_culane_v3 import (
    LaneLMHyperParams,
    collate_lanelm_batch_v3,
    build_frozen_clrernet_backbone,
)

from configs.clrernet.culane.dataset_culane_clrernet import (
    compose_cfg,
    crop_bbox,
    img_scale,
)


# Clean pipeline (no augmentation for overfit test)
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


def extract_full_fpn_feats(model, imgs: torch.Tensor):
    """Extract ALL FPN levels (P3, P4, P5) for richer visual information."""
    with torch.no_grad():
        feats = model.extract_feat(imgs)  # Returns [P3, P4, P5]
    return feats  # All 3 levels


def build_lanelm_model_v4(hparams, visual_in_channels):
    """Build LaneLM model with 2D PE enabled."""
    max_y_tokens = hparams.num_points + 1
    max_seq_len = hparams.num_points * 2  # Safe upper bound
    
    model = LaneLMModel(
        nbins_x=hparams.nbins_x,
        max_y_tokens=max_y_tokens,
        embed_dim=hparams.embed_dim,
        num_layers=hparams.num_layers,
        num_heads=8,
        ffn_dim=512,
        max_seq_len=max_seq_len,
        dropout=0.0,  # No dropout for overfit test
        visual_in_channels=visual_in_channels,  # Full FPN channels
    )
    return model


def build_clean_dataloader(data_root, list_path, batch_size, overfit_size=1):
    """Build dataloader with optional subset for overfit testing."""
    pipeline = [dict(type="albumentation", pipelines=clean_pipeline)]
    dataset = CulaneDataset(
        data_root=data_root,
        data_list=list_path,
        pipeline=pipeline,
        diff_file=None,
        test_mode=False,
    )
    
    # Take only first N images for overfit test
    if overfit_size > 0 and overfit_size < len(dataset):
        dataset = Subset(dataset, list(range(overfit_size)))
    
    print(f"Dataset size: {len(dataset)}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for overfit
        num_workers=0,  # Single worker for reproducibility
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_lanelm_batch_v3,
    )
    return dataloader, dataset


def visualize(model, clrernet_model, batch, tokenizer, device, epoch, save_dir, max_lanes=4):
    """Visualize predictions vs ground truth."""
    model.eval()
    imgs = batch["inputs"].to(device)
    
    with torch.no_grad():
        feats = extract_full_fpn_feats(clrernet_model, imgs)
        visual_tokens = model.encode_visual_tokens(feats)
        
        # Decode each lane
        T = tokenizer.cfg.num_steps
        all_preds = []
        
        # Y is FIXED (0, 1, 2, ..., T-1) - not predicted!
        y_fixed = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)
        
        for lane_idx in range(max_lanes):
            x_out = torch.zeros(1, T, dtype=torch.long, device=device)
            
            # Autoregressive decode X only (Y is fixed)
            for t in range(T):
                x_in = x_out.clone()
                # Shift right for teacher forcing style
                if t > 0:
                    x_in_shifted = torch.zeros_like(x_in)
                    x_in_shifted[:, 1:t+1] = x_out[:, :t]
                else:
                    x_in_shifted = x_in
                
                lane_ids = torch.tensor([lane_idx], device=device)
                logits_x, _ = model(visual_tokens[:1], x_in_shifted, y_fixed, lane_indices=lane_ids)
                
                pred_x = torch.argmax(logits_x[0, t], dim=-1)
                x_out[0, t] = pred_x
            
            all_preds.append((x_out[0].cpu().numpy(), y_fixed[0].cpu().numpy()))
        
        # Visualize first image
        img_vis = (imgs[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        
        # Draw predictions (colored) with smoothing
        # Filter out padding tokens (0) before decoding
        colors = [(0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
        for l_idx, (x_tokens, y_tokens) in enumerate(all_preds):
            # Filter: keep only non-padding tokens
            valid_mask = x_tokens > 0  # padding token is 0
            if valid_mask.sum() < 2:
                continue
            x_filtered = x_tokens[valid_mask]
            y_filtered = y_tokens[valid_mask]
            
            coords = tokenizer.decode_single_lane(x_filtered, y_filtered, smooth=True)
            if coords.shape[0] >= 2:
                for k in range(len(coords) - 1):
                    p1 = (int(coords[k][0]), int(coords[k][1]))
                    p2 = (int(coords[k+1][0]), int(coords[k+1][1]))
                    if 0 <= p1[0] < 800 and 0 <= p2[0] < 800:
                        cv2.line(img_vis, p1, p2, colors[l_idx % 4], 2)
        
        # Draw GT (GREEN)
        for lanes_points in batch["gt_points"][:1]:
            for lane in lanes_points[:max_lanes]:
                pts = np.array(lane, dtype=np.float32).reshape(-1, 2)
                if len(pts) >= 2:
                    for k in range(len(pts) - 1):
                        p1 = (int(pts[k][0]), int(pts[k][1]))
                        p2 = (int(pts[k+1][0]), int(pts[k+1][1]))
                        if 0 <= p1[0] < 800 and 0 <= p2[0] < 800 and 0 <= p1[1] < 320 and 0 <= p2[1] < 320:
                            cv2.line(img_vis, p1, p2, (0, 255, 0), 3)
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"ep{epoch:04d}.jpg")
        cv2.imwrite(save_path, img_vis)
    
    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--list-path", default="dataset/list/train_gt.txt")
    parser.add_argument("--work-dir", default="work_dirs/lanelm_v4_fixed")
    parser.add_argument("--overfit-size", type=int, default=1, help="Number of images for overfit test (0=all)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.work_dir, exist_ok=True)
    
    # ========== KEY CONFIG CHANGES ==========
    # 1. ABSOLUTE tokenization (simpler, no delta confusion)
    nbins_x = 200  # Quantize X to 200 bins
    
    # 2. Full FPN channels (P3=64, P4=64, P5=64 from CLRerNetFPN)
    visual_in_channels = (64, 64, 64)
    
    # 3. Model hyperparams
    hparams = LaneLMHyperParams(
        nbins_x=nbins_x,
        num_points=40,
        embed_dim=256,
        num_layers=4,
        max_lanes=4,
    )
    
    # Tokenizer with ABSOLUTE mode (key fix!)
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=hparams.img_w,
        img_h=hparams.img_h,
        num_steps=hparams.num_points,
        nbins_x=nbins_x,
        x_mode="absolute",  # NOT relative_disjoint!
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    
    # Build dataloader
    dataloader, dataset = build_clean_dataloader(
        args.data_root, args.list_path, batch_size=1, overfit_size=args.overfit_size
    )
    
    # Store first batch for visualization
    fixed_batch = next(iter(dataloader))
    
    # Build models
    print("Loading CLRerNet backbone (frozen)...")
    clrernet = build_frozen_clrernet_backbone(args.config, args.checkpoint, device)
    
    print("Building LaneLM v4 (Full FPN + 2D PE + Absolute Tokens)...")
    lanelm = build_lanelm_model_v4(hparams, visual_in_channels).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in lanelm.parameters())
    print(f"LaneLM parameters: {total_params:,}")
    
    # Check visual token count with first batch
    with torch.no_grad():
        test_imgs = fixed_batch["inputs"].to(device)
        test_feats = extract_full_fpn_feats(clrernet, test_imgs)
        test_vis_tokens = lanelm.encode_visual_tokens(test_feats)
        print(f"Visual tokens shape: {test_vis_tokens.shape}")
        print(f"  P3: {test_feats[0].shape} -> {test_feats[0].shape[2]*test_feats[0].shape[3]} tokens")
        print(f"  P4: {test_feats[1].shape} -> {test_feats[1].shape[2]*test_feats[1].shape[3]} tokens")
        print(f"  P5: {test_feats[2].shape} -> {test_feats[2].shape[2]*test_feats[2].shape[3]} tokens")
    
    # Optimizer
    optimizer = optim.Adam(lanelm.parameters(), lr=args.lr, weight_decay=0.0)
    
    # Loss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_cfg.pad_token_x, reduction='mean')
    
    print(f"\nStarting V4 OVERFIT Test ({args.overfit_size} images)...")
    print(f"Config: Full FPN + 2D PE + Absolute Tokenization")
    print(f"Target: Loss should approach 0 if model can learn\n")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        lanelm.train()
        total_loss = 0.0
        steps = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            imgs = batch["inputs"].to(device)
            feats = extract_full_fpn_feats(clrernet, imgs)
            
            # Collect all lanes data FIRST (before forward pass)
            all_img_indices = []
            all_x_tokens = []
            all_y_tokens = []
            all_lane_ids = []
            
            for img_idx, lanes_points in enumerate(batch["gt_points"]):
                lanes_points = [l for l in lanes_points if len(l) >= 4][:hparams.max_lanes]
                
                for l_idx, lane in enumerate(lanes_points):
                    pts = np.array(lane, dtype=np.float32).reshape(-1, 2)
                    x_np, y_np = tokenizer.encode_single_lane(pts)
                    
                    all_img_indices.append(img_idx)
                    all_x_tokens.append(torch.from_numpy(x_np).long())
                    all_y_tokens.append(torch.from_numpy(y_np).long())
                    all_lane_ids.append(torch.tensor(l_idx, dtype=torch.long))
            
            if not all_x_tokens:
                continue
            
            # Stack token tensors
            x_tokens = torch.stack(all_x_tokens).to(device)
            y_tokens = torch.stack(all_y_tokens).to(device)
            lane_ids = torch.stack(all_lane_ids).to(device)
            
            # Pure teacher forcing (no scheduled sampling)
            x_in = x_tokens.clone()
            x_in[:, 1:] = x_tokens[:, :-1]
            x_in[:, 0] = 0  # Start with padding token
            
            y_in = y_tokens.clone()
            y_in[:, 1:] = y_tokens[:, :-1]
            y_in[:, 0] = 0
            
            # CRITICAL: Encode visual tokens INSIDE the training loop (fresh graph!)
            visual_tokens = lanelm.encode_visual_tokens(feats)
            
            # Select visual tokens for each lane
            vis_tok_batch = torch.stack([visual_tokens[i] for i in all_img_indices]).to(device)
            
            # Forward (single pass for all lanes)
            logits_x, logits_y = lanelm(vis_tok_batch, x_in, y_in, lane_indices=lane_ids)
            
            # Loss (X only for now)
            B, T, V = logits_x.shape
            loss = loss_fn(logits_x.view(B * T, V), x_tokens.view(B * T))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lanelm.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
        
        if steps > 0:
            avg_loss = total_loss / steps
            
            if epoch % 10 == 0 or epoch <= 5:
                print(f"Ep {epoch}: Loss = {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "model_state_dict": lanelm.state_dict(),
                    "config": hparams.__dict__,
                    "epoch": epoch,
                    "loss": best_loss
                }, os.path.join(args.work_dir, "lanelm_v4_best.pth"))
            
            if epoch % 50 == 0:
                visualize(
                    lanelm, clrernet, fixed_batch, tokenizer, device, epoch,
                    os.path.join(args.work_dir, "vis")
                )
    
    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to {args.work_dir}/lanelm_v4_best.pth")
    
    # Final analysis
    if best_loss < 0.5:
        print("\n✅ SUCCESS: Model can learn from visual information!")
        print("   Next step: Scale up to more images")
    else:
        print("\n❌ FAILURE: Model still not learning properly")
        print("   Need to investigate further")


if __name__ == "__main__":
    main()

