"""
LaneLM v4 Full Dataset Training Script - RTX 3090 Optimized

Full CULane training set için optimize edilmiş versiyon:
- Batch size: 12 (RTX 3090 24GB VRAM için)
- Full dataset: dataset/list/train.txt (~88k images)
- Epochs: 50-100 (full dataset için yeterli)
- Learning rate: 3e-4 with cosine annealing
- Data augmentation: Clean pipeline (no augmentation)
- V5 Architecture: P5 Only + 2D PE + Absolute Tokenization
"""

import argparse
import os
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

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


# Clean pipeline (no augmentation for full training)
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
        dropout=0.0,  # No dropout for full training
        visual_in_channels=visual_in_channels,  # P5 Only: (64,)
    )
    return model


def build_full_dataloader(data_root, list_path, batch_size, num_workers=4):
    """Build dataloader for full dataset training."""
    pipeline = [dict(type="albumentation", pipelines=clean_pipeline)]
    dataset = CulaneDataset(
        data_root=data_root,
        data_list=list_path,
        pipeline=pipeline,
        diff_file=None,
        test_mode=False,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for full dataset
        num_workers=num_workers,  # Multiple workers for faster loading
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for consistent batch size
        collate_fn=collate_lanelm_batch_v3,
        persistent_workers=True if num_workers > 0 else False,
    )
    return dataloader, dataset


def visual_first_decode(model, visual_tokens, tokenizer, device, max_lanes):
    """V5 inference path: visual-first autoregressive decode (same as test)."""
    model_device = next(model.parameters()).device
    B, _, _ = visual_tokens.shape
    T = tokenizer.cfg.num_steps
    y_fixed = torch.arange(T, dtype=torch.long, device=model_device).unsqueeze(0).expand(B, -1)

    all_preds = []
    for lane_idx in range(max_lanes):
        lane_ids = torch.full((B,), lane_idx, dtype=torch.long, device=model_device)
        x_out = torch.zeros(B, T, dtype=torch.long, device=model_device)

        for t in range(T):
            x_in = torch.zeros_like(x_out)
            if t > 0:
                x_in[:, 1:t+1] = x_out[:, :t]
                x_in[:, 0] = x_out[:, 0]  # Keep first predicted token

            logits_x, _ = model(
                visual_tokens,
                x_in,
                y_fixed,
                lane_indices=lane_ids,
            )
            pred_x = torch.argmax(logits_x[:, t, :], dim=-1)
            x_out[:, t] = pred_x

        all_preds.append((x_out[0].cpu().numpy(), y_fixed[0].cpu().numpy()))
    return all_preds


def visualize(model, clrernet_model, batch, tokenizer, device, epoch, save_dir, max_lanes=4, use_p5_only=True):
    """Visualize predictions vs ground truth."""
    model.eval()
    imgs = batch["inputs"].to(device)
    
    with torch.no_grad():
        if use_p5_only:
            feats = extract_p5_feat(clrernet_model, imgs)
        else:
            raise NotImplementedError("Full FPN not supported in full training")
        visual_tokens = model.encode_visual_tokens(feats)
        
        all_preds = visual_first_decode(model, visual_tokens[:1], tokenizer, device, max_lanes)
        
        # Visualize first image
        img_vis = (imgs[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        
        # Draw predictions (colored) with smoothing
        colors = [(0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
        for l_idx, (x_tokens, y_tokens) in enumerate(all_preds):
            coords = tokenizer.decode_single_lane(x_tokens, y_tokens, smooth=True)
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
    parser = argparse.ArgumentParser(description="LaneLM v4 Full Dataset Training")
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--list-path", default="dataset/list/train.txt", help="Full training dataset list")
    parser.add_argument("--work-dir", default="work_dirs/lanelm_v4_full")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50 for full dataset)")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size (default: 12 for RTX 3090 24GB)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers (default: 4)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.work_dir, exist_ok=True)
    
    # ========== KEY CONFIG ==========
    # 1. ABSOLUTE tokenization (simpler, no delta confusion)
    nbins_x = 200  # Quantize X to 200 bins
    
    # 2. P5 ONLY (reduced noise, ~65 tokens with adaptive pooling)
    use_p5_only = True
    visual_in_channels = (64,)  # P5 Only
    
    # 3. Model hyperparams
    hparams = LaneLMHyperParams(
        nbins_x=nbins_x,
        num_points=40,
        embed_dim=256,
        num_layers=4,
        max_lanes=4,
    )
    
    # Tokenizer with ABSOLUTE mode
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=hparams.img_w,
        img_h=hparams.img_h,
        num_steps=hparams.num_points,
        nbins_x=nbins_x,
        x_mode="absolute",
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    
    # Build dataloader
    dataloader, dataset = build_full_dataloader(
        args.data_root, args.list_path, batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # Store first batch for visualization
    fixed_batch = next(iter(dataloader))
    
    # Build models
    print("Loading CLRerNet backbone (frozen)...")
    clrernet = build_frozen_clrernet_backbone(args.config, args.checkpoint, device)
    
    print("Building LaneLM v4 (P5 Only + 2D PE + Absolute Tokens + X-Loss Only)...")
    lanelm = build_lanelm_model_v4(hparams, visual_in_channels).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in lanelm.parameters())
    print(f"LaneLM parameters: {total_params:,}")
    
    # Check visual token count with first batch
    with torch.no_grad():
        test_imgs = fixed_batch["inputs"].to(device)
        test_feats = extract_p5_feat(clrernet, test_imgs)
        test_vis_tokens = lanelm.encode_visual_tokens(test_feats)
        print(f"Visual tokens shape: {test_vis_tokens.shape}")
        original_tokens = test_feats[0].shape[2] * test_feats[0].shape[3]
        actual_tokens = test_vis_tokens.shape[1]
        print(f"  P5 Only: {test_feats[0].shape} -> {original_tokens} tokens (original) -> {actual_tokens} tokens (V5 adaptive pooling)")
    
    # Optimizer
    optimizer = optim.Adam(lanelm.parameters(), lr=args.lr, weight_decay=0.0)
    
    # LR Scheduler (Cosine Annealing)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Loss functions
    loss_x_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_cfg.pad_token_x, reduction='mean')
    loss_x_pad_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    pad_y = tokenizer.T
    loss_y_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_y, reduction='mean')
    
    # Y-Loss disabled (X-loss only)
    use_y_loss = False
    y_loss_start_epoch = 999999
    y_loss_warmup_epochs = 50
    y_loss_final_weight = 0.05
    
    print(f"\nStarting V4 FULL DATASET Training...")
    print(f"Config: P5 Only + 2D PE + Absolute Tokenization + X-LOSS ONLY")
    print(f"Dataset: {len(dataset)} images")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print()
    
    best_loss = float('inf')
    
    # Scheduled sampling hyper-parameters
    ss_max_prob = 0.5
    ss_start_epoch = 30
    ss_warmup_epochs = 40

    # Autoregressive rollout loss
    ar_rollout_max_weight = 0.3
    ar_rollout_min_weight = 0.05

    for epoch in range(1, args.epochs + 1):
        lanelm.train()
        total_loss = 0.0
        total_loss_x = 0.0
        total_loss_y = 0.0
        total_loss_ar = 0.0
        steps = 0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            imgs = batch["inputs"].to(device)
            feats = extract_p5_feat(clrernet, imgs)
            
            # Collect all lanes data
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
            
            # Teacher Forcing inputs
            x_in_tf = x_tokens.clone()
            x_in_tf[:, 1:] = x_tokens[:, :-1]
            x_in_tf[:, 0] = x_tokens[:, 0]
            
            y_in = y_tokens.clone()
            y_in[:, 1:] = y_tokens[:, :-1]
            y_in[:, 0] = y_tokens[:, 0]
            
            # Encode visual tokens
            visual_tokens = lanelm.encode_visual_tokens(feats)
            vis_tok_batch = torch.stack([visual_tokens[i] for i in all_img_indices]).to(device)
            
            # Pure Teacher Forcing
            logits_x_tf, logits_y_tf = lanelm(
                vis_tok_batch, x_in_tf, y_in, lane_indices=lane_ids
            )
            
            pred_x_tf = torch.argmax(logits_x_tf, dim=-1)
            B_l, T_l = x_tokens.shape

            # Scheduled Sampling
            if epoch <= ss_start_epoch:
                ss_prob = 0.0
            else:
                if ss_warmup_epochs > 0:
                    progress = min(1.0, float(epoch - ss_start_epoch) / float(ss_warmup_epochs))
                else:
                    progress = 1.0
                ss_prob = ss_max_prob * progress
            
            if ss_prob > 0.0:
                x_in_ss = x_in_tf.clone()
                ss_mask = (torch.rand(B_l, T_l, device=device) < ss_prob)
                ss_mask[:, 0] = False
                
                gt_prev = x_tokens.clone()
                gt_prev[:, 0] = x_tokens[:, 0]
                pred_prev = pred_x_tf.clone()
                
                for t_step in range(1, T_l):
                    use_pred = ss_mask[:, t_step]
                    x_in_ss[:, t_step] = torch.where(
                        use_pred, pred_prev[:, t_step - 1], gt_prev[:, t_step - 1]
                    )
                
                x_in = x_in_ss
                logits_x, logits_y = lanelm(
                    vis_tok_batch, x_in, y_in, lane_indices=lane_ids
                )
            else:
                x_in = x_in_tf
                logits_x, logits_y = logits_x_tf, logits_y_tf
            
            # Loss calculation
            B, T, V = logits_x.shape
            loss_x = loss_x_fn(logits_x.view(B * T, V), x_tokens.view(B * T))
            
            if epoch >= y_loss_start_epoch:
                loss_y = loss_y_fn(logits_y.view(B * T, -1), y_tokens.view(B * T))
                if epoch < y_loss_start_epoch + y_loss_warmup_epochs:
                    warmup_progress = (epoch - y_loss_start_epoch) / y_loss_warmup_epochs
                    y_weight = 0.01 + 0.04 * warmup_progress
                else:
                    y_weight = y_loss_final_weight
                loss = (1.0 - y_weight) * loss_x + y_weight * loss_y
            else:
                loss = loss_x
                loss_y = torch.tensor(0.0, device=device)
            
            # AR Rollout Loss
            ar_loss = torch.tensor(0.0, device=device)
            if ar_rollout_max_weight > 0.0 and T_l > 1:
                x_in_roll = x_in_tf.clone()
                x_in_roll[:, 1:] = pred_x_tf[:, :-1]
                logits_x_roll, _ = lanelm(
                    vis_tok_batch, x_in_roll, y_in, lane_indices=lane_ids
                )
                roll_logits = logits_x_roll[:, 1:, :].reshape(-1, V)
                roll_targets = x_tokens[:, 1:].reshape(-1)
                roll_loss_all = torch.nn.functional.cross_entropy(
                    roll_logits,
                    roll_targets,
                    ignore_index=tokenizer_cfg.pad_token_x,
                    reduction="none",
                )
                t_ids = torch.arange(1, T_l, device=device).unsqueeze(0).expand(B_l, -1)
                t_norm = (t_ids - 1).float() / max(T_l - 2, 1)
                w_t = ar_rollout_min_weight + (1.0 - t_norm) * (ar_rollout_max_weight - ar_rollout_min_weight)
                w_flat = w_t.reshape(-1)
                valid_mask = roll_targets != tokenizer_cfg.pad_token_x
                if valid_mask.any():
                    ar_loss = (roll_loss_all[valid_mask] * w_flat[valid_mask]).sum() / valid_mask.sum()
                    loss = loss + ar_loss

            # Padding Loss
            pad_mask = (y_tokens == tokenizer.T)
            pad_mask_flat = pad_mask.view(B * T)
            pad_loss = torch.tensor(0.0, device=device)
            if pad_mask_flat.any():
                logits_x_flat = logits_x.view(B * T, V)
                logits_x_pad = logits_x_flat[pad_mask_flat]
                targets_pad = torch.zeros(logits_x_pad.size(0), dtype=torch.long, device=device)
                pad_loss = loss_x_pad_fn(logits_x_pad, targets_pad)
                loss = loss + 0.3 * pad_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lanelm.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_y += loss_y.item()
            total_loss_ar += ar_loss.item()
            steps += 1
            
            # Progress logging every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Ep {epoch} | Batch {batch_idx+1}/{len(dataloader)} | Loss={loss.item():.4f} (X={loss_x.item():.4f} AR={ar_loss.item():.4f})")
        
        if steps > 0:
            avg_loss = total_loss / steps
            avg_loss_x = total_loss_x / steps
            avg_loss_y = total_loss_y / steps
            avg_loss_ar = total_loss_ar / steps
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            if epoch % 5 == 0 or epoch <= 3:
                if epoch >= y_loss_start_epoch:
                    warmup_progress = min(1.0, (epoch - y_loss_start_epoch) / y_loss_warmup_epochs) if epoch < y_loss_start_epoch + y_loss_warmup_epochs else 1.0
                    y_weight = 0.01 + 0.04 * warmup_progress if epoch < y_loss_start_epoch + y_loss_warmup_epochs else y_loss_final_weight
                    print(f"Ep {epoch}: Loss = {avg_loss:.4f} (X={avg_loss_x:.4f} Y={avg_loss_y:.4f} AR={avg_loss_ar:.4f} Y-weight={y_weight:.3f}) LR={current_lr:.2e}")
                else:
                    print(f"Ep {epoch}: Loss = {avg_loss:.4f} (X={avg_loss_x:.4f} AR={avg_loss_ar:.4f} Y-loss disabled) LR={current_lr:.2e}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "model_state_dict": lanelm.state_dict(),
                    "config": hparams.__dict__,
                    "epoch": epoch,
                    "loss": best_loss
                }, os.path.join(args.work_dir, "lanelm_v4_best.pth"))
                print(f"  ✓ Saved best model (loss={best_loss:.4f})")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                    "model_state_dict": lanelm.state_dict(),
                    "config": hparams.__dict__,
                    "epoch": epoch,
                    "loss": avg_loss,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, os.path.join(args.work_dir, f"lanelm_v4_epoch_{epoch}.pth"))
                print(f"  ✓ Saved checkpoint at epoch {epoch}")
            
            # Visualize every 10 epochs
            if epoch % 10 == 0:
                visualize(
                    lanelm, clrernet, fixed_batch, tokenizer, device, epoch,
                    os.path.join(args.work_dir, "vis"), hparams.max_lanes, use_p5_only
                )
    
    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to {args.work_dir}/lanelm_v4_best.pth")


if __name__ == "__main__":
    main()

