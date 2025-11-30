"""
LaneLM 2K Subset Training Script.

Purpose:
  Scale-up test on 2000 images before full training.
  Crucially, this enables Y-LOSS to verify the model can learn vertical coordinates.
  
  Config:
  - P5 Only + Explicit BOS + Disjoint Vocab.
  - Y-Loss: ENABLED (Weight 1.0)
  - Data: train_gt_2k_subset.txt
  - Epochs: 50
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

def apply_logit_mask(logits_x, boundary_idx, bos_token_ids):
    # DISABLED for now as per previous success
    return logits_x

def autoregressive_decode_tokens_masked(
    lanelm_model,
    visual_tokens,
    num_points,
    nbins_x,
    max_lanes,
    temperature=0.0,
    boundary_idx=200,
    bos_token_ids=None
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
        
        # Now Y is also predicted autoregressively? 
        # No, LaneLM usually predicts X given Y (or predicts next (X,Y) pair).
        # Standard implementation: Input (X_t, Y_t) -> Output (X_t+1, Y_t+1)
        # But for inference, usually Y is fixed or generated.
        # Let's assume we generate Y as well or use fixed Y grid?
        # For strict LaneLM, Y is learned. Let's try to generate it.
        
        # Initial input
        curr_x = torch.full((B, 1), current_bos, dtype=torch.long, device=device)
        curr_y = torch.full((B, 1), 0, dtype=torch.long, device=device) # Y start token (usually 0 or special)
        
        # But wait, our model takes full sequence.
        # Inference loop:
        x_out = torch.full((B, T), pad_token_x, dtype=torch.long, device=device)
        y_out = torch.full((B, T), 0, dtype=torch.long, device=device)
        
        # Pre-fill BOS
        x_in = torch.full((B, T), pad_token_x, dtype=torch.long, device=device)
        x_in[:, 0] = current_bos
        y_in = torch.full((B, T), 0, dtype=torch.long, device=device) # Y BOS? Usually we use fixed Y sequence for CULane
        
        # If model learned Y, it should predict Y.
        # But for now, let's stick to the training logic: Input X, Y -> Output Next X, Y
        
        # Let's simplify: Assume we feed predicted tokens back
        for t in range(T):
            logits_x, logits_y = lanelm_model(
                visual_tokens, 
                x_tokens=x_in, 
                y_tokens=y_in, 
                lane_indices=lane_id_tensor
            )
            
            # Greedy X
            step_logits_x = logits_x[:, t, :]
            for bid in bos_token_ids: step_logits_x[:, bid] = -float('inf')
            pred_x = torch.argmax(step_logits_x, dim=-1)
            
            # Greedy Y
            step_logits_y = logits_y[:, t, :]
            pred_y = torch.argmax(step_logits_y, dim=-1)
            
            x_out[:, t] = pred_x
            y_out[:, t] = pred_y
            
            if t + 1 < T:
                x_in[:, t+1] = pred_x
                y_in[:, t+1] = pred_y # Feedback predicted Y
        
        all_x.append(x_out.cpu())
        all_y.append(y_out.cpu())

    return torch.stack(all_x, dim=1), torch.stack(all_y, dim=1)


def build_dataloader(data_root, list_path, batch_size, num_workers):
    pipeline = [dict(type="albumentation", pipelines=clean_pipeline)]
    dataset = CulaneDataset(
        data_root=data_root,
        data_list=list_path,
        pipeline=pipeline,
        diff_file=None,
        test_mode=False,
    )
    print(f"Loaded {len(dataset)} images from {list_path}")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True, # Drop incomplete batches for stability
        collate_fn=collate_lanelm_batch_v3,
    )
    return dataloader, dataset

def visualize(
    model, clrernet_model, batch, tokenizer, device, epoch, save_dir, bos_token_ids
):
    model.eval()
    imgs = batch["inputs"].to(device)
    
    with torch.no_grad():
        feats = extract_p5_feat(clrernet_model, imgs)
        visual_tokens = model.encode_visual_tokens(feats)
        
        x_tokens_all, y_tokens_all = autoregressive_decode_tokens_masked(
            lanelm_model=model,
            visual_tokens=visual_tokens,
            num_points=tokenizer.cfg.num_steps,
            nbins_x=model.nbins_x,
            max_lanes=4, 
            temperature=0.0,
            boundary_idx=tokenizer.cfg.nbins_x,
            bos_token_ids=bos_token_ids
        )
        
        x_tokens_np = x_tokens_all.cpu().numpy()
        y_tokens_np = y_tokens_all.cpu().numpy()
        
        for i in range(min(len(imgs), 4)):
            img_vis = (imgs[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
            
            for l in range(4):
                xt = x_tokens_np[i, l]
                yt = y_tokens_np[i, l]
                coords_resized = tokenizer.decode_single_lane(xt, yt)
                if coords_resized.shape[0] < 2:
                    continue
                for k in range(len(coords_resized) - 1):
                    p1 = (int(coords_resized[k][0]), int(coords_resized[k][1]))
                    p2 = (int(coords_resized[k+1][0]), int(coords_resized[k+1][1]))
                    cv2.line(img_vis, p1, p2, (0, 0, 255), 2)
            
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"ep{epoch:04d}_img{i}.jpg")
            cv2.imwrite(save_path, img_vis)
    
    model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--list-path", default="dataset/list/train_gt_2k_subset.txt") # 2K Subset
    parser.add_argument("--work-dir", default="work_dirs/lanelm_2k_subset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.work_dir, exist_ok=True)
    
    nbins_x = 200 
    max_abs_dx = 32 
    total_vocab_size = 300
    bos_token_ids = [296, 297, 298, 299]

    hparams = LaneLMHyperParams(
        nbins_x=total_vocab_size, 
        num_points=40,
        embed_dim=256,
        num_layers=4,
        max_lanes=4, 
    )
    
    # Load 2K Data
    dataloader, dataset = build_dataloader(
        args.data_root, args.list_path, batch_size=8, num_workers=4
    )
    fixed_vis_batch = next(iter(dataloader)) # For consistent visualization
    
    clrernet = build_frozen_clrernet_backbone(args.config, args.checkpoint, device)
    lanelm = build_lanelm_model_v3(hparams, visual_in_channels=(64,)).to(device)
    
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=hparams.img_w, img_h=hparams.img_h,
        num_steps=hparams.num_points, 
        nbins_x=nbins_x, 
        x_mode="relative_disjoint", 
        max_abs_dx=max_abs_dx
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    
    optimizer = optim.Adam(lanelm.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Losses
    # X Loss: Ignore Pad
    loss_x_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_cfg.pad_token_x, reduction='mean')
    # Y Loss: Ignore Pad (which is T=40 in our tokenizer usually, or 0? Let's check tokenizer)
    # Tokenizer uses 'T' as pad for Y.
    pad_y = tokenizer.T 
    loss_y_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_y, reduction='mean')

    print(f"Starting 2K-IMAGE Training (With Y-LOSS)...")
    
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        lanelm.train()
        total_loss = 0.0
        total_loss_x = 0.0
        total_loss_y = 0.0
        steps = 0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            imgs = batch["inputs"].to(device)
            feats = extract_p5_feat(clrernet, imgs)
            visual_tokens_imgs = lanelm.encode_visual_tokens(feats)
            
            lane_visual_tokens = []
            lane_x_list = []
            lane_y_list = []
            lane_id_list = []
            
            for img_idx, lanes_points in enumerate(batch["gt_points"]):
                 lanes_points = [l for l in lanes_points if len(l) >= 4][:4]
                 for l_idx, lane in enumerate(lanes_points):
                     pts = np.array(lane, dtype=np.float32).reshape(-1, 2)
                     x_np, y_np = tokenizer.encode_single_lane(pts)
                     lane_visual_tokens.append(visual_tokens_imgs[img_idx])
                     lane_x_list.append(torch.from_numpy(x_np).long())
                     lane_y_list.append(torch.from_numpy(y_np).long())
                     lane_id_list.append(torch.tensor(l_idx, dtype=torch.long)) 

            if not lane_visual_tokens:
                continue
                
            visual_tokens = torch.stack(lane_visual_tokens).to(device)
            x_tokens = torch.stack(lane_x_list).to(device)
            y_tokens = torch.stack(lane_y_list).to(device)
            lane_indices = torch.stack(lane_id_list).to(device)
            
            # Inputs shift right
            pad_x = tokenizer.cfg.pad_token_x
            
            x_in = x_tokens.clone()
            x_in[:, 1:] = x_tokens[:, :-1]
            for i in range(len(lane_indices)):
                lid = lane_indices[i].item()
                if lid < len(bos_token_ids):
                    x_in[i, 0] = bos_token_ids[lid]
                else:
                    x_in[i, 0] = bos_token_ids[-1]
            
            y_in = y_tokens.clone()
            y_in[:, 1:] = y_tokens[:, :-1]
            y_in[:, 0] = 0 # Y start token? Or Should we use Y[0]?
            # For Y, standard is 0..T-1 usually.
            # Let's assume Y_in starts with 0 (top of image or start of seq).
            
            logits_x, logits_y = lanelm(visual_tokens, x_in, y_in, lane_indices=lane_indices)
            
            B_flat, T, V = logits_x.shape
            
            # Loss X
            loss_x = loss_x_fn(logits_x.view(B_flat * T, V), x_tokens.view(B_flat * T))
            
            # Loss Y
            # logits_y shape: (B, T, T+1) usually (vocab size for Y is T+1 including pad)
            loss_y = loss_y_fn(logits_y.view(B_flat * T, -1), y_tokens.view(B_flat * T))
            
            loss = 0.5 * loss_x + 0.5 * loss_y
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lanelm.parameters(), max_norm=0.1)
            optimizer.step()
            
            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_y += loss_y.item()
            steps += 1
            
            if steps % 50 == 0:
                print(f"Ep {epoch} Step {steps}: L={loss.item():.4f} (X={loss_x.item():.4f} Y={loss_y.item():.4f})")
        
        scheduler.step()
        
        if steps > 0:
            avg_loss = total_loss / steps
            print(f"Ep {epoch} END: Avg L={avg_loss:.4f} (X={total_loss_x/steps:.4f} Y={total_loss_y/steps:.4f})")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "model_state_dict": lanelm.state_dict(),
                    "config": hparams.__dict__,
                    "epoch": epoch,
                    "loss": best_loss
                }, os.path.join(args.work_dir, "lanelm_2k_best.pth"))
        
        if epoch % 10 == 0:
            visualize(
                lanelm, clrernet, fixed_vis_batch, tokenizer, device, epoch, 
                os.path.join(args.work_dir, "vis"), bos_token_ids
            )

if __name__ == "__main__":
    main()

