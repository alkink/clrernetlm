"""
LaneLM Mini-Batch Overfit Script (Capacity Test).

Purpose:
  Verify that the model can memorize a small BATCH of images (8 images), not just one.
  This tests the model's capacity and verifies that it's not just "hardcoding" a single output.
  
  Config:
  - P5 Only + Explicit BOS + Disjoint Vocab.
  - Batch Size: 4
  - Overfit Size: 8 Images
  - Epochs: 2000 (Needs more time to learn 8 variations).
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
    lane_lm_loss_v3,
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
    T = logits_x.shape[1]
    logits_x[:, 0, boundary_idx:] = -1e9
    logits_x[:, 0, 0] = -1e9
    for bos_id in bos_token_ids:
        logits_x[:, :, bos_id] = -1e9
    if T > 1:
        logits_x[:, 1:, 1:boundary_idx] = -1e9
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
        
        y_tokens = torch.arange(T, device=device).unsqueeze(0).expand(B, -1).clone()
        x_tokens = torch.full((B, T), pad_token_x, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        current_input_x = torch.full((B, T), pad_token_x, dtype=torch.long, device=device)
        current_input_y = y_tokens.clone()
        current_input_x[:, 0] = current_bos 
        
        for t in range(T):
            logits_x, logits_y = lanelm_model(
                visual_tokens=visual_tokens,
                x_tokens=current_input_x,
                y_tokens=current_input_y,
                lane_indices=lane_id_tensor
            )
            step_logits_x = logits_x[:, t, :] 
            step_logits_y = logits_y[:, t, :]

            if t == 0:
                step_logits_x[:, boundary_idx:] = -float('inf')
                step_logits_x[:, 0] = -float('inf') 
            else:
                step_logits_x[:, 1:boundary_idx] = -float('inf')
                step_logits_x[:, 0] = -float('inf') 
            
            for bid in bos_token_ids:
                step_logits_x[:, bid] = -float('inf')

            if temperature > 0.0:
                probs_x = torch.softmax(step_logits_x / temperature, dim=-1)
                x_next = torch.multinomial(probs_x, num_samples=1).squeeze(1)
                probs_y = torch.softmax(step_logits_y / temperature, dim=-1)
                y_next = torch.multinomial(probs_y, num_samples=1).squeeze(1)
            else:
                x_next = torch.argmax(step_logits_x, dim=-1)
                y_next = torch.argmax(step_logits_y, dim=-1)

            still_running = ~finished
            x_tokens[still_running, t] = x_next[still_running]
            y_tokens[still_running, t] = y_next[still_running]
            
            if t + 1 < T:
                current_input_x[still_running, t+1] = x_next[still_running]

            is_eos = (x_next == pad_token_x) 
            finished = finished | is_eos

            if torch.all(finished):
                break

        all_x.append(x_tokens.cpu())
        all_y.append(y_tokens.cpu())

    return torch.stack(all_x, dim=1), torch.stack(all_y, dim=1)


def build_clean_dataloader(data_root, list_path, batch_size, num_workers, overfit_size=1):
    pipeline = [dict(type="albumentation", pipelines=clean_pipeline)]
    dataset = CulaneDataset(
        data_root=data_root,
        data_list=list_path,
        pipeline=pipeline,
        diff_file=None,
        test_mode=False,
    )
    indices = list(range(min(len(dataset), overfit_size)))
    subset = Subset(dataset, indices)
    print(f"Overfitting on {len(subset)} fixed images.")
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_lanelm_batch_v3,
    )
    return dataloader

def visualize_overfit(
    model, clrernet_model, batch, tokenizer, device, epoch, save_dir, bos_token_ids
):
    model.eval()
    imgs = batch["inputs"].to(device)
    boundary_idx = tokenizer.cfg.nbins_x 
    
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
            boundary_idx=boundary_idx,
            bos_token_ids=bos_token_ids
        )
        
        x_tokens_np = x_tokens_all.cpu().numpy()
        y_tokens_np = y_tokens_all.cpu().numpy()
        
        # Vis first 4 images
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
            save_path = os.path.join(save_dir, f"overfit_ep{epoch:04d}_img{i}.jpg")
            cv2.imwrite(save_path, img_vis)
    
    model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--list-path", default="dataset/list/train_gt.txt")
    parser.add_argument("--work-dir", default="work_dirs/lanelm_overfit_minibatch") # NEW DIR
    parser.add_argument("--epochs", type=int, default=2000) # MORE EPOCHS
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-y-loss", action="store_true", help="Disable Y token loss")
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
    
    # MINI-BATCH: 8 Images, Batch Size 4
    dataloader = build_clean_dataloader(
        args.data_root, args.list_path, batch_size=4, num_workers=0, overfit_size=8
    )
    
    clrernet = build_frozen_clrernet_backbone(args.config, args.checkpoint, device)
    lanelm = build_lanelm_model_v3(hparams, visual_in_channels=(64,)).to(device)
    
    for m in lanelm.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
            
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=hparams.img_w, img_h=hparams.img_h,
        num_steps=hparams.num_points, 
        nbins_x=nbins_x, 
        x_mode="relative_disjoint", 
        max_abs_dx=max_abs_dx
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    
    optimizer = optim.Adam(lanelm.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_cfg.pad_token_x, reduction='none')

    print(f"Starting MINI-BATCH (8 imgs) Overfit...")
    
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        lanelm.train()
        
        total_loss_epoch = 0.0
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
            
            pad_x = tokenizer.cfg.pad_token_x
            pad_y = tokenizer.T
            
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
            y_in[:, 0] = pad_y 
            
            logits_x, logits_y = lanelm(visual_tokens, x_in, y_in, lane_indices=lane_indices)
            logits_x = apply_logit_mask(logits_x, boundary_idx=nbins_x, bos_token_ids=bos_token_ids)
            
            B_flat, T, V = logits_x.shape
            logits_flat = logits_x.view(B_flat * T, V)
            targets_flat = x_tokens.view(B_flat * T)
            
            raw_loss = loss_fn(logits_flat, targets_flat)
            loss_x = raw_loss.mean()
            
            if not args.no_y_loss:
                loss_y_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_y)
                loss_y = loss_y_fn(logits_y.view(B_flat * T, -1), y_tokens.view(B_flat * T))
                loss = 0.5 * (loss_x + loss_y)
            else:
                loss = 0.5 * loss_x
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lanelm.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss_epoch += loss.item()
            steps += 1
        
        scheduler.step()
        
        if steps > 0:
            avg_loss = total_loss_epoch / steps
        else:
            avg_loss = 0.0
            
        if epoch % 10 == 0:
            print(f"Ep {epoch}: Avg Loss={avg_loss:.4f}")
        
        if avg_loss < best_loss and steps > 0:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": lanelm.state_dict(),
                "config": hparams.__dict__,
                "epoch": epoch,
                "loss": best_loss
            }, os.path.join(args.work_dir, "lanelm_overfit_minibatch_best.pth"))
            
        if epoch % 100 == 0 and steps > 0:
            visualize_overfit(
                lanelm, clrernet, batch, tokenizer, device, epoch, 
                os.path.join(args.work_dir, "vis"), bos_token_ids
            )

if __name__ == "__main__":
    main()
