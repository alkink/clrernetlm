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
from scipy.optimize import linear_sum_assignment

from libs.datasets import CulaneDataset
from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig
from tools.train_lanelm_culane_v3 import (
    LaneLMHyperParams,
    collate_lanelm_batch_v3,
    build_frozen_clrernet_backbone,
)
from mmdet.apis import init_detector
from mmdet.registry import MODELS

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

def extract_p5_feat(model, imgs: torch.Tensor):
    """Extract P5 Only (highest level, lowest resolution) for reduced noise."""
    with torch.no_grad():
        feats = model.extract_feat(imgs)  # Returns [P3, P4, P5]
        p5 = feats[-1]  # P5 is the last level
    return [p5]  # Return as list for compatibility


def build_lanelm_model_v4(
    hparams,
    visual_in_channels,
    x_embedding_scale: float = 1.0,
    lane_embedding_boost: float = 1.0,
):
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
        x_embedding_scale=float(x_embedding_scale),
        lane_embedding_boost=float(lane_embedding_boost),
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
    
    # For true training runs (overfit_size==0), shuffle improves generalization.
    shuffle = bool(overfit_size == 0)
    num_workers = 4 if overfit_size == 0 else 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_lanelm_batch_v3,
    )
    return dataloader, dataset

def visual_first_decode(model, visual_tokens, tokenizer, device, max_lanes):
    """V5 inference path: visual-first autoregressive decode (same as test)."""
    model_device = next(model.parameters()).device
    B, _, _ = visual_tokens.shape
    T = tokenizer.cfg.num_steps
    pad_token_x = tokenizer.cfg.pad_token_x
    y_fixed = torch.arange(T, dtype=torch.long, device=model_device).unsqueeze(0).expand(B, -1)

    all_preds = []
    for lane_idx in range(max_lanes):
        lane_ids = torch.full((B,), lane_idx, dtype=torch.long, device=model_device)
        x_out = torch.zeros(B, T, dtype=torch.long, device=model_device)

        for t in range(T):
            # 0-kp decode: BOS/pad hizası eğitim ile aynı olmalı.
            x_in = torch.full_like(x_out, pad_token_x)
            if t > 0:
                x_in[:, 1:t+1] = x_out[:, :t]
                # CRITICAL (V24): 0-kp modunda ilk token her zaman pad kalmalı.
                x_in[:, 0] = pad_token_x

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
    gt_points_batch = batch.get("gt_points", None)
    if gt_points_batch is None or len(gt_points_batch) == 0:
        print("[visualize] No gt_points in batch; skipping visualization.")
        model.train()
        return

    def _sort_lanes_left_to_right(lanes_points):
        lanes = []
        for lane in lanes_points:
            pts = np.array(lane, dtype=np.float32).reshape(-1, 2)
            if len(pts) >= 2:
                lanes.append(pts)
        if not lanes:
            return []
        means = [float(np.mean(l[:, 0])) for l in lanes]
        order = np.argsort(means)
        return [lanes[i] for i in order]

    def _teacher_forcing_decode_first_image(model, visual_tokens_1, tokenizer, lanes_points, max_lanes):
        """
        Teacher forcing argmax decode:
          - build GT tokens from gt_points
          - run model once with shifted GT input
          - take argmax over logits_x to obtain predicted tokens
        Returns:
          pred_list: list[(x_tokens_np, y_tokens_np)]
          gt_x: np.ndarray (L,T) GT tokens
          y_fixed: np.ndarray (T,) fixed y indices
        """
        model_device = next(model.parameters()).device
        T = tokenizer.cfg.num_steps
        pad_token_x = tokenizer.cfg.pad_token_x

        sorted_lanes = _sort_lanes_left_to_right(lanes_points)[:max_lanes]
        L = max_lanes

        gt_x_list = []
        for lane_idx in range(L):
            if lane_idx < len(sorted_lanes):
                x_t, y_t = tokenizer.encode_single_lane(sorted_lanes[lane_idx])
            else:
                x_t = np.full(T, pad_token_x, dtype=np.int64)
            gt_x_list.append(x_t)

        gt_x = torch.from_numpy(np.stack(gt_x_list, axis=0)).long().to(model_device)  # (L,T)
        x_in_tf = gt_x.clone()
        x_in_tf[:, 1:] = gt_x[:, :-1]
        x_in_tf[:, 0] = pad_token_x  # BOS/pad

        y_in = torch.arange(T, dtype=torch.long, device=model_device).unsqueeze(0).expand(L, -1)
        lane_ids = torch.arange(L, dtype=torch.long, device=model_device)

        # Expand visual tokens to per-lane batch
        vis_tok_batch = visual_tokens_1.expand(L, -1, -1).contiguous()

        logits_x, _ = model(vis_tok_batch, x_in_tf, y_in, lane_indices=lane_ids)
        pred_x = torch.argmax(logits_x, dim=-1)  # (L,T)

        pred_list = []
        y_fixed_np = np.arange(T, dtype=np.int64)
        for lane_idx in range(L):
            pred_list.append((pred_x[lane_idx].detach().cpu().numpy(), y_fixed_np))
        return pred_list, gt_x.detach().cpu().numpy(), y_fixed_np
    
    with torch.no_grad():
        if use_p5_only:
            feats = extract_p5_feat(clrernet_model, imgs)
        else:
            feats = extract_full_fpn_feats(clrernet_model, imgs)
        visual_tokens = model.encode_visual_tokens(feats)
        
        # Inference-like AR decode (this is what the user sees at test time)
        all_preds_ar = visual_first_decode(model, visual_tokens[:1], tokenizer, device, max_lanes)
        # Teacher forcing argmax decode (should match GT if model truly learned tokens)
        all_preds_tf, gt_x_tokens, _ = _teacher_forcing_decode_first_image(
            model, visual_tokens[:1], tokenizer, gt_points_batch[0], max_lanes
        )
        
        # Visualize first image
        img_vis = (imgs[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        img_vis_ar = img_vis.copy()
        img_vis_tf = img_vis.copy()
        
        # Draw predictions
        # NOTE: smooth=True is visualization-only; do not confuse with metric evaluation.
        colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]  # AR: red
        for l_idx, (x_tokens, y_tokens) in enumerate(all_preds_ar):
            coords = tokenizer.decode_single_lane(x_tokens, y_tokens, smooth=True)
            if coords.shape[0] >= 2:
                for k in range(len(coords) - 1):
                    p1 = (int(coords[k][0]), int(coords[k][1]))
                    p2 = (int(coords[k + 1][0]), int(coords[k + 1][1]))
                    if 0 <= p1[0] < 800 and 0 <= p2[0] < 800 and 0 <= p1[1] < 320 and 0 <= p2[1] < 320:
                        cv2.line(img_vis_ar, p1, p2, colors[l_idx % 4], 2)

        colors_tf = [(255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0)]  # TF: blue
        for l_idx, (x_tokens, y_tokens) in enumerate(all_preds_tf):
            coords = tokenizer.decode_single_lane(x_tokens, y_tokens, smooth=True)
            if coords.shape[0] >= 2:
                for k in range(len(coords) - 1):
                    p1 = (int(coords[k][0]), int(coords[k][1]))
                    p2 = (int(coords[k + 1][0]), int(coords[k + 1][1]))
                    if 0 <= p1[0] < 800 and 0 <= p2[0] < 800 and 0 <= p1[1] < 320 and 0 <= p2[1] < 320:
                        cv2.line(img_vis_tf, p1, p2, colors_tf[l_idx % 4], 2)
        
        # Draw GT (GREEN)
        for lane in gt_points_batch[:1][0][:max_lanes]:
                pts = np.array(lane, dtype=np.float32).reshape(-1, 2)
                if len(pts) >= 2:
                    for k in range(len(pts) - 1):
                        p1 = (int(pts[k][0]), int(pts[k][1]))
                        p2 = (int(pts[k+1][0]), int(pts[k+1][1]))
                        if 0 <= p1[0] < 800 and 0 <= p2[0] < 800 and 0 <= p1[1] < 320 and 0 <= p2[1] < 320:
                            cv2.line(img_vis_ar, p1, p2, (0, 255, 0), 3)
                            cv2.line(img_vis_tf, p1, p2, (0, 255, 0), 3)

        # Print token-level diagnostic to separate "exposure bias" vs "decode/plot bug".
        # Build GT token tensor (sorted) consistent with TF decode.
        try:
            T = tokenizer.cfg.num_steps
            pad_x = tokenizer.cfg.pad_token_x
            # Rebuild GT tokens in the same sorted-lanes order used above
            sorted_lanes = _sort_lanes_left_to_right(gt_points_batch[0])[:max_lanes]
            gt_x_list = []
            for lane_idx in range(max_lanes):
                if lane_idx < len(sorted_lanes):
                    x_t, y_t = tokenizer.encode_single_lane(sorted_lanes[lane_idx])
                else:
                    x_t = np.full(T, pad_x, dtype=np.int64)
                gt_x_list.append(x_t)
            gt_x_np = np.stack(gt_x_list, axis=0)
            valid = (gt_x_np != pad_x)

            # TF error vs GT
            tf_x_np = np.stack([p[0] for p in all_preds_tf], axis=0)
            if valid.any():
                tf_acc = float(((tf_x_np == gt_x_np) & valid).sum() / valid.sum())
                tf_mae = float(np.abs(tf_x_np[valid] - gt_x_np[valid]).mean())
            else:
                tf_acc, tf_mae = 0.0, 0.0

            # AR error vs GT
            ar_x_np = np.stack([p[0] for p in all_preds_ar], axis=0)
            if valid.any():
                ar_acc = float(((ar_x_np == gt_x_np) & valid).sum() / valid.sum())
                ar_mae = float(np.abs(ar_x_np[valid] - gt_x_np[valid]).mean())
            else:
                ar_acc, ar_mae = 0.0, 0.0

            print(
                f"[visualize] ep{epoch:04d} TF_ACC={tf_acc:.3f} TF_MAE_tok={tf_mae:.2f} | "
                f"AR_ACC={ar_acc:.3f} AR_MAE_tok={ar_mae:.2f}"
            )
        except Exception as e:
            print(f"[visualize] token diagnostic failed: {e}")
        
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f"ep{epoch:04d}_ar.jpg"), img_vis_ar)
        cv2.imwrite(os.path.join(save_dir, f"ep{epoch:04d}_tf.jpg"), img_vis_tf)
    
    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    parser.add_argument("--checkpoint", default="clrernet_culane_dla34_ema.pth")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--list-path", default="dataset/list/train_100.txt", help="Dataset list file (default: train_100.txt for 100-image subset)")
    parser.add_argument("--work-dir", default="work_dirs/lanelm_v4_fixed")
    parser.add_argument("--overfit-size", type=int, default=1, help="Number of images for overfit test (0=all)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4, was 1e-3 but too high for 100 images)")
    # ---- Debug / ablation flags (0-kp alignment + simplify losses) ----
    parser.add_argument(
        "--num-pseudo-points",
        type=int,
        default=0,
        help="CLRNet'ten alınacak başlangıç keypoint sayısı (0 => prompting/pseudo-label kapalı, saf GT).",
    )
    parser.add_argument(
        "--use-y-loss",
        action="store_true",
        help="Y-loss'u aç (debug için default kapalı).",
    )
    parser.add_argument(
        "--y-loss-weight",
        type=float,
        default=1.0,
        help="Y-loss ağırlığı. Kullanılan form: loss = loss_x + y_loss_weight * loss_y",
    )
    parser.add_argument(
        "--presence-weight",
        type=float,
        default=0.0,
        help="Presence loss ağırlığı (debug için default 0).",
    )
    parser.add_argument(
        "--ss-max-prob",
        type=float,
        default=0.0,
        help="Scheduled sampling maksimum olasılığı (debug için default 0).",
    )
    parser.add_argument(
        "--ar-rollout-max-weight",
        type=float,
        default=0.0,
        help="Autoregressive rollout loss max weight (debug için default 0).",
    )
    parser.add_argument(
        "--ar-rollout-min-weight",
        type=float,
        default=0.0,
        help="Autoregressive rollout loss min weight (debug için default 0).",
    )
    parser.add_argument(
        "--pad-loss-weight",
        type=float,
        default=0.0,
        help="Padding (no-lane) X=0 loss ağırlığı (debug için default 0).",
    )
    parser.add_argument(
        "--x-embedding-scale",
        type=float,
        default=1.0,
        help="KeypointEmbedding x_embedding_scale (overfit debug için 1.0 önerilir; default 1.0).",
    )
    parser.add_argument(
        "--lane-embedding-boost",
        type=float,
        default=1.0,
        help="KeypointEmbedding lane_embedding_boost (paper'a daha yakın: default 1.0; çok yüksek değer görseli bastırabilir).",
    )
    parser.add_argument(
        "--no-sort-gt-lanes",
        action="store_true",
        help="GT lane'leri soldan-sağa sıralamayı kapat (varsayılan: açık).",
    )
    parser.add_argument(
        "--y-direction",
        default="top_to_bottom",
        choices=["top_to_bottom", "bottom_to_top"],
        help="Tokenizer y sampling direction. Prompting/causal için train/test aynı olmalı.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.work_dir, exist_ok=True)
    
    # ========== KEY CONFIG CHANGES ==========
    # 1. ABSOLUTE tokenization (simpler, no delta confusion)
    # V7: Increase granularity to reduce zigzagging (PDF uses 800 bins)
    # 200 bins = 4px per bin (too coarse, causes zigzagging)
    # 400 bins = 2px per bin (better, reduces zigzagging)
    # 800 bins = 1px per bin (PDF standard, best but slower training)
    nbins_x = 800  # V7: PDF standard (line 570: "800 nbins and 100 training epochs")
    
    # 2. V18: PDF'de Full FPN (P3+P4+P5) kullanılıyor (Line 344-365, Table 5)
    # PDF: "{F0, F1, F2} = f(Xv)" - 3 seviye FPN
    # PDF Ablation Study (Table 5): FPN yok: 68.36, FPN var: 70.71 (+2.35 F1!)
    # PDF: "Li = Ev(Fi) + PEvision(Hi, Wi) + LE(i)" - Level Embedding (LE) var!
    # Bizde Level Embedding zaten var (model.py line 150-151), ama sadece Full FPN'de anlamlı
    use_p5_only = False  # V18: PDF'de Full FPN kullanılıyor!
    if use_p5_only:
        visual_in_channels = (64,)  # P5 Only
    else:
        visual_in_channels = (64, 64, 64)  # Full FPN (P3+P4+P5)
    
    # 3. Model hyperparams
    # V17: PDF'de LaneLM-512 (DLA34) için embed_dim=512 olmalı (Line 521-523, Table 1)
    # PDF: "With the different hidden size of the decoder, the visual encoders of LaneLM-128, LaneLM-256,
    # and LaneLM-512 are ResNet18, ResNet34 [38] and DLA34 [39]."
    # CULane için en iyi sonuç: LaneLM-512* (DLA34, embed_dim=512) - Total F1: 81.43 (Table 3)
    hparams = LaneLMHyperParams(
        nbins_x=nbins_x,
        num_points=40,
        embed_dim=512,  # V17: PDF'de LaneLM-512 için 512 (önceden 256 yanlıştı!)
        num_layers=3,  # V15: PDF'ye göre 3 layers (line 382: "consists of 3 layers of LaneLM blocks")
        max_lanes=4,
    )
    
    # Tokenizer with ABSOLUTE mode (key fix!)
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=hparams.img_w,
        img_h=hparams.img_h,
        num_steps=hparams.num_points,
        nbins_x=nbins_x,
        x_mode="absolute",  # NOT relative_disjoint!
        y_direction=str(args.y_direction),
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    
    # Build dataloader
    # V14: PDF'de batch_size=128 (line 570), ama overfit test için batch_size=8 kullanıyoruz
    # Batch size=1 çok küçük, gradient variance'ı artırır
    # NOTE: overfit_size==0 means "use full list". Batch size 1 makes full-list runs unnecessarily slow.
    batch_size = 8 if (args.overfit_size == 0 or args.overfit_size > 1) else 1
    dataloader, dataset = build_clean_dataloader(
        args.data_root, args.list_path, batch_size=batch_size, overfit_size=args.overfit_size
    )
    
    # Store first batch for visualization
    fixed_batch = next(iter(dataloader))
    
    # Build models
    print("Loading CLRerNet backbone (frozen)...")
    clrernet = build_frozen_clrernet_backbone(args.config, args.checkpoint, device)
    
    # V12: Full CLRNet model (head dahil) sadece pseudo-label / prompting gerekiyorsa yüklenmeli.
    # 0-kp debug modunda bunu yüklemek pahalı ve train/test mismatch'i gizleyebilir.
    clrernet_full = None
    if args.num_pseudo_points > 0:
        print("Loading full CLRerNet model for pseudo labels...")
        clrernet_full = init_detector(args.config, args.checkpoint, device=device)
        clrernet_full.eval()
        # Freeze all parameters
        for param in clrernet_full.parameters():
            param.requires_grad = False
    
    if use_p5_only:
        print("Building LaneLM v4 (P5 Only + 2D PE + Absolute Tokens + X-Loss Only)...")
    else:
        print("Building LaneLM v4 (Full FPN + 2D PE + Absolute Tokens)...")
    lanelm = build_lanelm_model_v4(
        hparams,
        visual_in_channels,
        x_embedding_scale=args.x_embedding_scale,
        lane_embedding_boost=args.lane_embedding_boost,
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in lanelm.parameters())
    print(f"LaneLM parameters: {total_params:,}")
    
    # Check visual token count with first batch
    with torch.no_grad():
        test_imgs = fixed_batch["inputs"].to(device)
        if use_p5_only:
            test_feats = extract_p5_feat(clrernet, test_imgs)
        else:
            test_feats = extract_full_fpn_feats(clrernet, test_imgs)
        test_vis_tokens = lanelm.encode_visual_tokens(test_feats)
        print(f"Visual tokens shape: {test_vis_tokens.shape}")
        if use_p5_only:
            # V5: With adaptive pooling, P5 (10,25) -> (5,13) = 65 tokens
            original_tokens = test_feats[0].shape[2] * test_feats[0].shape[3]
            actual_tokens = test_vis_tokens.shape[1]
            print(f"  P5 Only: {test_feats[0].shape} -> {original_tokens} tokens (original) -> {actual_tokens} tokens (V5 adaptive pooling)")
        else:
            print(f"  P3: {test_feats[0].shape} -> {test_feats[0].shape[2]*test_feats[0].shape[3]} tokens")
            print(f"  P4: {test_feats[1].shape} -> {test_feats[1].shape[2]*test_feats[1].shape[3]} tokens")
            print(f"  P5: {test_feats[2].shape} -> {test_feats[2].shape[2]*test_feats[2].shape[3]} tokens")
    
    # Optimizer
    optimizer = optim.Adam(lanelm.parameters(), lr=args.lr, weight_decay=0.0)
    
    # FIX 2: LR Scheduler (Cosine Annealing)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Loss functions
    # X Loss: Ignore padding token (iç bölge için)
    loss_x_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_cfg.pad_token_x, reduction='mean')
    # V5: Padding bölgesi için ayrı loss (X=0 öğrenilsin diye, ignore yok)
    loss_x_pad_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    # Y Loss: Ignore padding token (T=40 is used as pad for Y in tokenizer)
    pad_y = tokenizer.T  # T is used as padding/EOS for Y
    loss_y_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_y, reduction='mean')
    # V6: Lane presence loss (per-lane existence probability)
    loss_presence_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    # V6: Pixel-space alignment loss (resized 800x320) weight
    # ŞU AN SADECE DIAGNOSE AMAÇLI: weight=0 → loss'a eklenmiyor, sadece log için hesaplanıyor.
    pixel_loss_weight = 0.0
    # Bin merkezleri (X ekseninde) lazy olarak oluşturulacak
    bin_centers = None
    
    # Debug hedefi: 0-kp overfit'te önce X'i temizce fit etmek (loss ~ 0).
    # Bu nedenle default olarak Y-loss kapalı. İstenirse --use-y-loss ile açılabilir.
    use_y_loss = bool(args.use_y_loss)
    y_loss_weight = float(args.y_loss_weight)
    
    print(f"\nStarting V4 OVERFIT Test ({args.overfit_size} images)...")
    if use_p5_only:
        print(f"Config: P5 Only + 2D PE + Absolute Tokenization + X-LOSS ONLY")
    else:
        print(f"Config: Full FPN + 2D PE + Absolute Tokenization + {'Y-LOSS' if use_y_loss else 'X-LOSS ONLY'}")
    print(f"Target: Loss should approach 0 if model can learn\n")
    
    best_loss = float('inf')
    
    # Scheduled sampling hyper-parameters (for exposure bias mitigation)
    ss_max_prob = float(args.ss_max_prob)
    ss_start_epoch = 30        # V5: daha erken başla
    ss_warmup_epochs = 40      # V5: ss_prob'i 0 -> ss_max_prob arası yumuşak arttır

    # V5: Autoregressive rollout loss (full sequence, decaying weight)
    ar_rollout_max_weight = float(args.ar_rollout_max_weight)
    ar_rollout_min_weight = float(args.ar_rollout_min_weight)

    for epoch in range(1, args.epochs + 1):
        lanelm.train()
        total_loss = 0.0
        total_loss_x = 0.0
        total_loss_y = 0.0
        total_loss_ar = 0.0
        total_loss_presence = 0.0
        total_loss_pixel = 0.0
        total_acc = 0.0
        total_acc_count = 0.0
        steps = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            imgs = batch["inputs"].to(device)
            if use_p5_only:
                feats = extract_p5_feat(clrernet, imgs)
            else:
                feats = extract_full_fpn_feats(clrernet, imgs)
            
            # Collect per-image, per-lane sequences (including negative lanes)
            all_img_indices = []
            all_x_tokens = []
            all_y_tokens = []
            all_lane_ids = []
            all_presence_targets = []
            all_lq_masks = []

            pad_token_x = tokenizer_cfg.pad_token_x
            T_steps = tokenizer_cfg.num_steps
            
            # V12: PDF "*" Version for CULane - CLRNet Lq + Bipartite Matching (PDF Sayfa 867-871)
            # PDF: "Our model receives two adjacent keypoints output from CLRNet [6] as init prompts
            # for each lane and rollouts the remaining keypoints in the * version."
            # 
            # CRITICAL: PDF'de CULane için "*" versiyonu kullanılıyor, LLAMAS strategy değil!
            # LLAMAS strategy sadece LLAMAS dataset'i için (PDF Sayfa 887-891).
            # 
            # Training: CLRNet Lq ◦ GT Lgt (bipartite matching)
            # Test: CLRNet prompting (Lq from CLRNet) - already implemented
            # 
            # Avantajlar:
            # 1. Training/test uyumlu (Her ikisinde de CLRNet Lq)
            # 2. Model CLRNet keypoint'lerini yorumlamayı öğrenir
            # 3. PDF'de CULane için çalışıyor (F1@0.5 = 79.04)
            
            # Prompt/pseudo-label ayarı:
            # num_pseudo_points=0 => saf GT (0-kp), CLRNet prompting/pseudo-label tamamen kapalı.
            num_pseudo_points = int(args.num_pseudo_points)
            lq_noise_range = 0
            
            # Skip CLRNet inference when no pseudo points
            if num_pseudo_points > 0:
                with torch.no_grad():
                    from mmdet.structures import DetDataSample
                    batch_data_samples = []
                    for img_idx in range(len(batch["gt_points"])):
                        data_sample = DetDataSample()
                        data_sample.set_metainfo({
                            "img_shape": (img_scale[1], img_scale[0], 3),
                            "ori_shape": (img_scale[1], img_scale[0], 3),
                        })
                        batch_data_samples.append(data_sample)
                    if clrernet_full is None:
                        raise RuntimeError(
                            "num_pseudo_points>0 ama clrernet_full None. "
                            "CLRNet pseudo-label modeli yüklenmemiş."
                        )
                    clrernet_results = clrernet_full.predict(imgs, batch_data_samples, rescale=False)
            else:
                clrernet_results = [{"lanes": []} for _ in range(len(batch["gt_points"]))]
            
            for img_idx, lanes_points in enumerate(batch["gt_points"]):
                # Filter valid GT lanes for this image
                lanes_points = [l for l in lanes_points if len(l) >= 4]
                # Overfit/debug: GT lane slot permütasyonunu azaltmak için soldan-sağa sırala.
                # (Özellikle num_pseudo_points=0 modunda slot id -> lane eşleşmesi tutarlı olmalı.)
                if not args.no_sort_gt_lanes and len(lanes_points) > 1:
                    def _lane_sort_key(lane):
                        pts = np.array(lane, dtype=np.float32).reshape(-1, 2)
                        if pts.size == 0:
                            return float("inf")
                        # En alt noktayı (max y) al ve x'e göre sırala
                        idx = int(np.argmax(pts[:, 1]))
                        return float(pts[idx, 0])
                    lanes_points = sorted(lanes_points, key=_lane_sort_key)
                num_gt_lanes = min(len(lanes_points), hparams.max_lanes)
                
                # V13: CRITICAL FIX - Handle GT: 0 lanes case
                # Even if num_gt_lanes == 0, we still create negative lanes (presence_target=0.0)
                # This teaches the model to predict "no lane" for images with GT: 0 lanes
                
                # V12: Get CLRNet lanes for this image
                clr_result = clrernet_results[img_idx]
                clr_lanes = clr_result.get("lanes", [])
                
                # V12: Bipartite matching (CLRNet lanes ↔ GT lanes)
                # Match based on start point distance (PDF Eq. 10)
                if len(clr_lanes) > 0 and num_gt_lanes > 0:
                    # Extract start points (first point) from CLRNet lanes
                    clr_start_points = []
                    for clr_lane in clr_lanes:
                        if hasattr(clr_lane, 'points'):
                            points = clr_lane.points
                        else:
                            points = clr_lane
                        if isinstance(points, torch.Tensor):
                            points = points.cpu().numpy()
                        if len(points) > 0:
                            # CRITICAL: CLRNet points are normalized [0,1] in ori_img space (1640x590)
                            # Convert to resized space (800x320) for matching (same as test)
                            ori_img_w = crop_bbox[2] - crop_bbox[0]  # 1640
                            ori_img_h = crop_bbox[3] - crop_bbox[1]  # 590
                            # 1. Denormalize to original image space
                            x_start = points[0, 0] * ori_img_w  # Denormalize X (1640)
                            y_start = points[0, 1] * ori_img_h  # Denormalize Y (590)
                            # 2. Apply crop and resize
                            x_min, y_min, x_max, y_max = crop_bbox
                            x_start = x_start - x_min
                            y_start = y_start - y_min
                            x_scale = img_scale[0] / (x_max - x_min)
                            y_scale = img_scale[1] / (y_max - y_min)
                            x_start = x_start * x_scale
                            y_start = y_start * y_scale
                            clr_start_points.append([x_start, y_start])
                        else:
                            clr_start_points.append([0, 0])  # Padding
                    
                    # Extract start points from GT lanes
                    gt_start_points = []
                    for lane in lanes_points[:num_gt_lanes]:
                        pts = np.array(lane, dtype=np.float32).reshape(-1, 2)
                        if len(pts) > 0:
                            gt_start_points.append(pts[0].tolist())
                        else:
                            gt_start_points.append([0, 0])
                    
                    # Bipartite matching: minimize start point distance
                    num_clr = len(clr_start_points)
                    num_gt = len(gt_start_points)
                    if num_clr > 0 and num_gt > 0:
                        # Cost matrix: distance between start points
                        cost_matrix = np.zeros((num_clr, num_gt))
                        for i, clr_sp in enumerate(clr_start_points):
                            for j, gt_sp in enumerate(gt_start_points):
                                dist = np.sqrt((clr_sp[0] - gt_sp[0])**2 + (clr_sp[1] - gt_sp[1])**2)
                                cost_matrix[i, j] = dist
                        
                        # Hungarian algorithm (linear_sum_assignment)
                        row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        # Create mapping: clr_idx -> gt_idx
                        clr_to_gt = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
                    else:
                        clr_to_gt = {}
                else:
                    clr_to_gt = {}
                
                # For each lane slot, either use matched CLRNet Lq + GT Lgt or create negative lane
                for lane_slot in range(hparams.max_lanes):
                    if lane_slot < num_gt_lanes:
                        gt_lane = lanes_points[lane_slot]
                        gt_pts = np.array(gt_lane, dtype=np.float32).reshape(-1, 2)
                        
                        # V12: Try to find matching CLRNet lane
                        matched_clr_idx = None
                        for clr_idx, gt_idx in clr_to_gt.items():
                            if gt_idx == lane_slot and clr_idx < len(clr_lanes):
                                matched_clr_idx = clr_idx
                                break
                        
                        # Base tokens from GT (fixed-y formulation)
                        x_np, y_np = tokenizer.encode_single_lane(gt_pts)

                        # FIX (V42): In fixed-y tokenization, we cannot concatenate Lq ◦ Lgt by
                        # "packing tokens to the front". That destroys the meaning of timestep t.
                        # Instead, we place CLRNet prompt x-tokens at their corresponding y-step indices (t)
                        # and simply EXCLUDE those positions from loss (Lq mask).
                        lq_mask = np.zeros((T_steps,), dtype=bool)
                        if matched_clr_idx is not None and num_pseudo_points > 0:
                            clr_lane = clr_lanes[matched_clr_idx]
                            clr_points = clr_lane.points if hasattr(clr_lane, "points") else clr_lane
                            if isinstance(clr_points, torch.Tensor):
                                clr_points = clr_points.cpu().numpy()
                            clr_points = np.asarray(clr_points, dtype=np.float32)
                            if clr_points.ndim == 2 and clr_points.shape[1] == 2 and len(clr_points) >= num_pseudo_points:
                                first_k = clr_points[:num_pseudo_points].copy()  # normalized [0,1] in ori crop space

                                # Optional noise in pixel-x
                                ori_img_w = crop_bbox[2] - crop_bbox[0]
                                ori_img_h = crop_bbox[3] - crop_bbox[1]
                                if lq_noise_range > 0:
                                    noise_x = np.random.uniform(-lq_noise_range, lq_noise_range, size=(num_pseudo_points,))
                                else:
                                    noise_x = np.zeros((num_pseudo_points,), dtype=np.float32)

                                x_pix = first_k[:, 0] * ori_img_w + noise_x
                                y_pix = first_k[:, 1] * ori_img_h

                                # Crop + resize to tokenizer space (800x320)
                                x_min, y_min, x_max, y_max = crop_bbox
                                x_pix = x_pix - x_min
                                y_pix = y_pix - y_min
                                x_scale = img_scale[0] / (x_max - x_min)
                                y_scale = img_scale[1] / (y_max - y_min)
                                x_resized = np.clip(x_pix * x_scale, 0.0, float(img_scale[0] - 1))
                                y_resized = np.clip(y_pix * y_scale, 0.0, float(img_scale[1] - 1))

                                sample_ys = tokenizer._compute_sample_ys()  # (T,)
                                for kp_i in range(num_pseudo_points):
                                    # nearest y-step index
                                    t_idx = int(np.argmin(np.abs(sample_ys - float(y_resized[kp_i]))))
                                    x = float(x_resized[kp_i])
                                    if 0.0 <= x < float(img_scale[0]):
                                        bin_idx = int(round(x / (img_scale[0] - 1) * (tokenizer_cfg.nbins_x - 1)))
                                        bin_idx = max(1, min(tokenizer_cfg.nbins_x - 1, bin_idx))
                                        x_np[t_idx] = bin_idx
                                        y_np[t_idx] = t_idx
                                        lq_mask[t_idx] = True
                        
                        presence_target = 1.0
                    else:
                        # Negative lane: all padding tokens
                        x_np = np.full((T_steps,), pad_token_x, dtype=np.int64)
                        y_np = np.full((T_steps,), T_steps, dtype=np.int64)  # pad_y index = T
                        presence_target = 0.0
                        lq_mask = np.zeros((T_steps,), dtype=bool)

                    all_img_indices.append(img_idx)
                    all_x_tokens.append(torch.from_numpy(x_np).long())
                    all_y_tokens.append(torch.from_numpy(y_np).long())
                    all_lane_ids.append(torch.tensor(lane_slot, dtype=torch.long))
                    all_presence_targets.append(presence_target)
                    all_lq_masks.append(torch.from_numpy(lq_mask).bool())

            # V13: CRITICAL FIX - Don't skip if all_x_tokens is empty
            # Even if an image has GT: 0 lanes, we still want to train on negative lanes
            # This ensures the model learns to predict presence=0 for "no lane" cases
            if not all_x_tokens:
                # This should not happen now because we always create max_lanes slots
                # But keep this check as a safety measure
                continue

            # Stack token tensors (B_lanes, T)
            x_tokens = torch.stack(all_x_tokens).to(device)
            y_tokens = torch.stack(all_y_tokens).to(device)
            lane_ids = torch.stack(all_lane_ids).to(device)
            presence_targets = torch.tensor(all_presence_targets, dtype=torch.float32, device=device)
            
            # V42 FIX: Loss mask is per-timestep based on where we injected CLRNet prompts (Lq mask).
            # Do NOT "pack to the front" or mask by prefix length; that breaks fixed-y semantics.
            if all_lq_masks:
                lq_masks = torch.stack(all_lq_masks).to(device)  # (B_lanes, T) bool
                loss_mask = ~lq_masks
            else:
                loss_mask = torch.ones_like(x_tokens, dtype=torch.bool)
            
            # --- Teacher Forcing Girişleri ---
            # 0-kp modunda inference BOS=0 ile başlıyor; training de aynı olmalı.
            # Prompt'lu modda (num_pseudo_points>0) ilk token(lar) Lq olarak kalabilir.
            x_in_tf = x_tokens.clone()
            x_in_tf[:, 1:] = x_tokens[:, :-1]
            # Always BOS/pad at t=0 for fixed-y formulation (inference aligned).
            x_in_tf[:, 0] = pad_token_x
            
            # Y girişini inference ile hizala:
            # Inference tarafında y koordinatı fixed step index (y_fixed=t) olarak veriliyor.
            # Training'de de aynı y_in kullanılmalı; aksi halde train/test mismatch oluşur.
            y_in = torch.arange(T_steps, dtype=torch.long, device=device).unsqueeze(0).expand(x_tokens.shape[0], -1)
            
            # CRITICAL: Encode visual tokens INSIDE the training loop (fresh graph!)
            visual_tokens = lanelm.encode_visual_tokens(feats)

            # Select visual tokens for each lane (reuse per-image visual tokens across lane slots)
            vis_tok_batch = torch.stack([visual_tokens[i] for i in all_img_indices]).to(device)
            
            # --- 1. Geçiş: Pure Teacher Forcing ---
            logits_x_tf, logits_y_tf, presence_logits = lanelm(
                vis_tok_batch, x_in_tf, y_in, lane_indices=lane_ids, return_presence=True
            )
            
            # Öğretmen zorlama çıktısından argmax tahminlerini kaydet (her durumda)
            pred_x_tf = torch.argmax(logits_x_tf, dim=-1)  # (B, T)
            B_l, T_l = x_tokens.shape  # V5: rollout ve SS için her zaman tanımlı olsun
            # Debug metrik: token accuracy (valid target positions)
            with torch.no_grad():
                acc_mask = (x_tokens != pad_token_x) & loss_mask  # loss ile aynı maskeyi kullan
                if acc_mask.any():
                    correct = (pred_x_tf == x_tokens) & acc_mask
                    total_acc += correct.float().sum().item()
                    total_acc_count += acc_mask.float().sum().item()

            # --- Scheduled Sampling oranı hesapla ---
            if epoch <= ss_start_epoch:
                ss_prob = 0.0
            else:
                if ss_warmup_epochs > 0:
                    progress = min(
                        1.0, float(epoch - ss_start_epoch) / float(ss_warmup_epochs)
                    )
                else:
                    progress = 1.0
                ss_prob = ss_max_prob * progress
            
            # --- 2. Geçiş: Scheduled Sampling ile karışık giriş (opsiyonel) ---
            if ss_prob > 0.0:
                # x_in_ss: bazı adımlarda GT, bazı adımlarda model tahmini kullan
                x_in_ss = x_in_tf.clone()
                
                # t>=1 için Bernoulli mask
                ss_mask = (torch.rand(B_l, T_l, device=device) < ss_prob)
                # t=0'ı teacher forcing bırak (başlangıç sabit kalsın)
                ss_mask[:, 0] = False
                
                # Bir önceki adımın GT vs pred seçimi
                gt_prev = x_tokens.clone()
                gt_prev[:, 0] = x_tokens[:, 0]
                pred_prev = pred_x_tf.clone()
                
                # t>=1'de: x_in_ss[:, t] = ss_mask ? pred_prev[:, t-1] : gt_prev[:, t-1]
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
                # Scheduled sampling yoksa TF logits'lerini kullan
                x_in = x_in_tf
                logits_x, logits_y = logits_x_tf, logits_y_tf
            
            # Loss: X and Y (FIX 5: Aşamalı Y-loss) + V6 presence loss
            # V14: PDF'ye göre loss SADECE Lgt kısmında hesaplanmalı (Eq. 10, 11)
            # PDF: "Loss. We only adopt standard loss in the decoder-only language models."
            # Lq sadece input (query), Lgt output (answer) - Loss sadece Lgt'de!
            B, T, V = logits_x.shape
            # Flatten for loss computation
            logits_x_flat = logits_x.view(B * T, V)
            x_tokens_flat = x_tokens.view(B * T)
            loss_mask_flat = loss_mask.view(B * T)
            
            # V14: PDF'ye göre loss SADECE Lgt kısmında (loss_mask: False for Lq, True for Lgt)
            # Lq loss'u kaldırdık - PDF'de yok!
            lgt_mask = loss_mask_flat & (x_tokens_flat != pad_token_x)  # Lgt positions with valid tokens
            if lgt_mask.any():
                loss_x = loss_x_fn(
                    logits_x_flat[lgt_mask],
                    x_tokens_flat[lgt_mask]
                )
            else:
                loss_x = torch.tensor(0.0, device=device)
            # logits_y shape: (B, T, max_y_tokens) where max_y_tokens = T+1
            
            # Y-loss (opsiyonel)
            # NOT: Önceki implementasyonda convex-combination ((1-w)*X + w*Y) kullanımı
            # w=1 olduğunda X gradient'ini tamamen sıfırlayabiliyordu.
            # PDF Eq.11 ifadesi toplamsal: loss = loss_x + loss_y (isteğe bağlı ağırlıkla).
            if use_y_loss:
                logits_y_flat = logits_y.view(B * T, -1)
                y_tokens_flat = y_tokens.view(B * T)
                if lgt_mask.any():
                    loss_y = loss_y_fn(
                        logits_y_flat[lgt_mask],
                        y_tokens_flat[lgt_mask]
                    )
                else:
                    loss_y = torch.tensor(0.0, device=device)
                loss = loss_x + (y_loss_weight * loss_y)
                y_weight = y_loss_weight
            else:
                loss = loss_x
                loss_y = torch.tensor(0.0, device=device)
                y_weight = 0.0

            # Presence loss (debug için default 0)
            presence_logits_flat = presence_logits.view(-1)
            loss_presence = loss_presence_fn(presence_logits_flat, presence_targets)
            presence_weight = float(args.presence_weight)
            if presence_weight > 0.0:
                loss = loss + presence_weight * loss_presence

            # V6: Pixel-space alignment loss (resized 800x320)
            pixel_loss = torch.tensor(0.0, device=device)
            if pixel_loss_weight > 0.0:
                # Bin merkezlerini bir kez ve doğru cihazda oluştur
                if bin_centers is None or bin_centers.device != logits_x.device:
                    img_w = tokenizer_cfg.img_w
                    V_bins = logits_x.shape[-1]
                    bin_centers = torch.linspace(
                        0.0,
                        float(img_w - 1),
                        steps=V_bins,
                        device=logits_x.device,
                        dtype=logits_x.dtype,
                    ).view(1, 1, V_bins)  # (1,1,V)

                # Beklenen X (piksel uzayında): E[x] over bins
                probs_x = torch.softmax(logits_x, dim=-1)  # (B,T,V)
                x_exp = (probs_x * bin_centers).sum(dim=-1)  # (B,T)

                # GT X piksel uzayında (tokenlardan)
                V_bins = logits_x.shape[-1]
                img_w = tokenizer_cfg.img_w
                x_tokens_clamped = x_tokens.clamp(min=0, max=tokenizer_cfg.nbins_x - 1).float()
                x_gt = x_tokens_clamped / max(1, tokenizer_cfg.nbins_x - 1) * float(img_w - 1)

                # Sadece non-padding timestepler
                valid_mask_pix = (x_tokens != tokenizer_cfg.pad_token_x) & (y_tokens != tokenizer.T)
                if valid_mask_pix.any():
                    pixel_diff = (x_exp - x_gt).abs()
                    pixel_loss = pixel_diff[valid_mask_pix].mean()
                    loss = loss + pixel_loss_weight * pixel_loss
            
            # V5: Autoregressive rollout loss (full sequence, decaying weight over time)
            # V12: Only compute AR loss on Lgt part (not Lq)
            ar_loss = torch.tensor(0.0, device=device)
            if ar_rollout_max_weight > 0.0 and T_l > 1:
                # 1-step AR rollout için giriş dizisi: t>=1'de x_{t-1} = model tahmini
                x_in_roll = x_in_tf.clone()
                x_in_roll[:, 1:] = pred_x_tf[:, :-1]
                logits_x_roll, _ = lanelm(
                    vis_tok_batch, x_in_roll, y_in, lane_indices=lane_ids
                )
                # t=0 hariç, tüm timestepler için loss (but only on Lgt part)
                roll_logits = logits_x_roll[:, 1:, :].reshape(-1, V)          # (B*(T-1), V)
                roll_targets = x_tokens[:, 1:].reshape(-1)                    # (B*(T-1),)
                roll_loss_mask = loss_mask[:, 1:].reshape(-1)  # (B*(T-1),) - only Lgt positions
                
                # Eleman bazlı loss (ignore_index ile)
                roll_loss_all = torch.nn.functional.cross_entropy(
                    roll_logits,
                    roll_targets,
                    ignore_index=tokenizer_cfg.pad_token_x,
                    reduction="none",
                )
                # Zaman bazlı ağırlık maskesi: küçük t → yüksek weight, büyük t → düşük weight
                t_ids = torch.arange(1, T_l, device=device).unsqueeze(0).expand(B_l, -1)  # (B, T-1)
                # normalize edilmiş zaman [0,1]
                t_norm = (t_ids - 1).float() / max(T_l - 2, 1)
                w_t = ar_rollout_min_weight + (1.0 - t_norm) * (ar_rollout_max_weight - ar_rollout_min_weight)
                w_flat = w_t.reshape(-1)  # (B*(T-1),)
                # Only compute on Lgt positions (roll_loss_mask) and valid tokens
                valid_mask = (roll_targets != tokenizer_cfg.pad_token_x) & roll_loss_mask
                if valid_mask.any():
                    ar_loss = (roll_loss_all[valid_mask] * w_flat[valid_mask]).sum() / valid_mask.sum()
                    loss = loss + ar_loss

            # V5: Padding bölgelerinde X=0 öğrenilsin (no-lane timesteps)
            # Debug overfit koşularında bunu kapatmak isteyebiliriz (default 0).
            pad_mask = (y_tokens == tokenizer.T)  # padding Y
            pad_mask_flat = pad_mask.view(B * T)
            pad_loss = torch.tensor(0.0, device=device)
            if pad_mask_flat.any():
                logits_x_flat = logits_x.view(B * T, V)
                logits_x_pad = logits_x_flat[pad_mask_flat]
                targets_pad = torch.zeros(logits_x_pad.size(0), dtype=torch.long, device=device)
                pad_loss = loss_x_pad_fn(logits_x_pad, targets_pad)
                pad_loss_weight = float(args.pad_loss_weight)
                if pad_loss_weight > 0.0:
                    loss = loss + pad_loss_weight * pad_loss

            loss.backward()
            # FIX 2: Stricter gradient clipping (1.0 → 0.5)
            torch.nn.utils.clip_grad_norm_(lanelm.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_y += loss_y.item()
            total_loss_ar += ar_loss.item()
            total_loss_presence += loss_presence.item()
            total_loss_pixel += pixel_loss.item()
            steps += 1
        
        if steps > 0:
            avg_loss = total_loss / steps
            avg_loss_x = total_loss_x / steps
            avg_loss_y = total_loss_y / steps
            avg_loss_ar = total_loss_ar / steps
            avg_loss_presence = total_loss_presence / steps
            avg_loss_pixel = total_loss_pixel / steps
            avg_acc = (total_acc / total_acc_count) if total_acc_count > 0 else 0.0
            
            # LR'yi scheduler.step() öncesi logla (özellikle epochs=1 gibi koşularda
            # CosineAnnealingLR step sonrası hemen eta_min'e düşüp yanıltmasın).
            current_lr = optimizer.param_groups[0]["lr"]
            
            if epoch % 10 == 0 or epoch <= 5:
                if use_y_loss:
                    print(
                        f"Ep {epoch}: Loss = {avg_loss:.4f} "
                        f"(X={avg_loss_x:.4f} Y={avg_loss_y:.4f} "
                        f"AR={avg_loss_ar:.4f} PRES={avg_loss_presence:.4f} "
                        f"PIX={avg_loss_pixel:.4f} "
                        f"ACC={avg_acc:.3f} "
                        f"Y-weight={y_loss_weight:.3f}) LR={current_lr:.2e}"
                    )
                else:
                    print(
                        f"Ep {epoch}: Loss = {avg_loss:.4f} "
                        f"(X={avg_loss_x:.4f} Y={avg_loss_y:.4f} "
                        f"AR={avg_loss_ar:.4f} PRES={avg_loss_presence:.4f} "
                        f"PIX={avg_loss_pixel:.4f} "
                        f"ACC={avg_acc:.3f} "
                        f"Y-loss disabled) LR={current_lr:.2e}"
                    )
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                hparams_dict = dict(hparams.__dict__)
                hparams_dict["y_direction"] = str(args.y_direction)
                torch.save({
                    "model_state_dict": lanelm.state_dict(),
                    "config": hparams_dict,
                    "epoch": epoch,
                    "loss": best_loss
                }, os.path.join(args.work_dir, "lanelm_v4_best.pth"))
            
            # FIX 2: Update learning rate scheduler (epoch sonunda)
            scheduler.step()
            
            if epoch % 50 == 0:
                visualize(
                    lanelm, clrernet, fixed_batch, tokenizer, device, epoch,
                    os.path.join(args.work_dir, "vis"), hparams.max_lanes, use_p5_only
                )
    
    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to {args.work_dir}/lanelm_v4_best.pth")
    
    # Final analysis
    # Not: Çok kısa koşularda (örn. epochs=1) PASS/FAIL değerlendirmesi anlamlı değil.
    if args.epochs < 10:
        print("\nℹ️  SMOKE RUN: epochs<10 olduğu için bu koşuda PASS/FAIL değerlendirmesi yapılmadı.")
    else:
        if best_loss < 0.5:
            print("\n✅ SUCCESS: Model can learn from visual information!")
            print("   Next step: Scale up to more images")
        else:
            print("\n❌ FAILURE: Model still not learning properly")
            print("   Need to investigate further")


if __name__ == "__main__":
    main()

