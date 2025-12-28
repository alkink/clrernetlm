#!/usr/bin/env python3
"""
Visualize Teacher-Forcing (TF) vs Autoregressive (AR) decoding for a single sample.

Outputs:
  - tf.jpg: TF-argmax decode (blue) + GT (green)
  - ar.jpg: AR decode (red) + GT (green)

Also prints token-level diagnostics:
  TF_ACC / TF_MAE_tok and AR_ACC / AR_MAE_tok computed against GT tokens.
"""

import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from configs.clrernet.culane.dataset_culane_clrernet import compose_cfg, crop_bbox, img_scale
from libs.datasets import CulaneDataset
from libs.models.detectors.lanelm_detector import autoregressive_decode
from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig
from tools.train_lanelm_culane_v3 import build_frozen_clrernet_backbone
from tools.train_lanelm_v4_fixed import extract_full_fpn_feats, extract_p5_feat


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


def sort_lanes_left_to_right(gt_points_resized: List, max_lanes: int) -> List[np.ndarray]:
    lanes = []
    for lane in gt_points_resized:
        if len(lane) < 2:
            continue
        pts = np.array(lane, dtype=np.float32).reshape(-1, 2)
        lanes.append(pts)
    if not lanes:
        return []
    means = [float(np.mean(l[:, 0])) for l in lanes]
    order = np.argsort(means)
    lanes = [lanes[i] for i in order]
    return lanes[:max_lanes]


def draw_polyline(img_bgr: np.ndarray, coords: np.ndarray, color, thickness: int = 2):
    if coords is None or coords.shape[0] < 2:
        return
    for k in range(len(coords) - 1):
        p1 = (int(coords[k][0]), int(coords[k][1]))
        p2 = (int(coords[k + 1][0]), int(coords[k + 1][1]))
        if 0 <= p1[0] < img_bgr.shape[1] and 0 <= p2[0] < img_bgr.shape[1] and 0 <= p1[1] < img_bgr.shape[0] and 0 <= p2[1] < img_bgr.shape[0]:
            cv2.line(img_bgr, p1, p2, color, thickness)


def main():
    p = argparse.ArgumentParser(description="Visualize TF vs AR for a single sample")
    p.add_argument("--lanelm-ckpt", required=True)
    p.add_argument("--list-path", required=True)
    p.add_argument("--sample-idx", type=int, default=0, help="Start index for sweep (or single idx if --num-samples=1)")
    p.add_argument("--num-samples", type=int, default=1, help="How many GT-containing samples to process")
    p.add_argument("--data-root", default="dataset")
    p.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    p.add_argument("--clrernet-ckpt", default="clrernet_culane_dla34_ema.pth")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-dir", default="work_dirs/_debug_tf_vs_ar")
    p.add_argument("--smooth", action="store_true", help="Enable tokenizer smoothing for drawing")
    p.add_argument("--max-lanes", type=int, default=4)
    p.add_argument("--save-max", type=int, default=10, help="Max number of samples to save images for")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.lanelm_ckpt, map_location="cpu")
    cfg = ckpt["config"]
    sd = ckpt["model_state_dict"]

    # Determine visual levels from checkpoint
    n_levels = 1
    if "visual_encoder.level_embed.weight" in sd:
        n_levels = int(sd["visual_encoder.level_embed.weight"].shape[0])
    visual_in_channels = tuple([64] * n_levels)
    print(f"LaneLM visual levels: {n_levels} (visual_in_channels={visual_in_channels})")

    # Models
    clrernet = build_frozen_clrernet_backbone(args.config, args.clrernet_ckpt, device)
    lanelm = LaneLMModel(
        nbins_x=int(cfg["nbins_x"]),
        max_y_tokens=int(cfg["num_points"]) + 1,
        embed_dim=int(cfg["embed_dim"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        ffn_dim=int(cfg["ffn_dim"]),
        max_seq_len=80,
        dropout=float(cfg.get("dropout", 0.0)),
        visual_in_channels=visual_in_channels,
    )
    lanelm.load_state_dict(sd, strict=True)
    lanelm.to(device).eval()

    tokenizer = LaneTokenizer(
        LaneTokenizerConfig(
            img_w=int(cfg["img_w"]),
            img_h=int(cfg["img_h"]),
            num_steps=int(cfg["num_points"]),
            nbins_x=int(cfg["nbins_x"]),
            x_mode="absolute",
            y_direction=str(cfg.get("y_direction", "top_to_bottom")),
        )
    )
    T = tokenizer.T
    pad_x = tokenizer.cfg.pad_token_x

    dataset = CulaneDataset(
        data_root=args.data_root,
        data_list=args.list_path,
        pipeline=[dict(type="albumentation", pipelines=clean_pipeline)],
        diff_file=None,
        test_mode=False,
    )
    if args.sample_idx >= len(dataset):
        raise SystemExit(f"sample-idx {args.sample_idx} >= dataset size {len(dataset)}")

    os.makedirs(args.save_dir, exist_ok=True)

    processed = 0
    saved = 0
    n_tf_eq_ar = 0
    n_tf_perfect_ar_not = 0
    n_tf_not_perfect = 0

    idx = int(args.sample_idx)
    while idx < len(dataset) and processed < int(args.num_samples):
        sample = dataset[idx]
        sub = sample.get("sub_img_name", f"idx_{idx}")
        gt_points = sample.get("gt_points", [])

        lanes_sorted = sort_lanes_left_to_right(gt_points, max_lanes=args.max_lanes)
        if len(lanes_sorted) == 0:
            idx += 1
            continue

        # Image (resized) -> BGR for drawing
        img = sample["img"]
        img_vis = (img * 255).astype(np.uint8).copy()
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        img_ar = img_vis.copy()
        img_tf = img_vis.copy()

        # Visual tokens
        img_tensor = (
            torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
        )
        with torch.no_grad():
            if n_levels == 1:
                feats = extract_p5_feat(clrernet, img_tensor)
            else:
                feats = extract_full_fpn_feats(clrernet, img_tensor)
            visual_tokens = lanelm.encode_visual_tokens(feats)  # (1,N,D)

        # Build GT tokens (sorted)
        gt_x_list = []
        gt_y_list = []
        for lane_idx in range(args.max_lanes):
            if lane_idx < len(lanes_sorted):
                x_t, y_t = tokenizer.encode_single_lane(lanes_sorted[lane_idx])
            else:
                x_t = np.full(T, pad_x, dtype=np.int64)
                y_t = np.full(T, T, dtype=np.int64)
            gt_x_list.append(x_t)
            gt_y_list.append(y_t)
        gt_x_np = np.stack(gt_x_list, axis=0)
        gt_y_np = np.stack(gt_y_list, axis=0)
        valid = (gt_x_np != pad_x) & (gt_y_np < T)

        # -------- TF decode (single forward) --------
        with torch.no_grad():
            L = args.max_lanes
            vis_tok_batch = visual_tokens.expand(L, -1, -1).contiguous()
            lane_ids = torch.arange(L, device=device, dtype=torch.long)

            gt_x = torch.from_numpy(gt_x_np).long().to(device)
            x_in_tf = gt_x.clone()
            x_in_tf[:, 1:] = gt_x[:, :-1]
            x_in_tf[:, 0] = pad_x
            y_in = (
                torch.arange(T, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(L, -1)
            )

            logits_x, _ = lanelm(vis_tok_batch, x_in_tf, y_in, lane_indices=lane_ids)
            pred_tf = torch.argmax(logits_x, dim=-1).detach().cpu().numpy()  # (L,T)

        # -------- AR decode --------
        with torch.no_grad():
            x_tokens_all, _y_tokens_all = autoregressive_decode(
                lanelm_model=lanelm,
                visual_tokens=visual_tokens,
                tokenizer_cfg=tokenizer.cfg,
                max_lanes=args.max_lanes,
                temperature=0.0,
                use_presence_filter=False,  # isolate decoding
            )
        pred_ar = x_tokens_all[0].numpy()

        def _acc_mae(pred_np):
            if valid.any():
                acc = float(((pred_np == gt_x_np) & valid).sum() / valid.sum())
                mae = float(np.abs(pred_np[valid] - gt_x_np[valid]).mean())
                return acc, mae
            return 0.0, 0.0

        tf_acc, tf_mae = _acc_mae(pred_tf)
        ar_acc, ar_mae = _acc_mae(pred_ar)

        # Compare TF vs AR directly too
        if valid.any() and np.array_equal(pred_tf[valid], pred_ar[valid]):
            n_tf_eq_ar += 1
        if tf_acc >= 0.999 and ar_acc < 0.999:
            n_tf_perfect_ar_not += 1
        if tf_acc < 0.999:
            n_tf_not_perfect += 1

        print(f"[{idx}] {sub} | TF_ACC={tf_acc:.3f} TF_MAE={tf_mae:.2f} | AR_ACC={ar_acc:.3f} AR_MAE={ar_mae:.2f}")

        # Optionally save images for the first few processed samples
        if saved < int(args.save_max):
            # Draw GT (green)
            for lane in gt_points[: args.max_lanes]:
                pts = np.array(lane, dtype=np.float32).reshape(-1, 2)
                draw_polyline(img_ar, pts, (0, 255, 0), thickness=3)
                draw_polyline(img_tf, pts, (0, 255, 0), thickness=3)

            y_fixed = np.arange(T, dtype=np.int64)
            for l in range(args.max_lanes):
                coords_tf = tokenizer.decode_single_lane(pred_tf[l], y_fixed, smooth=bool(args.smooth))
                coords_ar = tokenizer.decode_single_lane(pred_ar[l], y_fixed, smooth=bool(args.smooth))
                draw_polyline(img_tf, coords_tf, (255, 0, 0), thickness=2)  # blue
                draw_polyline(img_ar, coords_ar, (0, 0, 255), thickness=2)  # red

            stem = sub.replace("/", "_").replace(".jpg", "")
            out_tf = os.path.join(args.save_dir, f"{stem}_tf.png")
            out_ar = os.path.join(args.save_dir, f"{stem}_ar.png")
            ok_tf = cv2.imwrite(out_tf, img_tf)
            ok_ar = cv2.imwrite(out_ar, img_ar)
            if not ok_tf or not ok_ar:
                raise RuntimeError(f"Failed to write images: ok_tf={ok_tf}, ok_ar={ok_ar}")
            saved += 1

        processed += 1
        idx += 1

    print("\n=== SUMMARY ===")
    print(f"processed={processed} saved={saved} smooth={bool(args.smooth)}")
    print(f"TF==AR (on valid tokens): {n_tf_eq_ar}/{processed}")
    print(f"TF perfect but AR not:   {n_tf_perfect_ar_not}/{processed}")
    print(f"TF not perfect:          {n_tf_not_perfect}/{processed}")


if __name__ == "__main__":
    main()


