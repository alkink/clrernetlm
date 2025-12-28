#!/usr/bin/env python3
"""
Teacher-Forcing Token Accuracy Debug for LaneLM (Model Prediction Debug)

Compute:
  - X-token accuracy on valid (non-pad) positions
  - mean absolute token error |pred_x - gt_x| on valid positions
  - optional per-timestep mean absolute token error

This isolates whether the model learns the discrete token targets.
"""

import argparse
import os
from typing import List

import numpy as np
import torch

from configs.clrernet.culane.dataset_culane_clrernet import compose_cfg, crop_bbox, img_scale
from libs.datasets import CulaneDataset
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


def main():
    p = argparse.ArgumentParser(description="Teacher forcing token ACC/e|err| for LaneLM")
    p.add_argument("--lanelm-ckpt", required=True)
    p.add_argument("--list-path", required=True)
    p.add_argument("--data-root", default="dataset")
    p.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    p.add_argument("--clrernet-ckpt", default="clrernet_culane_dla34_ema.pth")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-samples", type=int, default=50, help="GT içeren örnek sayısı")
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--per-step", action="store_true", help="Per-timestep mean abs token error raporla")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.lanelm_ckpt, map_location="cpu")
    cfg = ckpt["config"]
    sd = ckpt["model_state_dict"]

    # Determine visual levels (P5-only vs Full FPN)
    n_levels = 1
    if "visual_encoder.level_embed.weight" in sd:
        n_levels = int(sd["visual_encoder.level_embed.weight"].shape[0])
    visual_in_channels = tuple([64] * n_levels)

    # Build frozen CLRerNet for visual feats
    clrernet = build_frozen_clrernet_backbone(args.config, args.clrernet_ckpt, device)

    # Build LaneLM
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
        )
    )

    dataset = CulaneDataset(
        data_root=args.data_root,
        data_list=args.list_path,
        pipeline=[dict(type="albumentation", pipelines=clean_pipeline)],
        diff_file=None,
        test_mode=False,
    )

    T = tokenizer.T
    pad_x = tokenizer.cfg.pad_token_x

    total_correct = 0.0
    total_count = 0.0
    total_abs_err = 0.0

    per_t_err_sum = np.zeros(T, dtype=np.float64)
    per_t_err_cnt = np.zeros(T, dtype=np.float64)

    n_used = 0
    idx = int(args.start_idx)

    while idx < len(dataset) and n_used < int(args.num_samples):
        sample = dataset[idx]
        gt_points_resized = sample.get("gt_points", [])
        lanes = sort_lanes_left_to_right(gt_points_resized, max_lanes=int(cfg["max_lanes"]))
        if len(lanes) == 0:
            idx += 1
            continue

        # Build GT tokens for up to max_lanes, pad remaining lanes
        max_lanes = int(cfg["max_lanes"])
        x_tokens_all = []
        y_tokens_all = []
        for lane_i in range(max_lanes):
            if lane_i < len(lanes):
                x_t, y_t = tokenizer.encode_single_lane(lanes[lane_i])
            else:
                x_t = np.full(T, pad_x, dtype=np.int64)
                y_t = np.full(T, T, dtype=np.int64)
            x_tokens_all.append(x_t)
            y_tokens_all.append(y_t)
        x_tokens = torch.from_numpy(np.stack(x_tokens_all, axis=0)).long().to(device)  # (L,T)
        y_tokens = torch.from_numpy(np.stack(y_tokens_all, axis=0)).long().to(device)  # (L,T)

        # Visual tokens
        img = sample["img"]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            if n_levels == 1:
                feats = extract_p5_feat(clrernet, img_tensor)
            else:
                feats = extract_full_fpn_feats(clrernet, img_tensor)
            visual_tokens = lanelm.encode_visual_tokens(feats)  # (1,N,D)

        # Replicate visual tokens per-lane
        vis_tok_batch = visual_tokens.expand(max_lanes, -1, -1).contiguous()  # (L,N,D)
        lane_ids = torch.arange(max_lanes, device=device, dtype=torch.long)  # (L,)

        # Teacher forcing inputs
        x_in_tf = x_tokens.clone()
        x_in_tf[:, 1:] = x_tokens[:, :-1]
        x_in_tf[:, 0] = pad_x  # 0-kp BOS/pad

        y_in = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(max_lanes, -1)

        with torch.no_grad():
            logits_x, _ = lanelm(vis_tok_batch, x_in_tf, y_in, lane_indices=lane_ids)
            pred_x = torch.argmax(logits_x, dim=-1)  # (L,T)

        # Valid mask: ignore padding x=0 and y=T
        valid = (x_tokens != pad_x) & (y_tokens < T)
        if valid.any():
            correct = (pred_x == x_tokens) & valid
            total_correct += float(correct.sum().item())
            total_count += float(valid.sum().item())
            abs_err = (pred_x - x_tokens).abs()
            total_abs_err += float(abs_err[valid].sum().item())

            if args.per_step:
                abs_err_np = abs_err.detach().cpu().numpy()
                valid_np = valid.detach().cpu().numpy()
                for t in range(T):
                    m = valid_np[:, t]
                    if m.any():
                        per_t_err_sum[t] += float(abs_err_np[:, t][m].sum())
                        per_t_err_cnt[t] += float(m.sum())

        sub = sample.get("sub_img_name", f"idx_{idx}")
        print(f"[{idx}] {sub} | gt_lanes={len(lanes)}")
        n_used += 1
        idx += 1

    print("\n=== TOKEN ACC SUMMARY (teacher forcing) ===")
    print(f"samples_used={n_used}  list={args.list_path}")
    if total_count == 0:
        print("No valid GT tokens were found.")
        return
    acc = total_correct / total_count
    mean_abs_tok_err = total_abs_err / total_count
    print(f"X_token_ACC={acc:.4f}  mean_abs_token_error={mean_abs_tok_err:.3f}")

    if args.per_step:
        print("\nPer-timestep mean abs token error (valid only):")
        for t in range(T):
            if per_t_err_cnt[t] > 0:
                print(f"t={t:02d}: {per_t_err_sum[t]/per_t_err_cnt[t]:.3f}")


if __name__ == "__main__":
    main()


