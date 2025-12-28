#!/usr/bin/env python3
"""
Prefix -> next token sanity check (minimal drift locator).

Goal:
  For a single sample, per lane-slot:
    1) Teacher-prefix: feed GT prefix (shift-right) and ask "next token at t" correct?
    2) Self-feeding: greedy decode step-by-step and ask "first mismatch timestep" where drift starts.

This directly tests the user's hypothesis:
  "TF perfect, AR bad" => exposure-bias / drift starts at some early t.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

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


@torch.no_grad()
def next_token_with_prefix(
    model: LaneLMModel,
    visual_tokens_1: torch.Tensor,  # (1,N,D)
    gt_x: np.ndarray,               # (T,)
    lane_slot: int,
    t: int,
    pad_x: int,
) -> int:
    """Return argmax token at position t given a GT prefix of length t."""
    device = visual_tokens_1.device
    T = int(gt_x.shape[0])

    x_in = torch.full((1, T), pad_x, dtype=torch.long, device=device)
    # shift-right: x_in[t] sees gt_x[t-1]
    if t > 0:
        x_in[0, 1 : t + 1] = torch.from_numpy(gt_x[:t]).long().to(device)
    x_in[0, 0] = pad_x

    y_in = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)
    lane_ids = torch.tensor([lane_slot], dtype=torch.long, device=device)

    logits_x, _ = model(visual_tokens_1, x_in, y_in, lane_indices=lane_ids)
    pred = int(torch.argmax(logits_x[0, t], dim=-1).item())
    return pred


@torch.no_grad()
def greedy_decode_until(
    model: LaneLMModel,
    visual_tokens_1: torch.Tensor,  # (1,N,D)
    lane_slot: int,
    T: int,
    pad_x: int,
) -> np.ndarray:
    """Greedy decode a full lane token sequence (single lane slot)."""
    device = visual_tokens_1.device
    y_in = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)
    lane_ids = torch.tensor([lane_slot], dtype=torch.long, device=device)

    x_out = torch.zeros((1, T), dtype=torch.long, device=device)
    for t in range(T):
        x_in = torch.full_like(x_out, pad_x)
        if t > 0:
            x_in[:, 1 : t + 1] = x_out[:, :t]
        x_in[:, 0] = pad_x
        logits_x, _ = model(visual_tokens_1, x_in, y_in, lane_indices=lane_ids)
        x_out[:, t] = torch.argmax(logits_x[:, t, :], dim=-1)
    return x_out[0].detach().cpu().numpy()


def first_mismatch(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> int:
    idxs = np.where(valid_mask)[0]
    for i in idxs:
        if int(pred[i]) != int(gt[i]):
            return int(i)
    return -1


def main():
    p = argparse.ArgumentParser(description="Prefix->next-token sanity check (drift locator)")
    p.add_argument("--lanelm-ckpt", required=True)
    p.add_argument("--list-path", required=True)
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--data-root", default="dataset")
    p.add_argument("--config", default="configs/clrernet/culane/clrernet_culane_dla34_ema.py")
    p.add_argument("--clrernet-ckpt", default="clrernet_culane_dla34_ema.pth")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-lanes", type=int, default=4)
    p.add_argument("--save-json", default="", help="Optional path to save json summary")
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

    sample = dataset[int(args.sample_idx)]
    sub = sample.get("sub_img_name", f"idx_{args.sample_idx}")
    gt_points = sample.get("gt_points", [])
    lanes_sorted = sort_lanes_left_to_right(gt_points, max_lanes=int(args.max_lanes))
    if len(lanes_sorted) == 0:
        raise SystemExit("This sample has no GT lanes after filtering. Pick another sample-idx.")

    # Visual tokens
    img = sample["img"]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
    if n_levels == 1:
        feats = extract_p5_feat(clrernet, img_tensor)
    else:
        feats = extract_full_fpn_feats(clrernet, img_tensor)
    visual_tokens = lanelm.encode_visual_tokens(feats)  # (1,N,D)

    # GT tokens (per slot)
    gt_x_slots = []
    gt_y_slots = []
    for lane_idx in range(int(args.max_lanes)):
        if lane_idx < len(lanes_sorted):
            x_t, y_t = tokenizer.encode_single_lane(lanes_sorted[lane_idx])
        else:
            x_t = np.full(T, pad_x, dtype=np.int64)
            y_t = np.full(T, T, dtype=np.int64)
        gt_x_slots.append(x_t)
        gt_y_slots.append(y_t)
    gt_x_slots = np.stack(gt_x_slots, axis=0)  # (L,T)
    gt_y_slots = np.stack(gt_y_slots, axis=0)  # (L,T)
    valid = (gt_x_slots != pad_x) & (gt_y_slots < T)

    print(f"\nSample: {sub}")
    print(f"Tokenizer y_direction={tokenizer.cfg.y_direction} | T={T} | nbins_x={tokenizer.cfg.nbins_x}")

    summary = {
        "sample": sub,
        "y_direction": str(tokenizer.cfg.y_direction),
        "T": int(T),
        "max_lanes": int(args.max_lanes),
        "lanes": [],
    }

    for lane_slot in range(int(args.max_lanes)):
        gt_x = gt_x_slots[lane_slot]
        vmask = valid[lane_slot]
        if not vmask.any():
            print(f"\nLane slot {lane_slot}: no valid GT tokens (all pad) -> skipping")
            summary["lanes"].append({"lane_slot": lane_slot, "skipped": True})
            continue

        # Teacher-prefix curve
        pred_tf_prefix = np.full(T, -1, dtype=np.int64)
        for t in range(T):
            pred_tf_prefix[t] = next_token_with_prefix(
                lanelm, visual_tokens, gt_x, lane_slot=lane_slot, t=t, pad_x=pad_x
            )

        # Greedy AR
        pred_ar = greedy_decode_until(
            lanelm, visual_tokens, lane_slot=lane_slot, T=T, pad_x=pad_x
        )

        tf_first = first_mismatch(pred_tf_prefix, gt_x, vmask)
        ar_first = first_mismatch(pred_ar, gt_x, vmask)

        tf_acc = float(((pred_tf_prefix == gt_x) & vmask).sum() / max(1, vmask.sum()))
        ar_acc = float(((pred_ar == gt_x) & vmask).sum() / max(1, vmask.sum()))

        print(f"\nLane slot {lane_slot}:")
        print(f"  valid_tokens={int(vmask.sum())}")
        print(f"  TF-prefix next-token acc={tf_acc:.3f} | first_mismatch_t={tf_first}")
        print(f"  AR greedy acc={ar_acc:.3f}          | first_mismatch_t={ar_first}")
        if tf_first != -1:
            print(f"  TF mismatch example: t={tf_first} gt={int(gt_x[tf_first])} pred={int(pred_tf_prefix[tf_first])}")
        if ar_first != -1:
            print(f"  AR mismatch example: t={ar_first} gt={int(gt_x[ar_first])} pred={int(pred_ar[ar_first])}")

        summary["lanes"].append(
            {
                "lane_slot": int(lane_slot),
                "valid_tokens": int(vmask.sum()),
                "tf_prefix_acc": tf_acc,
                "tf_prefix_first_mismatch_t": int(tf_first),
                "ar_acc": ar_acc,
                "ar_first_mismatch_t": int(ar_first),
            }
        )

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        import json

        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()


