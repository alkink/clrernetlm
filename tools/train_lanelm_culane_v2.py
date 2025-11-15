"""
LaneLM v2 training script on CULane, closely following the LaneLM paper.

Key points:
  - Uses CLRerNet backbone + FPN as frozen visual encoder (teacher feature extractor).
  - Uses GT lane polylines as supervision and (optionally) as pseudo labels.
  - Encodes lanes into LaneLM-style token sequences (x_t, y_t) with LaneTokenizer.
  - Builds multi-lane sequences per image and trains a decoder-only LM with
    cross-entropy over x and y tokens (Eq. 11 in the paper).

This script is designed to be modular and faithful to the algorithmic
description in the LaneLM preprint, but does not attempt to reproduce all
engineering tricks (e.g., advanced KV caching) inside this file.
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mmdet.apis import init_detector

from libs.datasets import CulaneDataset
from libs.datasets.pipelines import Compose
from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig


@dataclass
class LaneLMHyperParams:
    """Hyper-parameters mirroring the LaneLM paper."""

    num_points: int = 40  # T
    nbins_x: int = 800
    max_lanes: int = 4
    img_w: int = 800
    img_h: int = 320
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 512


def collate_lanelm_batch_v2(batch):
    """Custom collate function to keep per-image GT lanes.

    Each item from CulaneDataset + Alaug + raw dict pipeline is expected to
    contain:
      - 'img': numpy array (H, W, 3)
      - 'gt_points': list of lanes, each as flat [x0, y0, x1, y1, ...]
      - 'sub_img_name': relative path used by CULaneMetric (for later tests)
      - 'ori_shape', 'img_shape': tuples

    We keep gt_points as a Python list per image to allow flexible tokenization.
    """
    imgs = [item["img"] for item in batch]
    gt_points = [item.get("gt_points", []) for item in batch]
    sub_img_names = [item.get("sub_img_name") for item in batch]
    ori_shapes = [item.get("ori_shape") for item in batch]
    img_shapes = [item.get("img_shape") for item in batch]

    imgs_tensor = torch.stack(
        [
            torch.from_numpy(img.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .contiguous()
            for img in imgs
        ]
    )

    return {
        "inputs": imgs_tensor,
        "gt_points": gt_points,
        "sub_img_name": sub_img_names,
        "ori_shape": ori_shapes,
        "img_shape": img_shapes,
    }


def build_culane_lanelm_dataloader_v2(
    data_root: str,
    list_path: str,
    diff_file: str,
    batch_size: int,
    num_workers: int,
    img_w: int,
    img_h: int,
) -> DataLoader:
    """CULane DataLoader for LaneLM v2 with GT lane polylines.

    We reuse the CULane augmentation pipeline (train_al_pipeline) from the
    CLRerNet dataset config, but we stop before PackCLRNetInputs and keep
    gt_points in the raw format for tokenization in this script.
    """
    from configs.clrernet.culane.dataset_culane_clrernet import (  # type: ignore
        compose_cfg,
        crop_bbox,
        img_scale,
        train_al_pipeline,
    )

    # Albumentations pipeline (crop + resize + aug), no PackCLRNetInputs
    train_pipeline = [
        dict(type="albumentation", pipelines=train_al_pipeline),
    ]

    # diff_file is optional; fall back to None if not found
    if diff_file and not os.path.exists(diff_file):
        diff_file = None

    dataset = CulaneDataset(
        data_root=data_root,
        data_list=list_path,
        pipeline=train_pipeline,
        diff_file=diff_file,
        diff_thr=15,
        test_mode=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_lanelm_batch_v2,
    )
    return dataloader


def build_frozen_clrernet_backbone(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
):
    """Initialize CLRerNet and freeze backbone + neck for visual encoding."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()

    # Freeze backbone and neck (visual encoder).
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.neck.parameters():
        param.requires_grad = False

    return model


def extract_pyramid_feats(model, imgs: torch.Tensor):
    """Extract FPN features {F0, F1, F2} from CLRerNet backbone + neck.

    Args:
        model: pretrained CLRerNet detector.
        imgs: (B, 3, H, W) tensor.
    Returns:
        feats: list of tensors [F0, F1, F2], each (B, C, H_i, W_i).
    """
    with torch.no_grad():
        feats = model.extract_feat(imgs)
    # CLRerNetFPN returns 3 levels, we keep all.
    return list(feats)


def build_lanelm_model_v2(
    hparams: LaneLMHyperParams,
    visual_in_channels: Tuple[int, int, int],
) -> LaneLMModel:
    """Construct a LaneLMModel consistent with the paper hyperparams."""
    max_y_tokens = hparams.num_points + 1  # including padding/EOS
    max_seq_len = hparams.num_points * hparams.max_lanes * 2
    model = LaneLMModel(
        nbins_x=hparams.nbins_x,
        max_y_tokens=max_y_tokens,
        embed_dim=hparams.embed_dim,
        num_layers=hparams.num_layers,
        num_heads=hparams.num_heads,
        ffn_dim=hparams.ffn_dim,
        max_seq_len=max_seq_len,
        visual_in_dim=None,
        visual_in_channels=visual_in_channels,
    )
    return model


def build_sequences_for_image(
    lanes_points: List[List[float]],
    tokenizer: LaneTokenizer,
    hparams: LaneLMHyperParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a multi-lane, multi-turn token sequence for one image.

    For now we approximate the (L_q, L_gt) pairs in the paper by using GT
    lanes both as "pseudo label" and target, i.e., L_q == L_gt. This still
    trains LaneLM as a conditional language model p(L | X_v).

    We:
      - take up to max_lanes GT lanes,
      - encode each lane to (x_tokens, y_tokens) of length T,
      - build a long sequence [L1; L2; ...; LN],
      - pad to max_seq_len if needed.
    """
    max_lanes = hparams.max_lanes
    T = tokenizer.T  # num_steps

    # Filter out degenerate lanes (< 2 points)
    lanes_points = [
        coords for coords in lanes_points if len(coords) >= 4
    ]
    lanes_points = lanes_points[:max_lanes]

    x_seqs: List[np.ndarray] = []
    y_seqs: List[np.ndarray] = []

    for lane in lanes_points:
        pts = np.array(lane, dtype=np.float32).reshape(-1, 2)
        xt, yt = tokenizer.encode_single_lane(pts)
        x_seqs.append(xt)
        y_seqs.append(yt)

    if not x_seqs:
        # No valid lanes: return an all-padding sequence of length T.
        x_tokens = np.full((T,), tokenizer.cfg.pad_token_x, dtype=np.int64)
        y_tokens = np.full((T,), tokenizer.T, dtype=np.int64)
    else:
        # Concatenate lanes along sequence dimension: [L1; L2; ...]
        x_tokens = np.concatenate(x_seqs, axis=0)  # (L*T,)
        y_tokens = np.concatenate(y_seqs, axis=0)  # (L*T,)

    max_seq_len = hparams.num_points * hparams.max_lanes * 2
    # Truncate if longer than model max_seq_len
    x_tokens = x_tokens[:max_seq_len]
    y_tokens = y_tokens[:max_seq_len]

    # Pad to max_seq_len
    if x_tokens.shape[0] < max_seq_len:
        pad_len = max_seq_len - x_tokens.shape[0]
        x_pad = np.full(
            (pad_len,), tokenizer.cfg.pad_token_x, dtype=np.int64
        )
        y_pad = np.full((pad_len,), tokenizer.T, dtype=np.int64)
        x_tokens = np.concatenate([x_tokens, x_pad], axis=0)
        y_tokens = np.concatenate([y_tokens, y_pad], axis=0)

    x_tokens_t = torch.from_numpy(x_tokens).long()
    y_tokens_t = torch.from_numpy(y_tokens).long()
    return x_tokens_t, y_tokens_t


def lane_lm_loss_v2(
    logits_x: torch.Tensor,
    logits_y: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    pad_token_x: int,
    pad_token_y: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-entropy loss over x and y tokens with padding ignored."""
    B, T, nbins_x = logits_x.shape
    _, _, max_y_tokens = logits_y.shape

    loss_fn_x = nn.CrossEntropyLoss(ignore_index=pad_token_x)
    loss_fn_y = nn.CrossEntropyLoss(ignore_index=pad_token_y)

    loss_x = loss_fn_x(
        logits_x.view(B * T, nbins_x),
        target_x.view(B * T),
    )
    loss_y = loss_fn_y(
        logits_y.view(B * T, max_y_tokens),
        target_y.view(B * T),
    )
    loss = 0.5 * (loss_x + loss_y)
    return loss, loss_x, loss_y


def train_one_epoch_v2(
    epoch: int,
    clrernet_model,
    lanelm_model: LaneLMModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    tokenizer: LaneTokenizer,
    hparams: LaneLMHyperParams,
    log_interval: int = 50,
) -> None:
    """Train LaneLM v2 for a single epoch."""
    lanelm_model.train()

    pad_token_x = tokenizer.cfg.pad_token_x
    pad_token_y = tokenizer.T  # T used as padding/EOS for y

    for step, batch in enumerate(dataloader):
        imgs = batch["inputs"].to(device, non_blocking=True)
        batch_gt_points = batch["gt_points"]

        B = imgs.shape[0]

        # Extract multi-level FPN features
        feats = extract_pyramid_feats(clrernet_model, imgs)
        visual_tokens = lanelm_model.encode_visual_tokens(feats)

        # Build token sequences per image
        x_list: List[torch.Tensor] = []
        y_list: List[torch.Tensor] = []
        for lanes_points in batch_gt_points:
            x_t, y_t = build_sequences_for_image(
                lanes_points=lanes_points,
                tokenizer=tokenizer,
                hparams=hparams,
            )
            x_list.append(x_t)
            y_list.append(y_t)

        x_tokens = torch.stack(x_list).to(device, non_blocking=True)
        y_tokens = torch.stack(y_list).to(device, non_blocking=True)

        # Teacher forcing: shift by one token
        x_in = x_tokens.clone()
        y_in = y_tokens.clone()
        x_in[:, 1:] = x_tokens[:, :-1]
        x_in[:, 0] = pad_token_x
        y_in[:, 1:] = y_tokens[:, :-1]
        y_in[:, 0] = pad_token_y

        logits_x, logits_y = lanelm_model(
            visual_tokens=visual_tokens,
            x_tokens=x_in,
            y_tokens=y_in,
            visual_padding_mask=None,
        )

        loss, loss_x, loss_y = lane_lm_loss_v2(
            logits_x,
            logits_y,
            x_tokens,
            y_tokens,
            pad_token_x=pad_token_x,
            pad_token_y=pad_token_y,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % log_interval == 0:
            print(
                f"[Epoch {epoch}] Step {step+1}/{len(dataloader)} "
                f"Loss: {loss.item():.4f} "
                f"(x: {loss_x.item():.4f}, y: {loss_y.item():.4f})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LaneLM v2 training on CULane using CLRerNet backbone as visual encoder."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/clrernet/culane/clrernet_culane_dla34_ema.py",
        help="Path to CLRerNet config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="clrernet_culane_dla34_ema.pth",
        help="Path to CLRerNet pretrained checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="CULane dataset root.",
    )
    parser.add_argument(
        "--train-list",
        type=str,
        default="dataset/list/train_gt.txt",
        help="CULane train list file relative to repo root.",
    )
    parser.add_argument(
        "--diff-file",
        type=str,
        default="dataset/list/train_diffs.npz",
        help="Path to frame difference npz file (optional).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for LaneLM head.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="work_dirs/lanelm_culane_dla34_v2",
        help="Directory to save LaneLM checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    hparams = LaneLMHyperParams()

    # DataLoader with GT lanes kept as lists of points
    dataloader = build_culane_lanelm_dataloader_v2(
        data_root=args.data_root,
        list_path=args.train_list,
        diff_file=args.diff_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_w=hparams.img_w,
        img_h=hparams.img_h,
    )

    # Frozen CLRerNet feature extractor
    clrernet_model = build_frozen_clrernet_backbone(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Visual feature dimension list from FPN
    # For CLRerNetFPN, in_channels is [128, 256, 512]; out_channels is 64 for each level
    # We need the OUTPUT channels, not the input channels
    visual_in_channels = (64, 64, 64)

    # LaneLM head
    lanelm_model = build_lanelm_model_v2(
        hparams=hparams,
        visual_in_channels=visual_in_channels,
    ).to(device)

    # Tokenizer consistent with training geometry
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=hparams.img_w,
        img_h=hparams.img_h,
        num_steps=hparams.num_points,
        nbins_x=hparams.nbins_x,
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)

    optimizer = optim.AdamW(lanelm_model.parameters(), lr=args.lr)

    os.makedirs(args.work_dir, exist_ok=True)

    # Simple loss log file for later analysis
    loss_log_path = os.path.join(args.work_dir, "loss_log.tsv")
    with open(loss_log_path, "w") as f:
        f.write("epoch\tstep\tloss\tloss_x\tloss_y\n")

    for epoch in range(1, args.epochs + 1):
        train_one_epoch_v2(
            epoch=epoch,
            clrernet_model=clrernet_model,
            lanelm_model=lanelm_model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            tokenizer=tokenizer,
            hparams=hparams,
        )

        # Append epoch-level summary losses (approximate: last batch)
        # Note: Detailed per-step logging is printed to stdout.
        # For now, we only record epoch index; step/loss columns are placeholders.
        with open(loss_log_path, "a") as f:
            f.write(f"{epoch}\t-1\t-1\t-1\t-1\n")

        # Save checkpoint after each epoch
        ckpt_path = os.path.join(
            args.work_dir,
            f"lanelm_culane_dla34_v2_epoch{epoch}.pth",
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": lanelm_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "num_points": hparams.num_points,
                    "nbins_x": hparams.nbins_x,
                    "embed_dim": hparams.embed_dim,
                    "num_layers": hparams.num_layers,
                    "num_heads": hparams.num_heads,
                    "ffn_dim": hparams.ffn_dim,
                    "max_lanes": hparams.max_lanes,
                },
            },
            ckpt_path,
        )
        print(f"Saved LaneLM v2 checkpoint to: {ckpt_path}")

    print("LaneLM v2 training finished.")


if __name__ == "__main__":
    main()
