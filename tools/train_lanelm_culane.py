"""
Train a LaneLM-style head on top of a frozen CLRerNet backbone on CULane.

This script:
  - Reuses the existing CULane data pipeline and augmentations.
  - Replaces the final PackCLRNetInputs transform with PackLaneLMInputs to
    produce LaneLM-style lane tokens.
  - Uses a pretrained CLRerNet model (DLA34 + FPN) as a frozen visual encoder.
  - Trains LaneLMModel to autoregressively model lane keypoint tokens.

Note:
  This is a standalone PyTorch training loop and does not modify the existing
  CLRerNet training/testing code paths.
"""

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mmdet.apis import init_detector

from libs.datasets import CulaneDataset
from libs.datasets.pipelines import Compose
from libs.datasets.pipelines.lane_formatting import PackLaneLMInputs
from libs.models.lanelm import LaneLMModel


def collate_lanelm_batch(batch):
    """Custom collate function for LaneLM batches.

    Handles metainfo as list of dicts and stacks tensor inputs.
    """
    inputs = torch.stack([item["inputs"] for item in batch])
    lane_tokens_x = torch.stack([item["lane_tokens_x"] for item in batch])
    lane_tokens_y = torch.stack([item["lane_tokens_y"] for item in batch])
    lane_valid_mask = torch.stack([item["lane_valid_mask"] for item in batch])
    metainfo = [item["metainfo"] for item in batch]

    return {
        "inputs": inputs,
        "lane_tokens_x": lane_tokens_x,
        "lane_tokens_y": lane_tokens_y,
        "lane_valid_mask": lane_valid_mask,
        "metainfo": metainfo,
    }


def build_culane_lanelm_dataloader(
    data_root: str,
    list_path: str,
    diff_file: str,
    batch_size: int,
    num_workers: int,
    img_w: int = 800,
    img_h: int = 320,
    max_lanes: int = 4,
    num_points: int = 40,
    nbins_x: int = 800,
) -> DataLoader:
    """Build DataLoader for CULane with LaneLM tokenization."""
    # Import the CULane augmentation pipeline from the existing config
    from configs.clrernet.culane.dataset_culane_clrernet import (  # type: ignore
        compose_cfg,
        crop_bbox,
        img_scale,
        train_al_pipeline,
    )

    # Reuse albumentations config but change the final packing step
    train_pipeline = [
        dict(type="albumentation", pipelines=train_al_pipeline),
        dict(
            type="PackLaneLMInputs",
            meta_keys=[
                "filename",
                "sub_img_name",
                "ori_shape",
                "img_shape",
            ],
            max_lanes=max_lanes,
            num_points=num_points,
            img_w=img_w,
            img_h=img_h,
            nbins_x=nbins_x,
        ),
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
        collate_fn=collate_lanelm_batch,
    )
    return dataloader


def build_frozen_clrernet(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
):
    """Initialize CLRerNet with pretrained weights and freeze backbone + neck."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()

    # Freeze backbone and neck: use CLRerNet only as a feature extractor
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.neck.parameters():
        param.requires_grad = False
    if hasattr(model, "bbox_head"):
        for param in model.bbox_head.parameters():
            param.requires_grad = False

    return model


def extract_visual_tokens(
    model,
    imgs: torch.Tensor,
) -> torch.Tensor:
    """Extract visual tokens from CLRerNet backbone + FPN.

    Args:
        model: CLRerNet detector.
        imgs: (B, 3, H, W) tensor.

    Returns:
        visual_tokens: (B, N, C_feat) tensor suitable for LaneLMModel.
    """
    # Extract FPN features
    with torch.no_grad():
        feats = model.extract_feat(imgs)

    # Use the highest-resolution FPN feature map
    feat = feats[0]  # shape: (B, C, Hf, Wf)
    B, C, Hf, Wf = feat.shape

    # Flatten spatial dimensions into token dimension
    visual_tokens = feat.view(B, C, Hf * Wf).permute(0, 2, 1).contiguous()
    # shape: (B, N, C)
    return visual_tokens


def build_lanelm_model(
    nbins_x: int,
    num_points: int,
    visual_in_dim: int,
    embed_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    ffn_dim: int = 512,
) -> LaneLMModel:
    """Construct LaneLMModel with parameters matching CULane settings."""
    max_y_tokens = num_points + 1  # including padding token
    model = LaneLMModel(
        nbins_x=nbins_x,
        max_y_tokens=max_y_tokens,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        max_seq_len=num_points,
        visual_in_dim=visual_in_dim,
    )
    return model


def lane_lm_loss(
    logits_x: torch.Tensor,
    logits_y: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    pad_token_x: int,
    pad_token_y: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute cross-entropy losses for x and y tokens with padding ignored."""
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
    loss = (loss_x + loss_y) * 0.5
    return loss, loss_x, loss_y


def train_one_epoch(
    epoch: int,
    clrernet_model,
    lanelm_model: LaneLMModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_points: int,
    nbins_x: int,
    log_interval: int = 50,
) -> None:
    """Train LaneLM for a single epoch."""
    lanelm_model.train()

    pad_token_x = 0
    pad_token_y = num_points  # T is used as padding for y

    for step, batch in enumerate(dataloader):
        imgs = batch["inputs"].to(device, non_blocking=True)
        lane_tokens_x = batch["lane_tokens_x"].to(device, non_blocking=True)
        lane_tokens_y = batch["lane_tokens_y"].to(device, non_blocking=True)
        lane_valid_mask = batch["lane_valid_mask"].to(device, non_blocking=True)

        B, L, T = lane_tokens_x.shape
        assert T == num_points, f"Expected {num_points} points, got {T}."

        # Extract visual tokens from CLRerNet
        visual_tokens_per_img = extract_visual_tokens(clrernet_model, imgs)
        _, N, C_feat = visual_tokens_per_img.shape

        # Repeat visual tokens for each lane in the image: (B, L, N, C) -> (B*L, N, C)
        visual_tokens = (
            visual_tokens_per_img.unsqueeze(1)
            .expand(-1, L, -1, -1)
            .reshape(B * L, N, C_feat)
        )

        # Flatten lane tokens: (B, L, T) -> (B*L, T)
        x_tgt = lane_tokens_x.view(B * L, T)
        y_tgt = lane_tokens_y.view(B * L, T)
        valid_mask_flat = lane_valid_mask.view(B * L)

        # Filter out lanes with no valid tokens to avoid wasting computation
        valid_idx = (valid_mask_flat > 0).nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            continue

        x_tgt = x_tgt[valid_idx]
        y_tgt = y_tgt[valid_idx]
        visual_tokens = visual_tokens[valid_idx]

        # Build input sequences by shifting targets to the right (teacher forcing)
        x_in = x_tgt.clone()
        y_in = y_tgt.clone()
        # Shift: first token is padding, each position sees previous target token
        x_in[:, 1:] = x_tgt[:, :-1]
        x_in[:, 0] = pad_token_x
        y_in[:, 1:] = y_tgt[:, :-1]
        y_in[:, 0] = pad_token_y

        logits_x, logits_y = lanelm_model(
            visual_tokens=visual_tokens,
            x_tokens=x_in,
            y_tokens=y_in,
            visual_padding_mask=None,
        )

        loss, loss_x, loss_y = lane_lm_loss(
            logits_x,
            logits_y,
            x_tgt,
            y_tgt,
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
        description="Train LaneLM head on CULane using CLRerNet backbone."
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
        help="CULane dataset root (as used in CLRerNet).",
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
        "--embed-dim",
        type=int,
        default=256,
        help="Embedding dimension for LaneLMModel.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of Transformer decoder layers.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=512,
        help="Hidden dimension of FFN in decoder layers.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=40,
        help="Number of lane points (T) for LaneLM token sequences.",
    )
    parser.add_argument(
        "--nbins-x",
        type=int,
        default=800,
        help="Vocabulary size for x-coordinate tokens.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="work_dirs/lanelm_culane_dla34",
        help="Directory to save LaneLM checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # DataLoader with LaneLM tokenization
    dataloader = build_culane_lanelm_dataloader(
        data_root=args.data_root,
        list_path=args.train_list,
        diff_file=args.diff_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_w=800,
        img_h=320,
        max_lanes=4,
        num_points=args.num_points,
        nbins_x=args.nbins_x,
    )

    # Frozen CLRerNet feature extractor
    clrernet_model = build_frozen_clrernet(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Visual feature dimension from FPN
    # CLRerNetFPN out_channels is 64 in base_clrernet.py
    visual_in_dim = 64

    # LaneLM head
    lanelm_model = build_lanelm_model(
        nbins_x=args.nbins_x,
        num_points=args.num_points,
        visual_in_dim=visual_in_dim,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
    ).to(device)

    optimizer = optim.AdamW(lanelm_model.parameters(), lr=args.lr)

    os.makedirs(args.work_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            epoch=epoch,
            clrernet_model=clrernet_model,
            lanelm_model=lanelm_model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            num_points=args.num_points,
            nbins_x=args.nbins_x,
        )

        # Save checkpoint after each epoch
        ckpt_path = os.path.join(
            args.work_dir,
            f"lanelm_culane_dla34_epoch{epoch}.pth",
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": lanelm_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "num_points": args.num_points,
                    "nbins_x": args.nbins_x,
                    "embed_dim": args.embed_dim,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "ffn_dim": args.ffn_dim,
                },
            },
            ckpt_path,
        )
        print(f"Saved LaneLM checkpoint to: {ckpt_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
