"""
LaneLM v3 training script on CULane (per-lane training).

Farkı ne?
  - v2'de her görüntüdeki lane'ler tek uzun sequence olarak concat ediliyordu
    (L1; L2; ...; LN). Inference'te ise lane başına ayrı decode yapıyoruz.
    Bu training/inference dağılımını uyumsuz hale getiriyordu.
  - v3'te her lane bağımsız bir sequence (B*L, T) olarak ele alınıyor.
    Yani model p(L | X_v) her lane için ayrı ayrı öğreniyor; train ve test
    formu bire bir aynı.

Bu script özellikle:
  - Küçük overfit deneyleri (1–2 görüntü) için,
  - LaneLM'in geometriyi prensipte öğrenip öğrenemediğini test etmek için
tasarlanmıştır.
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
    """Hyper-parameters for LaneLM v3 (per-lane training)."""

    num_points: int = 40  # T
    # Daha kolay öğrenilebilir hale getirmek için x vocab küçültüldü.
    nbins_x: int = 200
    max_lanes: int = 4
    img_w: int = 800
    img_h: int = 320
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 512
    dropout: float = 0.0  # Dropout rate for regularization


def collate_lanelm_batch_v3(batch):
    """Custom collate function to keep per-image GT lanes."""
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


def build_culane_lanelm_dataloader_v3(
    data_root: str,
    list_path: str,
    diff_file: str,
    batch_size: int,
    num_workers: int,
    img_w: int,
    img_h: int,
    no_aug: bool = False,
) -> DataLoader:
    """CULane DataLoader for LaneLM v3 (per-lane training).

    If no_aug=True, uses the validation pipeline (Crop+Resize, no random
    augmentations). This is intended for overfit/debug runs where we want
    the training distribution to match the visualization/test pipeline.
    Otherwise, uses the full train_al_pipeline with augmentations.
    """
    from configs.clrernet.culane.dataset_culane_clrernet import (  # type: ignore
        train_al_pipeline,
        val_al_pipeline,
    )

    if no_aug:
        pipelines = val_al_pipeline
    else:
        pipelines = train_al_pipeline

    train_pipeline = [
        dict(type="albumentation", pipelines=pipelines),
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
        collate_fn=collate_lanelm_batch_v3,
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
    """Extract FPN features {F0, F1, F2} from CLRerNet backbone + neck."""
    with torch.no_grad():
        feats = model.extract_feat(imgs)
    return list(feats)


def build_lanelm_model_v3(
    hparams: LaneLMHyperParams,
    visual_in_channels: Tuple[int, int, int],
) -> LaneLMModel:
    """Construct LaneLMModel for per-lane training.

    Sequence length = num_points (T); her lane tek bir sequence.
    """
    max_y_tokens = hparams.num_points + 1  # including padding/EOS
    max_seq_len = hparams.num_points
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


def lane_lm_loss_v3(
    logits_x: torch.Tensor,
    logits_y: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    pad_token_x: int,
    pad_token_y: int,
    y_loss_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-entropy loss over x and y tokens with padding ignored.

    y_loss_weight can be set to 0.0 for debug runs where we want to
    focus purely on the x-coordinate modeling.
    """
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
    loss = 0.5 * (loss_x + y_loss_weight * loss_y)
    return loss, loss_x, loss_y


def train_one_epoch_v3(
    epoch: int,
    clrernet_model,
    lanelm_model: LaneLMModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    tokenizer: LaneTokenizer,
    hparams: LaneLMHyperParams,
    log_interval: int = 10,
    y_loss_weight: float = 1.0,
    loss_log_path: str = "",
) -> None:
    """Train LaneLM v3 (per-lane) for a single epoch."""
    lanelm_model.train()

    pad_token_x = tokenizer.cfg.pad_token_x
    pad_token_y = tokenizer.T  # T used as padding/EOS for y

    # Accumulate losses for epoch-level logging
    total_loss = 0.0
    total_loss_x = 0.0
    total_loss_y = 0.0
    num_steps = 0

    for step, batch in enumerate(dataloader):
        imgs = batch["inputs"].to(device, non_blocking=True)
        batch_gt_points = batch["gt_points"]

        B = imgs.shape[0]

        # Visual tokens per image
        feats = extract_pyramid_feats(clrernet_model, imgs)
        visual_tokens_imgs = lanelm_model.encode_visual_tokens(feats)

        # Flatten lanes across batch: each lane becomes one sequence.
        lane_visual_tokens: List[torch.Tensor] = []
        lane_x_list: List[torch.Tensor] = []
        lane_y_list: List[torch.Tensor] = []

        for img_idx, lanes_points in enumerate(batch_gt_points):
            lanes_points = [
                coords for coords in lanes_points if len(coords) >= 4
            ]
            # Limit to max_lanes per image
            lanes_points = lanes_points[: hparams.max_lanes]

            for lane_idx, lane in enumerate(lanes_points):
                pts = np.array(lane, dtype=np.float32).reshape(-1, 2)

                # DEBUG: Check if points are in valid range
                if step == 0 and img_idx == 0 and lane_idx == 0:
                    print(f"\nDEBUG: First GT lane points")
                    print(f"  Points shape: {pts.shape}")
                    print(f"  X range: [{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}]")
                    print(f"  Y range: [{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}]")
                    print(f"  First 3 points: {pts[:3].tolist()}")

                x_np, y_np = tokenizer.encode_single_lane(pts)

                # DEBUG: Check encoded tokens
                if step == 0 and img_idx == 0 and lane_idx == 0:
                    print(f"\nDEBUG: Encoded tokens")
                    print(f"  x_tokens min/max: {x_np.min()}/{x_np.max()}")
                    print(f"  y_tokens min/max: {y_np.min()}/{y_np.max()}")
                    print(f"  x_tokens (first 10): {x_np[:10].tolist()}")
                    print(f"  y_tokens (first 10): {y_np[:10].tolist()}")
                    print(f"  Config: nbins_x={tokenizer.cfg.nbins_x}, img_w={tokenizer.cfg.img_w}, x_mode={tokenizer.cfg.x_mode}, max_abs_dx={tokenizer.cfg.max_abs_dx}")

                    # Validate tokens
                    invalid_mask = x_np >= tokenizer.cfg.nbins_x
                    if invalid_mask.any():
                        print(f"  ERROR: {invalid_mask.sum()} tokens >= nbins_x!")
                        invalid_indices = np.where(invalid_mask)[0]
                        print(f"  Invalid indices: {invalid_indices.tolist()}")
                        print(f"  Invalid tokens: {x_np[invalid_indices].tolist()}")

                lane_visual_tokens.append(visual_tokens_imgs[img_idx])
                lane_x_list.append(torch.from_numpy(x_np).long())
                lane_y_list.append(torch.from_numpy(y_np).long())

        if not lane_visual_tokens:
            # No valid lanes in this batch; skip.
            continue

        # Stack to tensors
        visual_tokens = torch.stack(lane_visual_tokens).to(device, non_blocking=True)
        x_tokens = torch.stack(lane_x_list).to(device, non_blocking=True)
        y_tokens = torch.stack(lane_y_list).to(device, non_blocking=True)

        # Teacher forcing
        x_in = x_tokens.clone()
        y_in = y_tokens.clone()
        x_in[:, 1:] = x_tokens[:, :-1]
        # t=0'da BOS yerine GT x0 vererek başlangıcı kolaylaştır
        x_in[:, 0] = x_tokens[:, 0]
        y_in[:, 1:] = y_tokens[:, :-1]
        y_in[:, 0] = pad_token_y

        logits_x, logits_y = lanelm_model(
            visual_tokens=visual_tokens,
            x_tokens=x_in,
            y_tokens=y_in,
            visual_padding_mask=None,
        )

        loss, loss_x, loss_y = lane_lm_loss_v3(
            logits_x,
            logits_y,
            x_tokens,
            y_tokens,
            pad_token_x=pad_token_x,
            pad_token_y=pad_token_y,
            y_loss_weight=y_loss_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_loss_x += loss_x.item()
        total_loss_y += loss_y.item()
        num_steps += 1

        # Per-step logging (for larger datasets)
        if (step + 1) % log_interval == 0:
            msg = (
                f"[Epoch {epoch}] Step {step+1}/{len(dataloader)} "
                f"Loss: {loss.item():.4f} "
                f"(x: {loss_x.item():.4f}, y: {loss_y.item():.4f})"
            )
            print(msg)

    # Epoch-level logging (always happens)
    if num_steps > 0:
        avg_loss = total_loss / num_steps
        avg_loss_x = total_loss_x / num_steps
        avg_loss_y = total_loss_y / num_steps

        msg = (
            f"[Epoch {epoch}] Average Loss: {avg_loss:.4f} "
            f"(x: {avg_loss_x:.4f}, y: {avg_loss_y:.4f})"
        )
        print(msg)

        if loss_log_path:
            with open(loss_log_path, "a") as f:
                f.write(
                    f"{epoch}\tepoch_avg\t"
                    f"{avg_loss:.6f}\t"
                    f"{avg_loss_x:.6f}\t"
                    f"{avg_loss_y:.6f}\n"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LaneLM v3 per-lane training on CULane using CLRerNet backbone."
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
        help="CULane train list file.",
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
        help="Batch size (number of images; lanes are flattened inside).",
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
        "--no-aug",
        action="store_true",
        help="Disable random data augmentations and use validation pipeline "
        "(Crop+Resize only). Recommended for overfit/debug runs.",
    )
    parser.add_argument(
        "--no-y-loss",
        action="store_true",
        help="Train only on X coordinate loss (ignore Y loss). "
        "Useful for debugging X tokenization issues.",
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
        default="work_dirs/lanelm_culane_dla34_v3",
        help="Directory to save LaneLM v3 checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    hparams = LaneLMHyperParams()

    y_loss_weight = 0.0 if args.no_y_loss else 1.0

    dataloader = build_culane_lanelm_dataloader_v3(
        data_root=args.data_root,
        list_path=args.train_list,
        diff_file=args.diff_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_w=hparams.img_w,
        img_h=hparams.img_h,
        no_aug=args.no_aug,
    )

    clrernet_model = build_frozen_clrernet_backbone(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Create tokenizer config first to determine vocab size
    tokenizer_cfg = LaneTokenizerConfig(
        img_w=hparams.img_w,
        img_h=hparams.img_h,
        num_steps=hparams.num_points,
        nbins_x=hparams.nbins_x,
        # Disjoint relative encoding with smaller dx for easier learning
        x_mode="relative_disjoint",
        max_abs_dx=32,
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)
    print(
        f"Tokenizer config -> x_mode={tokenizer_cfg.x_mode}, "
        f"max_abs_dx={tokenizer_cfg.max_abs_dx}, nbins_x={tokenizer_cfg.nbins_x}"
    )

    # Calculate required vocab size
    vocab_size_x = hparams.nbins_x
    if tokenizer_cfg.x_mode == "relative_disjoint":
        # Need: nbins_x (absolute) + 2*max_abs_dx+1 (relative deltas)
        vocab_size_x = hparams.nbins_x + 2 * tokenizer_cfg.max_abs_dx + 1
        print(f"Disjoint mode: vocab_size_x = {vocab_size_x}")
    else:
        print(f"Overlapping relative mode: vocab_size_x = {vocab_size_x}")

    # Build model with correct vocab size
    visual_in_channels = (64, 64, 64)

    # Temporarily set nbins_x for model
    original_nbins_x = hparams.nbins_x
    hparams.nbins_x = vocab_size_x

    lanelm_model = build_lanelm_model_v3(
        hparams=hparams,
        visual_in_channels=visual_in_channels,
    ).to(device)

    # Restore original
    hparams.nbins_x = original_nbins_x

    optimizer = optim.AdamW(lanelm_model.parameters(), lr=args.lr)

    # Y loss weight: 0.0 if --no-y-loss, else 1.0
    y_loss_weight = 0.0 if args.no_y_loss else 1.0
    if args.no_y_loss:
        print("Training with X-only loss (Y loss disabled)")

    os.makedirs(args.work_dir, exist_ok=True)

    loss_log_path = os.path.join(args.work_dir, "loss_log.tsv")
    with open(loss_log_path, "w") as f:
        f.write("epoch\tstep\tloss\tloss_x\tloss_y\n")

    for epoch in range(1, args.epochs + 1):
        train_one_epoch_v3(
            epoch=epoch,
            clrernet_model=clrernet_model,
            lanelm_model=lanelm_model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            tokenizer=tokenizer,
            hparams=hparams,
            y_loss_weight=y_loss_weight,
            loss_log_path=loss_log_path,
        )

        # Save checkpoint every 50 epochs and at the last epoch
        if epoch % 50 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(
                args.work_dir,
                f"lanelm_culane_dla34_v3_epoch{epoch}.pth",
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
                        "per_lane": True,
                        "x_mode": tokenizer_cfg.x_mode,
                        "max_abs_dx": tokenizer_cfg.max_abs_dx,
                    },
                },
                ckpt_path,
            )
            print(f"Saved LaneLM v3 checkpoint to: {ckpt_path}")

    print("LaneLM v3 training finished.")


if __name__ == "__main__":
    main()
    parser.add_argument(
        "--no-y-loss",
        action="store_true",
        help="Disable y-token loss (debug only; focus on x modeling).",
    )
