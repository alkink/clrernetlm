"""
LaneLM v2 CULane evaluation, aligned with the LaneLM paper and CLRerNet's
evaluation pipeline.

This script:
  - Uses CLRerNet backbone + FPN as frozen visual encoder.
  - Loads a LaneLM v2 checkpoint trained by tools/train_lanelm_culane_v2.py.
  - Runs autoregressive decoding with EOS to generate up to L lanes per image.
  - Maps lane coordinates back to original CULane space.
  - Evaluates the predictions with CULaneMetric (same code as CLRerNet test).
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmengine.config import Config

from libs.datasets import CulaneDataset
from libs.datasets.metrics.culane_metric import CULaneMetric
from libs.models.lanelm import LaneTokenizer, LaneTokenizerConfig, LaneLMModel
from libs.utils.lane_utils import Lane

from tools.train_lanelm_culane_v2 import (  # type: ignore
    build_frozen_clrernet_backbone,
    extract_pyramid_feats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test LaneLM v2 on CULane using CLRerNet backbone and CULaneMetric."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to CLRerNet config file.",
    )
    parser.add_argument(
        "clrernet_checkpoint",
        type=str,
        help="Path to CLRerNet pretrained checkpoint (backbone + FPN).",
    )
    parser.add_argument(
        "lanelm_checkpoint",
        type=str,
        help="Path to LaneLM v2 checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="CULane dataset root.",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default="dataset/list/test.txt",
        help="CULane test list file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--max-lanes",
        type=int,
        default=None,
        help="Max number of lanes to decode per image (default: from checkpoint config).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for token decoding (0.0 = greedy).",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Directory to save evaluation metrics.",
    )
    return parser.parse_args()


def collate_lanelm_test_batch_v2(batch):
    """Collate function for CULane test batches.

    Each item from CulaneDataset.prepare_test_img + Alaug has:
      - 'img': numpy array (H, W, 3)
      - 'sub_img_name': str
      - 'ori_shape', 'img_shape'
    """
    imgs = [item["img"] for item in batch]
    sub_img_names = [item.get("sub_img_name") for item in batch]

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
        "sub_img_name": sub_img_names,
    }


def build_culane_lanelm_test_dataloader_v2(
    data_root: str,
    list_path: str,
    batch_size: int,
    num_workers: int,
    img_w: int,
    img_h: int,
) -> DataLoader:
    """CULane DataLoader for LaneLM v2 test.

    Uses the val/test Albumentations pipeline (Crop + Resize) so that
    geometry is consistent with training and CLRerNet evaluation.
    """
    from configs.clrernet.culane.dataset_culane_clrernet import (  # type: ignore
        val_al_pipeline,
    )

    test_pipeline = [
        dict(type="albumentation", pipelines=val_al_pipeline),
    ]

    dataset = CulaneDataset(
        data_root=data_root,
        data_list=list_path,
        pipeline=test_pipeline,
        diff_file=None,
        test_mode=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_lanelm_test_batch_v2,
    )
    return dataloader


def autoregressive_decode_tokens_v2(
    lanelm_model: LaneLMModel,
    visual_tokens: torch.Tensor,
    num_points: int,
    nbins_x: int,
    max_lanes: int,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autoregressive decoding with EOS for LaneLM v2.

    Args:
        lanelm_model: trained LaneLMModel.
        visual_tokens: (B, N, D) visual tokens from encode_visual_tokens.
        num_points: T, sequence length per lane.
        nbins_x: vocabulary size for x tokens.
        max_lanes: max number of lanes to decode per image.
        temperature: sampling temperature; 0.0 = greedy.

    Returns:
        x_tokens_all: (B, L, T) long tensor.
        y_tokens_all: (B, L, T) long tensor.
    """
    device = visual_tokens.device
    B = visual_tokens.shape[0]
    T = num_points
    pad_token_x = 0
    pad_token_y = T

    all_x = []
    all_y = []

    for lane_idx in range(max_lanes):
        # Initialize tokens: all padding, y tokens are step indices [0..T-1].
        y_tokens = torch.arange(T, device=device).unsqueeze(0).expand(B, -1).clone()
        x_tokens = torch.full(
            (B, T),
            pad_token_x,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(T):
            logits_x, logits_y = lanelm_model(
                visual_tokens=visual_tokens,
                x_tokens=x_tokens,
                y_tokens=y_tokens,
                visual_padding_mask=None,
            )
            step_logits_x = logits_x[:, t, :]  # (B, nbins_x)
            step_logits_y = logits_y[:, t, :]  # (B, max_y_tokens)

            if temperature and temperature > 0.0:
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

            is_eos = (x_next == pad_token_x) | (y_next == pad_token_y)
            finished = finished | is_eos

            if torch.all(finished):
                break

        all_x.append(x_tokens.cpu())
        all_y.append(y_tokens.cpu())

    x_tokens_all = torch.stack(all_x, dim=1)
    y_tokens_all = torch.stack(all_y, dim=1)
    return x_tokens_all, y_tokens_all


def coords_to_lane_v2(
    coords_resized: np.ndarray,
    ori_img_w: int,
    ori_img_h: int,
    crop_bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> Lane:
    """Map coordinates from resized (800x320) back to original CULane space.

    This mirrors the Crop + Resize used in dataset_culane_clrernet.
    """
    if coords_resized.size == 0:
        return Lane(points=np.zeros((0, 2), dtype=np.float32))

    xs = coords_resized[:, 0]
    ys = coords_resized[:, 1]

    x_min, y_min, x_max, y_max = crop_bbox

    x_scale = float(ori_img_w) / float(img_w)
    y_scale = float(y_max - y_min) / float(img_h)

    x_orig = xs * x_scale
    y_orig = ys * y_scale + float(y_min)

    x_norm = x_orig / float(ori_img_w)
    y_norm = y_orig / float(ori_img_h)

    points = np.stack([x_norm, y_norm], axis=1).astype(np.float32)
    return Lane(points=points)


def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        os.makedirs(args.work_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Test configuration (original image size and crop).
    test_cfg = cfg.model.get("test_cfg", {})
    ori_img_w = int(test_cfg.get("ori_img_w", 1640))
    ori_img_h = int(test_cfg.get("ori_img_h", 590))

    from configs.clrernet.culane.dataset_culane_clrernet import (  # type: ignore
        crop_bbox,
        img_scale,
    )

    img_w, img_h = img_scale  # e.g., (800, 320)

    # Load LaneLM v2 checkpoint and config.
    ckpt = torch.load(args.lanelm_checkpoint, map_location="cpu")
    lm_cfg = ckpt.get("config", {})
    num_points = int(lm_cfg.get("num_points", 40))
    nbins_x = int(lm_cfg.get("nbins_x", 800))
    embed_dim = int(lm_cfg.get("embed_dim", 256))
    num_layers = int(lm_cfg.get("num_layers", 4))
    num_heads = int(lm_cfg.get("num_heads", 8))
    ffn_dim = int(lm_cfg.get("ffn_dim", 512))
    ckpt_max_lanes = int(lm_cfg.get("max_lanes", 4))

    max_lanes = args.max_lanes if args.max_lanes is not None else ckpt_max_lanes

    print(
        f"Loaded LaneLM v2 checkpoint '{args.lanelm_checkpoint}' "
        f"with config: num_points={num_points}, nbins_x={nbins_x}, "
        f"embed_dim={embed_dim}, num_layers={num_layers}, "
        f"num_heads={num_heads}, ffn_dim={ffn_dim}, max_lanes={ckpt_max_lanes}"
    )
    print(f"Decoding up to max_lanes={max_lanes} per image.")

    # DataLoader for CULane test split.
    test_loader = build_culane_lanelm_test_dataloader_v2(
        data_root=args.data_root,
        list_path=args.test_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_w=img_w,
        img_h=img_h,
    )

    # Frozen CLRerNet visual encoder.
    clrernet_model = build_frozen_clrernet_backbone(
        config_path=args.config,
        checkpoint_path=args.clrernet_checkpoint,
        device=device,
    )

    # LaneLM v2 head.
    visual_in_channels = (64, 64, 64)
    max_y_tokens = num_points + 1
    # Sequence length = T * max_lanes (no extra factor)
    max_seq_len = num_points * max_lanes

    lanelm_model = LaneLMModel(
        nbins_x=nbins_x,
        max_y_tokens=max_y_tokens,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
        visual_in_dim=None,
        visual_in_channels=visual_in_channels,
    )
    lanelm_model.load_state_dict(ckpt["model_state_dict"])
    lanelm_model.to(device)
    lanelm_model.eval()

    tokenizer_cfg = LaneTokenizerConfig(
        img_w=img_w,
        img_h=img_h,
        num_steps=num_points,
        nbins_x=nbins_x,
    )
    tokenizer = LaneTokenizer(tokenizer_cfg)

    # CULane metric (same as CLRerNet test).
    metric = CULaneMetric(
        data_root=args.data_root,
        data_list=args.test_list,
    )

    all_results: List[dict] = []

    print("Starting LaneLM v2 inference on CULane test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            imgs = batch["inputs"].to(device, non_blocking=True)
            sub_img_names = batch["sub_img_name"]

            B = imgs.shape[0]

            # Visual tokens from frozen CLRerNet.
            feats = extract_pyramid_feats(clrernet_model, imgs)
            visual_tokens = lanelm_model.encode_visual_tokens(feats)

            # Autoregressive decode with EOS.
            x_tokens_all, y_tokens_all = autoregressive_decode_tokens_v2(
                lanelm_model=lanelm_model,
                visual_tokens=visual_tokens,
                num_points=num_points,
                nbins_x=nbins_x,
                max_lanes=max_lanes,
                temperature=args.temperature,
            )

            x_tokens_np = x_tokens_all.numpy()  # (B, L, T)
            y_tokens_np = y_tokens_all.numpy()  # (B, L, T)

            for i in range(B):
                lanes: List[Lane] = []
                for l in range(x_tokens_np.shape[1]):
                    xt = x_tokens_np[i, l]
                    yt = y_tokens_np[i, l]

                    coords_resized = tokenizer.decode_single_lane(xt, yt)
                    if coords_resized.shape[0] == 0:
                        continue

                    lane = coords_to_lane_v2(
                        coords_resized=coords_resized,
                        ori_img_w=ori_img_w,
                        ori_img_h=ori_img_h,
                        crop_bbox=tuple(crop_bbox),
                        img_w=img_w,
                        img_h=img_h,
                    )
                    if lane.points.shape[0] > 0:
                        lanes.append(lane)

                sub_img_name = sub_img_names[i]
                all_results.append(
                    {
                        "lanes": lanes,
                        "scores": np.ones(len(lanes), dtype=np.float32),
                        "metainfo": {"sub_img_name": sub_img_name},
                    }
                )

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Processed {batch_idx + 1}/{len(test_loader)} "
                    f"batches ({(batch_idx + 1) * args.batch_size} images)..."
                )

    print("Finished inference. Running CULane evaluation...")
    results = metric.compute_metrics(all_results)

    print("CULane evaluation results (LaneLM v2 head):")
    for k in sorted(results.keys()):
        v = results[k]
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    if args.work_dir is not None:
        out_path = os.path.join(args.work_dir, "lanelm_culane_v2_metrics.txt")
        with open(out_path, "w") as f:
            for k in sorted(results.keys()):
                v = results[k]
                if isinstance(v, float):
                    f.write(f"{k}: {v:.6f}\n")
                else:
                    f.write(f"{k}: {v}\n")
        print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
