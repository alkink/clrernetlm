"""
Evaluate a LaneLM-style head on CULane using the same
evaluation logic as CLRerNet (CULaneMetric + .lines.txt files).

This script mirrors the CLI style of `tools/test.py` but runs a
custom PyTorch inference loop:
  - uses a frozen CLRerNet backbone + FPN as visual encoder
  - loads a LaneLMModel checkpoint trained by `tools/train_lanelm_culane.py`
  - autoregressively decodes lane keypoint tokens
  - decodes tokens into lane geometries and evaluates with CULaneMetric

Important notes:
  - LaneLM inference here is a minimal, greedy autoregressive decoder
    (1 lane per image by default). It is suitable as a smoke test and
    for qualitative analysis, not a final reproduction of the LaneLM paper.
  - The CULane evaluation *metric and data flow* (writing .lines.txt,
    categories, IoU computation) are identical to the CLRerNet test pipeline.
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
from libs.models.lanelm import LaneTokenizer, LaneTokenizerConfig
from libs.utils.lane_utils import Lane

from tools.train_lanelm_culane import (  # type: ignore
    build_frozen_clrernet,
    build_lanelm_model,
    extract_visual_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test LaneLM head on CULane using CLRerNet backbone and CULaneMetric."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to CLRerNet config file (same as for tools/test.py).",
    )
    parser.add_argument(
        "clrernet_checkpoint",
        type=str,
        help="Path to CLRerNet pretrained checkpoint (backbone + FPN).",
    )
    parser.add_argument(
        "lanelm_checkpoint",
        type=str,
        help="Path to LaneLM checkpoint produced by tools/train_lanelm_culane.py.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="CULane dataset root. Default: %(default)s",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default="dataset/list/test.txt",
        help="CULane test list file relative to repo root. Default: %(default)s",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for LaneLM inference. Default: %(default)s",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers. Default: %(default)s",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference. Default: %(default)s",
    )
    parser.add_argument(
        "--num-lanes-per-image",
        type=int,
        default=1,
        help=(
            "Number of LaneLM-generated lanes per image. "
            "Currently only 1 is supported in decoding; values >1 are treated as 1."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature for token decoding. "
            "0.0 means greedy argmax. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Optional directory to save evaluation logs/results.",
    )
    args = parser.parse_args()
    return args


def collate_lanelm_batch(batch):
    """Custom collate function for LaneLM batches.

    Handles metainfo as list of dicts instead of trying to stack them.
    """
    inputs = torch.stack([item["inputs"] for item in batch])
    metainfo = [item["metainfo"] for item in batch]

    # For test mode, we don't have lane tokens, just return inputs and metainfo
    return {
        "inputs": inputs,
        "metainfo": metainfo,
    }


def build_culane_lanelm_test_dataloader(
    data_root: str,
    list_path: str,
    batch_size: int,
    num_workers: int,
    img_w: int,
    img_h: int,
    max_lanes: int,
    num_points: int,
    nbins_x: int,
) -> DataLoader:
    """Build a CULane DataLoader for LaneLM inference.

    This mirrors the CULane val/test pipeline (Crop + Resize) by reusing
    `val_al_pipeline` from the CLRerNet dataset config and PackLaneLMInputs
    for image preprocessing. Ground-truth tokens are *not* used here.
    """
    from configs.clrernet.culane.dataset_culane_clrernet import (  # type: ignore
        val_al_pipeline,
    )
    from libs.datasets.pipelines.lane_formatting import PackLaneLMInputs

    # Albumentations-based deterministic test pipeline (Crop + Resize).
    test_pipeline = [
        dict(type="albumentation", pipelines=val_al_pipeline),
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
        collate_fn=collate_lanelm_batch,
    )
    return dataloader


def autoregressive_decode_tokens(
    lanelm_model,
    visual_tokens: torch.Tensor,
    num_points: int,
    nbins_x: int,
    temperature: float = 0.0,
    max_lanes: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy (or temperature) autoregressive decoding of LaneLM tokens.

    Args:
        lanelm_model: LaneLMModel in eval mode.
        visual_tokens: (B, N, C) visual feature tokens from CLRerNet FPN.
        num_points: Sequence length T used in training/tokenizer.
        nbins_x: Vocabulary size for x tokens.
        temperature: Sampling temperature; 0.0 = greedy argmax.

    Returns:
        x_tokens: (B, T) long tensor of x tokens.
        y_tokens: (B, T) long tensor of y/time-step tokens (0..T-1).

    Note:
        - This implementation is intentionally simple and computes a full
          forward pass for each time step. For T=40 this is acceptable.
        - y tokens are fixed to [0..T-1] for all positions to mimic the
          training setup where y encodes the step index.
    """
    device = visual_tokens.device
    B = visual_tokens.shape[0]
    T = num_points
    pad_token_x = 0
    pad_token_y = T  # EOS / padding for y

    # We decode up to max_lanes per image by repeating the decoding process.
    # Each lane has a fixed maximum length T, and we stop early when EOS
    # (x=0 or y=T) is predicted.
    all_x = []
    all_y = []

    for lane_idx in range(max_lanes):
        # Fixed y tokens encoding the step index [0..T-1].
        y_tokens = torch.arange(T, device=device).unsqueeze(0).expand(B, -1).clone()

        # Initialize all x tokens as padding.
        x_tokens = torch.full(
            (B, T),
            pad_token_x,
            dtype=torch.long,
            device=device,
        )

        # Autoregressive decoding with EOS criterion.
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

            # If EOS has not been reached yet, update tokens; otherwise keep padding.
            still_running = ~finished
            x_tokens[still_running, t] = x_next[still_running]
            y_tokens[still_running, t] = y_next[still_running]

            # EOS condition: x==0 or y==T
            is_eos = (x_next == pad_token_x) | (y_next == pad_token_y)
            finished = finished | is_eos

            # Early break if all sequences finished
            if torch.all(finished):
                break

        all_x.append(x_tokens.cpu())
        all_y.append(y_tokens.cpu())

    # Stack over lane dimension: (B, L, T)
    x_tokens_all = torch.stack(all_x, dim=1)
    y_tokens_all = torch.stack(all_y, dim=1)
    return x_tokens_all, y_tokens_all


def coords_to_lane(
    coords_resized: np.ndarray,
    ori_img_w: int,
    ori_img_h: int,
    crop_bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> Lane:
    """Map coordinates from cropped+resized space back to original CULane space.

    Args:
        coords_resized: (M, 2) array in the 800x320 (img_w x img_h) space used by LaneLM.
        ori_img_w: Original image width (e.g., 1640).
        ori_img_h: Original image height (e.g., 590).
        crop_bbox: (x_min, y_min, x_max, y_max) used in dataset pipeline.
        img_w: Width after resizing (e.g., 800).
        img_h: Height after resizing (e.g., 320).

    Returns:
        Lane instance whose internal points are normalized to [0, 1] in both axes,
        compatible with CULaneMetric.
    """
    if coords_resized.size == 0:
        return Lane(points=np.zeros((0, 2), dtype=np.float32))

    xs = coords_resized[:, 0]
    ys = coords_resized[:, 1]

    x_min, y_min, x_max, y_max = crop_bbox

    # In the current config: x is resized from [0, ori_w) -> [0, img_w),
    # y is cropped to [y_min, y_max) then resized to [0, img_h).
    # Invert this mapping.
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
    cut_height = int(test_cfg.get("cut_height", 270))

    # Dataset geometry (LaneLM operates in this resized space).
    from configs.clrernet.culane.dataset_culane_clrernet import (  # type: ignore
        crop_bbox,
        img_scale,
    )

    img_w, img_h = img_scale  # e.g., (800, 320)

    # Load LaneLM checkpoint and config.
    ckpt = torch.load(args.lanelm_checkpoint, map_location="cpu")
    lm_cfg = ckpt.get("config", {})
    num_points = int(lm_cfg.get("num_points", 40))
    nbins_x = int(lm_cfg.get("nbins_x", 800))
    embed_dim = int(lm_cfg.get("embed_dim", 256))
    num_layers = int(lm_cfg.get("num_layers", 4))
    num_heads = int(lm_cfg.get("num_heads", 8))
    ffn_dim = int(lm_cfg.get("ffn_dim", 512))

    print(
        f"Loaded LaneLM checkpoint '{args.lanelm_checkpoint}' "
        f"with config: num_points={num_points}, nbins_x={nbins_x}, "
        f"embed_dim={embed_dim}, num_layers={num_layers}, "
        f"num_heads={num_heads}, ffn_dim={ffn_dim}"
    )

    # DataLoader for CULane test split with LaneLM-style preprocessing.
    max_lanes = 4  # must match training pipeline
    test_loader = build_culane_lanelm_test_dataloader(
        data_root=args.data_root,
        list_path=args.test_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_w=img_w,
        img_h=img_h,
        max_lanes=max_lanes,
        num_points=num_points,
        nbins_x=nbins_x,
    )

    # Frozen CLRerNet feature extractor (same as in training).
    clrernet_model = build_frozen_clrernet(
        config_path=args.config,
        checkpoint_path=args.clrernet_checkpoint,
        device=device,
    )

    # Visual feature dimension (FPN out_channels).
    visual_in_dim = getattr(clrernet_model.neck, "out_channels", embed_dim)

    # LaneLM head and tokenizer.
    lanelm_model = build_lanelm_model(
        nbins_x=nbins_x,
        num_points=num_points,
        visual_in_dim=visual_in_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
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

    # CULane metric instance (same evaluation logic as CLRerNet).
    metric = CULaneMetric(
        data_root=args.data_root,
        data_list=args.test_list,
    )

    all_results: List[dict] = []

    print("Starting LaneLM inference on CULane test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            imgs = batch["inputs"].to(device, non_blocking=True)
            metas = batch["metainfo"]  # list of dicts

            B = imgs.shape[0]

            # Extract visual tokens from frozen CLRerNet.
            visual_tokens_per_img = extract_visual_tokens(clrernet_model, imgs)

            # Decode up to max_lanes sequences per image with EOS.
            x_tokens_all, y_tokens_all = autoregressive_decode_tokens(
                lanelm_model=lanelm_model,
                visual_tokens=visual_tokens_per_img,
                num_points=num_points,
                nbins_x=nbins_x,
                temperature=args.temperature,
                max_lanes=1,
            )

            x_tokens_np = x_tokens_all.cpu().numpy()  # (B, L, T)
            y_tokens_np = y_tokens_all.cpu().numpy()  # (B, L, T)

            for i in range(B):
                lanes: List[Lane] = []
                # Iterate over decoded lanes for this image
                for l in range(x_tokens_np.shape[1]):
                    xt = x_tokens_np[i, l]
                    yt = y_tokens_np[i, l]

                    # Decode tokens into 800x320 coordinate space.
                    coords_resized = tokenizer.decode_single_lane(xt, yt)
                    if coords_resized.shape[0] == 0:
                        continue

                    lane = coords_to_lane(
                        coords_resized=coords_resized,
                        ori_img_w=ori_img_w,
                        ori_img_h=ori_img_h,
                        crop_bbox=tuple(crop_bbox),
                        img_w=img_w,
                        img_h=img_h,
                    )
                    if lane.points.shape[0] > 0:
                        lanes.append(lane)

                sub_img_name = metas[i]["sub_img_name"]
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

    print("CULane evaluation results (LaneLM head):")
    for k in sorted(results.keys()):
        v = results[k]
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    if args.work_dir is not None:
        out_path = os.path.join(args.work_dir, "lanelm_culane_metrics.txt")
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
