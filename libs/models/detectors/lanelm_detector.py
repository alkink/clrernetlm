import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS

from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig
from libs.utils.lane_utils import Lane


def autoregressive_decode(lanelm_model, visual_tokens, tokenizer_cfg, max_lanes, temperature=0.0):
    """Greedy decode LaneLM (per-lane) with lane_indices support.
    
    CRITICAL: This EXACTLY matches train_lanelm_v4_fixed.py visual_first_decode() function (lines 113-141):
    - Uses ABSOLUTE tokenization (x_mode='absolute')
    - Uses padding token (0) for first position, NOT BOS tokens
    - Recreates x_in at each step with EXACT same logic as training
    - Matches training's visual_first_decode line-by-line
    """
    model_device = next(lanelm_model.parameters()).device
    B, _, _ = visual_tokens.shape
    T = tokenizer_cfg.num_steps
    pad_token_x = tokenizer_cfg.pad_token_x

    all_x = []
    all_y = []
    for lane_idx in range(max_lanes):
        # Fixed y sequence (0..T-1) - EXACT match to training
        y_fixed = torch.arange(T, dtype=torch.long, device=model_device).unsqueeze(0).expand(B, -1)
        
        # Initialize output tokens - EXACT match to training
        x_out = torch.zeros(B, T, dtype=torch.long, device=model_device)
        
        lane_indices = torch.full((B,), lane_idx, dtype=torch.long, device=model_device)

        # Visual-first autoregressive decode - EXACT match to training visual_first_decode (lines 125-138)
        for t in range(T):
            x_in = torch.zeros_like(x_out)
            if t > 0:
                x_in[:, 1:t+1] = x_out[:, :t]
                x_in[:, 0] = x_out[:, 0]  # Keep first predicted token - EXACT match to training line 129

            logits_x, _ = lanelm_model(
                visual_tokens,
                x_in,
                y_fixed,
                lane_indices=lane_indices,
            )

            # EXACT match to training line 137
            pred_x = torch.argmax(logits_x[:, t, :], dim=-1)
            pred_x = pred_x.clamp(0, lanelm_model.nbins_x - 1)
            x_out[:, t] = pred_x

        all_x.append(x_out.cpu())
        all_y.append(y_fixed.cpu())

    x_tokens_all = torch.stack(all_x, dim=1)
    y_tokens_all = torch.stack(all_y, dim=1)
    return x_tokens_all, y_tokens_all


def coords_to_lane_normalized(coords_resized, tokenizer_cfg, crop_bbox, img_w, img_h, ori_img_w, ori_img_h):
    """Convert decoded coords to normalized Lane (0..1).

    Notes:
    - decode_single_lane already de-quantizes x to resized pixel space [0, img_w).
    - CULaneMetric expects normalized coords in [0, 1). Values outside this
      range cause the line to be dropped. We therefore clamp to [0, 1).
    """
    if coords_resized.size == 0:
        return None

    xs = coords_resized[:, 0]
    ys = coords_resized[:, 1]
    x_min, y_min, x_max, y_max = crop_bbox

    # Clip to resized image bounds
    xs = np.clip(xs, 0.0, float(img_w - 1))
    ys = np.clip(ys, 0.0, float(img_h - 1))

    # Map resized x back to original-crop coordinates then normalize to [0,1)
    # Use same normalization as vis_lanelm_outputs_v2.py and test_lanelm_culane_v2.py
    x_scale = float(ori_img_w) / float(img_w)
    y_scale = float(y_max - y_min) / float(img_h)
    
    x_orig = xs * x_scale
    y_orig = ys * y_scale + float(y_min)
    
    x_norm = x_orig / float(ori_img_w)
    y_norm = y_orig / float(ori_img_h)

    # Final clamp to [0, 1)
    x_norm = np.clip(x_norm, 0.0, 0.999999)
    y_norm = np.clip(y_norm, 0.0, 0.999999)

    points = np.stack([x_norm, y_norm], axis=1).astype(np.float32)

    # Sort by y (primary) and x (secondary for same y)
    if len(points) > 1:
        # Sort by Y first, then by X for same Y values
        sort_idx = np.lexsort((points[:, 0], points[:, 1]))
        points = points[sort_idx]
        
        # Remove duplicate Y values (keep first occurrence, average X if needed)
        unique_y_mask = np.concatenate([[True], np.diff(points[:, 1]) > 1e-6])
        points = points[unique_y_mask]
        
        # Ensure Y is strictly increasing (required for spline)
        if len(points) > 1:
            # Check if Y is increasing
            y_diff = np.diff(points[:, 1])
            if (y_diff <= 0).any():
                # Remove points where Y doesn't increase
                keep = np.concatenate([[True], y_diff > 1e-6])
                points = points[keep]

    if len(points) < 2:
        return None
    
    # Final check: ensure Y is strictly increasing
    if len(points) > 1 and (np.diff(points[:, 1]) <= 0).any():
        return None  # Cannot create Lane with non-increasing Y
    
    return Lane(points=points)


@MODELS.register_module()
class LaneLMDetector(BaseDetector):
    """Inference-only LaneLM wrapper usable with MMDet runner."""
    
    # Class-level batch counter for progress tracking
    _batch_counter = 0
    _total_batches = None  # Will be set when first batch arrives

    def __init__(
        self,
        backbone,
        neck,
        lanelm_cfg,
        tokenizer_cfg,
        decode_cfg,
        clrernet_checkpoint=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Build backbone/neck (using registry)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)

        # Load CLRerNet weights if provided
        if clrernet_checkpoint:
            state = torch.load(clrernet_checkpoint, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            bb_state = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
            nk_state = {k.replace("neck.", ""): v for k, v in state.items() if k.startswith("neck.")}
            self.backbone.load_state_dict(bb_state, strict=False)
            self.neck.load_state_dict(nk_state, strict=False)
            print(f"✓ Loaded CLRerNet weights from {clrernet_checkpoint}")

        # Freeze backbone/neck
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.neck.parameters():
            p.requires_grad = False

        # Build LaneLM
        self.lanelm = LaneLMModel(
            nbins_x=lanelm_cfg.get("nbins_x", 200),
            max_y_tokens=lanelm_cfg.get("max_y_tokens", 41),
            embed_dim=lanelm_cfg.get("embed_dim", 256),
            num_layers=lanelm_cfg.get("num_layers", 4),
            num_heads=lanelm_cfg.get("num_heads", 8),
            ffn_dim=lanelm_cfg.get("ffn_dim", 512),
            max_seq_len=lanelm_cfg.get("max_seq_len", 160),
            visual_in_dim=None,
            visual_in_channels=lanelm_cfg.get("visual_in_channels", (64,)),
        )
        self.lanelm_ckpt_path = lanelm_cfg.get("ckpt_path", None)
        # LaneLM weights will be loaded in init_weights() after MMEngine loads backbone

        self.tokenizer_cfg = LaneTokenizerConfig(**tokenizer_cfg)
        self.tokenizer = LaneTokenizer(self.tokenizer_cfg)

        self.max_lanes = decode_cfg.get("max_lanes", 4)
        self.temperature = decode_cfg.get("temperature", 0.0)
        self.crop_bbox = decode_cfg.get("crop_bbox", (0, 270, 1640, 590))
        self.ori_img_w = decode_cfg.get("ori_img_w", 1640)
        self.ori_img_h = decode_cfg.get("ori_img_h", 590)
        self.img_w = decode_cfg.get("img_w", 800)
        self.img_h = decode_cfg.get("img_h", 320)

    def init_weights(self):
        """Load LaneLM weights after backbone/neck are initialized."""
        super().init_weights()
        self._load_lanelm_weights()
    
    def _load_lanelm_weights(self):
        """Load LaneLM checkpoint."""
        if hasattr(self, '_lanelm_loaded') and self._lanelm_loaded:
            return
        if self.lanelm_ckpt_path:
            state = torch.load(self.lanelm_ckpt_path, map_location="cpu")
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            missing, unexpected = self.lanelm.load_state_dict(state, strict=False)
            print(f"✓ Loaded LaneLM weights from {self.lanelm_ckpt_path}")
            print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            self._lanelm_loaded = True

    def loss(self, *args, **kwargs):
        raise NotImplementedError("LaneLMDetector is inference-only")

    def extract_feat(self, imgs):
        feats = self.backbone(imgs)
        feats = self.neck(feats)
        # Return all FPN levels for Full FPN, or just P5
        visual_channels = self.lanelm.visual_encoder.in_channels if self.lanelm.visual_encoder else (64,)
        if len(visual_channels) == 3:
            # Full FPN - return P3, P4, P5
            return feats
        else:
            # P5 only
            if isinstance(feats, (tuple, list)):
                feats = (feats[-1],)
            return feats

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        # Ensure LaneLM weights are loaded
        self._load_lanelm_weights()
        
        # Normalize input
        if isinstance(batch_inputs, list):
            imgs = batch_inputs[0] if len(batch_inputs) == 1 else torch.stack(batch_inputs)
        else:
            imgs = batch_inputs
        # Normalize to [0, 1] range to match training
        # Training uses images in [0, 1] range (train_lanelm_v4_fixed.py)
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0  # Normalize to [0, 1] range
        elif imgs.max() > 1.0:
            # If already float but in [0, 255] range, normalize
            imgs = imgs / 255.0

        device = imgs.device
        feats = self.extract_feat(imgs)
        visual_tokens = self.lanelm.encode_visual_tokens(feats)

        x_tokens_all, y_tokens_all = autoregressive_decode(
            lanelm_model=self.lanelm.to(device),
            visual_tokens=visual_tokens,
            tokenizer_cfg=self.tokenizer_cfg,
            max_lanes=self.max_lanes,
            temperature=self.temperature,
        )

        # Track batch progress (class-level counter)
        LaneLMDetector._batch_counter += 1
        batch_num = LaneLMDetector._batch_counter
        
        results = []
        batch_size = len(batch_data_samples)
        # Log progress: first sample, every 10% of batch, and last sample
        log_indices = set([0])
        if batch_size > 1:
            log_interval = max(1, batch_size // 10)  # Every 10%
            for idx in range(log_interval, batch_size, log_interval):
                log_indices.add(idx)
            log_indices.add(batch_size - 1)  # Last sample
        
        # Log batch-level progress every 10 batches or first batch
        log_batch_progress = (batch_num == 1 or batch_num % 10 == 0)
        
        for i, data_sample in enumerate(batch_data_samples):
            lanes_pred = []
            x_tok = x_tokens_all[i].numpy()
            y_tok = y_tokens_all[i].numpy()

            for l in range(x_tok.shape[0]):
                coords_resized = self.tokenizer.decode_single_lane(x_tok[l], y_tok[l], smooth=True)
                lane = coords_to_lane_normalized(
                    coords_resized=coords_resized,
                    tokenizer_cfg=self.tokenizer_cfg,
                    crop_bbox=self.crop_bbox,
                    img_w=self.img_w,
                    img_h=self.img_h,
                    ori_img_w=self.ori_img_w,
                    ori_img_h=self.ori_img_h,
                )
                if lane is not None and lane.points is not None and lane.points.shape[0] >= 2:
                    lanes_pred.append(lane)

            meta = getattr(data_sample, "metainfo", data_sample)
            sub_name = meta.get("sub_img_name") or meta.get("filename") or meta.get("img_path") or ""
            sub_name = str(Path(sub_name)).lstrip("/")

            # Progress logging: log at specific intervals
            if (i in log_indices or log_batch_progress) and lanes_pred:
                lp = lanes_pred[0].points
                sample_progress_pct = (i + 1) / batch_size * 100
                print(f"[LaneLMDetector] Batch #{batch_num} | Sample {i+1}/{batch_size} ({sample_progress_pct:.1f}%) | "
                      f"lane0 X[{lp[:,0].min():.3f},{lp[:,0].max():.3f}] "
                      f"Y[{lp[:,1].min():.3f},{lp[:,1].max():.3f}] points={lp.shape[0]}")

            results.append({"lanes": lanes_pred, "metainfo": {"sub_img_name": sub_name}})
        return results

    def _forward(self, *args, **kwargs):
        raise NotImplementedError
