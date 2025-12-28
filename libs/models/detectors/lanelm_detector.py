import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from mmengine.logging import MMLogger
from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS

from libs.models.lanelm import LaneLMModel, LaneTokenizer, LaneTokenizerConfig
from libs.models.detectors.clrernet import CLRerNet
from libs.utils.lane_utils import Lane


def autoregressive_decode(
    lanelm_model,
    visual_tokens,
    tokenizer_cfg,
    max_lanes,
    temperature=0.0,
    use_presence_filter=True,
    presence_threshold=0.5,
    initial_x_tokens=None,
    initial_y_tokens=None,
):
    """Greedy decode LaneLM (per-lane) with lane_indices support.
    
    CRITICAL: This EXACTLY matches train_lanelm_v4_fixed.py visual_first_decode() function (lines 113-141):
    - Uses ABSOLUTE tokenization (x_mode='absolute')
    - Uses padding token (0) for first position, NOT BOS tokens
    - Recreates x_in at each step with EXACT same logic as training
    - Matches training's visual_first_decode line-by-line
    
    V6 FIX: Added presence head filtering to prevent always predicting 4 lanes.
    V7: Added prompting strategy - initial_x_tokens and initial_y_tokens for first 2 keypoints from CLRNet.
    
    Args:
        initial_x_tokens: Optional (B, max_lanes, num_initial_points) tensor of initial X tokens from CLRNet.
        initial_y_tokens: Optional (B, max_lanes, num_initial_points) tensor of initial Y tokens from CLRNet.
    """
    model_device = next(lanelm_model.parameters()).device
    B, _, _ = visual_tokens.shape
    T = tokenizer_cfg.num_steps
    pad_token_x = tokenizer_cfg.pad_token_x

    min_valid_tokens = 2  # need at least 2 points to form a lane
    all_x = []
    all_y = []
    all_presence_scores = []  # V6: Track presence scores for filtering
    
    for lane_idx in range(max_lanes):
        # Initialize output tokens - EXACT match to training
        x_out = torch.zeros(B, T, dtype=torch.long, device=model_device)
        
        # Y sabit satır (t) kullan - training ile aynı (teacher forcing y = t)
        y_fixed = torch.arange(T, dtype=torch.long, device=model_device).unsqueeze(0).repeat(B, 1)

        # V7 (fixed-Y prompting FIX):
        # `initial_x_tokens` is expected to be a FULL-LENGTH tensor shaped (B, max_lanes, T),
        # containing x tokens placed at their correct y-step indices (t). This matches our
        # fixed-y formulation and avoids the old (incorrect) behavior of overwriting y tokens
        # or forcing prompt tokens into positions t=0..k-1.
        prompt_mask = torch.zeros(B, T, dtype=torch.bool, device=model_device)
        if initial_x_tokens is not None:
            if initial_x_tokens.dim() == 3 and initial_x_tokens.shape[-1] == T:
                init_full = initial_x_tokens[:, lane_idx, :].clone().to(model_device)  # (B,T)
                is_prompt = init_full != pad_token_x
                x_out = torch.where(is_prompt, init_full, x_out)
                prompt_mask = is_prompt
        
        lane_indices = torch.full((B,), lane_idx, dtype=torch.long, device=model_device)

        # EOS / padding stop:
        # Paper: (x,y) is an EOS token when x=0 or y=T. We currently use fixed y=t,
        # so use x=pad_token_x as EOS. To avoid stopping on a single noisy 0, require
        # `eos_consecutive` consecutive 0s. Keep defaults conservative.
        enable_eos_stop = getattr(tokenizer_cfg, "enable_eos_stop", False)
        eos_consecutive = int(getattr(tokenizer_cfg, "eos_consecutive", 2))
        eos_min_t = int(getattr(tokenizer_cfg, "eos_min_t", 5))
        eos_min_valid = int(getattr(tokenizer_cfg, "eos_min_valid", 2))
        eos_consecutive = max(1, eos_consecutive)
        eos_min_t = max(0, eos_min_t)
        eos_min_valid = max(0, eos_min_valid)
        alive = torch.ones(B, dtype=torch.bool, device=model_device)
        eos_run = torch.zeros(B, dtype=torch.long, device=model_device)
        valid_seen = torch.zeros(B, dtype=torch.long, device=model_device)

        # Visual-first autoregressive decode - EXACT match to training visual_first_decode (lines 125-138)
        presence_logits = None  # V6: Initialize presence logits
        for t in range(T):
            if enable_eos_stop and not alive.any():
                break
            # Skip prediction for prompted positions (per-sample).
            # If some samples have prompt at this t and others don't, we still run the model
            # and then keep prompted tokens unchanged.
            step_prompt = prompt_mask[:, t] if prompt_mask is not None else None
            
            x_in = torch.full_like(x_out, pad_token_x)  # Initialize with padding
            # CRITICAL FIX: Match training's shift-right logic (x_in_tf[:, 1:] = x_tokens[:, :-1])
            if t == 0:
                x_in[:, 0] = pad_token_x
            else:
                # t>0: Shift right (same as training)
                # x_in[:, 1:t+1] = x_out[:, 0:t] means x_in[t] = x_out[t-1]
                x_in[:, 1:t+1] = x_out[:, 0:t]
                # CRITICAL FIX (V24): 0-kp modunda BOS/pad hizası
                # Training (num_pseudo_points=0) için x_in_tf[:,0] her zaman pad_token_x idi.
                x_in[:, 0] = pad_token_x
                # Remaining positions [t+1, T-1] are already padding (from full_like)

            # V6: Get presence logits on final step (after full decode)
            if t == T - 1 and use_presence_filter:
                logits_x, _, presence_logits = lanelm_model(
                    visual_tokens,
                    x_in,
                    y_fixed,
                    lane_indices=lane_indices,
                    return_presence=True,
                )
            else:
                logits_x, _ = lanelm_model(
                    visual_tokens,
                    x_in,
                    y_fixed,
                    lane_indices=lane_indices,
                )

            # EXACT match to training line 137
            pred_x = torch.argmax(logits_x[:, t, :], dim=-1)
            pred_x = pred_x.clamp(0, lanelm_model.nbins_x - 1)
            if step_prompt is not None and step_prompt.any():
                pred_x = torch.where(step_prompt, x_out[:, t], pred_x)
            if enable_eos_stop:
                # Force already-finished samples to keep padding
                pred_x = torch.where(alive, pred_x, torch.full_like(pred_x, pad_token_x))
            x_out[:, t] = pred_x  # by default no early stopping unless eos enabled

            # Update EOS run and alive mask
            if enable_eos_stop:
                is_eos = (pred_x == pad_token_x)
                valid_seen = torch.where(alive & (~is_eos), valid_seen + 1, valid_seen)

                # Only allow EOS stopping after some minimum progress
                allow_eos = (t >= eos_min_t) & (valid_seen >= eos_min_valid)
                eos_run = torch.where(
                    alive & allow_eos & is_eos,
                    eos_run + 1,
                    torch.where(alive, torch.zeros_like(eos_run), eos_run),
                )
                alive = alive & (eos_run < eos_consecutive)
            
            # DEBUG: Log logits for first sample, first lane, first few timesteps
            if lane_idx == 0 and t < 3 and B > 0:
                logits_sample = logits_x[0, t, :].cpu().numpy()
                logits_max = logits_sample.max()
                logits_min = logits_sample.min()
                logits_mean = logits_sample.mean()
                probs = torch.softmax(logits_x[0, t, :], dim=-1).cpu().numpy()
                top5_probs = probs.argsort()[-5:][::-1]
                top5_values = probs[top5_probs]
                print(f"[DEBUG] Lane {lane_idx}, t={t}: pred_x={pred_x[0].item()}, "
                      f"logits_range=[{logits_min:.2f}, {logits_max:.2f}], mean={logits_mean:.2f}, "
                      f"top5_tokens={top5_probs.tolist()}, top5_probs={top5_values.tolist()}")

        # V6: Calculate presence score for this lane
        if use_presence_filter and presence_logits is not None:
            # presence_logits shape: (B, 1)
            presence_probs = torch.sigmoid(presence_logits).squeeze(-1)  # (B,)
            all_presence_scores.append(presence_probs.cpu())
            # DEBUG: Log presence scores for first sample, first few lanes
            if lane_idx < 4 and B > 0:
                print(f"[DEBUG] Lane {lane_idx}: presence_prob={presence_probs[0].item():.4f}, "
                      f"presence_logit={presence_logits[0, 0].item():.4f}")
        else:
            # If not using presence filter, assume all lanes are present
            all_presence_scores.append(torch.ones(B, dtype=torch.float32))
            if lane_idx < 4 and B > 0:
                print(f"[DEBUG] Lane {lane_idx}: presence_filter DISABLED (use_presence_filter={use_presence_filter})")

        # Check if this lane produced any non-padding tokens
        valid_mask = (x_out != pad_token_x)
        valid_counts = valid_mask.sum(dim=1)
        # If ALL samples in the batch have < min_valid_tokens, stop decoding
        if (valid_counts < min_valid_tokens).all():
            break

        all_x.append(x_out.cpu())
        all_y.append(y_fixed.cpu())

    # V6: Filter lanes based on presence scores
    if use_presence_filter and len(all_presence_scores) > 0 and len(all_x) > 0:
        presence_scores = torch.stack(all_presence_scores, dim=1)  # (B, max_lanes)
        presence_mask = presence_scores > presence_threshold  # (B, max_lanes)
        
        # DEBUG: Log presence filtering for first sample
        if B > 0:
            print(f"[DEBUG] Presence filtering: threshold={presence_threshold}")
            for lane_idx in range(min(4, presence_scores.shape[1])):
                print(f"[DEBUG]   Lane {lane_idx}: score={presence_scores[0, lane_idx].item():.4f}, "
                      f"pass={presence_mask[0, lane_idx].item()}")
            print(f"[DEBUG]   Total lanes passing filter: {presence_mask[0].sum().item()}/4")
        
        # For each sample in batch, filter out lanes with low presence
        filtered_x = []
        filtered_y = []
        for b in range(B):
            sample_x = []
            sample_y = []
            for lane_idx in range(len(all_x)):
                if presence_mask[b, lane_idx].item():
                    sample_x.append(all_x[lane_idx][b])
                    sample_y.append(all_y[lane_idx][b])
            if sample_x:
                filtered_x.append(torch.stack(sample_x, dim=0))
                filtered_y.append(torch.stack(sample_y, dim=0))
            else:
                # If no lanes pass presence filter, use the lane with highest presence score as fallback
                best_lane_idx = presence_scores[b].argmax().item()
                filtered_x.append(all_x[best_lane_idx][b].unsqueeze(0))
                filtered_y.append(all_y[best_lane_idx][b].unsqueeze(0))
        
        # Pad to same length for batching
        if len(filtered_x) > 0:
            max_lanes_in_batch = max(len(fx) for fx in filtered_x)
            x_tokens_all = []
            y_tokens_all = []
            for b in range(B):
                fx = filtered_x[b]
                fy = filtered_y[b]
                if len(fx) < max_lanes_in_batch:
                    # Pad with padding tokens
                    pad_x = torch.full((max_lanes_in_batch - len(fx), T), pad_token_x, dtype=torch.long)
                    pad_y = torch.full((max_lanes_in_batch - len(fy), T), T, dtype=torch.long)
                    fx = torch.cat([fx, pad_x], dim=0)
                    fy = torch.cat([fy, pad_y], dim=0)
                x_tokens_all.append(fx)
                y_tokens_all.append(fy)
            
            x_tokens_all = torch.stack(x_tokens_all, dim=0)  # (B, max_lanes_in_batch, T)
            y_tokens_all = torch.stack(y_tokens_all, dim=0)  # (B, max_lanes_in_batch, T)
        else:
            # Fallback: create empty batch with at least one padding lane per sample
            pad_lane_x = torch.full((1, T), pad_token_x, dtype=torch.long)
            pad_lane_y = torch.full((1, T), T, dtype=torch.long)
            x_tokens_all = pad_lane_x.unsqueeze(0).expand(B, -1, -1)  # (B, 1, T)
            y_tokens_all = pad_lane_y.unsqueeze(0).expand(B, -1, -1)  # (B, 1, T)
    elif len(all_x) > 0:
        # No presence filtering or presence scores not available, return all lanes
        x_tokens_all = torch.stack(all_x, dim=1)  # (B, max_lanes, T)
        y_tokens_all = torch.stack(all_y, dim=1)  # (B, max_lanes, T)
    else:
        # No lanes decoded at all, create max_lanes padding lanes
        pad_lane_x = torch.full((max_lanes, T), pad_token_x, dtype=torch.long)
        pad_lane_y = torch.full((max_lanes, T), T, dtype=torch.long)
        x_tokens_all = pad_lane_x.unsqueeze(0).expand(B, -1, -1)  # (B, max_lanes, T)
        y_tokens_all = pad_lane_y.unsqueeze(0).expand(B, -1, -1)  # (B, max_lanes, T)
    
    return x_tokens_all, y_tokens_all


def hallucination_removal(x_coords, y_coords, N=10):
    """Hallucination Removal (HR) algorithm from PDF Section 3.5, Algorithm 1.
    
    Filters out points with offsets of adjacent x-coordinates exceeding twice the 85th percentile.
    
    Args:
        x_coords: (N,) array of x-coordinates
        y_coords: (N,) array of y-coordinates (must match x_coords length)
        N: Minimum number of points required to apply HR (default: 10)
    
    Returns:
        x_coords_filtered: Filtered x-coordinates
        y_coords_filtered: Filtered y-coordinates (matching indices)
    """
    if len(x_coords) <= N:
        return x_coords, y_coords
    
    # Calculate absolute differences between adjacent x-coordinates
    diff = np.abs(np.diff(x_coords))
    
    # Threshold: 2 * 85th percentile
    theta = 2 * np.percentile(diff, 85)
    
    # Find first index where diff > theta.
    # NOTE: np.argmax is ambiguous because it returns 0 both when:
    #   - the first element is True, and
    #   - there are no True elements.
    # We therefore explicitly check for any violations.
    viol = np.where(diff > theta)[0]
    if viol.size > 0:
        p = int(viol[0])
        # Truncate at p+1 (keep points up to and including index p)
        x_coords = x_coords[: p + 1]
        y_coords = y_coords[: p + 1]
    
    return x_coords, y_coords


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
        # Optional postprocess smoothing in tokenizer decode.
        # Default False for evaluation (strict IoU can be sensitive to geometry changes).
        self.decode_smooth = bool(decode_cfg.get("smooth", False))
        # V36: EOS-stop config is specified in decode_cfg, but autoregressive_decode reads it from tokenizer_cfg.
        # Propagate here to avoid config wiring mistakes.
        self.tokenizer_cfg.enable_eos_stop = bool(decode_cfg.get("enable_eos_stop", False))
        self.tokenizer_cfg.eos_consecutive = int(decode_cfg.get("eos_consecutive", 2))
        self.tokenizer_cfg.eos_min_t = int(decode_cfg.get("eos_min_t", 5))
        self.tokenizer_cfg.eos_min_valid = int(decode_cfg.get("eos_min_valid", 2))
        # Optional Hallucination Removal (HR) postprocess.
        # Default True to match paper, but allow ablation.
        self.use_hr = bool(decode_cfg.get("use_hr", True))
        self.hr_min_points = int(decode_cfg.get("hr_min_points", 10))
        # Optional decode-time stabilization: keep only one contiguous run of non-pad tokens.
        # This reduces over-extended lanes caused by scattered non-pad tokens.
        self.use_contiguous_run = bool(decode_cfg.get("contiguous_run", False))
        self.contiguous_min_len = int(decode_cfg.get("contiguous_min_len", 2))
        # V6: Presence head filtering parameters
        self.use_presence_filter = decode_cfg.get("use_presence_filter", True)
        self.presence_threshold = decode_cfg.get("presence_threshold", 0.3)  # Lower threshold (was 0.5, too aggressive)
        
        # V7: Prompting Strategy - Build CLRerNet model for keypoint extraction
        # PDF (line 497-499): "A regression network is employed to provide the two initial keypoints"
        self.use_prompting = decode_cfg.get("use_prompting", True)  # Enable/disable prompting
        if self.use_prompting and clrernet_checkpoint:
            # Build CLRerNet model (with head) for keypoint extraction
            from libs.models.dense_heads.clrernet_head import CLRerHead
            from libs.core.anchor import CLRerNetAnchorGenerator
            
            # Build bbox_head config dict (CLRerHead) - same config as base_clrernet.py
            # CRITICAL: Store test_cfg separately to preserve it after MODELS.build
            test_cfg_dict = dict(
                conf_threshold=0.41,
                use_nms=True,
                as_lanes=True,
                extend_bottom=True,
                nms_thres=50,
                nms_topk=4,
                ori_img_w=self.ori_img_w,
                ori_img_h=self.ori_img_h,
                cut_height=self.crop_bbox[1],  # y_min from crop_bbox
            )
            
            bbox_head_cfg = dict(
                type="CLRerHead",
                anchor_generator=dict(
                    type="CLRerNetAnchorGenerator",
                    num_priors=192,
                    num_points=72,
                ),
                img_w=self.img_w,
                img_h=self.img_h,
                prior_feat_channels=64,
                fc_hidden_dim=64,
                num_fc=2,
                refine_layers=3,
                sample_points=36,
                attention=dict(type="ROIGather"),
                loss_cls=dict(type="KorniaFocalLoss", alpha=0.25, gamma=2, loss_weight=2.0),
                loss_bbox=dict(type="SmoothL1Loss", reduction="none", loss_weight=0.2),
                loss_iou=dict(
                    type="LaneIoULoss",
                    lane_width=7.5 / 800,
                    loss_weight=4.0,
                ),
                loss_seg=dict(
                    type="CLRNetSegLoss",
                    loss_weight=1.0,
                    num_classes=5,
                    ignore_label=255,
                    bg_weight=0.4,
                ),
                test_cfg=test_cfg_dict,  # Use the stored dict
            )

            # Build CLRerNet model (backbone + neck + head)
            # Note: SingleStageDetector expects bbox_head to be a dict (it calls bbox_head.update())
            # So we pass bbox_head_cfg as dict, and MODELS.build will convert nested test_cfg to ConfigDict
            self.clrernet_model = MODELS.build(dict(
                type='CLRerNet',
                backbone=backbone,
                neck=neck,
                bbox_head=bbox_head_cfg,  # Use config dict (test_cfg inside will be converted to ConfigDict)
                test_cfg=dict(),  # CLRerNet level test_cfg (not used, bbox_head uses its own)
            ))
            
            # After build, ensure bbox_head.test_cfg supports attribute access (test_cfg.as_lanes)
            # CRITICAL: CLRerHead.predict() uses test_cfg.as_lanes, but ConfigDict attribute access may fail
            # Solution: Create a wrapper class that supports both dict and attribute access
            # Use the stored test_cfg_dict instead of bbox_head_cfg.get() because MODELS.build may have modified it
            original_test_cfg = test_cfg_dict
            
            class TestCfgWrapper:
                """Wrapper that supports both dict-style and attribute-style access for test_cfg."""
                def __init__(self, d):
                    if not isinstance(d, dict):
                        raise TypeError(f"TestCfgWrapper requires dict, got {type(d)}")
                    # Store all keys as attributes for attribute access (test_cfg.as_lanes)
                    # Use __dict__.update() to ensure attributes are set correctly
                    self.__dict__.update(d)
                    # Also store as dict for dict access
                    self._dict = d.copy()
                
                def __getitem__(self, key):
                    return getattr(self, key)
                
                def get(self, key, default=None):
                    return getattr(self, key, default) if hasattr(self, key) else default
                
                def __contains__(self, key):
                    return hasattr(self, key) or key in self._dict
                
                def keys(self):
                    return self._dict.keys()
                
                def __getattr__(self, name):
                    # Fallback: if attribute not found, try dict
                    if name in self._dict:
                        return self._dict[name]
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
            # Replace test_cfg with wrapper that supports attribute access
            wrapped_cfg = TestCfgWrapper(original_test_cfg)
            self.clrernet_model.bbox_head.test_cfg = wrapped_cfg

            # Load CLRerNet head weights
            if clrernet_checkpoint:
                state = torch.load(clrernet_checkpoint, map_location="cpu")
                if "state_dict" in state:
                    state = state["state_dict"]
                head_state = {k.replace("bbox_head.", ""): v for k, v in state.items() if k.startswith("bbox_head.")}
                self.clrernet_model.bbox_head.load_state_dict(head_state, strict=False)
                print(f"✓ Loaded CLRerNet head weights from {clrernet_checkpoint}")
            
            # Freeze CLRerNet model
            for p in self.clrernet_model.parameters():
                p.requires_grad = False
            self.clrernet_model.eval()
        else:
            self.clrernet_model = None

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

        logger = MMLogger.get_current_instance()

        # V7: Prompting Strategy - Extract first 2 keypoints from CLRNet
        initial_x_tokens = None
        if self.use_prompting and self.clrernet_model is not None:
            with torch.no_grad():
                # CLRerNet predict (returns list of dicts with 'lanes' and 'scores')
                clrernet_results = self.clrernet_model.predict(imgs, batch_data_samples, rescale=False)
                
                # Extract first 2 keypoints from each CLRNet lane prediction
                batch_size = len(batch_data_samples)
                initial_x_list = []
                
                for b_idx, clr_result in enumerate(clrernet_results):
                    clr_lanes = clr_result.get("lanes", [])

                    # V42 FIX (analysis-driven):
                    # Training sorts GT lanes left->right and matches CLRNet lanes to those slots via Hungarian.
                    # Inference previously used raw CLRNet ordering, which can mismatch lane_slot ↔ lane_indices
                    # and cause prompting to *harm* predictions.
                    # Mitigation: sort CLRNet lanes left->right by bottom-most x (in resized space),
                    # which aligns with training's lane_slot ordering.
                    def _clr_lane_sort_key(clr_lane_obj) -> float:
                        try:
                            pts0 = clr_lane_obj.points if hasattr(clr_lane_obj, "points") else clr_lane_obj
                            if isinstance(pts0, torch.Tensor):
                                pts0 = pts0.detach().cpu().numpy()
                            pts0 = np.asarray(pts0, dtype=np.float32)
                            if pts0.ndim != 2 or pts0.shape[1] != 2 or len(pts0) < 2:
                                return float("inf")
                            # denorm to ori crop space (1640x590), then crop+resize to (800x320)
                            x_pix = pts0[:, 0] * self.ori_img_w
                            y_pix = pts0[:, 1] * self.ori_img_h
                            x_min, y_min, x_max, y_max = self.crop_bbox
                            x_pix = x_pix - x_min
                            y_pix = y_pix - y_min
                            x_scale = self.img_w / (x_max - x_min)
                            y_scale = self.img_h / (y_max - y_min)
                            x_res = x_pix * x_scale
                            y_res = y_pix * y_scale
                            # bottom-most point in resized space
                            idx = int(np.argmax(y_res))
                            return float(x_res[idx])
                        except Exception:
                            return float("inf")

                    if len(clr_lanes) > 1:
                        clr_lanes = sorted(clr_lanes, key=_clr_lane_sort_key)
                    
                    # DEBUG: Log CLRNet lane count and format
                    if b_idx == 0:
                        logger.info(f"[LaneLMDetector] CLRNet returned {len(clr_lanes)} lanes (after sort)")
                        if len(clr_lanes) > 0:
                            logger.info(f"[LaneLMDetector] First CLRNet lane type: {type(clr_lanes[0])}")
                            if hasattr(clr_lanes[0], 'points'):
                                points_debug = clr_lanes[0].points
                                logger.info(f"[LaneLMDetector] First CLRNet lane points type: {type(points_debug)}")
                                logger.info(f"[LaneLMDetector] First CLRNet lane points shape: {points_debug.shape if hasattr(points_debug, 'shape') else 'N/A'}")
                                if hasattr(points_debug, 'shape') and len(points_debug) >= 2:
                                    logger.info(f"[LaneLMDetector] First CLRNet lane points (first 2): {points_debug[:2]}")
                                    logger.info(f"[LaneLMDetector] First CLRNet lane points X range: [{points_debug[:, 0].min():.4f}, {points_debug[:, 0].max():.4f}]")
                                    logger.info(f"[LaneLMDetector] First CLRNet lane points Y range: [{points_debug[:, 1].min():.4f}, {points_debug[:, 1].max():.4f}]")
                    
                    sample_x_tokens = []
                    
                    # For each lane slot (max_lanes)
                    for lane_idx in range(self.max_lanes):
                        if lane_idx < len(clr_lanes) and clr_lanes[lane_idx] is not None:
                            # Get CLRNet lane points (normalized [0,1])
                            clr_lane = clr_lanes[lane_idx]
                            if hasattr(clr_lane, 'points'):
                                points = clr_lane.points  # (N, 2) in normalized [0,1]
                            else:
                                points = clr_lane  # (N, 2) tensor
                            
                            if isinstance(points, torch.Tensor):
                                points = points.cpu().numpy()
                            
                            # DEBUG: Log points before transformation
                            if b_idx == 0 and lane_idx == 0:
                                print(f"[DEBUG] Lane {lane_idx}: points shape={points.shape}, len={len(points)}")
                                if len(points) >= 2:
                                    print(f"[DEBUG] Lane {lane_idx}: first 2 points (normalized): {points[:2]}")
                            
                            if len(points) >= 2:
                                # Take first 2 keypoints (PDF uses first 2)
                                first_2_points = points[:2]  # (2, 2)
                                
                                # DEBUG: Log before transformation
                                if b_idx == 0 and lane_idx == 0:
                                    print(f"[DEBUG] Lane {lane_idx}: first_2_points (normalized): {first_2_points}")
                                
                                # Convert to resized space (800x320) for tokenization
                                # Points are in normalized [0,1], convert to resized pixel space
                                x_resized = first_2_points[:, 0] * self.ori_img_w  # Denormalize X
                                y_resized = first_2_points[:, 1] * self.ori_img_h  # Denormalize Y
                                
                                # DEBUG: Log after denormalization
                                if b_idx == 0 and lane_idx == 0:
                                    print(f"[DEBUG] Lane {lane_idx}: after denorm - x_resized={x_resized}, y_resized={y_resized}")
                                
                                # Apply crop and resize transformation (same as training)
                                x_min, y_min, x_max, y_max = self.crop_bbox
                                # Crop
                                x_resized = x_resized - x_min
                                y_resized = y_resized - y_min
                                # Resize
                                x_scale = self.img_w / (x_max - x_min)
                                y_scale = self.img_h / (y_max - y_min)
                                x_resized = x_resized * x_scale
                                y_resized = y_resized * y_scale
                                
                                # DEBUG: Log after crop/resize
                                if b_idx == 0 and lane_idx == 0:
                                    print(f"[DEBUG] Lane {lane_idx}: after crop/resize - x_resized={x_resized}, y_resized={y_resized}")
                                
                                # Clip to resized image bounds
                                x_resized = np.clip(x_resized, 0, self.img_w - 1)
                                y_resized = np.clip(y_resized, 0, self.img_h - 1)
                                
                                # DEBUG: Log after clipping
                                if b_idx == 0 and lane_idx == 0:
                                    print(f"[DEBUG] Lane {lane_idx}: after clip - x_resized={x_resized}, y_resized={y_resized}")
                                
                                # CRITICAL FIX: Direct tokenization for first 2 keypoints
                                # Fixed-Y prompting should place x tokens at their correct y-step indices (t),
                                # and keep y as implicit fixed grid. We therefore create a full-length
                                # (T,) prompt token sequence with sparse x tokens.
                                sample_ys = self.tokenizer._compute_sample_ys()  # T=40 positions
                                x_prompt_full = np.full(
                                    (self.tokenizer_cfg.num_steps,),
                                    self.tokenizer_cfg.pad_token_x,
                                    dtype=np.int64,
                                )
                                
                                # For each of the 2 keypoints, find closest sample_ys index
                                for kp_idx in range(2):
                                    y_kp = y_resized[kp_idx]
                                    x_kp = x_resized[kp_idx]
                                    
                                    # Find closest sample_ys index
                                    closest_t = np.argmin(np.abs(sample_ys - y_kp))
                                    
                                    # Tokenize x coordinate
                                    if 0 <= x_kp < self.img_w:
                                        # Quantize x to [1, nbins_x-1] (0 is padding)
                                        bin_idx = int(round(x_kp / (self.img_w - 1) * (self.tokenizer_cfg.nbins_x - 1)))
                                        bin_idx = max(1, min(self.tokenizer_cfg.nbins_x - 1, bin_idx))
                                        x_prompt_full[int(closest_t)] = bin_idx
                                
                                # DEBUG: Log tokens
                                if b_idx == 0 and lane_idx == 0:
                                    non_pad = np.where(x_prompt_full != self.tokenizer_cfg.pad_token_x)[0]
                                    logger.info(
                                        f"[LaneLMDetector] prompt lane0: prompt_t_indices={non_pad.tolist()} "
                                        f"prompt_x_tokens={x_prompt_full[non_pad].tolist()}"
                                    )
                                
                                sample_x_tokens.append(x_prompt_full)
                            else:
                                # Not enough points, use padding
                                sample_x_tokens.append(
                                    np.full((self.tokenizer_cfg.num_steps,), self.tokenizer_cfg.pad_token_x, dtype=np.int64)
                                )
                        else:
                            # No CLRNet lane for this slot, use padding
                            sample_x_tokens.append(
                                np.full((self.tokenizer_cfg.num_steps,), self.tokenizer_cfg.pad_token_x, dtype=np.int64)
                            )
                    
                    initial_x_list.append(np.stack(sample_x_tokens, axis=0))  # (max_lanes, T)
                
                if initial_x_list:
                    initial_x_tokens = torch.from_numpy(np.stack(initial_x_list, axis=0)).long()  # (B, max_lanes, T)
                    
                    # DEBUG: Log for first sample
                    if batch_size > 0:
                        logger.info(f"[LaneLMDetector] Prompting Strategy: Built sparse prompt tokens (T={initial_x_tokens.shape[2]})")
                        for lane_idx in range(min(4, self.max_lanes)):
                            non_pad = (initial_x_tokens[0, lane_idx] != self.tokenizer_cfg.pad_token_x).nonzero().view(-1)
                            logger.info(
                                f"[LaneLMDetector]   lane_slot={lane_idx}: prompt_t={non_pad.tolist()} "
                                f"prompt_x={initial_x_tokens[0, lane_idx][non_pad].tolist()}"
                            )

        # V6: Use presence head filtering from decode_cfg
        # V7: Pass initial keypoints as prompts
        x_tokens_all, y_tokens_all = autoregressive_decode(
            lanelm_model=self.lanelm.to(device),
            visual_tokens=visual_tokens,
            tokenizer_cfg=self.tokenizer_cfg,
            max_lanes=self.max_lanes,
            temperature=self.temperature,
            use_presence_filter=self.use_presence_filter,
            presence_threshold=self.presence_threshold,
            initial_x_tokens=initial_x_tokens,
            initial_y_tokens=None,
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

            # DEBUG: Log token statistics for first sample in first batch
            if batch_num == 1 and i == 0:
                print(f"\n[DEBUG] Sample {i}: x_tokens_all shape: {x_tokens_all.shape}")
                print(f"[DEBUG] Sample {i}: x_tok shape: {x_tok.shape}, y_tok shape: {y_tok.shape}")
                print(f"[DEBUG] Sample {i}: x_tok min/max: [{x_tok.min()}, {x_tok.max()}]")
                print(f"[DEBUG] Sample {i}: pad_token_x: {self.tokenizer_cfg.pad_token_x}")

            for l in range(x_tok.shape[0]):
                # CRITICAL FIX: Filter out padding lanes BEFORE decoding
                # Check if this lane has any non-padding tokens
                x_tokens_lane = x_tok[l].copy()
                pad_token_x = self.tokenizer_cfg.pad_token_x
                non_pad_mask = (x_tokens_lane != pad_token_x) & (x_tokens_lane != 0)
                valid_count = non_pad_mask.sum()

                # Optional: enforce a single contiguous segment of non-pad tokens.
                # Choose the longest run; tie-breaker: pick the bottom-most (largest index).
                if self.use_contiguous_run and valid_count >= 2:
                    mask = (x_tokens_lane != pad_token_x) & (x_tokens_lane != 0)
                    idxs = np.where(mask)[0]
                    if idxs.size > 0:
                        runs = []
                        start = int(idxs[0])
                        prev = int(idxs[0])
                        for j in idxs[1:]:
                            j = int(j)
                            if j == prev + 1:
                                prev = j
                            else:
                                runs.append((start, prev))
                                start = j
                                prev = j
                        runs.append((start, prev))
                        min_len = max(2, int(self.contiguous_min_len))
                        runs = [(a, b) for (a, b) in runs if (b - a + 1) >= min_len]
                        if runs:
                            runs.sort(key=lambda ab: ((ab[1] - ab[0] + 1), ab[1]))
                            a, b = runs[-1]
                            keep = np.zeros_like(mask, dtype=bool)
                            keep[a : b + 1] = True
                            x_tokens_lane[~keep] = pad_token_x
                            non_pad_mask = keep
                            valid_count = int(keep.sum())
                
                # DEBUG: Log for first sample
                if batch_num == 1 and i == 0:
                    print(f"[DEBUG] Lane {l}: valid tokens = {valid_count}/{len(x_tokens_lane)}, "
                          f"x_range=[{x_tokens_lane.min()}, {x_tokens_lane.max()}]")
                
                # Skip lanes that are all padding (no valid tokens)
                if valid_count < 2:  # Need at least 2 valid points for a lane
                    if batch_num == 1 and i == 0:
                        print(f"[DEBUG] Lane {l}: SKIPPED (valid_count < 2)")
                    continue
                
                coords_resized = self.tokenizer.decode_single_lane(
                    x_tokens_lane, y_tok[l], smooth=self.decode_smooth
                )
                
                # V7: Apply Hallucination Removal (HR) from PDF Section 3.5 (optional).
                # IMPORTANT: HR removes a suffix of the sequence after an abnormal x-jump.
                # To avoid truncating the lane bottom by accident, apply HR in bottom->top
                # order (descending y), then restore ordering.
                if self.use_hr and coords_resized.shape[0] > 0:
                    # Sort by y descending (bottom->top)
                    order = np.argsort(coords_resized[:, 1])[::-1]
                    coords_bt = coords_resized[order]
                    x_coords = coords_bt[:, 0]
                    y_coords = coords_bt[:, 1]
                    x_coords_hr, y_coords_hr = hallucination_removal(
                        x_coords, y_coords, N=self.hr_min_points
                    )
                    coords_bt_hr = np.stack([x_coords_hr, y_coords_hr], axis=1).astype(np.float32)
                    # Restore y-ascending order for downstream Lane(points) (expects increasing y)
                    coords_resized = coords_bt_hr[np.argsort(coords_bt_hr[:, 1])]
                
                # DEBUG: Log for first sample
                if batch_num == 1 and i == 0:
                    print(f"[DEBUG] Lane {l}: decoded coords shape: {coords_resized.shape}")
                    if coords_resized.shape[0] > 0:
                        print(f"[DEBUG] Lane {l}: coords X range: [{coords_resized[:, 0].min():.1f}, {coords_resized[:, 0].max():.1f}], "
                              f"Y range: [{coords_resized[:, 1].min():.1f}, {coords_resized[:, 1].max():.1f}]")
                
                # Additional check: decoded coords should have at least 2 points
                if coords_resized.shape[0] < 2:
                    if batch_num == 1 and i == 0:
                        print(f"[DEBUG] Lane {l}: SKIPPED (decoded coords < 2 points)")
                    continue
                
                lane = coords_to_lane_normalized(
                    coords_resized=coords_resized,
                    tokenizer_cfg=self.tokenizer_cfg,
                    crop_bbox=self.crop_bbox,
                    img_w=self.img_w,
                    img_h=self.img_h,
                    ori_img_w=self.ori_img_w,
                    ori_img_h=self.ori_img_h,
                )
                
                # DEBUG: Log for first sample
                if batch_num == 1 and i == 0:
                    if lane is not None:
                        print(f"[DEBUG] Lane {l}: Lane created! Points: {lane.points.shape[0]}")
                    else:
                        print(f"[DEBUG] Lane {l}: SKIPPED (coords_to_lane_normalized returned None)")
                
                if lane is not None and lane.points is not None and lane.points.shape[0] >= 2:
                    lanes_pred.append(lane)
            
            # DEBUG: Log final count for first sample
            if batch_num == 1 and i == 0:
                print(f"[DEBUG] Sample {i}: Final lanes_pred count: {len(lanes_pred)}\n")

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
