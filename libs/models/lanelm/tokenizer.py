from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class LaneTokenizerConfig:
    """Configuration for converting lane polylines to LaneLM-style token sequences.

    The design follows the LaneLM paper:
    - y is represented implicitly by the time-step index t in [0, T)
    - x is quantized to an integer token in [0, nbins)
      where 0 is reserved as a padding / "no lane" token.
    """

    img_w: int = 800
    img_h: int = 320
    num_steps: int = 40  # T in the paper
    nbins_x: int = 800   # vocabulary size for x (0 is padding)
    pad_token_x: int = 0
    pad_token_y: int = -1  # will be mapped to T internally
    # X tokenization mode:
    #   "absolute" - quantize absolute x in [0, img_w) as before.
    #   "relative" - quantize relative offsets Δx_t = x_t - x_{t-1}
    #                in pixel space, clipped to [-max_abs_dx, max_abs_dx].
    x_mode: str = "absolute"
    max_abs_dx: int = 64  # used only when x_mode == "relative"


class LaneTokenizer:
    """Utility to encode/decode lanes as integer token sequences.

    Lanes are expected as arrays of (x, y) points in pixel coordinates.
    The tokenizer:
      1. samples the lane at `num_steps` vertical positions between [0, img_h)
      2. quantizes x to [0, nbins_x), reserving 0 as "no lane"
      3. uses the step index t as y-token; a special value T is used as padding.
    """

    def __init__(self, config: Optional[LaneTokenizerConfig] = None) -> None:
        self.cfg = config or LaneTokenizerConfig()

    @property
    def T(self) -> int:
        return self.cfg.num_steps

    def _compute_sample_ys(self) -> np.ndarray:
        """Uniformly spaced y positions from top (0) to bottom (img_h)."""
        # Use linspace with endpoint=False to get T positions in [0, img_h)
        return np.linspace(0.0, float(self.cfg.img_h), num=self.cfg.num_steps, endpoint=False)

    def _fit_spline(self, points: np.ndarray):
        """Fit a 1D function x(y) using the provided lane points."""
        # Points are (x, y); ensure sorted by y
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2) with (x, y) coordinates.")
        # Sort by y ascending
        order = np.argsort(points[:, 1])
        points = points[order]

        from scipy.interpolate import InterpolatedUnivariateSpline

        # Require at least two points to define a lane
        if len(points) < 2:
            return None

        x = points[:, 0]
        y = points[:, 1]
        # Clamp y to valid range
        y = np.clip(y, 0.0, float(self.cfg.img_h - 1))

        # Remove duplicate y values to ensure strictly increasing
        # Keep the first x value for each unique y
        unique_y, unique_indices = np.unique(y, return_index=True)
        x = x[unique_indices]
        y = unique_y

        # Need at least 2 unique points after deduplication
        if len(y) < 2:
            return None

        # Spline over y -> x
        spline = InterpolatedUnivariateSpline(
            y,
            x,
            k=min(3, len(y) - 1),
        )
        return spline

    def encode_single_lane(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a single lane polyline into (x_tokens, y_tokens).

        Args:
            points: np.ndarray of shape (N, 2) with (x, y) in pixels.

        Returns:
            x_tokens: np.ndarray of shape (T,) with values in [0, nbins_x)
            y_tokens: np.ndarray of shape (T,) with values in [0, T] (T is padding)
        """
        sample_ys = self._compute_sample_ys()
        spline = self._fit_spline(points)

        # Initialize as padding (no lane)
        x_tokens = np.full(self.T, self.cfg.pad_token_x, dtype=np.int64)
        # y_token == T denotes padding / no lane at this step
        y_tokens = np.full(self.T, self.T, dtype=np.int64)

        if spline is None:
            return x_tokens, y_tokens

        # Evaluate spline at all sample_ys
        xs = spline(sample_ys)

        # Absolute x mode: identical to original behavior
        if self.cfg.x_mode == "absolute":
            for t in range(self.T):
                x = float(xs[t])
                # Check if the point lies inside the image horizontally
                if x < 0.0 or x >= float(self.cfg.img_w):
                    # keep padding tokens
                    continue
                # Quantize x to [1, nbins_x-1] so that 0 is reserved as padding
                # nbins_x=200 means tokens 1..199. 
                bin_idx = int(
                    round(x / (self.cfg.img_w - 1) * (self.cfg.nbins_x - 1))
                )
                bin_idx = max(1, min(self.cfg.nbins_x - 1, bin_idx))
                x_tokens[t] = bin_idx
                # y is represented by the step index; t in [0, T-1]
                y_tokens[t] = t
            return x_tokens, y_tokens

        # Relative x mode: encode Δx_t = x_t - x_{t-1} (in pixels), clipped.
        # t=0 is treated specially and encodes absolute x_0.
        # "relative" mode: overlaps vocab (ambiguous).
        # "relative_disjoint" mode: shifts relative tokens to non-overlapping range.
        if self.cfg.x_mode in ["relative", "relative_disjoint"]:
            max_dx = max(1, int(self.cfg.max_abs_dx))
            last_valid_x = None
            
            # Offset for relative tokens in disjoint mode
            # Absolute tokens use [1, nbins_x-1]
            # Relative tokens start at nbins_x + 1 (to leave room for nbins_x separator if needed)
            # Let's start at nbins_x
            rel_vocab_offset = self.cfg.nbins_x if self.cfg.x_mode == "relative_disjoint" else 0

            for t in range(self.T):
                x = float(xs[t])
                # Skip points outside image horizontally
                if x < 0.0 or x >= float(self.cfg.img_w):
                    continue

                if last_valid_x is None or t == 0:
                    # First valid point: encode absolute x
                    bin_idx = int(
                        round(x / (self.cfg.img_w - 1) * (self.cfg.nbins_x - 1))
                    )
                    bin_idx = max(1, min(self.cfg.nbins_x - 1, bin_idx))
                    code = bin_idx
                else:
                    # Subsequent points: encode relative offset Δx
                    dx = x - last_valid_x
                    dx_clipped = max(-max_dx, min(max_dx, dx))
                    dx_int = int(round(dx_clipped))
                    # Map Δx in [-max_dx, max_dx] to token in [1, 2*max_dx+1]
                    base_code = dx_int + max_dx + 1
                    
                    if self.cfg.x_mode == "relative_disjoint":
                        # Shift to disjoint range
                        code = base_code + rel_vocab_offset
                    else:
                         # Overlapping legacy mode
                        code = base_code
                    code = max(1, min(self.cfg.nbins_x - 1, code))

                x_tokens[t] = code
                y_tokens[t] = t
                last_valid_x = x

            return x_tokens, y_tokens

        # Fallback: if unknown mode, raise.
        raise ValueError(f"Unsupported x_mode: {self.cfg.x_mode}")

    def encode_lanes(self, lanes: Iterable[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Encode multiple lanes.

        Args:
            lanes: iterable of np.ndarray, each of shape (N_i, 2).

        Returns:
            List of (x_tokens, y_tokens) pairs.
        """
        sequences: List[Tuple[np.ndarray, np.ndarray]] = []
        for lane in lanes:
            x_tokens, y_tokens = self.encode_single_lane(lane)
            sequences.append((x_tokens, y_tokens))
        return sequences

    def decode_single_lane(
        self,
        x_tokens: np.ndarray,
        y_tokens: np.ndarray,
        smooth: bool = False,
    ) -> np.ndarray:
        """Decode a single lane from (x_tokens, y_tokens) back to (x, y) points.

        Args:
            x_tokens: np.ndarray of shape (T,)
            y_tokens: np.ndarray of shape (T,)

        Returns:
            np.ndarray of shape (M, 2) with continuous (x, y) coordinates.
        """
        sample_ys = self._compute_sample_ys()
        xs: List[float] = []
        ys: List[float] = []

        if self.cfg.x_mode == "absolute":
            for t in range(self.T):
                x_tok = int(x_tokens[t])
                y_tok = int(y_tokens[t])
                if x_tok == self.cfg.pad_token_x or y_tok >= self.T:
                    continue

                # De-quantize x
                x = x_tok / max(1, self.cfg.nbins_x - 1) * (self.cfg.img_w - 1)
                y = sample_ys[t]
                xs.append(float(x))
                ys.append(float(y))

        elif self.cfg.x_mode in ["relative", "relative_disjoint"]:
            max_dx = max(1, int(self.cfg.max_abs_dx))
            rel_vocab_offset = self.cfg.nbins_x if self.cfg.x_mode == "relative_disjoint" else 0
            
            current_x = None
            for t in range(self.T):
                x_tok = int(x_tokens[t])
                y_tok = int(y_tokens[t])
                if x_tok == self.cfg.pad_token_x or y_tok >= self.T:
                    continue

                # Decide if this token is absolute or relative
                is_absolute = False
                if self.cfg.x_mode == "relative_disjoint":
                    # If it's in the absolute range [1, nbins_x-1]
                    if x_tok < self.cfg.nbins_x:
                        is_absolute = True
                else:
                    # Legacy relative mode: rely on t=0 or first valid
                    if current_x is None or t == 0:
                        is_absolute = True

                if is_absolute:
                    # Absolute mapping
                    x = x_tok / max(1, self.cfg.nbins_x - 1) * (self.cfg.img_w - 1)
                    current_x = float(x)
                else:
                    # Relative mapping
                    if self.cfg.x_mode == "relative_disjoint":
                        base_code = x_tok - rel_vocab_offset
                    else:
                        base_code = x_tok
                        
                    dx_int = base_code - (max_dx + 1)
                    dx_int = max(-max_dx, min(max_dx, dx_int))
                    
                    if current_x is None:
                         # Should not happen if sequence is valid (starts with abs)
                         # But if it does, assume center or something? 
                         # Or just treat as absolute x?
                         current_x = float(self.cfg.img_w / 2)

                    current_x += float(dx_int)

                y = sample_ys[t]
                xs.append(float(current_x))
                ys.append(float(y))
        else:
            raise ValueError(f"Unsupported x_mode: {self.cfg.x_mode}")

        if not xs:
            return np.zeros((0, 2), dtype=np.float32)

        coords = np.stack([xs, ys], axis=1).astype(np.float32)

        if smooth and len(coords) >= 4:
            try:
                from scipy.signal import savgol_filter
                # Sort by y just in case
                sort_idx = np.argsort(coords[:, 1])
                coords = coords[sort_idx]
                
                x_eval = coords[:, 0]
                
                # Stronger smoothing: window_length=11 for better zigzag removal
                window_length = min(11, len(x_eval))
                if window_length % 2 == 0:
                    window_length -= 1
                
                if window_length >= 3:
                    x_smooth = savgol_filter(x_eval, window_length=window_length, polyorder=2)
                    coords[:, 0] = x_smooth
            except Exception:
                pass

        return coords

    def decode_lanes(
        self,
        sequences: Iterable[Tuple[np.ndarray, np.ndarray]],
        smooth: bool = False,
    ) -> List[np.ndarray]:
        """Decode multiple lanes from token sequences."""
        decoded: List[np.ndarray] = []
        for x_tokens, y_tokens in sequences:
            coords = self.decode_single_lane(x_tokens, y_tokens, smooth=smooth)
            if coords.shape[0] > 0:
                decoded.append(coords)
        return decoded
