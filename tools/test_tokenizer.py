"""Test tokenizer with a simple diagonal lane."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from libs.models.lanelm import LaneTokenizer, LaneTokenizerConfig

# Create tokenizer with same config as training
cfg = LaneTokenizerConfig(
    img_w=800,
    img_h=320,
    num_steps=40,
    nbins_x=800,
)
tokenizer = LaneTokenizer(cfg)

print("=" * 80)
print("TESTING TOKENIZER WITH SYNTHETIC DIAGONAL LANE")
print("=" * 80)

# Create a simple diagonal lane from bottom-left to top-right
# In resized space (800x320)
# Y goes from top (0) to bottom (320)
# X goes from left (100) to right (700)

num_points = 20
y_coords = np.linspace(0, 319, num_points)  # Top to bottom
x_coords = np.linspace(100, 700, num_points)  # Left to right

lane_points = np.stack([x_coords, y_coords], axis=1).astype(np.float32)

print(f"\nSynthetic lane points ({num_points} points):")
print(f"  X range: [{x_coords.min():.1f}, {x_coords.max():.1f}]")
print(f"  Y range: [{y_coords.min():.1f}, {y_coords.max():.1f}]")
print(f"  First 5 points:")
for i in range(5):
    print(f"    ({lane_points[i, 0]:.1f}, {lane_points[i, 1]:.1f})")

print(f"\n{'='*80}")
print("ENCODING")
print("=" * 80)

x_tokens, y_tokens = tokenizer.encode_single_lane(lane_points)

print(f"\nToken arrays shape: ({x_tokens.shape[0]},)")
non_pad_mask = x_tokens != 0
x_vals = x_tokens[non_pad_mask]
y_vals = y_tokens[non_pad_mask]

print(f"Non-padding tokens: {len(x_vals)} out of {len(x_tokens)}")
print(f"\nX tokens (first 20): {x_tokens[:20].tolist()}")
print(f"Y tokens (first 20): {y_tokens[:20].tolist()}")

print(f"\nX token statistics:")
print(f"  Range: [{x_vals.min()}, {x_vals.max()}]")
print(f"  Unique values: {len(np.unique(x_vals))}")

print(f"\nY token statistics:")
print(f"  Range: [{y_vals.min()}, {y_vals.max()}]")
print(f"  Expected: 0 to {tokenizer.T-1}")

# Check if X increases monotonically (should for diagonal lane)
diffs = np.diff(x_vals)
print(f"\nX token differences:")
print(f"  Min diff: {diffs.min()}")
print(f"  Max diff: {diffs.max()}")
print(f"  Mean diff: {diffs.mean():.2f}")
increasing = np.all(diffs >= 0)
print(f"  Monotonically increasing: {increasing}")

print(f"\n{'='*80}")
print("DECODING")
print("=" * 80)

decoded_coords = tokenizer.decode_single_lane(x_tokens, y_tokens)

print(f"\nDecoded coords shape: {decoded_coords.shape}")
if len(decoded_coords) > 0:
    print(f"X range: [{decoded_coords[:, 0].min():.1f}, {decoded_coords[:, 0].max():.1f}]")
    print(f"Y range: [{decoded_coords[:, 1].min():.1f}, {decoded_coords[:, 1].max():.1f}]")
    print(f"\nFirst 5 decoded points:")
    for i in range(min(5, len(decoded_coords))):
        print(f"  ({decoded_coords[i, 0]:.1f}, {decoded_coords[i, 1]:.1f})")

# Check reconstruction error
if len(decoded_coords) > 0:
    print(f"\n{'='*80}")
    print("RECONSTRUCTION ERROR")
    print("=" * 80)

    # Interpolate original lane at decoded Y positions
    from scipy.interpolate import interp1d
    f = interp1d(y_coords, x_coords, kind='linear', fill_value='extrapolate')

    decoded_ys = decoded_coords[:, 1]
    expected_xs = f(decoded_ys)

    x_errors = np.abs(decoded_coords[:, 0] - expected_xs)
    print(f"\nX coordinate errors:")
    print(f"  Mean: {x_errors.mean():.2f} pixels")
    print(f"  Max: {x_errors.max():.2f} pixels")
    print(f"  RMS: {np.sqrt(np.mean(x_errors**2)):.2f} pixels")
