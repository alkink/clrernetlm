"""Debug script to check training data preprocessing."""
import sys
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.clrernet.culane.dataset_culane_lanelm import (
    train_dataset,
    train_pipeline,
)
from libs.datasets import CulaneDataset

# Create dataset
dataset = CulaneDataset(
    data_root=train_dataset["data_root"],
    data_list=train_dataset["data_list"],
    pipeline=train_pipeline,
    test_mode=False,
)

print(f"Dataset size: {len(dataset)}")
print("\n" + "=" * 80)
print("CHECKING FIRST TRAINING SAMPLE")
print("=" * 80)

# Get first sample
data = dataset[0]

print(f"\nImage shape: {data['inputs'].shape}")
print(f"Lane tokens X shape: {data['lane_tokens_x'].shape}")
print(f"Lane tokens Y shape: {data['lane_tokens_y'].shape}")
print(f"Valid mask: {data['lane_valid_mask']}")

# Check first lane
lane_idx = 0
x_tokens = data['lane_tokens_x'][lane_idx].numpy()
y_tokens = data['lane_tokens_y'][lane_idx].numpy()
valid = data['lane_valid_mask'][lane_idx].item()

print(f"\n{'='*80}")
print(f"LANE {lane_idx} (valid={valid})")
print(f"{'='*80}")

if valid:
    # Find non-padding tokens
    non_pad_mask = x_tokens != 0
    x_vals = x_tokens[non_pad_mask]
    y_vals = y_tokens[non_pad_mask]

    print(f"\nNon-padding tokens: {len(x_vals)}")
    print(f"X tokens (first 20): {x_vals[:20].tolist()}")
    print(f"Y tokens (first 20): {y_vals[:20].tolist()}")

    print(f"\nX range: [{x_vals.min()}, {x_vals.max()}]")
    print(f"Y range: [{y_vals.min()}, {y_vals.max()}]")

    # Check if X values have diversity
    unique_x = np.unique(x_vals)
    print(f"\nUnique X values: {len(unique_x)} out of {len(x_vals)}")

    # Check Y sequence
    print(f"\nY sequence check:")
    print(f"  Expected: sequential from 0")
    print(f"  Actual first 10: {y_vals[:10].tolist()}")
    is_sequential = np.all(np.diff(y_vals[:len(x_vals)]) >= 0)
    print(f"  Sequential: {is_sequential}")
else:
    print("Lane is marked as invalid!")

# Check metadata
meta = data.get('metainfo', {})
print(f"\n{'='*80}")
print("METADATA")
print(f"{'='*80}")
for key, val in meta.items():
    if isinstance(val, (list, tuple, np.ndarray)):
        print(f"{key}: {type(val)} (len={len(val)})")
    else:
        print(f"{key}: {val}")
