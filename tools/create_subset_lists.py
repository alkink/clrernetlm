#!/usr/bin/env python3
"""
Create 100-image subsets for train and test datasets.
This allows faster iteration without full dataset training.
"""
import argparse
import os
from pathlib import Path


def create_subset_list(input_list_path, output_list_path, num_samples=100):
    """Create a subset list file with first N samples."""
    with open(input_list_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    # Take first N samples
    subset_lines = lines[:num_samples]
    
    # Write to output file
    os.makedirs(os.path.dirname(output_list_path), exist_ok=True)
    with open(output_list_path, 'w') as f:
        for line in subset_lines:
            f.write(line + '\n')
    
    print(f"✓ Created {output_list_path} with {len(subset_lines)} samples")
    return len(subset_lines)


def main():
    parser = argparse.ArgumentParser(description="Create subset list files")
    parser.add_argument("--train-list", default="dataset/list/train.txt", help="Input train list")
    parser.add_argument("--test-list", default="dataset/list/test.txt", help="Input test list")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples per subset")
    parser.add_argument("--output-dir", default="dataset/list", help="Output directory")
    args = parser.parse_args()
    
    # Create train subset
    train_subset_path = os.path.join(args.output_dir, f"train_{args.num_samples}.txt")
    create_subset_list(args.train_list, train_subset_path, args.num_samples)
    
    # Create test subset
    test_subset_path = os.path.join(args.output_dir, f"test_{args.num_samples}.txt")
    create_subset_list(args.test_list, test_subset_path, args.num_samples)
    
    print(f"\n✓ Created subset lists:")
    print(f"  Train: {train_subset_path}")
    print(f"  Test: {test_subset_path}")


if __name__ == "__main__":
    main()








