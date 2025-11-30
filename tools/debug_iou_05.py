#!/usr/bin/env python3
"""Debug IoU 0.5 issue - why no matches?"""

import os
import numpy as np
from libs.datasets.metrics.culane_metric import load_culane_img_data
from scipy.interpolate import interp1d

# İlk test görüntüsünü al
test_img = 'driver_100_30frame/05251517_0433.MP4/00930.jpg'
gt_path = os.path.join('dataset', test_img.replace('.jpg', '.lines.txt'))
pred_path = 'work_dirs/lanelm_v4_test_valid_gt/predictions/' + test_img.replace('.jpg', '.lines.txt')

print('=== IoU 0.5 Analizi ===')
print(f'Test image: {test_img}')
print()

# GT yükle
gt_data = load_culane_img_data(gt_path)
print(f'GT lanes: {len(gt_data)}')
for i, lane in enumerate(gt_data):
    xs = [p[0] for p in lane]
    ys = [p[1] for p in lane]
    print(f'  GT Lane {i}: {len(lane)} points, X:[{min(xs):.1f}, {max(xs):.1f}], Y:[{min(ys):.1f}, {max(ys):.1f}]')

# Prediction yükle
if not os.path.exists(pred_path):
    print(f'\n⚠️  Prediction file not found: {pred_path}')
    exit(1)

pred_data = load_culane_img_data(pred_path)
print(f'\nPred lanes: {len(pred_data)}')
for i, lane in enumerate(pred_data):
    xs = [p[0] for p in lane]
    ys = [p[1] for p in lane]
    print(f'  Pred Lane {i}: {len(lane)} points, X:[{min(xs):.1f}, {max(xs):.1f}], Y:[{min(ys):.1f}, {max(ys):.1f}]')

# X koordinatları karşılaştır
print(f'\n=== X Koordinatları Karşılaştırması ===')
for gt_idx, gt_lane in enumerate(gt_data):
    gt_arr = np.array(gt_lane, dtype=np.float32)
    gt_xs = gt_arr[:, 0]
    gt_ys = gt_arr[:, 1]
    
    print(f'\nGT Lane {gt_idx}:')
    print(f'  X range: [{gt_xs.min():.1f}, {gt_xs.max():.1f}]')
    print(f'  Y range: [{gt_ys.min():.1f}, {gt_ys.max():.1f}]')
    
    # En yakın prediction'ı bul
    best_pred_idx = -1
    min_mean_diff = float('inf')
    best_stats = None
    
    for pred_idx, pred_lane in enumerate(pred_data):
        pred_arr = np.array(pred_lane, dtype=np.float32)
        pred_xs = pred_arr[:, 0]
        pred_ys = pred_arr[:, 1]
        
        # Y overlap kontrolü
        y_min = max(gt_ys.min(), pred_ys.min())
        y_max = min(gt_ys.max(), pred_ys.max())
        
        if y_max <= y_min:
            continue
        
        # Y değerlerinde interpolate et
        y_eval = np.arange(y_min, y_max, 2)
        
        # GT interpolate
        try:
            gt_interp = interp1d(gt_ys, gt_xs, kind='linear', 
                                bounds_error=False, fill_value=np.nan)
            gt_xs_eval = gt_interp(y_eval)
            
            # Pred interpolate
            pred_interp = interp1d(pred_ys, pred_xs, kind='linear',
                                  bounds_error=False, fill_value=np.nan)
            pred_xs_eval = pred_interp(y_eval)
            
            # Valid points
            valid = ~(np.isnan(gt_xs_eval) | np.isnan(pred_xs_eval))
            if valid.sum() < 2:
                continue
            
            # X farklarını hesapla
            x_diffs = np.abs(gt_xs_eval[valid] - pred_xs_eval[valid])
            mean_diff = x_diffs.mean()
            
            if mean_diff < min_mean_diff:
                min_mean_diff = mean_diff
                best_pred_idx = pred_idx
                best_stats = {
                    'y_overlap': (y_min, y_max),
                    'valid_points': valid.sum(),
                    'mean_diff': mean_diff,
                    'max_diff': x_diffs.max(),
                    'points_30': (x_diffs < 30).sum(),
                    'points_50': (x_diffs < 50).sum(),
                    'points_100': (x_diffs < 100).sum(),
                    'total_points': len(x_diffs),
                }
    
    if best_pred_idx >= 0 and best_stats:
        print(f'  vs Pred Lane {best_pred_idx}:')
        print(f'    Y overlap: [{best_stats["y_overlap"][0]:.1f}, {best_stats["y_overlap"][1]:.1f}], {best_stats["valid_points"]} points')
        print(f'    Mean X diff: {best_stats["mean_diff"]:.1f} pixels')
        print(f'    Max X diff: {best_stats["max_diff"]:.1f} pixels')
        print(f'    Points < 30px: {best_stats["points_30"]}/{best_stats["total_points"]} ({100*best_stats["points_30"]/best_stats["total_points"]:.1f}%)')
        print(f'    Points < 50px: {best_stats["points_50"]}/{best_stats["total_points"]} ({100*best_stats["points_50"]/best_stats["total_points"]:.1f}%)')
        print(f'    Points < 100px: {best_stats["points_100"]}/{best_stats["total_points"]} ({100*best_stats["points_100"]/best_stats["total_points"]:.1f}%)')
        print(f'\n  Best match: Pred Lane {best_pred_idx}, Mean X diff: {min_mean_diff:.1f} pixels')
    else:
        print(f'  No valid match found!')


