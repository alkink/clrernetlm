_base_ = [
    './lanelm_v4_culane_test_v26_train100.py',
]

# V28 ablation: smoothing ON (can improve visuals but may hurt strict IoU).
model = dict(
    decode_cfg=dict(
        smooth=True,
    ),
)

test_evaluator = dict(
    result_dir='work_dirs/v26_test100_smooth/predictions',
)


