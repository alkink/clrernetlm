_base_ = [
    './lanelm_v4_culane_test_v26_train100.py',
]

# V30 ablation: HR OFF (hallucination removal kapalı) - strict IoU etkisini izole et.
model = dict(
    decode_cfg=dict(
        use_hr=False,
    ),
)

test_evaluator = dict(
    result_dir='work_dirs/v26_test100_nohr/predictions',
)


