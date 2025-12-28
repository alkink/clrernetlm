_base_ = ['./lanelm_v4_culane_test_v39_full2k_pad01.py']

# Ablation: disable Hallucination Removal to check whether lanes are being truncated/dropped.
model = dict(
    decode_cfg=dict(
        use_hr=False,
    ),
)

test_evaluator = dict(
    result_dir='work_dirs/v39_test100_full2k_pad01_nohr/predictions',
)


