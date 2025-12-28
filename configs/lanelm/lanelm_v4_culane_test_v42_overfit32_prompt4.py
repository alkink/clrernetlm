_base_ = ['./lanelm_v4_culane_test.py']

# V42 ablation: prompting with 4 keypoints (stronger hint).
model = dict(
    lanelm_cfg=dict(
        ckpt_path='work_dirs/v42_overfit32_prompt4/lanelm_v4_best.pth',
    ),
    decode_cfg=dict(
        use_prompting=True,
        use_presence_filter=False,
        smooth=False,
        enable_eos_stop=True,
        eos_consecutive=2,
        eos_min_t=5,
        eos_min_valid=2,
        use_hr=True,
    ),
)

test_evaluator = dict(
    result_dir='work_dirs/v42_test100_overfit32_prompt4/predictions',
)


