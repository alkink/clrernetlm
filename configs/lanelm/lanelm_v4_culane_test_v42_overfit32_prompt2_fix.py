_base_ = ['./lanelm_v4_culane_test.py']

# V42 (fix): prompt2 training/inference with fixed-Y sparse prompting.
model = dict(
    lanelm_cfg=dict(
        ckpt_path='work_dirs/v42_overfit32_prompt2_fix/lanelm_v4_best.pth',
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
    result_dir='work_dirs/v42_test100_overfit32_prompt2_fix/predictions',
)


