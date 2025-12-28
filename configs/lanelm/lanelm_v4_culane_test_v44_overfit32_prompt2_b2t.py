_base_ = ['./lanelm_v4_culane_test.py']

# V44: bottom-to-top y sampling + prompt2 (train/test consistent).
model = dict(
    lanelm_cfg=dict(
        ckpt_path='work_dirs/v44_overfit32_prompt2_b2t/lanelm_v4_best.pth',
    ),
    tokenizer_cfg=dict(
        y_direction='bottom_to_top',
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
    result_dir='work_dirs/v44_test100_overfit32_prompt2_b2t/predictions',
)


