_base_ = ['./lanelm_v4_culane_test_v42_overfit32_prompt2_fix.py']

# V44 (experiment): bottom-to-top y sampling to make CLRNet bottom keypoints land on early timesteps (causal-friendly).
model = dict(
    tokenizer_cfg=dict(
        y_direction='bottom_to_top',
    ),
)

test_evaluator = dict(
    result_dir='work_dirs/v44_test100_overfit32_prompt2_fix_b2t/predictions',
)


