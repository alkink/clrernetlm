_base_ = ['./lanelm_v4_culane_test_v42_overfit32_prompt2_fix.py']

# V47: decode-time stabilization to reduce over-extended lanes from scattered tokens.
model = dict(
    decode_cfg=dict(
        contiguous_run=True,
        contiguous_min_len=4,
    ),
)

test_evaluator = dict(
    result_dir='work_dirs/v47_test100_v42fix_contiguous/predictions',
)


