import numpy as np
import torch

from libs.models.lanelm import LaneTokenizer, LaneTokenizerConfig, LaneLMModel
from libs.datasets.pipelines.lane_formatting import PackLaneLMInputs


def test_lane_tokenizer_roundtrip_straight_line():
    cfg = LaneTokenizerConfig(
        img_w=800,
        img_h=320,
        num_steps=40,
        nbins_x=800,
    )
    tokenizer = LaneTokenizer(cfg)

    # Construct a simple straight lane: x=100 for all y
    ys = np.linspace(0, cfg.img_h - 1, num=10, endpoint=True, dtype=np.float32)
    xs = np.full_like(ys, 100.0)
    points = np.stack([xs, ys], axis=1)

    x_tokens, y_tokens = tokenizer.encode_single_lane(points)
    assert x_tokens.shape == (tokenizer.T,)
    assert y_tokens.shape == (tokenizer.T,)

    # There should be at least one non-padding sample
    assert np.any(x_tokens != tokenizer.cfg.pad_token_x)

    decoded = tokenizer.decode_single_lane(x_tokens, y_tokens)
    # Decoded lane should have points and roughly preserve x ~ 100
    assert decoded.shape[1] == 2
    assert np.allclose(decoded[:, 0], 100.0, atol=5.0)


def test_lanelm_model_forward_shapes():
    nbins_x = 800
    T = 40
    max_y_tokens = T + 1  # including padding token

    model = LaneLMModel(
        nbins_x=nbins_x,
        max_y_tokens=max_y_tokens,
        embed_dim=256,
        num_layers=2,
        num_heads=8,
        ffn_dim=512,
        max_seq_len=T,
        visual_in_dim=64,
    )

    batch_size = 2
    num_tokens = 100

    visual_tokens = torch.randn(batch_size, num_tokens, 64)
    x_tokens = torch.zeros(batch_size, T, dtype=torch.long)
    y_tokens = torch.full((batch_size, T), T, dtype=torch.long)

    logits_x, logits_y = model(
        visual_tokens=visual_tokens,
        x_tokens=x_tokens,
        y_tokens=y_tokens,
        visual_padding_mask=None,
    )

    assert logits_x.shape == (batch_size, T, nbins_x)
    assert logits_y.shape == (batch_size, T, max_y_tokens)


def test_pack_lanelm_inputs_basic():
    # Synthetic single-lane example on 800x320 image
    h, w = 320, 800
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Lane points with descending y (CULane convention)
    ys = np.linspace(h - 1, 0, num=10, dtype=np.float32)
    xs = np.full_like(ys, 200.0)
    coords = []
    for x, y in zip(xs, ys):
        coords.append(float(x))
        coords.append(float(y))

    results = dict(
        filename="dummy.jpg",
        sub_img_name="dummy.jpg",
        img=img,
        gt_points=[coords],
        img_shape=img.shape,
        ori_shape=img.shape,
    )

    pack = PackLaneLMInputs(
        max_lanes=4,
        num_points=40,
        img_w=w,
        img_h=h,
        nbins_x=800,
    )

    data = pack.transform(results)

    assert "inputs" in data
    assert "lane_tokens_x" in data
    assert "lane_tokens_y" in data
    assert "lane_valid_mask" in data

    x_tokens = data["lane_tokens_x"]
    y_tokens = data["lane_tokens_y"]
    valid_mask = data["lane_valid_mask"]

    assert x_tokens.shape == (4, 40)
    assert y_tokens.shape == (4, 40)
    assert valid_mask.shape == (4,)

    # At least one lane should be marked valid
    assert torch.any(valid_mask > 0)

