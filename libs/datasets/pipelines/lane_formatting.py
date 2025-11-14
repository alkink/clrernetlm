import math

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from libs.utils.lane_utils import sample_lane
from libs.models.lanelm import LaneTokenizer, LaneTokenizerConfig


@TRANSFORMS.register_module()
class PackCLRNetInputs(BaseTransform):
    def __init__(
        self,
        # keys=None,
        meta_keys=None,
        max_lanes=4,
        num_points=72,
        img_w=800,
        img_h=320,
    ):
        # self.keys = keys
        self.meta_keys = meta_keys
        self.max_lanes = max_lanes
        self.n_offsets = num_points
        self.n_strips = num_points - 1
        self.strip_size = img_h / self.n_strips
        self.offsets_ys = np.arange(img_h, -1, -self.strip_size)
        self.img_w = img_w

    def convert_targets(self, results):
        old_lanes = results["gt_points"]
        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 2, old_lanes)

        lanes = (
            np.ones((self.max_lanes, 2 + 1 + 1 + 2 + self.n_offsets), dtype=np.float32)
            * -1e5
        )
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            try:
                xs_outside_image, xs_inside_image = sample_lane(
                    lane, self.offsets_ys, self.img_w
                )
            except AssertionError:
                continue
            if len(xs_inside_image) <= 1:  # to calculate theta
                continue
            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = (
                    math.atan(
                        i
                        * self.strip_size
                        / (xs_inside_image[i] - xs_inside_image[0] + 1e-5)
                    )
                    / math.pi
                )
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)
            theta_far = sum(thetas) / len(thetas)

            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = (
                1 - len(xs_outside_image) / self.n_strips
            )  # y0, relative
            lanes[lane_idx, 3] = xs_inside_image[0]  # x0, absolute
            lanes[lane_idx, 4] = theta_far  # theta
            lanes[lane_idx, 5] = len(xs_inside_image)  # length
            lanes[lane_idx, 6 : 6 + len(all_xs)] = all_xs  # xs, absolute

        results["lanes"] = to_tensor(lanes)
        return results

    def transform(self, results):
        data = {}
        img_meta = {}
        data_sample = DetDataSample()
        instance_data = InstanceData()
        if "img" in results:
            img = results["img"]
            img = to_tensor(img).permute(2, 0, 1).contiguous()
        if "lanes" in self.meta_keys:  # training
            results = self.convert_targets(results)
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data_sample.gt_instances = instance_data
        data_sample.set_metainfo(img_meta)
        data["data_samples"] = data_sample
        data["inputs"] = img
        return data


@TRANSFORMS.register_module()
class PackLaneLMInputs(BaseTransform):
    """Pack inputs and LaneLM-style lane tokens for training a LaneLM head.

    This transform is intended for a separate LaneLM training pipeline and
    does not affect the existing CLRerNet training.
    """

    def __init__(
        self,
        meta_keys=None,
        max_lanes=4,
        num_points=40,
        img_w=800,
        img_h=320,
        nbins_x=800,
    ):
        self.meta_keys = meta_keys or [
            "filename",
            "sub_img_name",
            "ori_shape",
            "img_shape",
        ]
        self.max_lanes = max_lanes
        self.num_points = num_points
        self.img_w = img_w
        self.img_h = img_h
        self.nbins_x = nbins_x

        cfg = LaneTokenizerConfig(
            img_w=img_w,
            img_h=img_h,
            num_steps=num_points,
            nbins_x=nbins_x,
        )
        self.tokenizer = LaneTokenizer(cfg)

    def _encode_lanes(self, results):
        """Encode gt_points into LaneLM token sequences."""
        old_lanes = results.get("gt_points", [])
        # Filter out lanes with too few points
        old_lanes = [lane for lane in old_lanes if len(lane) > 3]

        T = self.tokenizer.T
        x_tokens = np.full(
            (self.max_lanes, T), self.tokenizer.cfg.pad_token_x, dtype=np.int64
        )
        y_tokens = np.full((self.max_lanes, T), T, dtype=np.int64)
        valid_mask = np.zeros((self.max_lanes,), dtype=np.int64)

        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break
            coords = np.array(lane, dtype=np.float32).reshape(-1, 2)
            xt, yt = self.tokenizer.encode_single_lane(coords)
            x_tokens[lane_idx] = xt
            y_tokens[lane_idx] = yt
            # Mark lane as valid if it has at least one non-padding token
            if np.any(xt != self.tokenizer.cfg.pad_token_x):
                valid_mask[lane_idx] = 1

        results["lane_tokens_x"] = to_tensor(x_tokens.astype(np.int64)).long()
        results["lane_tokens_y"] = to_tensor(y_tokens.astype(np.int64)).long()
        results["lane_valid_mask"] = to_tensor(valid_mask.astype(np.int64)).long()
        return results

    def transform(self, results):
        """Convert image and lane annotations into tensors and tokens."""
        data = {}
        if "img" in results:
            img = results["img"]
            # Convert to float and normalize to [0, 1]
            img = to_tensor(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        results = self._encode_lanes(results)

        # Minimal metadata pass-through for potential debugging
        meta = {key: results.get(key, None) for key in self.meta_keys}

        data["inputs"] = img
        data["lane_tokens_x"] = results["lane_tokens_x"]
        data["lane_tokens_y"] = results["lane_tokens_y"]
        data["lane_valid_mask"] = results["lane_valid_mask"]
        data["metainfo"] = meta
        return data
