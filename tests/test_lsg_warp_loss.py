import unittest


class LSGWarpLossTest(unittest.TestCase):
    def test_warping_defaults_wait_for_initialized_map(self):
        from utils.lsg_warp_loss import (
            get_lsg_feature_warping_config,
            should_prepare_lsg_warp_match_data,
        )

        cfg = get_lsg_feature_warping_config(
            {
                "Training": {
                    "lsg_slam": {"enabled": True},
                    "lsg_feature_warping": {"enabled": True},
                }
            }
        )

        allowed, reason = should_prepare_lsg_warp_match_data(
            cfg, initialized=False, pose_init_enabled=False
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "waiting_for_init")

    def test_warping_requires_accepted_pose_init_when_pose_init_is_enabled(self):
        from utils.lsg_warp_loss import should_prepare_lsg_warp_match_data

        cfg = {
            "enabled": True,
            "warp_after_init": True,
            "require_pose_init_accept": True,
        }

        allowed, reason = should_prepare_lsg_warp_match_data(
            cfg,
            initialized=True,
            pose_init_enabled=True,
            pose_init_accepted=False,
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "pose_init_not_accepted")

    def test_keypoint_warp_loss_backpropagates_to_pose_delta(self):
        try:
            import torch

            from utils.lsg_warp_loss import compute_lsg_warp_loss
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is not installed in this Python environment: {exc}")

        class FakeViewpoint:
            device = "cpu"
            fx = 100.0
            fy = 100.0
            cx = 50.0
            cy = 50.0
            image_width = 100
            image_height = 100

            def __init__(self):
                self.R = torch.eye(3)
                self.T = torch.zeros(3)
                self.cam_rot_delta = torch.nn.Parameter(torch.zeros(3))
                self.cam_trans_delta = torch.nn.Parameter(torch.zeros(3))

        viewpoint = FakeViewpoint()
        match_data = {
            "ref_idx": 0,
            "kpts_ref": torch.tensor([[50.0, 50.0], [55.0, 50.0]]),
            "kpts_cur": torch.tensor([[51.0, 50.0], [56.0, 50.0]]),
            "scores": torch.ones(2),
            "ref_depth": torch.full((100, 100), 2.0),
            "ref_R": torch.eye(3),
            "ref_T": torch.zeros(3),
            "ref_fx": 100.0,
            "ref_fy": 100.0,
            "ref_cx": 50.0,
            "ref_cy": 50.0,
        }

        loss, stats = compute_lsg_warp_loss(
            viewpoint,
            match_data,
            {
                "enabled": True,
                "weight_warp": 1.0,
                "min_matches_for_loss": 2,
                "min_depth": 0.1,
                "max_depth": 10.0,
                "robust_beta": 0.1,
                "normalize_pixels": False,
            },
        )
        loss.backward()

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertEqual(stats["used_matches"], 2)
        self.assertIsNotNone(viewpoint.cam_trans_delta.grad)
        self.assertGreater(float(viewpoint.cam_trans_delta.grad.abs().sum()), 0.0)

    def test_disabled_warp_loss_returns_zero_without_match_data(self):
        try:
            import torch

            from utils.lsg_warp_loss import compute_lsg_warp_loss
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is not installed in this Python environment: {exc}")

        class FakeViewpoint:
            device = "cpu"
            cam_rot_delta = torch.nn.Parameter(torch.zeros(3))
            cam_trans_delta = torch.nn.Parameter(torch.zeros(3))

        loss, stats = compute_lsg_warp_loss(
            FakeViewpoint(),
            None,
            {"enabled": False},
        )

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(stats["used_matches"], 0)


if __name__ == "__main__":
    unittest.main()
