import unittest

from utils.lsg_pose_init import (
    estimate_pose_from_2d3d,
    get_lsg_pose_init_config,
    pose_delta_reason,
)


class LSGPoseInitTest(unittest.TestCase):
    def test_master_switch_disables_pose_init(self):
        config = {
            "Training": {
                "lsg_slam": {"enabled": False},
                "lsg_pose_init": {"enabled": True},
            }
        }

        cfg = get_lsg_pose_init_config(config)

        self.assertFalse(cfg["enabled"])

    def test_default_pnp_thresholds_are_reachable(self):
        config = {
            "Training": {
                "lsg_slam": {"enabled": True},
                "lsg_pose_init": {"enabled": True},
            }
        }

        cfg = get_lsg_pose_init_config(config)

        self.assertGreaterEqual(cfg["min_pnp_points"], cfg["min_pnp_inliers"])

    def test_estimate_pose_from_2d3d_recovers_w2c_pose(self):
        try:
            import cv2
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy/cv2 is not installed in this Python environment")
        k = np.array(
            [[120.0, 0.0, 64.0], [0.0, 120.0, 48.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        xs, ys = np.meshgrid(np.linspace(-0.8, 0.8, 5), np.linspace(-0.5, 0.5, 4))
        points_world = np.stack(
            [xs.reshape(-1), ys.reshape(-1), np.full(xs.size, 4.0)], axis=1
        ).astype(np.float64)
        rvec_gt = np.array([[0.02], [-0.03], [0.01]], dtype=np.float64)
        rot_gt, _ = cv2.Rodrigues(rvec_gt)
        trans_gt = np.array([[0.15], [-0.04], [0.20]], dtype=np.float64)
        pixels, _ = cv2.projectPoints(points_world, rvec_gt, trans_gt, k, None)

        result = estimate_pose_from_2d3d(
            points_world,
            pixels.reshape(-1, 2),
            k,
            {
                "min_pnp_points": 8,
                "min_pnp_inliers": 8,
                "min_inlier_ratio": 0.5,
                "max_reproj_error": 2.0,
                "pnp_iterations": 100,
                "pnp_confidence": 0.99,
            },
        )

        self.assertTrue(result.accepted, result.reason)
        np.testing.assert_allclose(result.R, rot_gt, atol=2e-3)
        np.testing.assert_allclose(result.T.reshape(3, 1), trans_gt, atol=2e-3)

    def test_pose_delta_reason_rejects_large_translation(self):
        prev_R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        prev_T = [0.0, 0.0, 0.0]
        cur_R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        cur_T = [2.0, 0.0, 0.0]

        ok, reason = pose_delta_reason(
            prev_R,
            prev_T,
            cur_R,
            cur_T,
            {"max_translation_ratio": 0.5, "max_rotation_deg": 30.0},
            median_depth=2.0,
        )

        self.assertFalse(ok)
        self.assertIn("translation", reason)


if __name__ == "__main__":
    unittest.main()
