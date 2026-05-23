import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLUR_TRACKING = ROOT / "utils" / "blur_tracking.py"
SLAM_FRONTEND = ROOT / "utils" / "slam_frontend.py"
POSE_UTILS = ROOT / "utils" / "pose_utils.py"


def module_tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def function_names(tree):
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class BlurTrackingStaticTests(unittest.TestCase):
    def test_first_stage_config_keys_exist(self):
        source = BLUR_TRACKING.read_text(encoding="utf-8")

        self.assertIn('"init_motion_from_previous"', source)
        self.assertIn('"init_motion_scale"', source)
        self.assertIn('"min_tracking_iterations"', source)
        self.assertIn('"debug"', source)

    def test_first_stage_helpers_exist(self):
        names = function_names(module_tree(BLUR_TRACKING))

        self.assertIn("initialize_blur_motion_from_previous_frame", names)
        self.assertIn("should_stop_blur_tracking", names)
        self.assertIn("blur_motion_norms", names)

    def test_frontend_uses_helpers_only_in_blur_branch(self):
        source = SLAM_FRONTEND.read_text(encoding="utf-8")

        self.assertIn("initialize_blur_motion_from_previous_frame(", source)
        self.assertIn("should_stop_blur_tracking(", source)
        self.assertIn("record_blur_tracking_debug", source)
        self.assertIn("if use_blur_tracking:", source)

    def test_update_pose_aligns_w2c_dtype_to_pose_delta(self):
        source = POSE_UTILS.read_text(encoding="utf-8")

        self.assertIn("camera_w2c(camera).to(device=tau.device, dtype=tau.dtype)", source)


if __name__ == "__main__":
    unittest.main()
