import torch
from torch import nn

from gaussian_splatting.gaussian_renderer import render
from utils.pose_utils import get_virtual_camera_delta


DEFAULT_BLUR_TRACKING_CONFIG = {
    "enabled": False,
    "num_virtual_views": 3,
    "shutter_ratio": 1.0,
    "motion_l2": 0.001,
    "start_after_keyframes": 2,
}


def _parse_bool(value, name):
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be a bool, got {type(value).__name__}")


def _parse_positive_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int, got bool")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}")
    return parsed


def _parse_nonnegative_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int, got bool")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0, got {parsed}")
    return parsed


def _parse_nonnegative_float(value, name):
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0, got {parsed}")
    return parsed


def get_blur_tracking_config(config):
    blur_cfg = dict(DEFAULT_BLUR_TRACKING_CONFIG)
    user_cfg = config.get("Training", {}).get("blur_aware_tracking", {})
    if user_cfg is None:
        user_cfg = {}
    if not isinstance(user_cfg, dict):
        raise ValueError("Training.blur_aware_tracking must be a mapping")
    blur_cfg.update(user_cfg)
    blur_cfg["enabled"] = _parse_bool(
        blur_cfg["enabled"], "Training.blur_aware_tracking.enabled"
    )
    blur_cfg["num_virtual_views"] = _parse_positive_int(
        blur_cfg["num_virtual_views"],
        "Training.blur_aware_tracking.num_virtual_views",
    )
    blur_cfg["shutter_ratio"] = _parse_nonnegative_float(
        blur_cfg["shutter_ratio"], "Training.blur_aware_tracking.shutter_ratio"
    )
    blur_cfg["motion_l2"] = _parse_nonnegative_float(
        blur_cfg["motion_l2"], "Training.blur_aware_tracking.motion_l2"
    )
    blur_cfg["start_after_keyframes"] = _parse_nonnegative_int(
        blur_cfg["start_after_keyframes"],
        "Training.blur_aware_tracking.start_after_keyframes",
    )
    return blur_cfg


def blur_aware_tracking_ready(blur_cfg, num_keyframes):
    return blur_cfg.get("enabled", False) and (
        num_keyframes >= blur_cfg["start_after_keyframes"]
    )


def ensure_blur_motion_params(viewpoint):
    if getattr(viewpoint, "blur_rot_delta", None) is None:
        viewpoint.blur_rot_delta = nn.Parameter(torch.zeros_like(viewpoint.cam_rot_delta))
    if getattr(viewpoint, "blur_trans_delta", None) is None:
        viewpoint.blur_trans_delta = nn.Parameter(
            torch.zeros_like(viewpoint.cam_trans_delta)
        )


def blur_motion_l2(viewpoint, blur_cfg):
    weight = blur_cfg["motion_l2"]
    if weight <= 0:
        return viewpoint.blur_rot_delta.sum() * 0.0
    return weight * (
        viewpoint.blur_rot_delta.square().sum()
        + viewpoint.blur_trans_delta.square().sum()
    )


def blur_sample_factors(num_virtual_views, shutter_ratio, device, dtype):
    if num_virtual_views <= 1:
        return torch.zeros(1, device=device, dtype=dtype)
    return (
        torch.linspace(-0.5, 0.5, num_virtual_views, device=device, dtype=dtype)
        * shutter_ratio
    )


def _render_with_pose_delta(
    viewpoint, gaussians, pipeline_params, background, rot_delta, trans_delta
):
    return render(
        viewpoint,
        gaussians,
        pipeline_params,
        background,
        pose_delta_override=(rot_delta, trans_delta),
    )


def render_blur_aware_tracking(
    viewpoint, gaussians, pipeline_params, background, blur_cfg
):
    ensure_blur_motion_params(viewpoint)
    center_rot, center_trans = get_virtual_camera_delta(viewpoint, 0.0)
    center_pkg = _render_with_pose_delta(
        viewpoint, gaussians, pipeline_params, background, center_rot, center_trans
    )
    if center_pkg is None or blur_cfg["num_virtual_views"] <= 1:
        return center_pkg

    samples = blur_sample_factors(
        blur_cfg["num_virtual_views"],
        blur_cfg["shutter_ratio"],
        viewpoint.cam_rot_delta.device,
        viewpoint.cam_rot_delta.dtype,
    )
    renders = []
    center_index = blur_cfg["num_virtual_views"] // 2
    has_center_sample = blur_cfg["num_virtual_views"] % 2 == 1
    for sample_idx, sample in enumerate(samples):
        if has_center_sample and sample_idx == center_index:
            sample_pkg = center_pkg
        else:
            rot_delta, trans_delta = get_virtual_camera_delta(viewpoint, sample)
            sample_pkg = _render_with_pose_delta(
                viewpoint,
                gaussians,
                pipeline_params,
                background,
                rot_delta,
                trans_delta,
            )
        renders.append(sample_pkg["render"])

    render_pkg = dict(center_pkg)
    render_pkg["render"] = torch.stack(renders, dim=0).mean(dim=0)
    render_pkg["blur_sample_count"] = blur_cfg["num_virtual_views"]
    return render_pkg
