import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_GAUSTAR_STAGE1_CFG = {
    "enabled": False,
    "use_metric3d_depth": True,
    "use_flow_pose_init": True,
    "use_rendered_depth_filter": True,
    "depth_scale_align": "median",
    "depth_consistency_threshold": 0.12,
    "flow_fb_pixel_threshold": 3.0,
    "flow_depth_fb_threshold": 0.05,
    "depth_edge_kernel": 7,
    "depth_edge_threshold": 0.1,
    "min_pose_correspondences": 200,
    "min_pose_inliers": 80,
    "max_pose_reproj_error": 4.0,
    "max_pose_correspondences": 5000,
    "tracking_filter_min_ratio": 0.05,
    "apply_filter_to_tracking": True,
    "apply_filter_to_keyframe_depth": True,
    "apply_filter_to_render_guided_densification": True,
    "require_rendered_depth_for_keyframe_depth": True,
    "flow_pose_after_init": True,
    "flow_pose_compare_init_loss": True,
    "flow_pose_loss_improvement": 0.0,
    "max_flow_pose_translation_ratio": 0.5,
    "max_flow_pose_rotation_deg": 20.0,
    "use_depth_scale_for_flow_pose": True,
    "apply_filter_after_init": True,
    "tracking_filter_soft_weight": 0.3,
    "min_depth": 0.01,
    "metric3d_model": "metric3d_vit_large",
    "metric3d_image_size": [616, 1064],
    "metric3d_max_depth": 300.0,
    "cache_metric3d_depth": True,
    "precompute_flow": False,
    "raft_root": "",
    "raft_checkpoint": "",
    "flow_iters": 20,
    "flow_skip_existing": True,
    "flow_small": False,
    "flow_mixed_precision": False,
    "flow_alternate_corr": False,
}


def get_gaustar_stage1_config(config):
    cfg = dict(DEFAULT_GAUSTAR_STAGE1_CFG)
    cfg.update(config.get("Training", {}).get("gaustar_stage1", {}))
    return cfg


def gaustar_stage1_enabled(config):
    return get_gaustar_stage1_config(config).get("enabled", False)


def lsg_metric3d_depth_enabled(config):
    training = config.get("Training", {})
    lsg_cfg = training.get("lsg_slam", {})
    pose_cfg = training.get("lsg_pose_init", {})
    return bool(
        lsg_cfg.get("enabled", False)
        and pose_cfg.get("enabled", False)
        and pose_cfg.get("use_metric3d_depth", False)
    )


def metric3d_depth_requested(config):
    cfg = get_gaustar_stage1_config(config)
    return bool(
        (cfg.get("enabled", False) and cfg.get("use_metric3d_depth", True))
        or lsg_metric3d_depth_enabled(config)
    )


def _first_npz_array(npz_data, preferred_keys):
    for key in preferred_keys:
        if key in npz_data:
            return npz_data[key]
    return npz_data[npz_data.files[0]]


def _load_array(path, preferred_keys):
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            return _first_npz_array(data, preferred_keys)
    if path.lower().endswith(".npy"):
        return np.load(path, allow_pickle=False)
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read array image: {path}")
    return image


def resize_depth(depth, target_shape):
    target_h, target_w = target_shape
    if depth.shape[:2] == (target_h, target_w):
        return depth
    return cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def sanitize_depth(depth, target_shape=None):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if target_shape is not None:
        depth = resize_depth(depth, target_shape)
    finite = np.isfinite(depth)
    depth = np.where(finite & (depth > 0), depth, 0.0).astype(np.float32)
    return depth


def resize_flow(flow, target_shape):
    target_h, target_w = target_shape
    src_h, src_w = flow.shape[:2]
    if (src_h, src_w) == (target_h, target_w):
        return flow
    scale_x = float(target_w) / max(float(src_w), 1.0)
    scale_y = float(target_h) / max(float(src_h), 1.0)
    flow = cv2.resize(flow, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= scale_x
    flow[..., 1] *= scale_y
    return flow


def sanitize_flow(flow, target_shape=None):
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim == 3 and flow.shape[0] == 2 and flow.shape[-1] != 2:
        flow = np.moveaxis(flow, 0, -1)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Expected flow shape HxWx2 or 2xHxW, got {flow.shape}")
    if target_shape is not None:
        flow = resize_flow(flow, target_shape)
    finite = np.isfinite(flow).all(axis=-1)
    flow = np.where(finite[..., None], flow, 0.0).astype(np.float32)
    return flow


def load_metric_depth_file(path, target_shape=None):
    return sanitize_depth(_load_array(path, ("depth", "metric_depth", "arr_0")), target_shape)


def get_metric3d_model_name(config):
    cfg = get_gaustar_stage1_config(config)
    return config.get("mono_prior", {}).get(
        "depth", cfg.get("metric3d_model", "metric3d_vit_large")
    )


def get_metric3d_estimator(config, device):
    depth_model = get_metric3d_model_name(config)
    if "metric3d_vit" not in depth_model:
        raise NotImplementedError(f"Unsupported Metric3D model: {depth_model}")
    model = torch.hub.load("yvanyin/metric3d", depth_model, pretrain=True)
    return model.to(device).eval()


@torch.no_grad()
def predict_metric3d_depth(model, input_tensor, config, device):
    cfg = get_gaustar_stage1_config(config)
    image_size = cfg.get("metric3d_image_size", [616, 1064])
    image_size = (int(image_size[0]), int(image_size[1]))
    input_tensor = input_tensor.to(device=device, dtype=torch.float32)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor[None]
    h, w = input_tensor.shape[-2:]
    scale = min(image_size[0] / h, image_size[1] / w)
    resized_h = max(1, int(h * scale))
    resized_w = max(1, int(w * scale))

    img_tensor = F.interpolate(
        input_tensor,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=img_tensor.dtype)[
        None, :, None, None
    ]
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=img_tensor.dtype)[
        None, :, None, None
    ]
    img_tensor = (img_tensor - mean) / std

    pad_h = image_size[0] - resized_h
    pad_w = image_size[1] - resized_w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    img_tensor = F.pad(
        img_tensor,
        (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half),
        mode="constant",
        value=0.0,
    )

    pred_depth, _, _ = model.inference({"input": img_tensor})
    pred_depth = pred_depth.squeeze()
    end_h = pred_depth.shape[0] - (pad_h - pad_h_half)
    end_w = pred_depth.shape[1] - (pad_w - pad_w_half)
    pred_depth = pred_depth[pad_h_half:end_h, pad_w_half:end_w]
    pred_depth = F.interpolate(
        pred_depth[None, None], size=(h, w), mode="bicubic", align_corners=False
    ).squeeze()

    fx = float(config["Dataset"]["Calibration"]["fx"])
    pred_depth = pred_depth * (fx / 1000.0)
    return torch.clamp(pred_depth, 0, float(cfg.get("metric3d_max_depth", 300.0)))


def save_metric_depth_file(depth, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    depth_np = depth.detach().cpu().float().numpy().astype(np.float32)
    np.save(path, depth_np)


def load_flow_file(path, target_shape=None):
    return sanitize_flow(_load_array(path, ("flow", "arr_0")), target_shape)


def load_prior_mask_file(path, target_shape=None):
    mask = _load_array(path, ("mask", "valid_mask", "arr_0"))
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if target_shape is not None and mask.shape[:2] != tuple(target_shape):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask.astype(bool)


def depth_edge_response_np(depth, kernel=7):
    depth = np.asarray(depth, dtype=np.float32)
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    valid = depth > 0
    safe_depth = np.where(valid, depth, 0.0)
    mean = cv2.blur(safe_depth, (kernel, kernel))
    mean_sq = cv2.blur(safe_depth * safe_depth, (kernel, kernel))
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var) / np.maximum(mean, 1e-6)


def depth_edge_response_torch(depth, kernel=7):
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    depth_2d = depth.squeeze().float()
    valid = depth_2d > 0
    safe_depth = torch.where(valid, depth_2d, torch.zeros_like(depth_2d))
    pad = kernel // 2
    depth_4d = safe_depth[None, None]
    mean = torch.nn.functional.avg_pool2d(depth_4d, kernel, stride=1, padding=pad)[0, 0]
    mean_sq = torch.nn.functional.avg_pool2d(
        depth_4d * depth_4d, kernel, stride=1, padding=pad
    )[0, 0]
    var = torch.clamp(mean_sq - mean * mean, min=0.0)
    return torch.sqrt(var) / torch.clamp(mean, min=1e-6)


def sample_array_at_pixels(array, pixels):
    h, w = array.shape[:2]
    x = np.rint(pixels[:, 0]).astype(np.int64)
    y = np.rint(pixels[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    return array[y, x], valid


def estimate_depth_scale(metric_depth, render_depth, opacity=None, mask=None, cfg=None):
    if cfg is None:
        cfg = DEFAULT_GAUSTAR_STAGE1_CFG
    metric = metric_depth.squeeze().float()
    rendered = render_depth.squeeze().float()
    valid = (metric > cfg.get("min_depth", 0.01)) & (rendered > cfg.get("min_depth", 0.01))
    if opacity is not None:
        valid = valid & (opacity.squeeze() > 0.95)
    if mask is not None:
        valid = valid & mask.squeeze().bool()
    if int(valid.count_nonzero().item()) == 0:
        return None
    if cfg.get("depth_scale_align", "median") == "median":
        scale = torch.median(rendered[valid] / torch.clamp(metric[valid], min=1e-6))
    else:
        scale = torch.tensor(1.0, dtype=rendered.dtype, device=rendered.device)
    if not torch.isfinite(scale) or float(scale.item()) <= 0:
        return None
    return scale


def metric_depth_to_tensor(metric_depth, device, dtype=torch.float32):
    if metric_depth is None:
        return None
    if isinstance(metric_depth, torch.Tensor):
        tensor = metric_depth.to(device=device, dtype=dtype)
    else:
        tensor = torch.from_numpy(metric_depth).to(device=device, dtype=dtype)
    return tensor.squeeze()[None]


def build_depth_consistency_mask(
    metric_depth,
    render_depth,
    opacity,
    cfg,
    prior_valid_mask=None,
):
    metric = metric_depth_to_tensor(metric_depth, render_depth.device, render_depth.dtype)
    if metric is None:
        return None, None
    rendered = render_depth.squeeze()
    metric_2d = metric.squeeze()
    valid = (metric_2d > cfg.get("min_depth", 0.01)) & (
        rendered > cfg.get("min_depth", 0.01)
    )
    valid = valid & (opacity.squeeze() > 0.95)
    if prior_valid_mask is not None:
        if isinstance(prior_valid_mask, torch.Tensor):
            prior_mask = prior_valid_mask.to(device=render_depth.device, dtype=torch.bool)
        else:
            prior_mask = torch.as_tensor(
                prior_valid_mask, device=render_depth.device, dtype=torch.bool
            )
        valid = valid & prior_mask.squeeze()

    edge = depth_edge_response_torch(
        metric_2d, kernel=cfg.get("depth_edge_kernel", 7)
    )
    valid = valid & (edge < cfg.get("depth_edge_threshold", 0.1))
    scale = estimate_depth_scale(metric_2d, rendered, opacity, valid, cfg)
    if scale is None:
        return None, None
    aligned = metric_2d * scale
    rel_diff = torch.abs(rendered - aligned) / torch.clamp(aligned, min=1e-6)
    mask = valid & (rel_diff < cfg.get("depth_consistency_threshold", 0.12))
    return mask[None], scale


def _camera_matrix(viewpoint):
    return np.array(
        [[viewpoint.fx, 0.0, viewpoint.cx], [0.0, viewpoint.fy, viewpoint.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def initialize_pose_from_flow(prev_viewpoint, viewpoint, cfg):
    priors = getattr(viewpoint, "priors", {}) or {}
    prev_priors = getattr(prev_viewpoint, "priors", {}) or {}
    flow = priors.get("flow_prev_to_cur")
    flow_back = priors.get("flow_cur_to_prev")
    metric_depth = prev_priors.get("metric_depth")
    if metric_depth is None and prev_viewpoint.depth is not None:
        metric_depth = prev_viewpoint.depth
    if flow is None or flow_back is None or metric_depth is None:
        return False, "missing priors"

    flow = flow.detach().cpu().numpy() if isinstance(flow, torch.Tensor) else np.asarray(flow)
    flow_back = (
        flow_back.detach().cpu().numpy()
        if isinstance(flow_back, torch.Tensor)
        else np.asarray(flow_back)
    )
    metric_depth = (
        metric_depth.detach().cpu().numpy()
        if isinstance(metric_depth, torch.Tensor)
        else np.asarray(metric_depth)
    )
    metric_depth = sanitize_depth(metric_depth, (viewpoint.image_height, viewpoint.image_width))
    if cfg.get("use_depth_scale_for_flow_pose", True):
        depth_scale = prev_priors.get("depth_scale")
        if depth_scale is not None:
            metric_depth = metric_depth * float(depth_scale)

    h, w = metric_depth.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    pix_prev = np.stack([xs, ys], axis=-1).reshape(-1, 2)
    flow_flat = flow.reshape(-1, 2)
    pix_cur = pix_prev + flow_flat
    depth_flat = metric_depth.reshape(-1)

    valid = depth_flat > cfg.get("min_depth", 0.01)
    valid &= np.isfinite(flow_flat).all(axis=1)
    valid &= (pix_cur[:, 0] >= 1) & (pix_cur[:, 0] < w - 2)
    valid &= (pix_cur[:, 1] >= 1) & (pix_cur[:, 1] < h - 2)

    prior_valid_mask = prev_priors.get("prior_valid_mask")
    if prior_valid_mask is not None:
        prior_valid = (
            prior_valid_mask.detach().cpu().numpy()
            if isinstance(prior_valid_mask, torch.Tensor)
            else np.asarray(prior_valid_mask)
        ).astype(bool)
        valid &= prior_valid.reshape(-1)

    edge = depth_edge_response_np(
        metric_depth, kernel=cfg.get("depth_edge_kernel", 7)
    ).reshape(-1)
    valid &= edge < cfg.get("depth_edge_threshold", 0.1)

    back_sample, back_valid = sample_array_at_pixels(flow_back, pix_cur)
    pix_prev_back = pix_cur + back_sample
    valid &= back_valid
    valid &= (
        np.linalg.norm(pix_prev_back - pix_prev, axis=1)
        < cfg.get("flow_fb_pixel_threshold", 3.0)
    )
    depth_prev_back, depth_back_valid = sample_array_at_pixels(metric_depth, pix_prev_back)
    depth_fb_rel = np.abs(depth_prev_back - depth_flat) / np.maximum(depth_flat, 1e-6)
    valid &= depth_back_valid
    valid &= depth_fb_rel < cfg.get("flow_depth_fb_threshold", 0.05)

    if int(valid.sum()) < cfg.get("min_pose_correspondences", 200):
        return False, "not enough correspondences"

    valid_idx = np.flatnonzero(valid)
    max_corr = int(cfg.get("max_pose_correspondences", 5000))
    if valid_idx.shape[0] > max_corr:
        rng = np.random.default_rng(int(viewpoint.uid))
        valid_idx = rng.choice(valid_idx, size=max_corr, replace=False)

    u = pix_prev[valid_idx, 0]
    v = pix_prev[valid_idx, 1]
    z = depth_flat[valid_idx]
    xyz_cam = np.stack(
        [
            (u - prev_viewpoint.cx) / prev_viewpoint.fx * z,
            (v - prev_viewpoint.cy) / prev_viewpoint.fy * z,
            z,
        ],
        axis=1,
    )
    prev_R = prev_viewpoint.R.detach().cpu().numpy()
    prev_T = prev_viewpoint.T.detach().cpu().numpy()
    xyz_world = (prev_R.T @ (xyz_cam - prev_T).T).T.astype(np.float64)
    img_points = pix_cur[valid_idx].astype(np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        xyz_world,
        img_points,
        _camera_matrix(viewpoint),
        None,
        reprojectionError=float(cfg.get("max_pose_reproj_error", 4.0)),
        iterationsCount=100,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None or len(inliers) < cfg.get("min_pose_inliers", 80):
        return False, "pnp rejected"

    rot, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(
        xyz_world[inliers[:, 0]], rvec, tvec, _camera_matrix(viewpoint), None
    )
    reproj = np.linalg.norm(projected.reshape(-1, 2) - img_points[inliers[:, 0]], axis=1)
    if float(np.mean(reproj)) > cfg.get("max_pose_reproj_error", 4.0):
        return False, "reprojection too high"

    viewpoint.update_RT(
        torch.from_numpy(rot.astype(np.float32)).to(device=viewpoint.device),
        torch.from_numpy(tvec.reshape(3).astype(np.float32)).to(device=viewpoint.device),
    )
    return True, f"inliers={len(inliers)}, reproj={float(np.mean(reproj)):.3f}"
