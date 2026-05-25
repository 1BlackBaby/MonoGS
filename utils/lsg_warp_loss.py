DEFAULT_LSG_WARP_LOSS_CFG = {
    "enabled": False,
    "weight_warp": 0.05,
    "min_matches_for_loss": 20,
    "min_depth": 0.1,
    "max_depth": 50.0,
    "robust_beta": 0.1,
    "normalize_pixels": True,
    "warp_after_init": True,
    "require_pose_init_accept": True,
    "log_every_frame": True,
}


def get_lsg_feature_warping_config(config):
    training = config.get("Training", {})
    cfg = dict(DEFAULT_LSG_WARP_LOSS_CFG)
    cfg.update(training.get("lsg_feature_warping", {}))
    master_enabled = training.get("lsg_slam", {}).get("enabled", False)
    cfg["enabled"] = bool(master_enabled and cfg.get("enabled", False))
    return cfg


def should_prepare_lsg_warp_match_data(
    cfg, initialized, pose_init_enabled=False, pose_init_accepted=None
):
    if not cfg.get("enabled", False):
        return False, "disabled"
    if cfg.get("warp_after_init", True) and not initialized:
        return False, "waiting_for_init"
    if (
        pose_init_enabled
        and cfg.get("require_pose_init_accept", True)
        and pose_init_accepted is not True
    ):
        return False, "pose_init_not_accepted"
    return True, "ok"


def current_w2c_with_pose_delta(viewpoint):
    import torch

    from utils.pose_utils import SE3_exp, camera_w2c

    tau = torch.cat([viewpoint.cam_trans_delta, viewpoint.cam_rot_delta], dim=0)
    return SE3_exp(tau) @ camera_w2c(viewpoint).to(device=tau.device, dtype=tau.dtype)


def _as_tensor(value, device, dtype):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def build_lsg_warp_match_data(ref_idx, match_data, ref_viewpoint, ref_depth, device):
    import torch

    ref_R = ref_viewpoint.R.detach().to(device=device)
    ref_T = ref_viewpoint.T.detach().to(device=device)
    return {
        "ref_idx": ref_idx,
        "kpts_cur": torch.as_tensor(
            match_data["kpts_cur"], device=device, dtype=torch.float32
        ),
        "kpts_ref": torch.as_tensor(
            match_data["kpts_ref"], device=device, dtype=torch.float32
        ),
        "scores": torch.as_tensor(
            match_data.get("scores", []), device=device, dtype=torch.float32
        ),
        "ref_depth": torch.as_tensor(ref_depth, device=device, dtype=torch.float32),
        "ref_R": ref_R,
        "ref_T": ref_T,
        "ref_fx": float(ref_viewpoint.fx),
        "ref_fy": float(ref_viewpoint.fy),
        "ref_cx": float(ref_viewpoint.cx),
        "ref_cy": float(ref_viewpoint.cy),
        "num_matches": int(match_data.get("num_matches", 0)),
    }


def _zero_loss(viewpoint):
    return (viewpoint.cam_trans_delta.sum() + viewpoint.cam_rot_delta.sum()) * 0.0


def compute_lsg_warp_loss(viewpoint, match_data, cfg):
    import torch
    import torch.nn.functional as F

    stats = {"used_matches": 0, "reason": ""}
    if not cfg.get("enabled", False) or match_data is None:
        prev_stats = getattr(viewpoint, "lsg_warp_stats", {}) or {}
        stats["reason"] = prev_stats.get("reason", "disabled_or_missing")
        return _zero_loss(viewpoint), stats

    device = viewpoint.cam_trans_delta.device
    dtype = viewpoint.cam_trans_delta.dtype
    kpts_ref = _as_tensor(match_data["kpts_ref"], device, dtype).reshape(-1, 2)
    kpts_cur = _as_tensor(match_data["kpts_cur"], device, dtype).reshape(-1, 2)
    if kpts_ref.shape[0] != kpts_cur.shape[0]:
        stats["reason"] = "shape_mismatch"
        return _zero_loss(viewpoint), stats

    min_matches = int(cfg.get("min_matches_for_loss", 20))
    if kpts_ref.shape[0] < min_matches:
        stats["reason"] = "not_enough_matches"
        return _zero_loss(viewpoint), stats

    depth = _as_tensor(match_data["ref_depth"], device, dtype).squeeze()
    h, w = depth.shape[-2:]
    x = torch.round(kpts_ref[:, 0]).long()
    y = torch.round(kpts_ref[:, 1]).long()
    in_depth = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x_safe = torch.clamp(x, 0, w - 1)
    y_safe = torch.clamp(y, 0, h - 1)
    z = depth[y_safe, x_safe]
    valid = (
        in_depth
        & torch.isfinite(z)
        & (z > float(cfg.get("min_depth", 0.1)))
        & (z < float(cfg.get("max_depth", 50.0)))
    )
    if int(valid.count_nonzero().item()) < min_matches:
        stats["reason"] = "not_enough_valid_depth"
        return _zero_loss(viewpoint), stats

    kpts_ref = kpts_ref[valid]
    kpts_cur = kpts_cur[valid]
    z = z[valid]

    ref_fx = float(match_data["ref_fx"])
    ref_fy = float(match_data["ref_fy"])
    ref_cx = float(match_data["ref_cx"])
    ref_cy = float(match_data["ref_cy"])
    xyz_ref = torch.stack(
        [
            (kpts_ref[:, 0] - ref_cx) / ref_fx * z,
            (kpts_ref[:, 1] - ref_cy) / ref_fy * z,
            z,
        ],
        dim=1,
    )
    ref_R = _as_tensor(match_data["ref_R"], device, dtype)
    ref_T = _as_tensor(match_data["ref_T"], device, dtype).reshape(3)
    xyz_world = (xyz_ref - ref_T) @ ref_R

    cur_w2c = current_w2c_with_pose_delta(viewpoint)
    xyz_cur = xyz_world @ cur_w2c[:3, :3].transpose(0, 1) + cur_w2c[:3, 3]
    z_cur = xyz_cur[:, 2]
    projected = torch.stack(
        [
            xyz_cur[:, 0] / torch.clamp(z_cur, min=1e-6) * float(viewpoint.fx)
            + float(viewpoint.cx),
            xyz_cur[:, 1] / torch.clamp(z_cur, min=1e-6) * float(viewpoint.fy)
            + float(viewpoint.cy),
        ],
        dim=1,
    )

    in_front = z_cur > float(cfg.get("min_depth", 0.1))
    in_image = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < float(viewpoint.image_width))
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < float(viewpoint.image_height))
    )
    valid_proj = in_front & in_image & torch.isfinite(projected).all(dim=1)
    if int(valid_proj.count_nonzero().item()) < min_matches:
        stats["reason"] = "not_enough_projected"
        return _zero_loss(viewpoint), stats

    projected = projected[valid_proj]
    target = kpts_cur[valid_proj]
    scores = match_data.get("scores")
    if scores is not None and len(scores) > 0:
        scores = _as_tensor(scores, device, dtype).reshape(-1)[valid][valid_proj]
        scores = scores / torch.clamp(scores.mean(), min=1e-6)
        projected = projected * scores[:, None]
        target = target * scores[:, None]

    if cfg.get("normalize_pixels", False):
        scale = torch.tensor(
            [float(viewpoint.image_width), float(viewpoint.image_height)],
            device=device,
            dtype=dtype,
        )
        projected = projected / scale
        target = target / scale

    stats["used_matches"] = int(projected.shape[0])
    stats["reason"] = "ok"
    loss = F.smooth_l1_loss(
        projected,
        target,
        beta=float(cfg.get("robust_beta", 0.1)),
        reduction="mean",
    )
    return loss * float(cfg.get("weight_warp", 1.0)), stats
