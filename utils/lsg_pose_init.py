from dataclasses import dataclass
import math


DEFAULT_LSG_POSE_INIT_CFG = {
    "enabled": False,
    "pose_after_init": True,
    "max_keypoints": 1024,
    "match_topk": 512,
    "min_matches": 20,
    "min_pnp_points": 20,
    "min_pnp_inliers": 20,
    "min_inlier_ratio": 0.25,
    "min_depth": 0.1,
    "max_depth": 50.0,
    "max_reproj_error": 10.0,
    "pnp_iterations": 200,
    "pnp_confidence": 0.99,
    "max_translation_ratio": 0.5,
    "max_rotation_deg": 20.0,
    "compare_init_loss": True,
    "loss_improvement": 0.0,
    "use_essential_filter": True,
    "essential_threshold": 1.0,
    "log_every_frame": True,
    "feature_model_root": "",
    "use_metric3d_depth": False,
    "metric3d_for_keyframe_depth": True,
    "metric3d_filter_with_rendered_depth": True,
    "metric3d_depth_priority": "after_lsg_keyframe",
}


@dataclass
class PoseInitResult:
    accepted: bool
    reason: str
    R: object = None
    T: object = None
    num_points: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    reproj_error: float = 0.0


def get_lsg_pose_init_config(config):
    training = config.get("Training", {})
    cfg = dict(DEFAULT_LSG_POSE_INIT_CFG)
    cfg.update(training.get("lsg_pose_init", {}))
    master_enabled = training.get("lsg_slam", {}).get("enabled", False)
    cfg["enabled"] = bool(master_enabled and cfg.get("enabled", False))
    return cfg


def _trace3(mat):
    return float(mat[0][0] + mat[1][1] + mat[2][2])


def _transpose3(mat):
    return [[mat[j][i] for j in range(3)] for i in range(3)]


def _matmul3(a, b):
    return [
        [sum(float(a[i][k]) * float(b[k][j]) for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _norm3(vec):
    return math.sqrt(sum(float(v) * float(v) for v in vec))


def pose_delta_reason(prev_R, prev_T, cur_R, cur_T, cfg, median_depth=None):
    rel_R = _matmul3(cur_R, _transpose3(prev_R))
    cos_theta = max(-1.0, min(1.0, (_trace3(rel_R) - 1.0) * 0.5))
    rot_deg = math.degrees(math.acos(cos_theta))
    trans = _norm3(float(cur_T[i]) - float(prev_T[i]) for i in range(3))

    max_rot = float(cfg.get("max_rotation_deg", 0.0))
    if max_rot > 0.0 and rot_deg > max_rot:
        return False, f"rotation too large ({rot_deg:.3f} deg)"

    max_trans_ratio = float(cfg.get("max_translation_ratio", 0.0))
    if max_trans_ratio > 0.0 and median_depth is not None:
        max_trans = max_trans_ratio * float(median_depth)
        if trans > max_trans:
            return False, f"translation too large ({trans:.3f} > {max_trans:.3f})"

    return True, f"delta trans={trans:.3f}, rot={rot_deg:.3f}"


def camera_matrix_from_viewpoint(viewpoint):
    import numpy as np

    return np.array(
        [
            [float(viewpoint.fx), 0.0, float(viewpoint.cx)],
            [0.0, float(viewpoint.fy), float(viewpoint.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _to_numpy(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _depth_2d(depth):
    import numpy as np

    depth = _to_numpy(depth).astype(np.float64)
    if depth.ndim == 3:
        depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got {depth.shape}")
    return depth


def matched_pixels_to_world_points(mkpts_cur, mkpts_ref, ref_viewpoint, ref_depth, cfg):
    import numpy as np

    kpts_cur = _to_numpy(mkpts_cur).astype(np.float64).reshape(-1, 2)
    kpts_ref = _to_numpy(mkpts_ref).astype(np.float64).reshape(-1, 2)
    depth = _depth_2d(ref_depth)
    h, w = depth.shape
    min_depth = float(cfg.get("min_depth", 0.1))
    max_depth = float(cfg.get("max_depth", 50.0))

    ref_R = _to_numpy(ref_viewpoint.R).astype(np.float64)
    ref_T = _to_numpy(ref_viewpoint.T).astype(np.float64).reshape(3)

    points_world = []
    pixels_cur = []
    for uv_cur, uv_ref in zip(kpts_cur, kpts_ref):
        x_ref = int(round(float(uv_ref[0])))
        y_ref = int(round(float(uv_ref[1])))
        if x_ref < 0 or x_ref >= w or y_ref < 0 or y_ref >= h:
            continue
        z = float(depth[y_ref, x_ref])
        if not math.isfinite(z) or z < min_depth or z > max_depth:
            continue
        xyz_ref = np.array(
            [
                (float(uv_ref[0]) - float(ref_viewpoint.cx)) / float(ref_viewpoint.fx) * z,
                (float(uv_ref[1]) - float(ref_viewpoint.cy)) / float(ref_viewpoint.fy) * z,
                z,
            ],
            dtype=np.float64,
        )
        xyz_world = ref_R.T @ (xyz_ref - ref_T)
        points_world.append(xyz_world)
        pixels_cur.append(uv_cur)

    if not points_world:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return np.asarray(points_world, dtype=np.float64), np.asarray(
        pixels_cur, dtype=np.float64
    )


def estimate_pose_from_2d3d(points_world, pixels_cur, camera_matrix, cfg):
    import cv2
    import numpy as np

    points_world = np.ascontiguousarray(points_world, dtype=np.float64).reshape(-1, 3)
    pixels_cur = np.ascontiguousarray(pixels_cur, dtype=np.float64).reshape(-1, 2)
    num_points = int(points_world.shape[0])
    min_points = int(cfg.get("min_pnp_points", 20))
    if num_points < min_points:
        return PoseInitResult(False, "not enough 2d-3d points", num_points=num_points)

    flags = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_world,
            pixels_cur.reshape(-1, 1, 2),
            np.asarray(camera_matrix, dtype=np.float64),
            None,
            reprojectionError=float(cfg.get("max_reproj_error", 10.0)),
            iterationsCount=int(cfg.get("pnp_iterations", 200)),
            confidence=float(cfg.get("pnp_confidence", 0.99)),
            flags=flags,
        )
    except cv2.error as exc:
        return PoseInitResult(False, f"pnp exception: {exc}", num_points=num_points)

    if not ok or inliers is None:
        return PoseInitResult(False, "pnp failed", num_points=num_points)

    num_inliers = int(len(inliers))
    inlier_ratio = float(num_inliers) / max(float(num_points), 1.0)
    if num_inliers < int(cfg.get("min_pnp_inliers", 50)):
        return PoseInitResult(
            False,
            "not enough pnp inliers",
            num_points=num_points,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
        )
    if inlier_ratio < float(cfg.get("min_inlier_ratio", 0.25)):
        return PoseInitResult(
            False,
            "pnp inlier ratio too low",
            num_points=num_points,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
        )

    rot, _ = cv2.Rodrigues(rvec)
    inlier_idx = inliers[:, 0]
    projected, _ = cv2.projectPoints(
        points_world[inlier_idx],
        rvec,
        tvec,
        np.asarray(camera_matrix, dtype=np.float64),
        None,
    )
    reproj = np.linalg.norm(projected.reshape(-1, 2) - pixels_cur[inlier_idx], axis=1)
    reproj_error = float(np.mean(reproj))
    if reproj_error > float(cfg.get("max_reproj_error", 10.0)):
        return PoseInitResult(
            False,
            "reprojection too high",
            num_points=num_points,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            reproj_error=reproj_error,
        )

    return PoseInitResult(
        True,
        "accepted",
        R=rot.astype(np.float32),
        T=tvec.reshape(3).astype(np.float32),
        num_points=num_points,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        reproj_error=reproj_error,
    )


def estimate_pose_from_matches(
    mkpts_cur,
    mkpts_ref,
    ref_viewpoint,
    cur_viewpoint,
    ref_depth,
    cfg,
):
    points_world, pixels_cur = matched_pixels_to_world_points(
        mkpts_cur, mkpts_ref, ref_viewpoint, ref_depth, cfg
    )
    return estimate_pose_from_2d3d(
        points_world, pixels_cur, camera_matrix_from_viewpoint(cur_viewpoint), cfg
    )
