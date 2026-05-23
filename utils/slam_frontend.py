import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.blur_tracking import (
    blur_aware_tracking_ready,
    blur_motion_l2,
    ensure_blur_motion_params,
    get_blur_tracking_config,
    render_blur_aware_tracking,
)
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.lsg_features import LSGFeatureMatcher
from utils.lsg_pose_init import (
    estimate_pose_from_matches,
    get_lsg_pose_init_config,
    pose_delta_reason,
)
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import reset_blur_motion, update_pose
from utils.slam_utils import (
    build_rendered_depth_consistency_mask,
    get_loss_tracking,
    get_metric_depth_for_initialization,
    get_median_depth,
    log_uncertainty_debug_stats,
)
from utils.mono_priors.gaustar_stage1 import (
    gaustar_stage1_enabled,
    get_gaustar_stage1_config,
    initialize_pose_from_flow,
)


class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        self.uncer_network = None
        self.uncertainty_state_syncs = 0
        self._logged_tracking_uncertainty_warmup = False
        self._logged_tracking_uncertainty_enabled = False
        self.gaustar_stage1_cfg = get_gaustar_stage1_config(config)
        self._logged_flow_pose_init = False
        self._logged_flow_pose_fallback = False
        self._logged_flow_pose_reject = False
        self._gaustar_recent_priors = {}
        self.blur_tracking_cfg = get_blur_tracking_config(config)
        self._logged_blur_tracking_warmup = False
        self._logged_blur_tracking_enabled = False
        self.lsg_pose_init_cfg = get_lsg_pose_init_config(config)
        self.lsg_feature_matcher = None

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
        uncertainty_cfg = self.config.get("Training", {}).get("uncertainty", {})
        self.tracking_uncertainty_warmup_keyframes = uncertainty_cfg.get(
            "tracking_warmup_keyframes", 0
        )
        self.tracking_uncertainty_warmup_backend_syncs = uncertainty_cfg.get(
            "tracking_warmup_backend_syncs", 0
        )
        self.blur_tracking_cfg = get_blur_tracking_config(self.config)
        self.lsg_pose_init_cfg = get_lsg_pose_init_config(self.config)

    def use_blur_aware_tracking(self):
        if not self.blur_tracking_cfg["enabled"]:
            return False
        if not blur_aware_tracking_ready(self.blur_tracking_cfg, len(self.kf_indices)):
            if not self._logged_blur_tracking_warmup:
                print(
                    "[BlurTrack] warmup: "
                    f"keyframes={len(self.kf_indices)}/"
                    f"{self.blur_tracking_cfg['start_after_keyframes']}"
                )
                self._logged_blur_tracking_warmup = True
            return False
        if not self._logged_blur_tracking_enabled:
            print(
                "[BlurTrack] enabled: "
                f"num_virtual_views={self.blur_tracking_cfg['num_virtual_views']}, "
                f"shutter_ratio={self.blur_tracking_cfg['shutter_ratio']}"
            )
            self._logged_blur_tracking_enabled = True
        return True

    def get_tracking_uncertainty_network(self):
        uncertainty_cfg = self.config.get("Training", {}).get("uncertainty", {})
        tracking_uncertainty_enabled = (
            uncertainty_cfg.get("enabled", False)
            and (
                uncertainty_cfg.get("apply_to_tracking_rgb", True)
                or uncertainty_cfg.get("apply_to_tracking_depth", False)
            )
            and self.uncer_network is not None
        )
        if not tracking_uncertainty_enabled:
            return self.uncer_network

        warmup_done = (
            self.initialized
            and len(self.kf_indices) >= self.tracking_uncertainty_warmup_keyframes
            and self.uncertainty_state_syncs
            >= self.tracking_uncertainty_warmup_backend_syncs
        )
        if not warmup_done:
            if not self._logged_tracking_uncertainty_warmup:
                print(
                    "[Uncertainty] tracking warmup: "
                    f"keyframes={len(self.kf_indices)}/"
                    f"{self.tracking_uncertainty_warmup_keyframes}, "
                    f"backend_syncs={self.uncertainty_state_syncs}/"
                    f"{self.tracking_uncertainty_warmup_backend_syncs}"
                )
                self._logged_tracking_uncertainty_warmup = True
            return None

        if not self._logged_tracking_uncertainty_enabled:
            print(
                "[Uncertainty] tracking warmup done: "
                f"keyframes={len(self.kf_indices)}, "
                f"backend_syncs={self.uncertainty_state_syncs}"
            )
            self._logged_tracking_uncertainty_enabled = True
        return self.uncer_network

    def flow_pose_delta_reason(self, prev_R, prev_T, cur_R, cur_T):
        cfg = self.gaustar_stage1_cfg
        rel_R = cur_R @ prev_R.transpose(0, 1)
        cos_theta = torch.clamp((torch.trace(rel_R) - 1.0) * 0.5, -1.0, 1.0)
        rot_deg = float(torch.rad2deg(torch.arccos(cos_theta)).detach().cpu().item())
        trans = float(torch.linalg.norm(cur_T - prev_T).detach().cpu().item())

        max_rot = float(cfg.get("max_flow_pose_rotation_deg", 0.0))
        if max_rot > 0 and rot_deg > max_rot:
            return False, f"rotation too large ({rot_deg:.3f} deg)"

        max_trans_ratio = float(cfg.get("max_flow_pose_translation_ratio", 0.0))
        if max_trans_ratio > 0 and hasattr(self, "median_depth"):
            max_trans = max_trans_ratio * float(self.median_depth.detach().cpu().item())
            if trans > max_trans:
                return False, f"translation too large ({trans:.3f} > {max_trans:.3f})"
        return True, f"delta trans={trans:.3f}, rot={rot_deg:.3f}"

    def tracking_init_loss(self, viewpoint):
        with torch.no_grad():
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            loss = get_loss_tracking(
                self.config,
                image,
                depth,
                opacity,
                viewpoint,
                uncertainty_network=None,
            )
        return float(loss.detach().cpu().item())

    def get_lsg_feature_matcher(self):
        if self.lsg_feature_matcher is None:
            self.lsg_feature_matcher = LSGFeatureMatcher(
                self.lsg_pose_init_cfg, self.device
            )
        return self.lsg_feature_matcher

    def get_lsg_reference_keyframe(self):
        if len(self.current_window) > 0:
            return self.current_window[0]
        if len(self.kf_indices) > 0:
            return self.kf_indices[-1]
        return None

    def get_lsg_reference_depth(self, viewpoint):
        priors = getattr(viewpoint, "priors", {}) or {}
        for key in ("lsg_keyframe_depth", "rendered_depth", "metric_depth"):
            if key in priors and priors[key] is not None:
                return priors[key]
        return viewpoint.depth

    def log_lsg_pose_init(
        self,
        cur_frame_idx,
        ref_idx,
        matches=0,
        inliers=0,
        accepted=False,
        reason="",
    ):
        if not self.lsg_pose_init_cfg.get("log_every_frame", True):
            return
        print(
            "[LSG PoseInit] "
            f"frame={cur_frame_idx}, ref={ref_idx}, matches={matches}, "
            f"inliers={inliers}, accepted={accepted}, reason={reason}"
        )

    def try_lsg_pose_init(self, cur_frame_idx, viewpoint, prev):
        cfg = self.lsg_pose_init_cfg
        if not cfg.get("enabled", False):
            return False
        if cfg.get("pose_after_init", True) and not self.initialized:
            self.log_lsg_pose_init(
                cur_frame_idx, None, accepted=False, reason="waiting for init"
            )
            return False

        ref_idx = self.get_lsg_reference_keyframe()
        ref = self.cameras.get(ref_idx) if ref_idx is not None else None
        if ref is None:
            self.log_lsg_pose_init(
                cur_frame_idx, ref_idx, accepted=False, reason="missing reference"
            )
            return False

        ref_depth = self.get_lsg_reference_depth(ref)
        if ref_depth is None:
            self.log_lsg_pose_init(
                cur_frame_idx, ref_idx, accepted=False, reason="missing ref depth"
            )
            return False

        matcher = self.get_lsg_feature_matcher()
        match_data, match_reason = matcher.match(viewpoint, ref)
        if match_data is None:
            self.log_lsg_pose_init(
                cur_frame_idx, ref_idx, accepted=False, reason=match_reason
            )
            return False

        result = estimate_pose_from_matches(
            match_data["kpts_cur"],
            match_data["kpts_ref"],
            ref,
            viewpoint,
            ref_depth,
            cfg,
        )
        if not result.accepted:
            self.log_lsg_pose_init(
                cur_frame_idx,
                ref_idx,
                matches=match_data["num_matches"],
                inliers=result.num_inliers,
                accepted=False,
                reason=result.reason,
            )
            return False

        prev_R = prev.R.detach().cpu()
        prev_T = prev.T.detach().cpu()
        median_depth = None
        if hasattr(self, "median_depth"):
            median_depth = float(self.median_depth.detach().cpu().item())
        delta_ok, delta_reason = pose_delta_reason(
            prev_R.tolist(),
            prev_T.tolist(),
            result.R.tolist(),
            result.T.tolist(),
            cfg,
            median_depth=median_depth,
        )
        if not delta_ok:
            self.log_lsg_pose_init(
                cur_frame_idx,
                ref_idx,
                matches=match_data["num_matches"],
                inliers=result.num_inliers,
                accepted=False,
                reason=delta_reason,
            )
            return False

        candidate_R = torch.from_numpy(result.R).to(device=viewpoint.device)
        candidate_T = torch.from_numpy(result.T).to(device=viewpoint.device)

        if cfg.get("compare_init_loss", True) and self.gaussians is not None:
            viewpoint.update_RT(candidate_R, candidate_T)
            candidate_loss = self.tracking_init_loss(viewpoint)
            viewpoint.update_RT(prev.R, prev.T)
            fallback_loss = self.tracking_init_loss(viewpoint)
            improvement = float(cfg.get("loss_improvement", 0.0))
            if candidate_loss > fallback_loss * (1.0 - improvement):
                self.log_lsg_pose_init(
                    cur_frame_idx,
                    ref_idx,
                    matches=match_data["num_matches"],
                    inliers=result.num_inliers,
                    accepted=False,
                    reason=(
                        "init loss rejected "
                        f"({candidate_loss:.6f} >= {fallback_loss:.6f})"
                    ),
                )
                return False

        viewpoint.update_RT(candidate_R, candidate_T)
        self.log_lsg_pose_init(
            cur_frame_idx,
            ref_idx,
            matches=match_data["num_matches"],
            inliers=result.num_inliers,
            accepted=True,
            reason=(
                f"inlier_ratio={result.inlier_ratio:.3f}, "
                f"reproj={result.reproj_error:.3f}, {delta_reason}"
            ),
        )
        return True

    def initialize_tracking_pose(self, cur_frame_idx, viewpoint, prev):
        viewpoint.update_RT(prev.R, prev.T)
        if self.try_lsg_pose_init(cur_frame_idx, viewpoint, prev):
            return
        if not gaustar_stage1_enabled(self.config):
            return
        if not getattr(prev, "priors", None) and prev.uid in self._gaustar_recent_priors:
            prev.priors = self._gaustar_recent_priors[prev.uid]
        cfg = self.gaustar_stage1_cfg
        if not cfg.get("use_flow_pose_init", True):
            return
        if cfg.get("flow_pose_after_init", True) and not self.initialized:
            return
        prev_R = prev.R.detach().clone()
        prev_T = prev.T.detach().clone()
        ok, reason = initialize_pose_from_flow(prev, viewpoint, cfg)
        if ok:
            delta_ok, delta_reason = self.flow_pose_delta_reason(
                prev_R, prev_T, viewpoint.R, viewpoint.T
            )
            if not delta_ok:
                viewpoint.update_RT(prev_R, prev_T)
                if not self._logged_flow_pose_reject:
                    print(
                        "[GauSTAR Stage1] flow pose initialization rejected "
                        f"({delta_reason}); using previous-frame pose."
                    )
                    self._logged_flow_pose_reject = True
                return

            if cfg.get("flow_pose_compare_init_loss", True):
                pnp_R = viewpoint.R.detach().clone()
                pnp_T = viewpoint.T.detach().clone()
                pnp_loss = self.tracking_init_loss(viewpoint)
                viewpoint.update_RT(prev_R, prev_T)
                prev_loss = self.tracking_init_loss(viewpoint)
                improvement = float(cfg.get("flow_pose_loss_improvement", 0.0))
                if pnp_loss > prev_loss * (1.0 - improvement):
                    if not self._logged_flow_pose_reject:
                        print(
                            "[GauSTAR Stage1] flow pose initialization rejected "
                            f"(pnp_loss={pnp_loss:.6f}, prev_loss={prev_loss:.6f}); "
                            "using previous-frame pose."
                        )
                        self._logged_flow_pose_reject = True
                    return
                viewpoint.update_RT(pnp_R, pnp_T)

            if not self._logged_flow_pose_init:
                print(
                    "[GauSTAR Stage1] flow pose initialization enabled "
                    f"({reason})"
                )
                self._logged_flow_pose_init = True
            return
        if not self._logged_flow_pose_fallback:
            print(
                "[GauSTAR Stage1] flow pose initialization unavailable "
                f"({reason}); using previous-frame pose."
            )
            self._logged_flow_pose_fallback = True

    def store_lsg_keyframe_depth(self, viewpoint, depth_map):
        if not self.lsg_pose_init_cfg.get("enabled", False) or depth_map is None:
            return
        if hasattr(depth_map, "detach"):
            depth_map = depth_map.detach().cpu().numpy()
        viewpoint.priors["lsg_keyframe_depth"] = np.asarray(depth_map).copy()

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            gaustar_cfg = self.gaustar_stage1_cfg
            if (
                gaustar_stage1_enabled(self.config)
                and gaustar_cfg.get("use_metric3d_depth", True)
                and gaustar_cfg.get("apply_filter_to_keyframe_depth", True)
            ):
                metric_initial_depth = get_metric_depth_for_initialization(
                    self.config,
                    viewpoint,
                    depth=depth,
                    opacity=opacity,
                    mask=valid_rgb,
                )
                if metric_initial_depth is not None:
                    depth_map = metric_initial_depth.cpu().numpy()[0]
                    self.store_lsg_keyframe_depth(viewpoint, depth_map)
                    return depth_map
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            depth_map = initial_depth.cpu().numpy()[0]
            self.store_lsg_keyframe_depth(viewpoint, depth_map)
            return depth_map
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        depth_map = initial_depth[0].numpy()
        self.store_lsg_keyframe_depth(viewpoint, depth_map)
        return depth_map

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        self.initialize_tracking_pose(cur_frame_idx, viewpoint, prev)
        use_blur_tracking = self.use_blur_aware_tracking()
        if use_blur_tracking:
            ensure_blur_motion_params(viewpoint)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        if use_blur_tracking:
            opt_params.append(
                {
                    "params": [viewpoint.blur_rot_delta],
                    "lr": self.config["Training"]["lr"].get(
                        "blur_rot_delta",
                        self.config["Training"]["lr"]["cam_rot_delta"],
                    ),
                    "name": "blur_rot_{}".format(viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.blur_trans_delta],
                    "lr": self.config["Training"]["lr"].get(
                        "blur_trans_delta",
                        self.config["Training"]["lr"]["cam_trans_delta"],
                    ),
                    "name": "blur_trans_{}".format(viewpoint.uid),
                }
            )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        try:
            for tracking_itr in range(self.tracking_itr_num):
                if use_blur_tracking:
                    render_pkg = render_blur_aware_tracking(
                        viewpoint,
                        self.gaussians,
                        self.pipeline_params,
                        self.background,
                        self.blur_tracking_cfg,
                    )
                else:
                    render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background
                    )
                image, depth, opacity = (
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )
                pose_optimizer.zero_grad()
                if getattr(viewpoint, "priors", None) is not None:
                    viewpoint.priors["slam_initialized"] = self.initialized
                loss_tracking = get_loss_tracking(
                    self.config,
                    image,
                    depth,
                    opacity,
                    viewpoint,
                    uncertainty_network=self.get_tracking_uncertainty_network(),
                )
                if use_blur_tracking:
                    loss_tracking = loss_tracking + blur_motion_l2(
                        viewpoint, self.blur_tracking_cfg
                    )
                loss_tracking.backward()

                with torch.no_grad():
                    pose_optimizer.step()
                    converged = update_pose(viewpoint)

                if tracking_itr % 10 == 0:
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            current_frame=viewpoint,
                            gtcolor=viewpoint.original_image,
                            gtdepth=viewpoint.depth
                            if not self.monocular
                            else np.zeros(
                                (viewpoint.image_height, viewpoint.image_width)
                            ),
                        )
                    )
                if converged:
                    break
        finally:
            if use_blur_tracking:
                reset_blur_motion(viewpoint)

        self.median_depth = get_median_depth(depth, opacity)
        if getattr(viewpoint, "priors", None) is not None:
            viewpoint.priors["rendered_depth"] = depth.detach().cpu()
            consistency_mask = build_rendered_depth_consistency_mask(
                self.config, depth, opacity, viewpoint
            )
            if consistency_mask is not None:
                viewpoint.priors["rendered_depth_consistency_mask"] = (
                    consistency_mask.detach().cpu()
                )
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        uncer_state = data[4] if len(data) > 4 else None
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())
        if self.uncer_network is not None and uncer_state is not None:
            self.uncer_network.load_state_dict(
                {k: v.to(self.device) for k, v in uncer_state.items()}
            )
            self.uncer_network.eval()
            if data[0] != "init":
                self.uncertainty_state_syncs += 1

    def cleanup(self, cur_frame_idx):
        if gaustar_stage1_enabled(self.config):
            priors = getattr(self.cameras[cur_frame_idx], "priors", {}) or {}
            retained = {
                key: priors[key]
                for key in ("metric_depth", "prior_valid_mask")
                if key in priors
            }
            if retained:
                retained["gaustar_stage1"] = True
                self._gaustar_recent_priors[cur_frame_idx] = retained
            for old_idx in list(self._gaustar_recent_priors.keys()):
                if old_idx < cur_frame_idx - self.use_every_n_frames:
                    del self._gaustar_recent_priors[old_idx]
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    if hasattr(self.dataset, "log_feature_stats"):
                        self.dataset.log_feature_stats(" (frontend)")
                    log_uncertainty_debug_stats(" (frontend)")
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                # 2. 处于“平稳运行期” (当滑窗已满)
                # 当滑窗填满后，代码就不会进入这个 if 分支了。
                # 此时 create_kf 就完全听从 is_keyframe 的指挥。
                # 因为此时地图已经稳了，物理距离的判断变得可靠，系统可以更灵活地根据位移来创建关键帧。
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                # 1. 处于“初始化期” (当 len(self.current_window) < self.window_size)
                # 此时滑窗还没填满（比如滑窗大小是 8，现在才攒了 3 帧），系统处于极其脆弱的“打地基”阶段。
                # 为什么要重写？：在 is_keyframe 内部，判断逻辑包含了“物理距离”。但在刚开机时，由于尺度（Scale）还没算准，median_depth 可能一直在跳，导致物理距离的判断不准。
                # 这里的策略：在这个阶段，系统改用了一套更保守、更死板的逻辑：
                # 必须满足 check_time（强制要求每隔固定帧数才能建一个）。
                # 必须满足 point_ratio < kf_overlap（画面确实变了）。
                # 它故意忽略了 is_keyframe 里的“大步流星”逻辑，防止在初始化还没稳的时候，因为相机晃动一下就乱建一堆关键帧。
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                # 如果是单线程模式，系统会进一步“加码”限制：不管你重合度多低，必须先满足时间间隔。这是为了防止前端产生关键帧太快，导致单线程的后端处理不过来，引发系统卡死。
                if self.single_thread:
                    create_kf = check_time and create_kf
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
