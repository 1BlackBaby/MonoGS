import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping, log_uncertainty_debug_stats


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.uncer_network = None
        self.uncer_optimizer = None

    def _require_nonnegative_number(self, cfg, key, display_key):
        value = cfg.get(key)
        if value is None or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"Training.{display_key} must be a non-negative number")
        return value

    def _load_render_guided_config(self):
        cfg = self.config.get("Training", {}).get("render_guided_densification", {})
        if not cfg.get("enabled", False):
            return cfg

        for key in ("opacity_threshold", "rgb_error_threshold", "min_depth"):
            self._require_nonnegative_number(
                cfg, key, f"render_guided_densification.{key}"
            )
        for key in ("max_new_points_per_kf", "max_new_points_per_update"):
            value = cfg.get(key, cfg.get("max_new_points_per_kf", 3000))
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Training.render_guided_densification.{key} must be a non-negative integer"
                )
        for key in ("trigger_every", "downsample"):
            value = cfg.get(key)
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"Training.render_guided_densification.{key} must be an integer >= 1"
                )
        pcd_downsample = cfg.get("pcd_downsample", 4)
        if not isinstance(pcd_downsample, int) or pcd_downsample < 1:
            raise ValueError(
                "Training.render_guided_densification.pcd_downsample must be an integer >= 1"
            )
        trigger_offset = cfg.get("trigger_offset")
        if not isinstance(trigger_offset, int) or trigger_offset < 0:
            raise ValueError(
                "Training.render_guided_densification.trigger_offset must be a non-negative integer"
            )
        initial_opacity = cfg.get("initial_opacity", 0.8)
        if (
            not isinstance(initial_opacity, (int, float))
            or initial_opacity <= 0
            or initial_opacity >= 1
        ):
            raise ValueError(
                "Training.render_guided_densification.initial_opacity must be in (0, 1)"
            )
        if cfg.get("use_depth_error", False):
            self._require_nonnegative_number(
                cfg,
                "depth_error_ratio_threshold",
                "render_guided_densification.depth_error_ratio_threshold",
            )
        return cfg

    def _load_forgetting_regularization_config(self):
        cfg = self.config.get("Training", {}).get("forgetting_regularization", {})
        if not cfg.get("enabled", False):
            return cfg

        update_every = cfg.get("update_every_kf", 1)
        if not isinstance(update_every, int) or update_every < 1:
            raise ValueError(
                "Training.forgetting_regularization.update_every_kf must be an integer >= 1"
            )
        for key in ("lambda_color", "lambda_scaling", "lambda_xyz", "lambda_opacity"):
            value = cfg.get(key, 0.0)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(
                    f"Training.forgetting_regularization.{key} must be a non-negative number"
                )
        return cfg

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
        self.uncertainty_cfg = self.config.get("Training", {}).get("uncertainty", {})
        self.render_guided_cfg = self._load_render_guided_config()
        self.forgetting_reg_cfg = self._load_forgetting_regularization_config()
        if self.uncertainty_cfg.get("enabled", False) and self.uncer_network is not None:
            self.uncer_optimizer = torch.optim.Adam(
                self.uncer_network.parameters(),
                lr=self.uncertainty_cfg.get("lr", 0.0004),
                weight_decay=self.uncertainty_cfg.get("weight_decay", 0.00001),
            )

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        return self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def add_render_guided_gaussians(
        self, frame_idx, viewpoint, image, depth, opacity, max_new_points
    ):
        cfg = self.render_guided_cfg
        if not cfg.get("enabled", False) or max_new_points <= 0:
            return 0

        with torch.no_grad():
            gt_image = viewpoint.original_image.to(device=image.device)
            image_ab = torch.exp(viewpoint.exposure_a) * image + viewpoint.exposure_b
            image_ab = torch.clamp(image_ab, 0.0, 1.0)
            rgb_error_map = torch.abs(image_ab - gt_image).mean(dim=0)
            opacity_map = opacity.squeeze()

            densify_mask = torch.zeros_like(opacity_map, dtype=torch.bool)
            score_map = torch.zeros_like(opacity_map, dtype=rgb_error_map.dtype)
            opacity_candidates = 0
            rgb_candidates = 0
            depth_candidates = 0
            if cfg.get("use_opacity_hole", True):
                opacity_score = torch.clamp(
                    cfg.get("opacity_threshold", 0.5) - opacity_map, min=0.0
                )
                opacity_candidate_mask = opacity_map < cfg.get(
                    "opacity_threshold", 0.5
                )
                densify_mask = torch.logical_or(
                    densify_mask,
                    opacity_candidate_mask,
                )
                score_map = score_map + opacity_score
                opacity_candidates = int(opacity_candidate_mask.count_nonzero().item())
            if cfg.get("use_rgb_error", True):
                rgb_score = torch.clamp(
                    rgb_error_map - cfg.get("rgb_error_threshold", 0.25), min=0.0
                )
                rgb_candidate_mask = rgb_error_map > cfg.get(
                    "rgb_error_threshold", 0.25
                )
                densify_mask = torch.logical_or(
                    densify_mask,
                    rgb_candidate_mask,
                )
                score_map = score_map + rgb_score
                rgb_candidates = int(rgb_candidate_mask.count_nonzero().item())

            if cfg.get("use_depth_error", False) and viewpoint.depth is not None:
                gt_depth_for_error = torch.from_numpy(viewpoint.depth).to(
                    dtype=torch.float32, device=image.device
                )
                depth_diff_ratio = torch.abs(gt_depth_for_error - depth.squeeze()) / (
                    gt_depth_for_error + 1e-6
                )
                depth_score = torch.clamp(
                    depth_diff_ratio
                    - cfg.get("depth_error_ratio_threshold", 0.1),
                    min=0.0,
                )
                depth_candidate_mask = depth_diff_ratio > cfg.get(
                    "depth_error_ratio_threshold", 0.1
                )
                densify_mask = torch.logical_or(
                    densify_mask,
                    depth_candidate_mask,
                )
                score_map = score_map + depth_score
                depth_candidates = int(depth_candidate_mask.count_nonzero().item())

            rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
            valid_rgb = gt_image.sum(dim=0) > rgb_boundary_threshold
            candidate_before_valid = int(densify_mask.count_nonzero().item())
            valid_rgb_count = int(valid_rgb.count_nonzero().item())
            densify_mask = torch.logical_and(densify_mask, valid_rgb)

            if viewpoint.depth is not None:
                depth_source = torch.from_numpy(viewpoint.depth).to(
                    dtype=torch.float32, device=image.device
                )
            else:
                depth_source = depth.squeeze().detach()
            min_depth = cfg.get("min_depth", 0.01)
            valid_depth = depth_source > min_depth
            valid_depth_count = int(valid_depth.count_nonzero().item())
            densify_mask = torch.logical_and(densify_mask, valid_depth)
            selected_after_valid = int(densify_mask.count_nonzero().item())

            downsample = max(1, int(cfg.get("downsample", 1)))
            if downsample > 1:
                grid_y = torch.arange(
                    densify_mask.shape[0], device=densify_mask.device
                ).view(-1, 1)
                grid_x = torch.arange(
                    densify_mask.shape[1], device=densify_mask.device
                ).view(1, -1)
                sample_mask = torch.logical_and(
                    grid_y % downsample == 0, grid_x % downsample == 0
                )
                densify_mask = torch.logical_and(densify_mask, sample_mask)

            selected_count = int(densify_mask.count_nonzero().item())
            selected_before_budget = selected_count
            if selected_count == 0:
                Log(
                    "Render-guided diagnostics "
                    f"kf={frame_idx}: opacity={opacity_candidates}, "
                    f"rgb={rgb_candidates}, depth={depth_candidates}, "
                    f"candidate={candidate_before_valid}, "
                    f"valid_rgb={valid_rgb_count}, valid_depth={valid_depth_count}, "
                    f"after_valid={selected_after_valid}, "
                    f"after_pixel_downsample={selected_count}, added=0"
                )
                return 0

            if max_new_points > 0 and selected_count > max_new_points:
                candidate_idx = densify_mask.flatten().nonzero(as_tuple=False).squeeze(1)
                candidate_score = score_map.flatten()[candidate_idx]
                if torch.any(candidate_score > 0):
                    topk = torch.topk(
                        candidate_score, k=max_new_points, largest=True
                    ).indices
                else:
                    topk = torch.arange(
                        max_new_points, device=candidate_idx.device
                    )
                limited_mask = torch.zeros_like(densify_mask.flatten(), dtype=torch.bool)
                limited_mask[candidate_idx[topk]] = True
                densify_mask = limited_mask.view_as(densify_mask)
                selected_count = int(densify_mask.count_nonzero().item())

            mask_np = densify_mask.detach().cpu().numpy().astype(bool)
            depth_np = depth_source.detach().cpu().numpy().astype("float32")

        pcd_downsample = max(1, int(cfg.get("pcd_downsample", 4)))
        added, point_count = self.gaussians.extend_from_pcd_seq(
            viewpoint,
            kf_id=frame_idx,
            init=False,
            depthmap=depth_np,
            mask=mask_np,
            reset_densification_stats=False,
            opacity_value=cfg.get(
                "initial_opacity", max(0.5, float(self.gaussian_th) + 0.05)
            ),
            use_valid_depth_median=True,
            pcd_downsample_factor=pcd_downsample,
            return_point_count=True,
        )
        Log(
            "Render-guided diagnostics "
            f"kf={frame_idx}: opacity={opacity_candidates}, "
            f"rgb={rgb_candidates}, depth={depth_candidates}, "
            f"candidate={candidate_before_valid}, "
            f"valid_rgb={valid_rgb_count}, valid_depth={valid_depth_count}, "
            f"after_valid={selected_after_valid}, "
            f"after_pixel_downsample={selected_before_budget}, "
            f"after_budget={selected_count}, "
            f"pcd_downsample={pcd_downsample}, "
            f"pcd_before={point_count['pcd_before_downsample']}, "
            f"pcd_after={point_count['pcd_after_downsample']}, added={added}"
        )
        if added > 0:
            Log("Render-guided densification added", added, "Gaussians")
        return added

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        self.gaussians.reset_forgetting_statistics()
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config,
                image,
                depth,
                viewpoint,
                opacity,
                initialization=True,
                uncertainty_network=self.uncer_network,
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                if self.uncer_optimizer is not None:
                    self.uncer_optimizer.step()
                    self.uncer_optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        render_guided_enabled = self.render_guided_cfg.get("enabled", False)

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []
            render_guided_candidates = [] if render_guided_enabled else None

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config,
                    image,
                    depth,
                    viewpoint,
                    opacity,
                    uncertainty_network=self.uncer_network,
                    regularization_viewpoints=viewpoint_stack[
                        max(0, cam_idx - 2) : min(len(viewpoint_stack), cam_idx + 3)
                    ],
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)
                if render_guided_enabled:
                    render_guided_candidates.append(
                        (
                            viewpoint.uid,
                            viewpoint,
                            image.detach(),
                            depth.detach(),
                            opacity.detach(),
                        )
                    )

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config,
                    image,
                    depth,
                    viewpoint,
                    opacity,
                    uncertainty_network=self.uncer_network,
                    regularization_viewpoints=viewpoint_stack[: min(len(viewpoint_stack), 5)],
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            if self.forgetting_reg_cfg.get("enabled", False) and not prune:
                loss_mapping += self.gaussians.forgetting_regularization_loss(
                    self.forgetting_reg_cfg
                )
            loss_mapping.backward()
            if self.forgetting_reg_cfg.get("enabled", False) and not prune:
                update_every = max(
                    1, int(self.forgetting_reg_cfg.get("update_every_kf", 1))
                )
                if self.iteration_count % update_every == 0:
                    self.gaussians.update_forgetting_importance(
                        visibility_filter_acm,
                        normalize=self.forgetting_reg_cfg.get(
                            "normalize_importance", True
                        ),
                    )
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    if self.uncer_optimizer is not None:
                        self.uncer_optimizer.zero_grad(set_to_none=True)
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    trigger_every = int(
                        self.render_guided_cfg.get(
                            "trigger_every", self.gaussian_update_every
                        )
                    )
                    trigger_offset = int(
                        self.render_guided_cfg.get(
                            "trigger_offset", self.gaussian_update_offset
                        )
                    )
                    run_render_guided = (
                        render_guided_enabled
                        and trigger_every > 0
                        and self.iteration_count % trigger_every == trigger_offset
                    )
                    if run_render_guided:
                        remaining_budget = int(
                            self.render_guided_cfg.get(
                                "max_new_points_per_update",
                                self.render_guided_cfg.get(
                                    "max_new_points_per_kf", 3000
                                ),
                            )
                        )
                        per_kf_limit = int(
                            self.render_guided_cfg.get("max_new_points_per_kf", 3000)
                        )
                        for (
                            frame_idx,
                            viewpoint,
                            image_rg,
                            depth_rg,
                            opacity_rg,
                        ) in render_guided_candidates:
                            if remaining_budget <= 0:
                                break
                            added = self.add_render_guided_gaussians(
                                frame_idx,
                                viewpoint,
                                image_rg,
                                depth_rg,
                                opacity_rg,
                                min(per_kf_limit, remaining_budget),
                            )
                            remaining_budget -= added
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                if self.forgetting_reg_cfg.get("enabled", False):
                    self.gaussians.capture_forgetting_snapshot()
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                if self.uncer_optimizer is not None:
                    self.uncer_optimizer.step()
                    self.uncer_optimizer.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        return gaussian_split

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        uncer_state = None
        if self.uncer_network is not None:
            uncer_state = {
                k: v.detach().cpu() for k, v in self.uncer_network.state_dict().items()
            }
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes, uncer_state]
        self.frontend_queue.put(msg)

    def release_resources(self):
        self.gaussians = None
        self.viewpoints = {}
        self.occ_aware_visibility = {}
        self.current_window = []
        self.keyframe_optimizers = None
        self.uncer_network = None
        self.uncer_optimizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    log_uncertainty_debug_stats(" (backend)")
                    self.release_resources()
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
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
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        return
