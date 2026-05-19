#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p


class GaussianModel:
    def __init__(self, sh_degree: int, config=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")

        self.unique_kfIDs = torch.empty(0).int()
        self.n_obs = torch.empty(0).int()

        self.config = config
        self.ply_input = None

        self.isotropic = False
        self.forgetting_enabled = (
            config is not None
            and config.get("Training", {})
            .get("forgetting_regularization", {})
            .get("enabled", False)
        )
        self._clear_forgetting_statistics()
        if self.forgetting_enabled:
            self._init_empty_forgetting_statistics()

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_pcd_from_image(
        self,
        cam_info,
        init=False,
        scale=2.0,
        depthmap=None,
        mask=None,
        opacity_value=0.5,
        use_valid_depth_median=False,
        pcd_downsample_factor=None,
        return_point_count=False,
    ):
        cam = cam_info
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            if mask is not None:
                depthmap = depthmap.copy()
                depthmap[~mask.astype(bool)] = 0.0
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            if mask is not None:
                depth_raw = depth_raw.copy()
                depth_raw[~mask.astype(bool)] = 0.0

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        return self.create_pcd_from_image_and_depth(
            cam,
            rgb,
            depth,
            init,
            opacity_value=opacity_value,
            use_valid_depth_median=use_valid_depth_median,
            pcd_downsample_factor=pcd_downsample_factor,
            return_point_count=return_point_count,
        )

    def create_pcd_from_image_and_depth(
        self,
        cam,
        rgb,
        depth,
        init=False,
        opacity_value=0.5,
        use_valid_depth_median=False,
        pcd_downsample_factor=None,
        return_point_count=False,
    ):
        if pcd_downsample_factor is not None:
            downsample_factor = pcd_downsample_factor
        elif init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                if use_valid_depth_median:
                    depth_values = np.asarray(depth)
                    valid_depth = depth_values[depth_values > 0]
                    if valid_depth.shape[0] > 0:
                        point_size = min(0.05, point_size * np.median(valid_depth))
                else:
                    point_size = min(0.05, point_size * np.median(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        pcd_count_before_downsample = len(pcd_tmp.points)
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)
        point_count = {
            "pcd_before_downsample": pcd_count_before_downsample,
            "pcd_after_downsample": new_xyz.shape[0],
        }

        if new_xyz.shape[0] == 0:
            empty_xyz = torch.empty(0, 3, device="cuda")
            empty_features = torch.empty(
                0, 3, (self.max_sh_degree + 1) ** 2, device="cuda"
            )
            empty_scaling = torch.empty(0, 3, device="cuda")
            if self.isotropic:
                empty_scaling = torch.empty(0, 1, device="cuda")
            empty_rotation = torch.empty(0, 4, device="cuda")
            empty_opacity = torch.empty(0, 1, device="cuda")
            result = (
                empty_xyz,
                empty_features,
                empty_scaling,
                empty_rotation,
                empty_opacity,
            )
            if return_point_count:
                return result, point_count
            return result

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacity_value = float(np.clip(opacity_value, 0.0001, 0.9999))
        opacities = inverse_sigmoid(
            opacity_value
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        result = (fused_point_cloud, features, scales, rots, opacities)
        if return_point_count:
            return result, point_count
        return result

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(
        self,
        fused_point_cloud,
        features,
        scales,
        rots,
        opacities,
        kf_id,
        reset_densification_stats=True,
    ):
        if fused_point_cloud.shape[0] == 0:
            return 0

        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
            reset_densification_stats=reset_densification_stats,
        )
        return new_xyz.shape[0]

    def extend_from_pcd_seq(
        self,
        cam_info,
        kf_id=-1,
        init=False,
        scale=2.0,
        depthmap=None,
        mask=None,
        reset_densification_stats=True,
        opacity_value=0.5,
        use_valid_depth_median=False,
        pcd_downsample_factor=None,
        return_point_count=False,
    ):
        pcd_result = self.create_pcd_from_image(
            cam_info,
            init,
            scale=scale,
            depthmap=depthmap,
            mask=mask,
            opacity_value=opacity_value,
            use_valid_depth_median=use_valid_depth_median,
            pcd_downsample_factor=pcd_downsample_factor,
            return_point_count=return_point_count,
        )
        if return_point_count:
            pcd_result, point_count = pcd_result
        fused_point_cloud, features, scales, rots, opacities = pcd_result
        added = self.extend_from_pcd(
            fused_point_cloud,
            features,
            scales,
            rots,
            opacities,
            kf_id,
            reset_densification_stats=reset_densification_stats,
        )
        if return_point_count:
            return added, point_count
        return added

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()
        self.reset_forgetting_statistics()

    def _clear_forgetting_statistics(self):
        self.seen_times = None
        self.last_xyz = None
        self.last_features_dc = None
        self.last_scaling = None
        self.last_opacity = None
        self.xyz_importance_weights = None
        self.features_dc_importance_weights = None
        self.scaling_importance_weights = None
        self.opacity_importance_weights = None

    def _init_empty_forgetting_statistics(self):
        self.seen_times = torch.empty(0, 1, device="cuda")
        self.last_xyz = torch.empty(0, 3, device="cuda")
        self.last_features_dc = torch.empty(0, 1, 3, device="cuda")
        self.last_scaling = torch.empty(0, 3, device="cuda")
        self.last_opacity = torch.empty(0, 1, device="cuda")
        self.xyz_importance_weights = torch.empty(0, 3, device="cuda")
        self.features_dc_importance_weights = torch.empty(0, 1, 3, device="cuda")
        self.scaling_importance_weights = torch.empty(0, 3, device="cuda")
        self.opacity_importance_weights = torch.empty(0, 1, device="cuda")

    def reset_forgetting_statistics(self):
        if not self.forgetting_enabled:
            self._clear_forgetting_statistics()
            return
        self.seen_times = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.last_xyz = self._xyz.detach().clone()
        self.last_features_dc = self._features_dc.detach().clone()
        self.last_scaling = self._scaling.detach().clone()
        self.last_opacity = self._opacity.detach().clone()
        self.xyz_importance_weights = torch.zeros_like(self._xyz)
        self.features_dc_importance_weights = torch.zeros_like(self._features_dc)
        self.scaling_importance_weights = torch.zeros_like(self._scaling)
        self.opacity_importance_weights = torch.zeros_like(self._opacity)

    def _ensure_forgetting_statistics(self):
        if not self.forgetting_enabled:
            return
        if self.seen_times is None or self.seen_times.shape[0] != self.get_xyz.shape[0]:
            self.reset_forgetting_statistics()

    def _append_forgetting_statistics(
        self, new_xyz, new_features_dc, new_scaling, new_opacity
    ):
        if not self.forgetting_enabled:
            return
        self._ensure_forgetting_statistics()
        count = new_xyz.shape[0]
        if count == 0:
            return
        self.seen_times = torch.cat(
            (self.seen_times, torch.zeros((count, 1), device="cuda")), dim=0
        )
        self.last_xyz = torch.cat((self.last_xyz, new_xyz.detach().clone()), dim=0)
        self.last_features_dc = torch.cat(
            (self.last_features_dc, new_features_dc.detach().clone()), dim=0
        )
        self.last_scaling = torch.cat(
            (self.last_scaling, new_scaling.detach().clone()), dim=0
        )
        self.last_opacity = torch.cat(
            (self.last_opacity, new_opacity.detach().clone()), dim=0
        )
        self.xyz_importance_weights = torch.cat(
            (self.xyz_importance_weights, torch.zeros_like(new_xyz)), dim=0
        )
        self.features_dc_importance_weights = torch.cat(
            (
                self.features_dc_importance_weights,
                torch.zeros_like(new_features_dc),
            ),
            dim=0,
        )
        self.scaling_importance_weights = torch.cat(
            (self.scaling_importance_weights, torch.zeros_like(new_scaling)), dim=0
        )
        self.opacity_importance_weights = torch.cat(
            (self.opacity_importance_weights, torch.zeros_like(new_opacity)), dim=0
        )

    def _prune_forgetting_statistics(self, valid_points_mask):
        if not self.forgetting_enabled:
            return
        self._ensure_forgetting_statistics()
        mask = valid_points_mask.to(device="cuda")
        self.seen_times = self.seen_times[mask]
        self.last_xyz = self.last_xyz[mask]
        self.last_features_dc = self.last_features_dc[mask]
        self.last_scaling = self.last_scaling[mask]
        self.last_opacity = self.last_opacity[mask]
        self.xyz_importance_weights = self.xyz_importance_weights[mask]
        self.features_dc_importance_weights = self.features_dc_importance_weights[
            mask
        ]
        self.scaling_importance_weights = self.scaling_importance_weights[
            mask
        ]
        self.opacity_importance_weights = self.opacity_importance_weights[
            mask
        ]

    def capture_forgetting_snapshot(self):
        if not self.forgetting_enabled:
            return
        self._ensure_forgetting_statistics()
        self.last_xyz = self._xyz.detach().clone()
        self.last_features_dc = self._features_dc.detach().clone()
        self.last_scaling = self._scaling.detach().clone()
        self.last_opacity = self._opacity.detach().clone()

    def update_forgetting_importance(self, visibility_filters, normalize=True):
        if not self.forgetting_enabled:
            return
        self._ensure_forgetting_statistics()
        if self.get_xyz.shape[0] == 0:
            return

        visible = torch.zeros((self.get_xyz.shape[0]), dtype=torch.bool, device="cuda")
        for visibility_filter in visibility_filters:
            if visibility_filter.shape[0] != visible.shape[0]:
                print(
                    "Warning: skipping forgetting importance update due to "
                    "visibility/gaussian shape mismatch."
                )
                return
            visible = torch.logical_or(visible, visibility_filter)
        if not visible.any():
            return

        self.seen_times[visible] += 1
        seen_count = torch.clamp(self.seen_times, min=1.0)

        if self._xyz.grad is not None:
            xyz_grad = torch.abs(self._xyz.grad.detach())
            if normalize:
                self.xyz_importance_weights[visible] = (
                    self.xyz_importance_weights[visible]
                    * (self.seen_times[visible] - 1)
                    + xyz_grad[visible]
                ) / seen_count[visible]
            else:
                self.xyz_importance_weights[visible] += xyz_grad[visible]

        if self._features_dc.grad is not None:
            features_grad = torch.abs(self._features_dc.grad.detach())
            if normalize:
                self.features_dc_importance_weights[visible] = (
                    self.features_dc_importance_weights[visible]
                    * (self.seen_times[visible].view(-1, 1, 1) - 1)
                    + features_grad[visible]
                ) / seen_count[visible].view(-1, 1, 1)
            else:
                self.features_dc_importance_weights[visible] += features_grad[visible]

        if self._scaling.grad is not None:
            scaling_grad = torch.abs(self._scaling.grad.detach())
            if normalize:
                self.scaling_importance_weights[visible] = (
                    self.scaling_importance_weights[visible]
                    * (self.seen_times[visible] - 1)
                    + scaling_grad[visible]
                ) / seen_count[visible]
            else:
                self.scaling_importance_weights[visible] += scaling_grad[visible]

        if self._opacity.grad is not None:
            opacity_grad = torch.abs(self._opacity.grad.detach())
            if normalize:
                self.opacity_importance_weights[visible] = (
                    self.opacity_importance_weights[visible]
                    * (self.seen_times[visible] - 1)
                    + opacity_grad[visible]
                ) / seen_count[visible]
            else:
                self.opacity_importance_weights[visible] += opacity_grad[visible]

    def forgetting_regularization_loss(self, cfg):
        if not self.forgetting_enabled:
            return self.get_xyz.sum() * 0.0
        self._ensure_forgetting_statistics()
        if self.get_xyz.shape[0] == 0:
            return self.get_xyz.sum() * 0.0

        loss = self.get_xyz.sum() * 0.0
        lambda_color = cfg.get("lambda_color", 0.01)
        lambda_scaling = cfg.get("lambda_scaling", 0.01)
        lambda_xyz = cfg.get("lambda_xyz", 0.0)
        lambda_opacity = cfg.get("lambda_opacity", 0.0)

        if lambda_color > 0:
            loss = loss + lambda_color * (
                self.features_dc_importance_weights
                * torch.abs(self._features_dc - self.last_features_dc)
            ).mean()
        if lambda_scaling > 0:
            loss = loss + lambda_scaling * (
                self.scaling_importance_weights
                * torch.abs(self._scaling - self.last_scaling)
            ).mean()
        if lambda_xyz > 0:
            loss = loss + lambda_xyz * (
                self.xyz_importance_weights * torch.abs(self._xyz - self.last_xyz)
            ).mean()
        if lambda_opacity > 0:
            loss = loss + lambda_opacity * (
                self.opacity_importance_weights
                * torch.abs(self._opacity - self.last_opacity)
            ).mean()
        return loss

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        self._prune_forgetting_statistics(valid_points_mask)
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
        reset_densification_stats=True,
    ):
        self._append_forgetting_statistics(
            new_xyz, new_features_dc, new_scaling, new_opacities
        )
        old_xyz_gradient_accum = self.xyz_gradient_accum
        old_denom = self.denom
        old_max_radii2D = self.max_radii2D

        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if reset_densification_stats or old_xyz_gradient_accum.shape[0] == 0:
            self.xyz_gradient_accum = torch.zeros(
                (self.get_xyz.shape[0], 1), device="cuda"
            )
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        else:
            self.xyz_gradient_accum = torch.cat(
                (
                    old_xyz_gradient_accum,
                    torch.zeros((new_xyz.shape[0], 1), device="cuda"),
                ),
                dim=0,
            )
            self.denom = torch.cat(
                (old_denom, torch.zeros((new_xyz.shape[0], 1), device="cuda")),
                dim=0,
            )
            self.max_radii2D = torch.cat(
                (
                    old_max_radii2D,
                    torch.zeros((new_xyz.shape[0]), device="cuda"),
                ),
                dim=0,
            )
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
