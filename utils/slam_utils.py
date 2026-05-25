import torch
import torch.nn.functional as F

from utils.dyn_uncertainty import mapping_utils
from utils.lsg_pose_init import get_lsg_pose_init_config
from utils.lsg_warp_loss import (
    compute_lsg_warp_loss,
    get_lsg_feature_warping_config,
)
from utils.mono_priors.gaustar_stage1 import (
    build_depth_consistency_mask,
    gaustar_stage1_enabled,
    get_gaustar_stage1_config,
    metric_depth_to_tensor,
)


_uncertainty_debug_stats = {
    "tracking_used": 0,
    "tracking_fallback": 0,
    "mapping_used": 0,
    "mapping_fallback": 0,
}


def _record_uncertainty_debug(name):
    _uncertainty_debug_stats[name] += 1
    if _uncertainty_debug_stats[name] == 1:
        print(f"[Uncertainty] {name.replace('_', ' ')}")


def log_uncertainty_debug_stats(tag=""):
    total = sum(_uncertainty_debug_stats.values())
    if total == 0:
        return
    print(
        f"[Uncertainty] loss stats{tag}: "
        f"tracking_used={_uncertainty_debug_stats['tracking_used']}, "
        f"tracking_fallback={_uncertainty_debug_stats['tracking_fallback']}, "
        f"mapping_used={_uncertainty_debug_stats['mapping_used']}, "
        f"mapping_fallback={_uncertainty_debug_stats['mapping_fallback']}"
    )


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def uncertainty_enabled(config):
    return config.get("Training", {}).get("uncertainty", {}).get("enabled", False)


def get_uncertainty_config(config):
    return config.get("Training", {}).get("uncertainty", {})


def build_rendered_depth_consistency_mask(config, depth, opacity, viewpoint):
    if not gaustar_stage1_enabled(config):
        return None
    cfg = get_gaustar_stage1_config(config)
    if not cfg.get("use_rendered_depth_filter", True):
        return None
    priors = getattr(viewpoint, "priors", {}) or {}
    metric_depth = priors.get("metric_depth")
    if metric_depth is None:
        return None
    prior_valid_mask = priors.get("prior_valid_mask")
    with torch.no_grad():
        mask, scale = build_depth_consistency_mask(
            metric_depth, depth, opacity, cfg, prior_valid_mask=prior_valid_mask
        )
        if mask is None:
            return None
        min_ratio = cfg.get("tracking_filter_min_ratio", 0.05)
        if float(mask.float().mean().item()) < min_ratio:
            return None
        priors["depth_scale"] = float(scale.detach().cpu().item())
        return mask.to(device=depth.device, dtype=torch.bool)


def get_metric_depth_for_initialization(
    config, viewpoint, depth=None, opacity=None, mask=None, use_lsg_metric3d=False
):
    cfg = get_gaustar_stage1_config(config)
    if use_lsg_metric3d:
        lsg_cfg = get_lsg_pose_init_config(config)
        if (
            not lsg_cfg.get("enabled", False)
            or not lsg_cfg.get("use_metric3d_depth", False)
            or not lsg_cfg.get("metric3d_for_keyframe_depth", True)
        ):
            return None
        filter_with_rendered_depth = lsg_cfg.get(
            "metric3d_filter_with_rendered_depth", True
        )
        require_rendered_depth = filter_with_rendered_depth
    else:
        if not gaustar_stage1_enabled(config):
            return None
        if not cfg.get("use_metric3d_depth", True):
            return None
        require_rendered_depth = cfg.get(
            "require_rendered_depth_for_keyframe_depth", True
        )
        filter_with_rendered_depth = depth is not None and opacity is not None
        if filter_with_rendered_depth and not cfg.get("use_rendered_depth_filter", True):
            return None
    if (
        require_rendered_depth
        and (depth is None or opacity is None)
    ):
        return None
    priors = getattr(viewpoint, "priors", {}) or {}
    metric_depth = priors.get("metric_depth")
    if metric_depth is None:
        return None
    device = depth.device if depth is not None else viewpoint.original_image.device
    metric = metric_depth_to_tensor(metric_depth, device=device)
    if metric is None:
        return None
    valid = metric > cfg.get("min_depth", 0.01)
    if mask is not None:
        valid = torch.logical_and(valid, mask.to(device=device, dtype=torch.bool))

    if filter_with_rendered_depth and depth is not None and opacity is not None:
        prior_valid_mask = priors.get("prior_valid_mask")
        consistency_mask, scale = build_depth_consistency_mask(
            metric, depth, opacity, cfg, prior_valid_mask=prior_valid_mask
        )
        if consistency_mask is None or scale is None:
            return None
        priors["depth_scale"] = float(scale.detach().cpu().item())
        valid = torch.logical_and(valid, consistency_mask)
        metric = metric * float(scale)

    min_ratio = cfg.get("tracking_filter_min_ratio", 0.05)
    if float(valid.float().mean().item()) < min_ratio:
        return None
    metric = metric.detach().clone()
    metric[~valid] = 0.0
    return metric


def has_uncertainty_features(viewpoint):
    return getattr(viewpoint, "features", None) is not None


def uncertainty_to_weight(uncertainty, uncertainty_config):
    min_uncertainty = uncertainty_config.get("min_uncertainty", 0.1)
    eps = uncertainty_config.get("eps", 1e-3)
    weights = 0.5 / (torch.clamp(uncertainty, min=min_uncertainty) + eps) ** 2
    weights = torch.clamp(weights, min=0.0, max=1.0)
    min_weight_threshold = uncertainty_config.get("min_weight_threshold", 0.1)
    if min_weight_threshold is not None and min_weight_threshold > 0:
        weights = torch.where(weights < min_weight_threshold, torch.zeros_like(weights), weights)
    return weights


def predict_uncertainty_weight(
    uncertainty_network, viewpoint, target_shape, uncertainty_config, no_grad=False
):
    if uncertainty_network is None or not has_uncertainty_features(viewpoint):
        return None, None

    features = viewpoint.features.to(device=next(uncertainty_network.parameters()).device)
    def _predict():
        uncertainty = uncertainty_network(features)
        weight = uncertainty_to_weight(uncertainty, uncertainty_config)
        weight = F.interpolate(
            weight[None, None],
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )[0]
        return uncertainty, weight

    if no_grad:
        with torch.no_grad():
            return _predict()
    return _predict()


def sample_dino_regularization_inputs(
    uncertainty, features, uncertainty_config, device
):
    reg_stride = max(1, uncertainty_config.get("reg_stride", 2))
    features = features.reshape(-1, features.shape[-1])
    uncertainty = uncertainty.reshape(-1, 1)
    if features.shape[0] == 0 or features.shape[0] != uncertainty.shape[0]:
        return None, None

    sample_count = max(1, features.shape[0] // (reg_stride**4))
    reg_max_samples = uncertainty_config.get("reg_max_samples", 4096)
    if reg_max_samples is not None and reg_max_samples > 0:
        sample_count = min(sample_count, reg_max_samples)
    sample_count = min(sample_count, features.shape[0])

    sample_idx = torch.randperm(features.shape[0], device=device)[:sample_count]
    return uncertainty[sample_idx].unsqueeze(0), features[sample_idx].unsqueeze(0)


def get_loss_tracking(
    config,
    image,
    depth,
    opacity,
    viewpoint,
    initialization=False,
    uncertainty_network=None,
):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        loss = get_loss_tracking_rgb(
            config, image_ab, depth, opacity, viewpoint, uncertainty_network
        )
    else:
        loss = get_loss_tracking_rgbd(
            config,
            image_ab,
            depth,
            opacity,
            viewpoint,
            uncertainty_network=uncertainty_network,
        )
    warp_cfg = get_lsg_feature_warping_config(config)
    if warp_cfg.get("enabled", False):
        warp_loss, warp_stats = compute_lsg_warp_loss(
            viewpoint, getattr(viewpoint, "lsg_match_data", None), warp_cfg
        )
        viewpoint.lsg_warp_stats = warp_stats
        loss = loss + warp_loss
    return loss


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint, uncertainty_network=None):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    gaustar_cfg = get_gaustar_stage1_config(config)
    if gaustar_cfg.get("apply_filter_to_tracking", True):
        priors = getattr(viewpoint, "priors", {}) or {}
        initialized = bool(priors.get("slam_initialized", False))
        consistency_mask = None
        if initialized or not gaustar_cfg.get("apply_filter_after_init", True):
            consistency_mask = build_rendered_depth_consistency_mask(
                config, depth, opacity, viewpoint
            )
        if consistency_mask is not None:
            soft_weight = float(gaustar_cfg.get("tracking_filter_soft_weight", 0.0))
            if soft_weight > 0:
                weight = torch.where(
                    consistency_mask.bool(),
                    torch.ones_like(rgb_pixel_mask, dtype=gt_image.dtype),
                    torch.full_like(rgb_pixel_mask, soft_weight, dtype=gt_image.dtype),
                )
                rgb_pixel_mask = rgb_pixel_mask * weight
            else:
                rgb_pixel_mask = torch.logical_and(
                    rgb_pixel_mask.bool(), consistency_mask.bool()
                ).to(dtype=gt_image.dtype)
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    uncertainty_cfg = get_uncertainty_config(config)
    if (
        uncertainty_cfg.get("enabled", False)
        and uncertainty_cfg.get("apply_to_tracking_rgb", True)
        and uncertainty_network is not None
    ):
        _, weights = predict_uncertainty_weight(
            uncertainty_network, viewpoint, (h, w), uncertainty_cfg, no_grad=True
        )
        if weights is not None:
            _record_uncertainty_debug("tracking_used")
            l1 = l1 * weights.to(device=l1.device, dtype=l1.dtype)
        else:
            _record_uncertainty_debug("tracking_fallback")
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False, uncertainty_network=None
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(
        config, image, depth, opacity, viewpoint, uncertainty_network
    )
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    uncertainty_cfg = get_uncertainty_config(config)
    if (
        uncertainty_cfg.get("enabled", False)
        and uncertainty_cfg.get("apply_to_tracking_depth", False)
        and uncertainty_network is not None
    ):
        _, weights = predict_uncertainty_weight(
            uncertainty_network,
            viewpoint,
            depth.shape[-2:],
            uncertainty_cfg,
            no_grad=True,
        )
        if weights is not None:
            l1_depth = l1_depth * weights.to(device=l1_depth.device, dtype=l1_depth.dtype)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(
    config,
    image,
    depth,
    viewpoint,
    opacity,
    initialization=False,
    uncertainty_network=None,
    regularization_viewpoints=None,
):
    uncertainty_cfg = get_uncertainty_config(config)
    if (
        uncertainty_cfg.get("enabled", False)
        and uncertainty_network is not None
        and has_uncertainty_features(viewpoint)
        and (not initialization or uncertainty_cfg.get("apply_during_init", True))
    ):
        _record_uncertainty_debug("mapping_used")
        return get_loss_mapping_uncertainty(
            config,
            image,
            depth,
            viewpoint,
            opacity,
            uncertainty_network,
            initialization,
            regularization_viewpoints=regularization_viewpoints,
        )

    if uncertainty_cfg.get("enabled", False) and uncertainty_network is not None:
        _record_uncertainty_debug("mapping_fallback")

    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_uncertainty(
    config,
    image,
    depth,
    viewpoint,
    opacity,
    uncertainty_network,
    initialization=False,
    regularization_viewpoints=None,
):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    uncertainty_cfg = get_uncertainty_config(config)
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)

    if viewpoint.depth is None:
        ref_depth = depth.detach()
    else:
        ref_depth = torch.from_numpy(viewpoint.depth).to(
            dtype=torch.float32, device=image.device
        )[None]

    features = viewpoint.features.to(device=image.device)
    uncertainty = uncertainty_network(features)
    uncer_loss, resized_uncertainty, l1_rgb, l1_depth = (
        mapping_utils.compute_mapping_loss_components(
            gt_image,
            image_ab,
            ref_depth,
            depth,
            uncertainty,
            opacity.view(*mask_shape),
            train_fraction=uncertainty_cfg.get("train_frac_fix", 0.3),
            ssim_fraction=uncertainty_cfg.get("train_frac_fix", 0.3),
            uncertainty_config=uncertainty_cfg,
            mask=rgb_pixel_mask,
        )
    )

    weights = uncertainty_to_weight(resized_uncertainty, uncertainty_cfg).view(*mask_shape)
    if uncertainty_cfg.get("apply_to_mapping_rgb", True):
        l1_rgb = l1_rgb * weights

    if uncertainty_cfg.get("apply_to_mapping_depth", True):
        l1_depth = l1_depth * weights

    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    if config["Training"]["monocular"]:
        loss = l1_rgb.mean()
    else:
        loss = alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()

    ssim_mult = uncertainty_cfg.get("ssim_mult", 0.5)
    loss = loss + ssim_mult * uncer_loss.mean()

    reg_mult = uncertainty_cfg.get("reg_mult", 0.5)
    if reg_mult > 0:
        local_features = [features]
        if regularization_viewpoints is not None:
            for reg_viewpoint in regularization_viewpoints:
                if reg_viewpoint is viewpoint or not has_uncertainty_features(reg_viewpoint):
                    continue
                local_features.append(reg_viewpoint.features.to(device=image.device))

        feature_dim = features.shape[-1]
        feature_samples = torch.cat(
            [
                local_feature[:: max(1, uncertainty_cfg.get("reg_stride", 2)), :: max(1, uncertainty_cfg.get("reg_stride", 2))]
                .reshape(-1, feature_dim)
                for local_feature in local_features
            ],
            dim=0,
        )
        sampled_feature_input = feature_samples.reshape(1, feature_samples.shape[0], 1, feature_dim)
        sampled_uncertainty_full = uncertainty_network(sampled_feature_input).reshape(-1, 1)
        sampled_uncertainty, sampled_features = sample_dino_regularization_inputs(
            sampled_uncertainty_full,
            feature_samples,
            uncertainty_cfg,
            image.device,
        )
        if sampled_uncertainty is not None:
            loss = loss + reg_mult * mapping_utils.compute_dino_regularization_loss(
                sampled_uncertainty,
                sampled_features,
            )
    return loss


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
