from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils.dyn_uncertainty.median_filter import MedianPool2d


EPSILON = torch.finfo(torch.float32).eps
SSIM_C1 = 0.01**2
SSIM_C2 = 0.03**2
SSIM_C3 = SSIM_C2 / 2
GAUSSIAN_SIGMA = 1.5
SSIM_MAX_CLIP = 0.98
TOP_K_FEATURES = 128
SIMILARITY_THRESHOLD = 0.75


def resample_tensor_to_shape(tensor, target_shape, interpolation_mode="bilinear"):
    tensor = tensor.view((1, 1) + tensor.shape[:2])
    return F.interpolate(
        tensor, size=target_shape, mode=interpolation_mode, align_corners=False
    ).squeeze(0).squeeze(0)


def compute_bias_factor(x, s):
    return x / (1 + (1 - x) * (1 / s - 2))


def _gaussian_kernel(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def _gaussian_window(window_size, num_channels):
    window_1d = _gaussian_kernel(window_size, GAUSSIAN_SIGMA).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return Variable(window_2d.expand(num_channels, 1, window_size, window_size).contiguous())


def compute_ssim_components(img1, img2, window_size=11):
    num_channels = img1.size(-3)
    window = _gaussian_window(window_size, num_channels).to(
        device=img1.device, dtype=img1.dtype
    )
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        squeeze_orig = True
    else:
        squeeze_orig = False

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=num_channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=num_channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=num_channels
    ) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=num_channels
    ) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2, groups=num_channels
    ) - mu1_mu2

    eps = torch.tensor([EPSILON], device=img1.device, dtype=img1.dtype)
    sigma1_sq = torch.maximum(eps, sigma1_sq)
    sigma2_sq = torch.maximum(eps, sigma2_sq)
    sigma12 = torch.sign(sigma12) * torch.minimum(
        torch.sqrt(sigma1_sq * sigma2_sq), torch.abs(sigma12)
    )

    luminance = (2 * mu1_mu2 + SSIM_C1) / (mu1_sq + mu2_sq + SSIM_C1)
    contrast = (2 * torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + SSIM_C2) / (
        sigma1_sq + sigma2_sq + SSIM_C2
    )
    structure = (sigma12 + SSIM_C3) / (
        torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + SSIM_C3
    )
    contrast = torch.clamp(contrast, max=SSIM_MAX_CLIP)
    structure = torch.clamp(structure, max=SSIM_MAX_CLIP)

    if squeeze_orig:
        return luminance.mean(1).squeeze(), contrast.mean(1).squeeze(), structure.mean(1).squeeze()
    return luminance.mean(1), contrast.mean(1), structure.mean(1)


def compute_mapping_loss_components(
    gt_img,
    rendered_img,
    ref_depth,
    rendered_depth,
    uncertainty,
    opacity,
    train_fraction,
    ssim_fraction,
    uncertainty_config,
    mask,
):
    _, h, w = gt_img.shape
    rgb_l1_loss = torch.abs(rendered_img * mask - gt_img * mask)

    median_depth = ref_depth[ref_depth > 0.01].median() if (ref_depth > 0.01).any() else ref_depth.median()
    depth_threshold = min(10 * median_depth.item(), 50)
    depth_mask = ((ref_depth > 0.01) & (ref_depth < depth_threshold)).view(*rendered_depth.shape)
    depth_l1_loss = torch.abs(rendered_depth * depth_mask - ref_depth * depth_mask)

    min_uncertainty = uncertainty_config.get("min_uncertainty", 0.1)
    eps = uncertainty_config.get("eps", 1e-3)
    processed_uncertainty = torch.clamp(uncertainty, min=min_uncertainty) + eps
    resized_uncertainty = resample_tensor_to_shape(processed_uncertainty.detach(), (h, w))
    data_rate = 1 + compute_bias_factor(train_fraction, 0.8)
    resized_uncertainty = (resized_uncertainty - min_uncertainty) * data_rate + min_uncertainty

    resized_opacity = opacity.detach().view((h, w))
    small_opacity = resample_tensor_to_shape(resized_opacity, uncertainty.shape)

    ssim_weight = 100 + 900 * compute_bias_factor(ssim_fraction, 0.8)
    luminance, contrast, structure = compute_ssim_components(
        gt_img, rendered_img, window_size=uncertainty_config.get("ssim_window_size", 7)
    )
    ssim_loss = torch.clamp(
        resized_opacity * ssim_weight * (1 - luminance) * (1 - structure) * (1 - contrast),
        max=5.0,
    )

    median_filter = MedianPool2d(
        kernel_size=uncertainty_config.get("ssim_median_filter_size", 5),
        stride=1,
        padding=0,
        same=True,
    )
    small_ssim_loss = resample_tensor_to_shape(ssim_loss.detach(), uncertainty.shape)
    filtered_ssim_loss = median_filter(
        small_ssim_loss.unsqueeze(0).unsqueeze(0)
    ).squeeze(0).squeeze(0)

    small_depth_loss = resample_tensor_to_shape(
        torch.clamp(depth_l1_loss.squeeze(), max=5.0).detach(), uncertainty.shape, "bicubic"
    )
    small_depth = resample_tensor_to_shape(ref_depth.squeeze().detach(), uncertainty.shape, "bicubic")
    small_depth_loss[small_depth > depth_threshold] = 0.0

    uncertainty_loss = (
        filtered_ssim_loss / processed_uncertainty**2
        + 0.5 * torch.log(processed_uncertainty)
        + uncertainty_config.get("uncer_depth_mult", 0.2)
        * small_depth_loss
        / processed_uncertainty**2
    )
    uncertainty_loss[
        small_opacity < uncertainty_config.get("opacity_th_for_uncer_loss", 0.9)
    ] = 0
    return uncertainty_loss, resized_uncertainty, rgb_l1_loss, depth_l1_loss


def compute_dino_regularization_loss(uncertainty_buffer, feature_buffer):
    uncertainty = torch.stack(uncertainty_buffer) if isinstance(uncertainty_buffer, list) else uncertainty_buffer
    features = torch.stack(feature_buffer) if isinstance(feature_buffer, list) else feature_buffer
    feature_dim = features.shape[-1]
    uncertainty_flat = uncertainty.reshape(-1, 1)
    features_flat = features.contiguous().reshape(-1, feature_dim)
    features_normalized = F.normalize(features_flat, p=2, dim=-1)
    similarity_matrix = torch.matmul(features_normalized, features_normalized.T)
    k = min(TOP_K_FEATURES, similarity_matrix.shape[-1])
    top_similarities, top_indices = torch.topk(similarity_matrix, k=k, dim=-1)
    similarity_mask = (top_similarities > SIMILARITY_THRESHOLD).float()
    neighbor_uncertainties = uncertainty_flat[top_indices] * similarity_mask.unsqueeze(-1)
    uncertainty_sums = torch.sum(neighbor_uncertainties, dim=1)
    valid_counts = torch.sum(similarity_mask, dim=-1, keepdim=True) + EPSILON
    uncertainty_means = uncertainty_sums / valid_counts
    squared_differences = (
        neighbor_uncertainties - uncertainty_means.unsqueeze(-1)
    ) ** 2 * similarity_mask.unsqueeze(-1)
    uncertainty_variances = torch.sum(squared_differences, dim=1) / valid_counts
    return torch.mean(uncertainty_variances)

