import os
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class Fit3DModels(torch.nn.Module):
    def __init__(self, extractor_model, device):
        super().__init__()
        self.model = torch.hub.load("ywyue/FiT3D", extractor_model).to(device).eval()

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n=1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        if not hasattr(self.model, "_intermediate_layers"):
            kwargs = {
                "n": n,
                "reshape": reshape,
                "return_class_token": return_class_token,
                "norm": norm,
            }
            signature = inspect.signature(self.model.get_intermediate_layers)
            supported_kwargs = {
                key: value for key, value in kwargs.items() if key in signature.parameters
            }
            return self.model.get_intermediate_layers(x, **supported_kwargs)

        outputs = self.model._intermediate_layers(x, n)
        if norm:
            outputs = [self.model.norm(out) for out in outputs]
        if return_class_token:
            prefix_tokens = [out[:, 0] for out in outputs]
        else:
            prefix_tokens = [
                out[:, 0 : self.model.num_prefix_tokens] for out in outputs
            ]
        outputs = [out[:, self.model.num_prefix_tokens :] for out in outputs]

        if reshape:
            _, _, h, w = x.shape
            grid_size = (
                (h - self.model.patch_embed.patch_size[0])
                // self.model.patch_embed.proj.stride[0]
                + 1,
                (w - self.model.patch_embed.patch_size[1])
                // self.model.patch_embed.proj.stride[1]
                + 1,
            )
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]

        if return_prefix_tokens or return_class_token:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)


def get_feature_extractor(config, device="cuda"):
    extractor_model = (
        config.get("mono_prior", {}).get("feature_extractor")
        or config.get("Training", {}).get("uncertainty", {}).get(
            "feature_extractor", "dinov2_reg_small_fine"
        )
    )

    if extractor_model in ["dinov2_reg_small_fine", "dinov2_small_fine"]:
        return Fit3DModels(extractor_model, device)
    if extractor_model in ["dinov2_vits14", "dinov2_vits14_reg"]:
        return torch.hub.load("facebookresearch/dinov2", extractor_model).to(device).eval()
    raise NotImplementedError(f"Unsupported feature extractor: {extractor_model}")


@torch.no_grad()
def predict_img_features(
    model: nn.Module,
    idx: int,
    input_tensor: torch.Tensor,
    config,
    device: str,
    output_dir: str,
    save_feat: bool = True,
    suffix: str = "",
) -> torch.Tensor:
    extractor_model = (
        config.get("mono_prior", {}).get("feature_extractor")
        or config.get("Training", {}).get("uncertainty", {}).get(
            "feature_extractor", "dinov2_reg_small_fine"
        )
    )
    stride = 14
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    image_resized = process_image(input_tensor, stride, normalize, device)

    if extractor_model in ["dinov2_reg_small_fine", "dinov2_small_fine"]:
        features = model.get_intermediate_layers(
            image_resized,
            n=[8, 9, 10, 11],
            reshape=True,
            return_prefix_tokens=False,
            return_class_token=False,
            norm=True,
        )
        features = features[-1].squeeze().permute((1, 2, 0))
    elif extractor_model in ["dinov2_vits14", "dinov2_vits14_reg"]:
        features_dict = model.forward_features(image_resized)
        features = features_dict["x_norm_patchtokens"].view(
            image_resized.shape[2] // stride,
            image_resized.shape[3] // stride,
            -1,
        )
    else:
        raise NotImplementedError(f"Unsupported feature extractor: {extractor_model}")

    if save_feat:
        os.makedirs(output_dir, exist_ok=True)
        feature_path = os.path.join(output_dir, f"{idx:05d}{suffix}.npy")
        np.save(feature_path, features.detach().cpu().float().numpy())

    return features


def process_image(
    image: torch.Tensor, stride: int, normalize: nn.Module, device: str = "cuda"
) -> torch.Tensor:
    image_tensor = normalize(image).float().to(device)
    h, w = image_tensor.shape[2:]
    height_int = (h // stride) * stride
    width_int = (w // stride) * stride
    return F.interpolate(image_tensor, size=(height_int, width_int), mode="bilinear")
