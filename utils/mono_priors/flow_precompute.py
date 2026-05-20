import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

from utils.mono_priors.gaustar_stage1 import (
    gaustar_stage1_enabled,
    get_gaustar_stage1_config,
)


class RaftArgs(SimpleNamespace):
    def __contains__(self, key):
        return hasattr(self, key)


def _resolve_raft_core(raft_root):
    if not raft_root:
        return None
    if os.path.basename(os.path.normpath(raft_root)) == "core":
        return raft_root
    core_path = os.path.join(raft_root, "core")
    return core_path if os.path.isdir(core_path) else raft_root


def _import_raft(raft_root):
    core_path = _resolve_raft_core(raft_root)
    if core_path is None or not os.path.isdir(core_path):
        raise FileNotFoundError(f"RAFT core directory not found: {raft_root}")

    saved_path = list(sys.path)
    saved_utils = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "utils" or name.startswith("utils.")
    }
    for name in saved_utils:
        del sys.modules[name]

    sys.path.insert(0, core_path)
    try:
        from raft import RAFT
        from utils.utils import InputPadder
    finally:
        for name in [
            name
            for name in list(sys.modules.keys())
            if name == "utils" or name.startswith("utils.")
        ]:
            del sys.modules[name]
        sys.modules.update(saved_utils)
        sys.path = saved_path

    return RAFT, InputPadder


def _flow_output_root(dataset, config):
    flow_path = config.get("Dataset", {}).get("flow_path", "")
    if flow_path:
        return flow_path
    if hasattr(dataset, "get_mono_prior_root"):
        return os.path.join(dataset.get_mono_prior_root(), "flow_bi")
    dataset_path = config.get("Dataset", {}).get("dataset_path", "")
    return os.path.join(dataset_path, "mono_priors", "flow_bi")


def _read_dataset_image(dataset, idx, device):
    image = np.array(Image.open(dataset.color_paths[idx])).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = image[..., :3]
    if getattr(dataset, "disorted", False):
        image = cv2.remap(image, dataset.map1x, dataset.map1y, cv2.INTER_LINEAR)
    return torch.from_numpy(image).permute(2, 0, 1).float()[None].to(device)


def _load_raft_model(cfg, device):
    raft_root = cfg.get("raft_root", "")
    checkpoint = cfg.get("raft_checkpoint", "")
    if not checkpoint:
        raise FileNotFoundError("Training.gaustar_stage1.raft_checkpoint is empty")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"RAFT checkpoint not found: {checkpoint}")

    RAFT, _ = _import_raft(raft_root)
    args = RaftArgs(
        small=bool(cfg.get("flow_small", False)),
        mixed_precision=bool(cfg.get("flow_mixed_precision", False)),
        alternate_corr=bool(cfg.get("flow_alternate_corr", False)),
        dropout=0,
    )
    model = torch.nn.DataParallel(RAFT(args))
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.module.to(device).eval()
    return model


@torch.no_grad()
def precompute_flow_priors(dataset, config, device="cuda"):
    if not gaustar_stage1_enabled(config):
        return
    cfg = get_gaustar_stage1_config(config)
    if not cfg.get("use_flow_pose_init", True) or not cfg.get("precompute_flow", False):
        return
    if not hasattr(dataset, "color_paths"):
        print("[GauSTAR Stage1] Flow precompute skipped: dataset has no color_paths.")
        return

    color_paths = getattr(dataset, "color_paths", [])
    num_imgs = min(len(color_paths), int(getattr(dataset, "num_imgs", len(color_paths))))
    if num_imgs < 2:
        print("[GauSTAR Stage1] Flow precompute skipped: fewer than two frames.")
        return

    flow_root = _flow_output_root(dataset, config)
    os.makedirs(flow_root, exist_ok=True)
    skip_existing = cfg.get("flow_skip_existing", True)
    missing_pairs = []
    for idx in range(num_imgs - 1):
        flow_f = os.path.join(flow_root, f"{idx:05d}_f.npz")
        flow_b = os.path.join(flow_root, f"{idx:05d}_b.npz")
        if skip_existing and os.path.isfile(flow_f) and os.path.isfile(flow_b):
            continue
        missing_pairs.append((idx, flow_f, flow_b))

    if not missing_pairs:
        print(f"[GauSTAR Stage1] Flow precompute skipped: cache complete at {flow_root}")
        return

    try:
        _, InputPadder = _import_raft(cfg.get("raft_root", ""))
        model = _load_raft_model(cfg, device)
    except Exception as exc:
        print(f"[GauSTAR Stage1] Flow precompute unavailable: {exc}")
        return

    iters = int(cfg.get("flow_iters", 20))
    print(
        "[GauSTAR Stage1] Precomputing bidirectional RAFT flow: "
        f"missing_pairs={len(missing_pairs)}, cache root={flow_root}"
    )
    for count, (idx, flow_f, flow_b) in enumerate(missing_pairs, start=1):
        image1 = _read_dataset_image(dataset, idx, device)
        image2 = _read_dataset_image(dataset, idx + 1, device)
        padder = InputPadder(image1.shape)
        image1_pad, image2_pad = padder.pad(image1, image2)

        _, f_flow_up = model(image1_pad, image2_pad, iters=iters, test_mode=True)
        f_flow = padder.unpad(f_flow_up[0]).permute(1, 2, 0).detach().cpu().numpy()
        np.savez_compressed(flow_f, flow=f_flow.astype(np.float32))

        _, b_flow_up = model(image2_pad, image1_pad, iters=iters, test_mode=True)
        b_flow = padder.unpad(b_flow_up[0]).permute(1, 2, 0).detach().cpu().numpy()
        np.savez_compressed(flow_b, flow=b_flow.astype(np.float32))

        if count == 1 or count == len(missing_pairs) or count % 25 == 0:
            print(
                "[GauSTAR Stage1] Flow precompute progress: "
                f"{count}/{len(missing_pairs)} pairs"
            )
