import contextlib
import os
import sys


@contextlib.contextmanager
def _pushd(path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class LSGFeatureMatcher:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.sp_extractor = None
        self.lg_matcher = None
        self.load_error = None

    def _model_root(self):
        configured = self.cfg.get("feature_model_root", "")
        candidates = []
        if configured:
            candidates.append(configured)
        env_root = os.environ.get("LSG_SLAM_ROOT")
        if env_root:
            candidates.append(env_root)
        candidates.append(os.getcwd())

        for root in candidates:
            if os.path.basename(os.path.normpath(root)) == "sp_lg":
                root = os.path.dirname(os.path.normpath(root))
            if os.path.isdir(os.path.join(root, "sp_lg")):
                return root
        return None

    def _ensure_models(self):
        if self.sp_extractor is not None and self.lg_matcher is not None:
            return True, "ready"
        if self.load_error is not None:
            return False, self.load_error

        root = self._model_root()
        if root is None:
            self.load_error = "missing sp_lg model directory"
            return False, self.load_error

        if root not in sys.path:
            sys.path.insert(0, root)

        try:
            import torch
            from sp_lg.lightglue import LightGlue
            from sp_lg.superpoint import SuperPoint

            with _pushd(root):
                self.sp_extractor = SuperPoint(
                    max_num_keypoints=int(self.cfg.get("max_keypoints", 1024))
                ).to(self.device)
                self.lg_matcher = LightGlue(pretrained="superpoint").to(self.device)
            self.sp_extractor.eval()
            self.lg_matcher.eval()
        except Exception as exc:
            self.sp_extractor = None
            self.lg_matcher = None
            self.load_error = f"unable to load SuperPoint/LightGlue: {exc}"
            return False, self.load_error

        return True, "ready"

    def _image_tensor(self, viewpoint):
        import torch

        image = getattr(viewpoint, "original_image", None)
        if image is None:
            return None
        image = image.detach().clone().to(self.device).float()
        depth = self._reference_depth(viewpoint)
        if depth is not None and self.cfg.get("mask_invalid_depth_for_features", True):
            depth_t = torch.as_tensor(depth, device=self.device).squeeze()
            invalid = (depth_t < float(self.cfg.get("min_depth", 0.1))) | (
                depth_t > float(self.cfg.get("max_depth", 50.0))
            )
            if invalid.shape == image.shape[-2:]:
                image[:, invalid] = 0.0
        return image

    def _reference_depth(self, viewpoint):
        priors = getattr(viewpoint, "priors", {}) or {}
        depth = priors.get("lsg_keyframe_depth")
        if depth is not None:
            return depth
        depth = priors.get("rendered_depth")
        if depth is not None:
            return depth
        depth = priors.get("metric_depth")
        if depth is not None:
            return depth
        return getattr(viewpoint, "depth", None)

    def extract(self, viewpoint):
        ok, reason = self._ensure_models()
        if not ok:
            return None, reason

        cached = getattr(viewpoint, "lsg_local_features", None)
        if cached is not None:
            return cached, "cached"

        image = self._image_tensor(viewpoint)
        if image is None:
            return None, "missing image"

        import torch

        with torch.no_grad():
            feats, _ = self.sp_extractor({"image": image[None]})
        feats = {
            key: value.detach()
            for key, value in feats.items()
            if hasattr(value, "detach")
        }
        cached = {"features": feats}
        viewpoint.lsg_local_features = cached
        return cached, "extracted"

    def match(self, cur_viewpoint, ref_viewpoint):
        import numpy as np
        import torch

        cur_feats, cur_reason = self.extract(cur_viewpoint)
        if cur_feats is None:
            return None, cur_reason
        ref_feats, ref_reason = self.extract(ref_viewpoint)
        if ref_feats is None:
            return None, ref_reason

        cur_image = self._image_tensor(cur_viewpoint)
        ref_image = self._image_tensor(ref_viewpoint)
        if cur_image is None or ref_image is None:
            return None, "missing image"

        data = {"image0": cur_image[None], "image1": ref_image[None]}
        pred = {
            **{f"{key}0": value for key, value in cur_feats["features"].items()},
            **{f"{key}1": value for key, value in ref_feats["features"].items()},
            **data,
        }
        with torch.no_grad():
            pred = {**pred, **self.lg_matcher(pred)}
        pred = {
            key: value.detach()[0] if isinstance(value, torch.Tensor) else value
            for key, value in pred.items()
        }

        matches0 = pred["matches0"]
        scores0 = pred["matching_scores0"]
        valid = matches0 > -1
        if int(valid.count_nonzero().item()) < int(self.cfg.get("min_matches", 20)):
            return None, "not enough matches"

        matches = torch.stack([torch.where(valid)[0], matches0[valid]], dim=-1)
        scores = scores0[valid]
        scores, order = scores.sort(dim=0, descending=True)
        topk = int(self.cfg.get("match_topk", 512))
        order = order[:topk]
        scores = scores[:topk]
        matches = matches[order]

        kpts_cur = pred["keypoints0"][matches[:, 0]].detach().cpu().numpy()
        kpts_ref = pred["keypoints1"][matches[:, 1]].detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        if self.cfg.get("use_essential_filter", True) and kpts_cur.shape[0] >= 8:
            try:
                import cv2

                k_mat = np.array(
                    [
                        [cur_viewpoint.fx, 0.0, cur_viewpoint.cx],
                        [0.0, cur_viewpoint.fy, cur_viewpoint.cy],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                _, mask = cv2.findEssentialMat(
                    kpts_cur,
                    kpts_ref,
                    k_mat,
                    cv2.RANSAC,
                    0.999,
                    float(self.cfg.get("essential_threshold", 1.0)),
                )
                if mask is not None:
                    mask = mask.reshape(-1).astype(bool)
                    kpts_cur = kpts_cur[mask]
                    kpts_ref = kpts_ref[mask]
                    scores_np = scores_np[mask]
            except Exception:
                pass

        if kpts_cur.shape[0] < int(self.cfg.get("min_matches", 20)):
            return None, "not enough verified matches"

        return {
            "kpts_cur": kpts_cur.astype(np.float64),
            "kpts_ref": kpts_ref.astype(np.float64),
            "scores": scores_np,
            "num_matches": int(kpts_cur.shape[0]),
        }, "matched"
