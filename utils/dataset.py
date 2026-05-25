import csv
import glob
import os

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from gaussian_splatting.utils.graphics_utils import focal2fov
from utils.mono_priors.gaustar_stage1 import (
    gaustar_stage1_enabled,
    get_gaustar_stage1_config,
    get_metric3d_estimator,
    load_flow_file,
    load_metric_depth_file,
    load_prior_mask_file,
    metric3d_depth_requested,
    predict_metric3d_depth,
    save_metric_depth_file,
)

try:
    import pyrealsense2 as rs
except Exception:
    pass


class ReplicaParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/traj.txt")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "transform_matrix": pose.tolist(),
            }

            frames.append(frame)
        self.frames = frames


class TUMParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }

            self.frames.append(frame)


class EuRoCParser:
    def __init__(self, input_folder, start_idx=0):
        self.input_folder = input_folder
        self.start_idx = start_idx
        self.color_paths = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam0/data/*.png")
        )
        self.color_paths_r = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam1/data/*.png")
        )
        assert len(self.color_paths) == len(self.color_paths_r)
        self.color_paths = self.color_paths[start_idx:]
        self.color_paths_r = self.color_paths_r[start_idx:]
        self.n_img = len(self.color_paths)
        self.load_poses(
            f"{self.input_folder}/mav0/state_groundtruth_estimate0/data.csv"
        )

    def associate(self, ts_pose):
        pose_indices = []
        for i in range(self.n_img):
            color_ts = float((self.color_paths[i].split("/")[-1]).split(".")[0])
            k = np.argmin(np.abs(ts_pose - color_ts))
            pose_indices.append(k)

        return pose_indices

    def load_poses(self, path):
        self.poses = []
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [list(map(float, row)) for row in reader]
        data = np.array(data)
        T_i_c0 = np.array(
            [
                [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        pose_ts = data[:, 0]
        pose_indices = self.associate(pose_ts)

        frames = []
        for i in range(self.n_img):
            trans = data[pose_indices[i], 1:4]
            quat = data[pose_indices[i], 4:8]
            quat = quat[[1, 2, 3, 0]]
            
            
            T_w_i = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T_w_i[:3, 3] = trans
            T_w_c = np.dot(T_w_i, T_i_c0)

            self.poses += [np.linalg.inv(T_w_c)]

            frame = {
                "file_path": self.color_paths[i],
                "transform_matrix": (np.linalg.inv(T_w_c)).tolist(),
            }

            frames.append(frame)
        self.frames = frames


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999
        uncertainty_cfg = config.get("Training", {}).get("uncertainty", {})
        self.use_uncertainty = uncertainty_cfg.get("enabled", False)
        self.feature_path = config.get("Dataset", {}).get("feature_path", "")
        self.feature_format = config.get("Dataset", {}).get("feature_format", "npy")
        self.feature_suffix = uncertainty_cfg.get("feature_suffix", "")
        self.extract_features_online = uncertainty_cfg.get(
            "extract_features_online", False
        )
        self.feature_output_root = uncertainty_cfg.get("feature_output_root", "output")
        self._warned_missing_features = False
        self._warned_feature_fallbacks = set()
        self.feature_stats = {"hit": 0, "missing": 0, "bad": 0}
        self._logged_first_feature_hit = False
        self._feature_extractor = None
        self._feature_extractor_failed = False
        self.use_gaustar_stage1 = gaustar_stage1_enabled(config)
        self.use_mono_priors = self.use_gaustar_stage1 or metric3d_depth_requested(
            config
        )
        self.gaustar_stage1_cfg = get_gaustar_stage1_config(config)
        dataset_cfg = config.get("Dataset", {})
        self.mono_prior_path = dataset_cfg.get("mono_prior_path", "")
        self.metric3d_depth_path = dataset_cfg.get("metric3d_depth_path", "")
        self.flow_path = dataset_cfg.get("flow_path", "")
        self.prior_mask_path = dataset_cfg.get("prior_mask_path", "")
        self.prior_format = dataset_cfg.get("prior_format", "npy")
        self._warned_gaustar_fallbacks = set()
        self.gaustar_prior_stats = {"metric_depth": 0, "flow": 0, "mask": 0}
        self._metric_depth_estimator = None
        self._metric_depth_estimator_failed = False
        self._logged_metric3d_online_prediction = False

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass

    def warn_feature_fallback(self, key, message):
        if key in self._warned_feature_fallbacks:
            return
        print(f"[Warning] {message} Falling back to original MonoGS losses.")
        self._warned_feature_fallbacks.add(key)

    def log_feature_stats(self, tag=""):
        if not self.use_uncertainty:
            return
        total = (
            self.feature_stats["hit"]
            + self.feature_stats["missing"]
            + self.feature_stats["bad"]
        )
        print(
            f"[Uncertainty] feature stats{tag}: "
            f"hit={self.feature_stats['hit']}, "
            f"missing={self.feature_stats['missing']}, "
            f"bad={self.feature_stats['bad']}, total_checked={total}"
        )

    def warn_gaustar_fallback(self, key, message):
        if key in self._warned_gaustar_fallbacks:
            return
        print(f"[GauSTAR Stage1] {message} Falling back to original MonoGS behavior.")
        self._warned_gaustar_fallbacks.add(key)

    def get_mono_prior_root(self):
        if self.mono_prior_path:
            return self.mono_prior_path
        dataset_path = self.config["Dataset"].get("dataset_path", self.path)
        return os.path.join(dataset_path, "mono_priors")

    def _prior_file(self, root, names):
        for name in names:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                return path
        return os.path.join(root, names[0])

    def get_metric_depth_output_root(self):
        if self.metric3d_depth_path:
            return self.metric3d_depth_path
        return os.path.join(self.get_mono_prior_root(), "depths")

    def get_online_metric_depth_estimator(self):
        if self._metric_depth_estimator_failed:
            return None
        if self._metric_depth_estimator is None:
            try:
                self._metric_depth_estimator = get_metric3d_estimator(
                    self.config, self.device
                )
                print("[GauSTAR Stage1] Metric3D online depth estimator initialized")
            except Exception as exc:
                self._metric_depth_estimator_failed = True
                self.warn_gaustar_fallback(
                    "metric3d_estimator_init",
                    f"Unable to initialize Metric3D online depth estimator: {exc}.",
                )
                return None
        return self._metric_depth_estimator

    def predict_metric_depth_online(self, idx, image, output_file):
        if image is None:
            return None
        model = self.get_online_metric_depth_estimator()
        if model is None:
            return None
        try:
            depth = predict_metric3d_depth(model, image, self.config, self.device)
        except Exception as exc:
            self.warn_gaustar_fallback(
                "metric3d_predict",
                f"Unable to predict Metric3D depth for frame {idx}: {exc}.",
            )
            return None
        if self.gaustar_stage1_cfg.get("cache_metric3d_depth", True):
            try:
                save_metric_depth_file(depth, output_file)
            except OSError as exc:
                self.warn_gaustar_fallback(
                    "metric3d_cache",
                    f"Unable to cache Metric3D depth file {output_file}: {exc}.",
                )
        if not self._logged_metric3d_online_prediction:
            print(
                "[GauSTAR Stage1] Metric3D online depth prediction enabled; "
                f"cache root={os.path.dirname(output_file)}"
            )
            self._logged_metric3d_online_prediction = True
        return depth.detach().cpu()

    def load_metric_depth(self, idx, target_shape, image=None):
        if not metric3d_depth_requested(self.config):
            return None
        if self.metric3d_depth_path:
            roots = [self.metric3d_depth_path]
        else:
            mono_prior_root = self.get_mono_prior_root()
            roots = [
                os.path.join(mono_prior_root, "metric3d_depth"),
                os.path.join(mono_prior_root, "depths"),
            ]
        names = [
            f"{idx:05d}.{self.prior_format}",
            f"{idx:06d}.{self.prior_format}",
            f"{idx}.{self.prior_format}",
        ]
        if self.prior_format != "npy":
            names += [f"{idx:05d}.npy", f"{idx:06d}.npy", f"{idx}.npy"]
        depth_file = None
        for root in roots:
            candidate = self._prior_file(root, names)
            if os.path.isfile(candidate):
                depth_file = candidate
                break
        if depth_file is None:
            depth_file = self._prior_file(roots[0], names)
        if not os.path.isfile(depth_file):
            output_file = os.path.join(
                self.get_metric_depth_output_root(), f"{idx:05d}.npy"
            )
            depth = self.predict_metric_depth_online(idx, image, output_file)
            if depth is None:
                self.warn_gaustar_fallback(
                    "metric_depth_missing",
                    f"Metric3D depth file is missing. Checked roots: {roots}.",
                )
                return None
            self.gaustar_prior_stats["metric_depth"] += 1
            return depth
        try:
            depth = load_metric_depth_file(depth_file, target_shape=target_shape)
        except (OSError, ValueError, RuntimeError) as exc:
            self.warn_gaustar_fallback(
                "metric_depth_bad",
                f"Unable to load Metric3D depth file {depth_file}: {exc}.",
            )
            return None
        self.gaustar_prior_stats["metric_depth"] += 1
        return torch.from_numpy(depth)

    def load_flow_pair(self, idx, target_shape):
        if not self.use_gaustar_stage1 or idx <= 0:
            return None, None
        root = self.flow_path or os.path.join(self.get_mono_prior_root(), "flow_bi")
        prev_idx = idx - 1
        flow_f_file = self._prior_file(
            root,
            [
                f"{prev_idx:05d}_f.npz",
                f"{prev_idx:06d}_f.npz",
                f"{prev_idx}_f.npz",
                f"{prev_idx:05d}_f.npy",
            ],
        )
        flow_b_file = self._prior_file(
            root,
            [
                f"{prev_idx:05d}_b.npz",
                f"{prev_idx:06d}_b.npz",
                f"{prev_idx}_b.npz",
                f"{prev_idx:05d}_b.npy",
            ],
        )
        if not os.path.isfile(flow_f_file) or not os.path.isfile(flow_b_file):
            self.warn_gaustar_fallback(
                "flow_missing",
                f"Bidirectional flow files are missing: {flow_f_file}, {flow_b_file}.",
            )
            return None, None
        try:
            flow_f = load_flow_file(flow_f_file, target_shape=target_shape)
            flow_b = load_flow_file(flow_b_file, target_shape=target_shape)
        except (OSError, ValueError, RuntimeError) as exc:
            self.warn_gaustar_fallback(
                "flow_bad",
                f"Unable to load bidirectional flow for frame {idx}: {exc}.",
            )
            return None, None
        self.gaustar_prior_stats["flow"] += 1
        return torch.from_numpy(flow_f), torch.from_numpy(flow_b)

    def load_prior_valid_mask(self, idx, target_shape):
        if not self.use_gaustar_stage1:
            return None
        root = self.prior_mask_path or os.path.join(
            self.get_mono_prior_root(), "valid_mask"
        )
        names = [
            f"{idx:05d}.npy",
            f"{idx:06d}.npy",
            f"{idx:05d}.npz",
            f"{idx:06d}.npz",
            f"{idx:05d}.png",
            f"{idx:06d}.png",
        ]
        mask_file = self._prior_file(root, names)
        if not os.path.isfile(mask_file):
            return None
        try:
            mask = load_prior_mask_file(mask_file, target_shape=target_shape)
        except (OSError, ValueError, RuntimeError) as exc:
            self.warn_gaustar_fallback(
                "mask_bad",
                f"Unable to load prior valid mask {mask_file}: {exc}.",
            )
            return None
        self.gaustar_prior_stats["mask"] += 1
        return torch.from_numpy(mask)

    def load_gaustar_priors(self, idx, target_shape, image=None):
        if not self.use_mono_priors:
            return None
        priors = {"mono_priors": True}
        if self.use_gaustar_stage1:
            priors["gaustar_stage1"] = True
        metric_depth = self.load_metric_depth(idx, target_shape, image=image)
        if metric_depth is not None:
            priors["metric_depth"] = metric_depth
        flow_f, flow_b = self.load_flow_pair(idx, target_shape)
        if flow_f is not None and flow_b is not None:
            priors["flow_prev_to_cur"] = flow_f
            priors["flow_cur_to_prev"] = flow_b
        prior_valid_mask = self.load_prior_valid_mask(idx, target_shape)
        if prior_valid_mask is not None:
            priors["prior_valid_mask"] = prior_valid_mask
        return priors

    def get_feature_root(self):
        if self.feature_path:
            return self.feature_path
        if self.extract_features_online:
            scene = self.config.get("Dataset", {}).get("scene", "")
            if not scene:
                dataset_path = self.config["Dataset"].get("dataset_path", self.path)
                scene = os.path.basename(os.path.normpath(dataset_path)) or "scene"
            return os.path.join(
                self.feature_output_root, scene, "mono_priors", "features"
            )
        dataset_path = self.config["Dataset"].get("dataset_path", self.path)
        return os.path.join(dataset_path, "mono_priors", "features")

    def get_online_feature_extractor(self):
        if self._feature_extractor_failed:
            return None
        if self._feature_extractor is None:
            try:
                from utils.mono_priors.img_feature_extractors import (
                    get_feature_extractor,
                )

                self._feature_extractor = get_feature_extractor(
                    self.config, device=self.device
                )
                print("[Uncertainty] online feature extractor initialized")
            except Exception as exc:
                self._feature_extractor_failed = True
                self.warn_feature_fallback(
                    "feature_extractor_init",
                    f"Unable to initialize online feature extractor: {exc}.",
                )
                return None
        return self._feature_extractor

    def extract_features_online_for_image(self, idx, image, feature_root):
        if not self.extract_features_online or image is None:
            return None
        extractor = self.get_online_feature_extractor()
        if extractor is None:
            return None
        try:
            from utils.mono_priors.img_feature_extractors import predict_img_features

            features = predict_img_features(
                extractor,
                idx,
                image.unsqueeze(0),
                self.config,
                self.device,
                feature_root,
                save_feat=True,
                suffix=self.feature_suffix,
            )
        except Exception as exc:
            self.warn_feature_fallback(
                "feature_extract",
                f"Unable to extract online uncertainty features for frame {idx}: {exc}.",
            )
            return None
        return features.detach().cpu().float()

    def load_features(self, idx, image=None):
        if not self.use_uncertainty:
            return None
        if self.feature_format != "npy":
            raise ValueError(f"Unsupported feature_format: {self.feature_format}")

        feature_root = self.get_feature_root()
        feature_file = os.path.join(feature_root, f"{idx:05d}{self.feature_suffix}.npy")
        if not os.path.isfile(feature_file):
            features = self.extract_features_online_for_image(idx, image, feature_root)
            if features is not None:
                self.feature_stats["hit"] += 1
                if not self._logged_first_feature_hit:
                    print(
                        f"[Uncertainty] first feature extracted: {feature_file}, "
                        f"shape={tuple(features.shape)}"
                    )
                    self._logged_first_feature_hit = True
                return features

            self.feature_stats["missing"] += 1
            if not self._warned_missing_features:
                print(
                    f"[Warning] Uncertainty enabled but feature file is missing: {feature_file}. "
                    "Falling back to original MonoGS losses."
                )
                self._warned_missing_features = True
            return None

        try:
            features = np.load(feature_file, allow_pickle=False)
        except (OSError, ValueError, RuntimeError) as exc:
            self.feature_stats["bad"] += 1
            self.warn_feature_fallback(
                "read_error",
                f"Unable to load uncertainty feature file {feature_file}: {exc}.",
            )
            return None

        expected_dim = self.config.get("Training", {}).get("uncertainty", {}).get(
            "feature_dim", 384
        )
        if not isinstance(features, np.ndarray):
            self.feature_stats["bad"] += 1
            self.warn_feature_fallback(
                "not_array",
                f"Uncertainty feature file {feature_file} did not contain an ndarray.",
            )
            return None
        if features.ndim != 3 or features.shape[0] <= 0 or features.shape[1] <= 0:
            self.feature_stats["bad"] += 1
            self.warn_feature_fallback(
                "bad_shape",
                f"Uncertainty feature file {feature_file} has invalid shape {features.shape}; expected HxWxC.",
            )
            return None
        if features.shape[-1] != expected_dim:
            self.feature_stats["bad"] += 1
            self.warn_feature_fallback(
                "bad_feature_dim",
                f"Uncertainty feature file {feature_file} has feature_dim={features.shape[-1]}, expected {expected_dim}.",
            )
            return None

        try:
            features = features.astype(np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            self.feature_stats["bad"] += 1
            self.warn_feature_fallback(
                "bad_dtype",
                f"Uncertainty feature file {feature_file} cannot be converted to float32: {exc}.",
            )
            return None
        if not np.isfinite(features).all():
            self.feature_stats["bad"] += 1
            self.warn_feature_fallback(
                "non_finite",
                f"Uncertainty feature file {feature_file} contains NaN or Inf values.",
            )
            return None

        self.feature_stats["hit"] += 1
        if not self._logged_first_feature_hit:
            print(
                f"[Uncertainty] first feature loaded: {feature_file}, "
                f"shape={features.shape}"
            )
            self._logged_first_feature_hit = True
        return torch.from_numpy(features)


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path)) / self.depth_scale

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        features = self.load_features(idx, image=image)
        priors = self.load_gaustar_priors(idx, (self.height, self.width), image=image)
        if self.use_uncertainty:
            if self.use_mono_priors:
                return image, depth, pose, features, priors
            return image, depth, pose, features
        if self.use_mono_priors:
            return image, depth, pose, priors
        return image, depth, pose


class StereoDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        self.width = calibration["width"]
        self.height = calibration["height"]

        cam0raw = calibration["cam0"]["raw"]
        cam0opt = calibration["cam0"]["opt"]
        cam1raw = calibration["cam1"]["raw"]
        cam1opt = calibration["cam1"]["opt"]
        # Camera prameters
        self.fx_raw = cam0raw["fx"]
        self.fy_raw = cam0raw["fy"]
        self.cx_raw = cam0raw["cx"]
        self.cy_raw = cam0raw["cy"]
        self.fx = cam0opt["fx"]
        self.fy = cam0opt["fy"]
        self.cx = cam0opt["cx"]
        self.cy = cam0opt["cy"]

        self.fx_raw_r = cam1raw["fx"]
        self.fy_raw_r = cam1raw["fy"]
        self.cx_raw_r = cam1raw["cx"]
        self.cy_raw_r = cam1raw["cy"]
        self.fx_r = cam1opt["fx"]
        self.fy_r = cam1opt["fy"]
        self.cx_r = cam1opt["cx"]
        self.cy_r = cam1opt["cy"]

        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K_raw = np.array(
            [
                [self.fx_raw, 0.0, self.cx_raw],
                [0.0, self.fy_raw, self.cy_raw],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.Rmat = np.array(calibration["cam0"]["R"]["data"]).reshape(3, 3)
        self.K_raw_r = np.array(
            [
                [self.fx_raw_r, 0.0, self.cx_raw_r],
                [0.0, self.fy_raw_r, self.cy_raw_r],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K_r = np.array(
            [[self.fx_r, 0.0, self.cx_r], [0.0, self.fy_r, self.cy_r], [0.0, 0.0, 1.0]]
        )
        self.Rmat_r = np.array(calibration["cam1"]["R"]["data"]).reshape(3, 3)

        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [cam0raw["k1"], cam0raw["k2"], cam0raw["p1"], cam0raw["p2"], cam0raw["k3"]]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K_raw,
            self.dist_coeffs,
            self.Rmat,
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        self.dist_coeffs_r = np.array(
            [cam1raw["k1"], cam1raw["k2"], cam1raw["p1"], cam1raw["p2"], cam1raw["k3"]]
        )
        self.map1x_r, self.map1y_r = cv2.initUndistortRectifyMap(
            self.K_raw_r,
            self.dist_coeffs_r,
            self.Rmat_r,
            self.K_r,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        color_path_r = self.color_paths_r[idx]

        pose = self.poses[idx]
        image = cv2.imread(color_path, 0)
        image_r = cv2.imread(color_path_r, 0)
        depth = None
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)
        stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
        stereo.setUniquenessRatio(40)
        disparity = stereo.compute(image, image_r) / 16.0
        disparity[disparity == 0] = 1e10
        depth = 47.90639384423901 / (
            disparity
        )  ## Following ORB-SLAM2 config, baseline*fx
        depth[depth < 0] = 0
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)

        features = self.load_features(idx, image=image)
        priors = self.load_gaustar_priors(idx, (self.height, self.width), image=image)
        if self.use_uncertainty:
            if self.use_mono_priors:
                return image, depth, pose, features, priors
            return image, depth, pose, features
        if self.use_mono_priors:
            return image, depth, pose, priors
        return image, depth, pose


class TUMDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = TUMParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class ReplicaDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ReplicaParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class EurocDataset(StereoDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = EuRoCParser(dataset_path, start_idx=config["Dataset"]["start_idx"])
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.color_paths_r = parser.color_paths_r
        self.poses = parser.poses


class RealsenseDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        self.pipeline = rs.pipeline()
        self.h, self.w = 720, 1280
        
        self.depth_scale = 0
        if self.config["Dataset"]["sensor_type"] == "depth":
            self.has_depth = True 
        else: 
            self.has_depth = False

        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, 30)
        if self.has_depth:
            self.rs_config.enable_stream(rs.stream.depth)

        self.profile = self.pipeline.start(self.rs_config)

        if self.has_depth:
            self.align_to = rs.stream.color
            self.align = rs.align(self.align_to)

        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        # rgb_sensor.set_option(rs.option.enable_auto_white_balance, True)
        self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        self.rgb_sensor.set_option(rs.option.exposure, 200)
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        self.rgb_intrinsics = self.rgb_profile.get_intrinsics()
        
        self.fx = self.rgb_intrinsics.fx
        self.fy = self.rgb_intrinsics.fy
        self.cx = self.rgb_intrinsics.ppx
        self.cy = self.rgb_intrinsics.ppy
        self.width = self.rgb_intrinsics.width
        self.height = self.rgb_intrinsics.height
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.disorted = True
        self.dist_coeffs = np.asarray(self.rgb_intrinsics.coeffs)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, np.eye(3), self.K, (self.w, self.h), cv2.CV_32FC1
        )

        if self.has_depth:
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale  = self.depth_sensor.get_depth_scale()
            self.depth_profile = rs.video_stream_profile(
                self.profile.get_stream(rs.stream.depth)
            )
            self.depth_intrinsics = self.depth_profile.get_intrinsics()
        
        


    def __getitem__(self, idx):
        pose = torch.eye(4, device=self.device, dtype=self.dtype)
        depth = None

        frameset = self.pipeline.wait_for_frames()

        if self.has_depth:
            aligned_frames = self.align.process(frameset)
            rgb_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth = np.array(aligned_depth_frame.get_data())*self.depth_scale
            depth[depth < 0] = 0
            np.nan_to_num(depth, nan=1000)
        else:
            rgb_frame = frameset.get_color_frame()

        image = np.asanyarray(rgb_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        features = self.load_features(idx, image=image)
        priors = self.load_gaustar_priors(idx, (self.height, self.width), image=image)
        if self.use_uncertainty:
            if self.use_mono_priors:
                return image, depth, pose, features, priors
            return image, depth, pose, features
        if self.use_mono_priors:
            return image, depth, pose, priors
        return image, depth, pose


def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":
        return TUMDataset(args, path, config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(args, path, config)
    elif config["Dataset"]["type"] == "euroc":
        return EurocDataset(args, path, config)
    elif config["Dataset"]["type"] == "realsense":
        return RealsenseDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
