# 将 LSG-SLAM 四个创新点嵌入 MonoGS 的分阶段计划

本文档只给实施方案，不修改代码。计划基于当前 MonoGS 代码结构，以及 LSG-SLAM 源码中以下实现线索：

- MonoGS 前端跟踪：`utils/slam_frontend.py` 的 `initialize_tracking_pose()`、`tracking()`。
- MonoGS 跟踪/建图损失：`utils/slam_utils.py` 的 `get_loss_tracking*()`、`get_loss_mapping*()`。
- MonoGS 后端建图：`utils/slam_backend.py` 的 `map()`、致密化、opacity reset、isotropic loss。
- MonoGS 高斯结构：`gaussian_splatting/scene/gaussian_model.py` 的 `GaussianModel.isotropic`、`_scaling`、`_rotation`、`unique_kfIDs`。
- LSG-SLAM 特征/PnP/warping：`scripts/feature_matching.py`。
- LSG-SLAM 在线 tracking 与 loop 调用：`scripts/loop_closure.py`。
- LSG-SLAM pose graph 与 structure refinement：`tools/loop_closure/pose_graph_part_optim.py`。

## 总体原则

四个创新点不要一次性全部合入。正确路线是先把“几何增强 tracking”做成可独立开关，再做“建图结构精修”，最后接回环闭合。原因是 8.1 和 8.2 直接影响每帧 tracking 的稳定性；8.4 主要影响最终渲染质量；8.6 依赖稳定的关键帧、特征缓存、相对位姿估计和高斯点归属信息。

建议所有新增能力都挂在配置项下，默认关闭，保证原 MonoGS baseline 可以原样复现实验：

```yaml
Training:
  lsg_pose_init:
    enabled: false
  lsg_feature_warping:
    enabled: false
  lsg_structure_refine:
    enabled: false
  lsg_loop_closure:
    enabled: false
```

## 阶段 0：基线、接口与实验闭环

目标：先建立可对照的实验闭环，避免后续每改一个模块都无法判断收益来源。

实施内容：

1. 固定一组小规模验证序列和一组长序列。小规模建议先用项目现有 RGB-D/monocular 配置；长序列再考虑 KITTI/EuRoC 风格数据。
2. 记录 baseline 指标：ATE RMSE、tracking lost/reset 次数、关键帧数量、最终 PSNR/SSIM/LPIPS、FPS、显存峰值。
3. 为每个新增模块增加独立 ablation 开关：只开 8.1、只开 8.2、只开 8.4、8.1+8.2、8.1+8.2+8.4、最后再加 8.6。
4. 先整理关键帧数据缓存接口。后续 8.1、8.2、8.6 都需要关键帧图像、深度、内参、位姿、局部特征、描述子、匹配结果。

建议新增模块：

- `utils/lsg_features.py`：SuperPoint/LightGlue 封装、特征缓存、匹配过滤。
- `utils/lsg_pose_init.py`：PnP/RANSAC、点云配准 fallback、候选初值评分。
- `utils/lsg_warp_loss.py`：feature warping loss。
- `utils/lsg_structure_refine.py`：球形到椭球转换、离线/半离线精修。
- `utils/lsg_loop_closure.py`：回环候选、验证、pose graph 接口。

验收标准：

- 关闭全部 `lsg_*` 开关时，结果与原 MonoGS 基本一致。
- 每个新增模块可以单独打开和关闭。
- 日志至少记录：PnP 是否成功、inlier 数量、warping loss 是否生效、structure refine 迭代数、回环候选和通过验证的回环数量。

## 阶段 1：8.1 多模态位姿初值

目标：在 `tracking()` 优化 `cam_rot_delta`、`cam_trans_delta` 之前，为当前帧提供比上一帧 pose copy / motion model 更可靠的初值。

### 1.1 接入位置

MonoGS 当前入口是：

- `utils/slam_frontend.py::tracking()` 调用 `initialize_tracking_pose(cur_frame_idx, viewpoint, prev)`。
- `initialize_tracking_pose()` 当前先 `viewpoint.update_RT(prev.R, prev.T)`，并已有 GauSTAR flow pose initialization 逻辑。

建议把 LSG-SLAM 初值作为 `initialize_tracking_pose()` 内的一个候选源，而不是替换已有逻辑。最终形成候选优先级：

1. SuperPoint/LightGlue + PnP/RANSAC 成功，且通过 sanity check。
2. 可选：RGB-D/stereo 下点云配准 fallback 成功，且通过 sanity check。
3. 已有 GauSTAR flow pose init 成功。
4. 原始 previous pose / motion model。

### 1.2 特征提取与匹配

参考 LSG-SLAM：

- `scripts/feature_matching.py::get_local_features()`
- `extract_feature()`
- `match_feature()`

MonoGS 中建议：

1. 给关键帧缓存 `local_features`、`descriptors`、`keypoints`、`feature_scores`。
2. 当前帧进入 tracking 前提取 SuperPoint 特征，最多保留 512 或 1024 个点。
3. 当前帧与最近关键帧匹配，必要时再与滑窗内 overlap 最高的 1-2 个关键帧匹配。
4. 用 EssentialMat/RANSAC 或 LightGlue score 做第一层过滤，低于 `min_matches` 直接 fallback。

### 1.3 PnP 初值

参考 LSG-SLAM：

- `scripts/feature_matching.py::estimate_pnp()`
- `cv2.solvePnPRansac(..., flags=cv2.SOLVEPNP_SQPNP)`

MonoGS 中的 2D-3D 来源：

1. RGB-D/stereo：直接用参考关键帧 `viewpoint.depth` 反投影匹配点。
2. monocular：优先用关键帧保存的 `rendered_depth` 或初始化时的 keyframe depth；没有可靠尺度时只启用旋转/方向约束，或者直接 fallback。
3. 如果当前已有 `priors["rendered_depth"]` 或 metric depth，可作为可选补充，但要保留深度范围过滤。

PnP 结果写回方式：

- PnP 得到当前帧 W2C 后调用 `viewpoint.update_RT(R, T)`。
- 不直接写 `cam_rot_delta/cam_trans_delta`，让后续 Adam tracking 从新初值继续优化。

### 1.4 初值质量检查

必须加入 reject 逻辑，否则错误 PnP 会把 tracking 拉崩：

- `num_inliers >= min_pnp_inliers`，初始建议 RGB-D 50，KITTI/EuRoC 100。
- inlier ratio 达到阈值，例如 0.25。
- 相对上一帧平移、旋转不超过数据集合理上限。
- 可选：像当前 GauSTAR flow init 一样，比较 PnP 初值和 fallback 初值的 `tracking_init_loss()`，只有 PnP loss 更低才接受。

### 1.5 点云配准 fallback

这部分先作为第二轮增强，不建议第一版就做。RGB-D/stereo 下可用 Open3D ICP：

1. 从当前帧深度和参考关键帧深度生成稀疏点云。
2. 用 PnP 或上一帧位姿作为 ICP 初值。
3. ICP fitness / inlier RMSE 达标才接受。

### 1.6 阶段验收

最小可验收版本：

- 只实现最近关键帧 SuperPoint/LightGlue + PnP。
- PnP 失败时无缝 fallback 到原 MonoGS。
- 输出每帧 `matches/inliers/accepted/rejected_reason`。

指标预期：

- ATE RMSE 应先下降，尤其低帧率、大视角变化、快速运动序列。
- PSNR/SSIM/LPIPS 可能间接改善，但不作为本阶段主指标。

### 1.7 Metric3D 深度先验辅助 LSG-PnP

PnP 本质需要当前帧 2D 匹配点和参考帧 3D 世界点。对 RGB-D/stereo 输入，参考关键帧可以直接使用观测深度或双目深度；但对 monocular 输入，如果只依赖 MonoGS 初始化深度或渲染深度，尺度和局部几何误差会直接影响 PnP 平移估计。因此建议把 `gaustar_stage1` 中已有的 Metric3D 深度能力抽象为可被 LSG-SLAM 复用的单目深度先验，而不是让 LSG-PnP 强依赖 GauSTAR flow pose init。

开关设计建议：

```yaml
Training:
  lsg_slam:
    enabled: true
  lsg_pose_init:
    enabled: true
    use_metric3d_depth: true
    metric3d_for_keyframe_depth: true
    metric3d_filter_with_rendered_depth: true
    metric3d_depth_priority: "after_lsg_keyframe"
```

Metric3D 是否启用不应只由 `gaustar_stage1.enabled` 决定。更合理的触发条件是：

```text
Metric3D 可用 =
  gaustar_stage1.enabled && gaustar_stage1.use_metric3d_depth
  OR
  lsg_slam.enabled && lsg_pose_init.use_metric3d_depth
```

这样可以在 `gaustar_stage1.enabled=false` 时，单独验证 `LSG-PnP + Metric3D depth prior` 的收益，避免 GauSTAR flow pose init 和 LSG-PnP 的贡献混在一起。

推荐数据流：

1. Dataset 层根据“GauSTAR 或 LSG 任一模块请求 Metric3D”来读取或在线预测 `metric_depth`。
2. `Camera.init_from_dataset()` 保留 `viewpoint.priors["metric_depth"]`。
3. `FrontEnd.add_new_keyframe()` 在 monocular 模式下，如果 `lsg_pose_init.use_metric3d_depth=true`，优先用 Metric3D 生成关键帧参考深度。
4. 如果配置 `metric3d_filter_with_rendered_depth=true`，则使用当前 rendered depth、opacity 和 RGB mask 做一致性过滤与尺度对齐，再保存为 `viewpoint.priors["lsg_keyframe_depth"]`。
5. LSG-PnP 读取参考深度时优先使用 `lsg_keyframe_depth`，再按传感器类型 fallback 到其他深度来源。

深度优先级建议按输入类型区分：

```text
RGB-D:
  lsg_keyframe_depth -> viewpoint.depth -> metric_depth -> rendered_depth

Monocular:
  lsg_keyframe_depth -> metric_depth -> rendered_depth

Stereo:
  lsg_keyframe_depth -> stereo/depth prior -> metric_depth -> rendered_depth
```

这样可以避免 Metric3D 覆盖 RGB-D 真实传感器深度，同时让 monocular 场景获得 PnP 所需的相对稳定 3D 点。

实现边界建议：

- `gaustar_stage1.enabled` 继续控制 GauSTAR 自身的 flow pose init、rendered depth filter 和相关增强。
- `lsg_pose_init.use_metric3d_depth` 只表示 LSG-PnP 请求 Metric3D 深度先验。
- Dataset 层只关心是否有模块请求 Metric3D，不关心该深度最终服务于 GauSTAR 还是 LSG。
- LSG-PnP 不应直接要求 `gaustar_stage1.enabled=true`，否则消融实验无法隔离 LSG 创新点。

验收与消融建议：

```text
A. Baseline
B. LSG-PnP only
C. LSG-PnP + Metric3D depth
D. LSG-PnP + Metric3D depth + rendered-depth consistency filter
```

主要观察 `PnP accepted rate`、`matches/inliers`、`reproj_error`、`ATE RMSE`、tracking reset/lost 次数和 FPS。预期收益主要体现在 monocular 或缺少可靠深度的配置；RGB-D 上 Metric3D 不应作为主深度来源，只作为 fallback 或对照实验。

## 阶段 2：8.2 Feature Warping Tracking Loss

目标：tracking 优化过程中不只依赖渲染 RGB/depth loss，还用稀疏但可靠的特征几何约束持续拉住位姿。

### 2.1 接入位置

MonoGS 当前 loss 入口：

- `utils/slam_frontend.py::tracking()` 中调用 `get_loss_tracking(...)`。
- `utils/slam_utils.py::get_loss_tracking()` 分发到 RGB/RGB-D。

建议实现方式：

1. 在 `tracking()` 开始前准备 `viewpoint.lsg_match_data`，包含参考关键帧 id、匹配点、匹配分数、参考深度、参考描述子/feature map。
2. `get_loss_tracking()` 保持原签名时，可以从 `viewpoint` 上读取 match data；或显式增加可选参数。
3. `loss_tracking = rendering_loss + weight_warp * feature_warp_loss`。

### 2.2 Warping loss 计算

参考 LSG-SLAM：

- `scripts/feature_matching.py::get_loss_from_match()`
- 核心逻辑：用当前位姿把当前/参考特征点反投影和重投影，再计算 keypoint 坐标 SmoothL1 与 feature patch SmoothL1。

MonoGS 建议采用“参考关键帧点反投影到 3D，再投影到当前帧”的形式：

1. 对参考关键帧匹配点 `uv_ref` 读取参考深度 `d_ref`。
2. 用参考内参反投影到参考相机坐标。
3. 用参考 W2C 逆矩阵变到世界坐标。
4. 用当前待优化的 `viewpoint.R/T` 加上 `cam_rot_delta/cam_trans_delta` 对应的当前 W2C 投影到当前图像。
5. 与当前匹配点 `uv_cur` 做 SmoothL1。
6. 可选：在当前/参考 feature map 上用 `grid_sample` 采样描述子，做 SmoothL1 或 cosine distance。

### 2.3 梯度链路注意点

关键是 warping loss 必须对当前 pose delta 可导。MonoGS 的渲染器内部会读取 `viewpoint.cam_rot_delta`、`cam_trans_delta`，但自写 warping loss 不能只用已经离散更新后的 `viewpoint.R/T`，否则梯度不会回到 pose delta。

实施时建议抽一个工具函数：

- 输入 `viewpoint.R/T + cam_rot_delta/cam_trans_delta`。
- 用和 `utils/pose_utils.py::SE3_exp()` 一致的方式构造当前临时 W2C。
- 不调用 `update_pose()`，只返回可导矩阵。

### 2.4 Mask 与权重

初始配置建议：

- `weight_warp`: RGB-D 1-10 起步，KITTI 可试 10-50；不要直接硬套 LSG-SLAM 的固定 50。
- `max_depth`: 参考 LSG-SLAM 过滤远点，先设 `min(dataset.depth_filter_far, 15.0)` 或 MonoGS 配置项。
- `min_matches_for_loss`: 20。
- `robust_beta`: 0.1 或按像素归一化后设 0.01。

需要过滤：

- 深度无效或过远的点。
- 投影出图像边界的点。
- match score 过低的点。
- 动态物体或高不确定区域，后续可接入已有 uncertainty/grad mask。

### 2.5 阶段验收

最小可验收版本：

- 只做 keypoint 2D reprojection SmoothL1，不做 feature descriptor loss。
- loss 可反传到 `cam_rot_delta/cam_trans_delta`。
- PnP 初值关闭时，warping loss 也能独立打开。

增强版本：

- 加 descriptor/feature map SmoothL1。
- 加多参考关键帧 warping loss。
- 加动态权重：前 10-20 次 tracking iteration warping 权重大，后续降低，让渲染 loss 精修。

指标预期：

- ATE RMSE 进一步下降。
- tracking 失败次数降低。
- 如果权重过大，可能牺牲局部 photometric alignment，需要通过 ablation 调参。

## 阶段 3：8.4 球形建图 + 椭球结构精修

目标：在线阶段用更稳定的球形高斯降低漂浮和早期拉伸；单全局地图的关键帧轨迹稳定后，把球形高斯扩展成椭球并做结构精修，提高最终渲染质量。

### 3.1 在线球形建图

MonoGS 已有基础：

- `GaussianModel.isotropic` 控制 `_scaling` 是 `[N, 1]` 还是 `[N, 3]`。
- `create_pcd_from_image_and_depth()` 已根据 `self.isotropic` 创建单轴 scale。
- `utils/slam_backend.py::map()` 当前额外加了 isotropic loss：`10 * abs(scaling - mean(scaling)).mean()`。

第一步建议：

1. 增加配置 `Training.lsg_structure_refine.online_isotropic=true`。
2. 初始化 `GaussianModel` 时根据配置设置 `gaussians.isotropic = true`。
3. 在线 mapping 阶段如果 `_scaling` 已是 `[N, 1]`，就关闭或跳过现有 isotropic loss，避免重复约束。
4. 保持 densify/prune/reset 逻辑先不大改，确保在线 SLAM 稳定。

### 3.2 球形到椭球转换

参考 LSG-SLAM：

- `scripts/loop_closure.py::initialize_params()` 中 `gaussian_distribution="isotropic"` 时 `log_scales` 为 `[N,1]`。
- `tools/loop_closure/pose_graph_part_optim.py` 中如果 `log_scales.shape[1] == 1` 且精修用 anisotropic，则 tile 成 `[N,3]`。

MonoGS 建议新增 `GaussianModel.convert_isotropic_to_anisotropic()`：

1. 如果 `_scaling.shape[1] == 1`，复制为 `[N,3]`。
2. 保持 `_rotation` 为单位四元数或当前值。
3. 重建 optimizer param groups，因为 `_scaling` 参数形状变了。
4. 设置 `gaussians.isotropic = false`。

### 3.3 结构精修入口

可选两个入口：

1. 离线入口：主流程结束后，在 `slam.py` 的 `eval_rendering/color_refinement` 附近触发。
2. 半离线入口：后端收到 `["structure_refinement"]` 消息后运行，完成后 `sync_backend` 给前端。

第一版建议做离线入口，工程风险最低。

### 3.4 结构精修 loss

参考 LSG-SLAM：

- `pose_graph_part_optim.py` 中 structure refine 使用 `0.8 * L1 + 0.2 * (1 - SSIM)`。
- anisotropic 后加 `compute_min_scale_loss()`，鼓励至少一个轴变小，形成贴合表面的薄片。

MonoGS 中建议：

1. 随机采样关键帧，不优化相机位姿，只优化高斯参数。
2. loss 使用 RGB L1 + SSIM；RGB-D 模式可保留小权重 depth loss。
3. 椭球 scale regularization：
   - `min_scale_loss = min(get_scaling, dim=1).mean()`，类似 LSG-SLAM。
   - 另加上限约束，防止某一轴无限变大。
4. 精修期间允许优化：`_xyz`、`_features_dc/rest`、`_opacity`、`_scaling`、`_rotation`。
5. 精修期间可选择关闭新增点，只做已有点优化；第二版再启用 densify。

### 3.5 阶段验收

最小可验收版本：

- 在线球形高斯可以跑完序列。
- 结束后成功转换为椭球。
- 离线精修 1000-5000 iterations，输出 before/after 渲染指标。

指标预期：

- PSNR/SSIM 上升，LPIPS 下降。
- ATE RMSE 基本不直接改善。
- 若在线球形过强导致细节不足，精修前 PSNR 可能下降，但精修后应补回来。

## 阶段 4：8.6 回环闭合

目标：在 MonoGS 单全局 `GaussianModel` 的前提下，给长序列提供全局一致性，降低累计漂移，并把 pose graph 优化后的关键帧位姿同步回同一个高斯地图。这里不引入 LSG-SLAM 的连续 GS 子地图设计，回环只作用在“关键帧位姿图 + 全局高斯点变形”上。

### 4.1 前置依赖

不建议在 8.1/8.2 稳定前做回环。回环至少依赖：

- 稳定的关键帧位姿。
- 可复用的 SuperPoint/LightGlue 特征缓存。
- 关键帧 global descriptor。
- 每个高斯点的创建关键帧归属。MonoGS 已有 `unique_kfIDs`，但需要确认 densify/split/clone 后归属传播正确。
- 单地图下的全局高斯点更新机制。回环优化后不能创建第二张地图，也不能把地图切成子块；只能对当前 `GaussianModel` 的 `_xyz` 和必要的 `_rotation` 做一致性变形。

### 4.2 回环候选检索

参考 LSG-SLAM：

- `scripts/loop_closure.py::find_loops()`
- 数据集中预存 `global_features/*.npy`。
- 用 KDTree/top-k 检索非邻近关键帧。

MonoGS 建议两级实现：

第一版：

1. 离线或在线为每个关键帧提取 global descriptor。
2. 使用轻量向量检索，跳过最近 `loop_before_interval` 个关键帧。
3. 每隔 `loop_detect_interval` 个关键帧检测一次。

第二版：

1. 接入 TransVPR/NetVLAD/DINO 全局特征。
2. 支持更长历史窗口检索和更严格的时序间隔过滤，避免相邻关键帧被误判为回环。

### 4.3 回环验证

参考 LSG-SLAM：

- global feature 只给候选。
- 再用 SuperPoint/LightGlue 匹配。
- `estimate_pnp_for_loop()` 用候选参考帧深度 PnP。
- 对当前帧相邻帧再做一致性检查，筛假回环。

MonoGS 建议：

1. 当前关键帧和候选关键帧做局部特征匹配。
2. 用候选关键帧深度反投影，PnP 求 `T_query_ref`。
3. 阈值：`pnp_inliers >= 100` 起步，室内可降到 50。
4. 用 query 的前后相邻关键帧验证相对位姿一致性。
5. 可选：用 8.2 的 feature warping + 渲染 loss 对回环相对位姿做小规模优化。

### 4.4 Pose graph 优化

参考 LSG-SLAM：

- `tools/loop_closure/pose_graph_part_optim.py::PoseGraphManager`
- GTSAM `PriorFactorPose3`、`BetweenFactorPose3`、loop factor。

MonoGS 有两条实现路线：

1. 快速路线：引入 `gtsam`，按 LSG-SLAM 写 PoseGraphManager。
2. 保守路线：先用 `scipy.optimize.least_squares` 实现 SE3 pose graph，少一个依赖，但工程量略高。

建议第一版用 GTSAM，如果环境安装困难，再退回 scipy。

图结构：

- node：关键帧 pose。
- odometry edge：相邻关键帧相对位姿。
- loop edge：通过验证的回环相对位姿。
- fixed prior：第一个关键帧。

### 4.5 单全局高斯地图同步

这是回环闭合能否真正改善渲染的关键。

参考 LSG-SLAM：

- `pose_graph_part_optim.py` 先把每个高斯按 `variables['timestep']` 找到创建时的旧 pose。
- 将高斯点变到创建帧相机坐标，再用优化后的 pose 变回世界坐标。

MonoGS 可使用 `GaussianModel.unique_kfIDs`：

1. 保存 pose graph 优化前每个关键帧的 `W2C_old`。
2. pose graph 输出每个关键帧的 `W2C_new` 或 `C2W_new`。
3. 对每个高斯点，根据 `unique_kfIDs` 找创建关键帧。
4. `xyz_cam = W2C_old[kf] @ xyz_world_old`。
5. `xyz_world_new = C2W_new[kf] @ xyz_cam`。
6. 更新 `_xyz`，必要时同步旋转 `_rotation`。

单地图约束下的实现重点：

1. `GaussianModel` 仍然只有一个实例，前端和后端同步协议保持 `clone_obj(self.gaussians)` 这一套，不新增额外地图列表。
2. pose graph 只优化关键帧位姿，不直接优化高斯参数；高斯参数通过关键帧位姿变化做一次全局 deformation。
3. deformation 完成后，建议追加短轮次 mapping/color refinement，让被回环拉动后的地图重新贴合图像观测。
4. 如果回环修正幅度很大，应暂停前端 tracking，等待后端完成位姿图优化、地图变形和前端同步，避免前端继续用旧地图追踪。

注意：

- densify/split/clone 产生的新点必须继承 parent 的 `unique_kfIDs`。
- 如果某些点没有有效归属，先按最近可见关键帧或最近空间关键帧处理；第一版可直接跳过并记录比例。

### 4.6 单地图回环流程

MonoGS 没有子地图，因此回环闭合的第一版流程应明确限制为：

1. 前端照常维护 `self.cameras`、`self.kf_indices`、`self.current_window` 和单个 `self.gaussians`。
2. 后端或离线脚本从全部历史关键帧中检索回环候选，不做地图分组。
3. 回环候选通过局部特征匹配和 PnP 验证后，写入关键帧 pose graph 的 loop edge。
4. pose graph 优化得到每个关键帧的新全局位姿。
5. 用 `unique_kfIDs` 把所有高斯点绑定到创建关键帧，按“旧创建帧坐标 -> 新世界坐标”更新同一张高斯地图。
6. 将更新后的关键帧位姿和同一个 `GaussianModel` 同步给前端。

不建议在本创新点中引入额外地图管理器。原因是这会把 8.6 回环闭合扩展成“地图拆分管理 + 回环闭合”两个创新点，工程边界会变模糊，也不符合当前 MonoGS 单地图架构。

单地图方案的限制：

- 不能解决超长序列下单个 `GaussianModel` 的显存增长问题。
- 大幅回环修正会让远离创建关键帧的高斯点出现局部拉扯，需要后续 mapping/refinement 缓解。
- 回环后当前活动窗口中的关键帧位姿必须一起更新，否则前端、后端和 GUI 显示会短暂不一致。

### 4.7 阶段验收

最小可验收版本：

- 能检测并保存回环候选。
- 至少一个回环通过局部特征 + PnP 验证。
- pose graph 优化后 ATE 下降。
- 单个 `GaussianModel` 中的高斯点根据优化后关键帧位姿完成一次 deformation，且没有新增子地图结构。

指标预期：

- 长序列 ATE RMSE 明显下降。
- 有回到旧区域的序列，重影和地图断裂减少。
- 短序列或无回环序列不应明显变差。

## 推荐实施顺序

1. 基线与配置开关：建立可复现实验与日志。
2. 8.1 最小版：最近关键帧匹配 + PnP 初值 + fallback。
3. 8.2 最小版：keypoint reprojection warping loss。
4. 8.1/8.2 联合调参：确认 ATE、失败率、FPS。
5. 8.4 在线球形建图：打开 `GaussianModel.isotropic` 并处理 optimizer/正则逻辑。
6. 8.4 离线椭球结构精修：球形 scale 扩成三轴 scale，做 L1+SSIM+scale regularization。
7. 8.6 离线回环：global descriptor 检索、局部特征验证、pose graph、地图 deformation。
8. 8.6 在线化：把回环检测放入后端消息循环，定期触发 pose graph 和地图同步。

## 关键风险与规避

1. PnP 错误初值风险最高。必须有 inlier、运动幅度、tracking init loss 三层 reject。
2. Feature warping loss 权重过大会压过渲染 loss。先做小权重和前半段 iteration 生效。
3. monocular 模式深度尺度不稳定。8.1/8.2 优先在 RGB-D/stereo 或有 metric depth prior 的配置上验证。
4. 球形高斯可能降低在线细节。它的收益主要是稳定，最终质量依赖椭球精修。
5. 回环闭合不能只改相机轨迹。必须同步 deformation 高斯点，否则 ATE 下降但渲染地图会错位。
6. GTSAM/LightGlue/SuperPoint 会带来依赖复杂度。建议把依赖失败时的 fallback 写清楚，不让主流程崩溃。

## 消融实验矩阵

| 实验 | 8.1 PnP 初值 | 8.2 Warping | 8.4 结构精修 | 8.6 回环 | 主看指标 |
|---|---:|---:|---:|---:|---|
| Baseline | 关 | 关 | 关 | 关 | 全部 |
| PoseInit | 开 | 关 | 关 | 关 | ATE/FPS |
| WarpOnly | 关 | 开 | 关 | 关 | ATE/失败率 |
| PoseInit+Warp | 开 | 开 | 关 | 关 | ATE/失败率/FPS |
| StructureRefine | 开 | 开 | 开 | 关 | PSNR/SSIM/LPIPS |
| LoopClosure | 开 | 开 | 可选 | 开 | ATE/全局一致性 |

## 最小可投稿/可展示版本建议

如果工程时间有限，优先做：

```text
SuperPoint/LightGlue + PnP pose initialization
+ Feature warping tracking loss
+ offline anisotropic structure refinement
```

回环闭合放在长序列增强版。它收益高，但工程依赖链最长；在 8.1 和 8.2 不稳定时提前做回环，容易把问题混在一起，难以定位。
