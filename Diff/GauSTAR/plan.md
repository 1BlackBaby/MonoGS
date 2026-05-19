# GauSTAR 第一阶段实验嵌入 MonoGS 计划

## Summary

目标是在 MonoGS 中以可开关、可消融的方式接入：

- `Metric3D depth`：作为单目初始化深度、关键帧补点深度源、渲染深度一致性参考。
- `optical flow pose initialization`：用相邻帧光流和深度建立 3D-2D 对应，给前端 tracking 提供比“上一帧位姿”更稳的初值。
- `rendered-depth consistency filtering`：借鉴 GauSTAR 的深度边缘、双向光流一致性、深度一致性过滤思想，过滤不可靠像素，减少动态/遮挡/深度错误区域对 tracking 和补点的影响。

`compare_GauSTAR.md` 复核后确认：第一阶段不迁移 GauSTAR 的 mesh / unbinding / re-meshing / HumanRF 训练流程，只保留对 MonoGS 侵入较小的 Metric3D 深度、光流位姿初值和渲染深度一致性过滤。结合 WildGS-SLAM 的 Metric3D 接入方式，数据加载需要兼容 `mono_priors/depths/{idx:05d}.npy`，并且当已有 rendered depth 可用于对齐时，Metric3D 只能在尺度对齐和一致性检查成功后参与关键帧初始化/补点，否则回退原生 MonoGS。

## Key Interfaces

- `Dataset` 新增可选单目先验配置：

```yaml
Dataset:
  mono_prior_path: ""
  metric3d_depth_path: ""
  flow_path: ""
  prior_format: "npy"

Training:
  gaustar_stage1:
    enabled: false
    use_metric3d_depth: true
    use_flow_pose_init: true
    use_rendered_depth_filter: true
    depth_scale_align: "median"
    depth_consistency_threshold: 0.12
    flow_fb_pixel_threshold: 3.0
    flow_depth_fb_threshold: 0.05
    depth_edge_kernel: 7
    depth_edge_threshold: 0.1
    min_pose_correspondences: 200
    min_pose_inliers: 80
    max_pose_reproj_error: 4.0
    tracking_filter_min_ratio: 0.05
    apply_filter_to_tracking: true
    apply_filter_to_keyframe_depth: true
    apply_filter_to_render_guided_densification: true
```

- `Camera` 增加 `priors` 字段，保存：
  - `metric_depth`: `H x W` float32
  - `flow_prev_to_cur`: `(H, W, 2)`，当前帧加载上一帧到当前帧的光流
  - `flow_cur_to_prev`: `(H, W, 2)`，用于 forward-backward check
  - `prior_valid_mask`: 可选外部 mask
  - `depth_scale`: 当前帧 Metric3D 到 MonoGS/rendered depth 的鲁棒尺度

- `utils/mono_priors/` 新增先验工具模块：
  - 加载 Metric3D depth / RAFT flow 文件。
  - 对 Metric3D depth 做 resize、finite check、边缘过滤。
  - 计算双向光流一致性 mask。
  - 计算 rendered depth 与 Metric3D depth 的 scale alignment 和 consistency mask。
  - 计算 flow-based PnP 位姿初值。

## Implementation Changes

- 数据读取：
  - 在 `utils/dataset.py` 的 `BaseDataset` 增加 `load_metric_depth(idx)`、`load_flow_pair(idx)`、`load_gaustar_priors(idx)`。
  - 默认读取路径：
    - `mono_priors/metric3d_depth/{idx:05d}.npy`
    - `mono_priors/depths/{idx:05d}.npy`（兼容 WildGS-SLAM 的 Metric3D 输出）
    - `mono_priors/flow_bi/{idx-1:05d}_f.npz`
    - `mono_priors/flow_bi/{idx-1:05d}_b.npz`
  - 文件缺失时只 warning 一次并回退到原 MonoGS 行为。

- 前端位姿初始化：
  - 在 `utils/slam_frontend.py` 的 `tracking()` 开头，将当前 `viewpoint.update_RT(prev.R, prev.T)` 包装为 `initialize_tracking_pose(...)`。
  - 若 `use_flow_pose_init=true` 且当前帧包含上一帧到当前帧的双向光流：
    - 从上一帧 Metric3D depth 或上一帧 rendered depth 反投影得到世界点。
    - 用 flow 得到当前帧 2D 对应点。
    - 使用 GauSTAR 风格过滤：图像边界、深度边缘、forward-backward pixel error、depth consistency、最大运动阈值。
    - 调用 `cv2.solvePnPRansac` 得到当前帧 W2C 初值。
    - 满足 inlier 数和重投影误差阈值时更新 `viewpoint.R/T`；否则回退上一帧位姿。

- Metric3D 深度接入：
  - 初始化第一帧和新关键帧时，`add_new_keyframe()` 优先使用 scale-aligned Metric3D depth。
  - 如果已有 rendered depth，则用有效区域中位数比例对齐：`scale = median(render_depth / metric_depth)`。
  - 若已有 rendered depth 但尺度对齐失败或一致性有效像素比例过低，禁止使用未对齐 Metric3D，回退现有 rendered depth / MonoGS 随机初始深度逻辑。
  - RGB-D 模式默认不替换真实深度，只可选用于过滤。

- rendered-depth consistency filtering：
  - 在 `utils/slam_utils.py` 增加 `build_rendered_depth_consistency_mask(...)`。
  - mask 组成：
    - `opacity > 0.95`
    - `metric_depth > min_depth`
    - `abs(render_depth - scale * metric_depth) / max(scale * metric_depth, eps) < threshold`
    - depth edge response 小于阈值
    - 可选 prior_valid_mask
  - tracking loss 中，在原 `rgb_pixel_mask * grad_mask` 后再乘该 consistency mask。
  - RGB-D depth loss 默认不被 Metric3D mask 覆盖，除非配置显式开启。

- 与现有 `render_guided_densification` 合并：
  - 不新建第二套补点逻辑，扩展当前 `BackEnd.add_render_guided_gaussians(...)`。
  - 当 `gaustar_stage1.use_metric3d_depth=true` 时，补点深度源优先使用 scale-aligned Metric3D depth。
  - 当 `use_rendered_depth_filter=true` 时，`densify_mask` 额外受 consistency mask 限制，避免在遮挡边界和深度不可信区域补点。
  - 保留现有 `opacity hole`、`rgb error`、`depth error` 消融项。

- GauSTAR 源码借鉴点：
  - RAFT 双向 flow 保存格式与一致性思想参考 `data_process/RAFT/demo_GauSTAR.py`。
  - flow warp、depth edge、forward-backward pixel/depth check 参考 `gaustar_tools/warp_mesh.py`。
  - rendered depth / observed depth 差异和边缘过滤思想参考 `gaustar_trainers/refined_mesh.py`。
  - 不迁移 GauSTAR 的 mesh warp、SuGaR surface、HumanRF、多视角动态人体训练流程。

## Test Plan

- 兼容性：
  - `Training.gaustar_stage1.enabled=false` 时，MonoGS 结果和现有代码路径一致。
  - 缺失 Metric3D depth 或 flow 文件时，不崩溃，清晰 warning，并回退原 tracking / keyframe depth / densification。
  - `uncertainty`、`render_guided_densification`、`forgetting_regularization` 关闭或开启时都能组合运行。

- 单元级验证：
  - Metric3D depth 加载后 shape 与图像一致，NaN/Inf/负深度被过滤。
  - flow forward-backward mask 在边界、遮挡、无效 flow 上正确置 false。
  - depth scale alignment 在 synthetic depth 上恢复预期比例。
  - PnP 初始化在模拟相机运动下输出正确方向，inlier 不足时回退。

- SLAM 路径验证：
  - 单目 TUM 上跑四组消融：
    - baseline
    - `+ Metric3D depth`
    - `+ Metric3D depth + flow pose init`
    - `+ Metric3D depth + flow pose init + rendered-depth filtering`
  - 记录 ATE RMSE、tracking 平均迭代次数、tracking loss、关键帧数、最终高斯数、FPS、显存峰值。
  - 对 `render_guided_densification` 再做 filtering on/off 对比，确认补点没有明显爆炸。

## Assumptions

- 第一版采用预计算 Metric3D depth 和 RAFT flow，避免把 Metric3D/RAFT 大模型直接放入 MonoGS 主循环。
- Metric3D depth 作为单目先验，RGB-D 真实深度优先级更高。
- MonoGS 当前已有 `render_guided_densification`，本计划是在其上扩展 GauSTAR 风格 filtering，而不是重复实现补点模块。
- WildGS-SLAM 的 Metric3D 更强调预计算 metric depth 和跨帧过滤；本阶段只兼容其默认深度目录并采用 fail-closed 的尺度/一致性策略，不引入 WildGS 的完整 depth video / DINO 动态过滤流水线。
