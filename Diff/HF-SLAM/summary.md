# HF-SLAM 模块迁移任务总结

## 1. 目标

本次任务目标是根据 `Diff/HF-SLAM/plan.md` 中第 4 阶段和第 5 阶段的计划，将 HF-SLAM 中两个可迁移思想接入 MonoGS 后端：

1. `render_guided_densification`：由 opacity 空洞检测、RGB 渲染误差，以及可选 depth error 共同引导新增高斯点。
2. `forgetting_regularization`：维护高斯历史快照和重要性权重，在 mapping loss 中加入防遗忘正则。

两个模块必须分别由独立总开关控制：

- `Training.render_guided_densification.enabled` 控制所有新增的渲染引导致密化逻辑。
- `Training.forgetting_regularization.enabled` 控制所有新增的防遗忘统计、importance 更新、snapshot 和正则 loss。

当两个开关都关闭时，应尽量回到原生 MonoGS 行为，不改变普通关键帧初始化、原生 `densify_and_prune` 和 prune 路径。

## 2. 最终改动

### `gaussian_splatting/scene/gaussian_model.py`

- 扩展 `create_pcd_from_image(...)` / `extend_from_pcd_seq(...)`，支持基于像素 `mask` 的局部 RGB-D 反投影补点。
- 为 render-guided 补点加入可配置 `opacity_value`，并对其做范围 clamp，避免 `inverse_sigmoid` 产生无穷值。
- 仅在 render-guided mask 补点路径使用 valid-depth median，普通路径恢复原生 `np.median(depth)` 的 `adaptive_pointsize` 行为。
- 增加防遗忘统计字段：
  - `seen_times`
  - `last_xyz`
  - `last_features_dc`
  - `last_scaling`
  - `last_opacity`
  - 对应 importance weights
- 增加防遗忘相关方法：
  - `reset_forgetting_statistics`
  - `_append_forgetting_statistics`
  - `_prune_forgetting_statistics`
  - `capture_forgetting_snapshot`
  - `update_forgetting_importance`
  - `forgetting_regularization_loss`
- 所有防遗忘统计维护均受 `forgetting_regularization.enabled` 短路控制。
- 新增点为空时安全返回，避免 Open3D 反投影无点导致后续 optimizer 拼接异常。
- render-guided 补点可选择不重置原生 densification stats，用于与原生梯度致密化共存。

### `utils/slam_backend.py`

- 增加 render-guided 和 forgetting 配置读取与合法性校验。
- 增加 `add_render_guided_gaussians(...)`：
  - 使用 exposure 校正后的 render RGB 与 GT RGB 计算 `rgb_error_map`。
  - 使用 `opacity_map` 检测 opacity hole。
  - 支持 RGB-D 下可选 `depth_error_ratio`。
  - 使用 RGB 边界、有效 depth、downsample 约束补点区域。
  - 使用 opacity / RGB / depth 的组合 score 做 top-k 选点。
- 在 `update_gaussian` 分支中先执行原生 `densify_and_prune(...)`，再执行 render-guided 补点，避免补点改变 Gaussian 数量后破坏原生梯度统计 shape。
- 增加单次 update 总补点预算 `max_new_points_per_update`，避免窗口内多个 keyframe 同时加满导致高斯数量暴涨。
- 防遗忘正则只在非 prune 的普通 mapping step 中加入 loss、更新 importance。
- optimizer step 后，在防遗忘开关开启时更新参数快照。

### 配置文件

在以下配置中加入默认关闭的配置块：

- `configs/live/realsense.yaml`
- `configs/live/realsense_rgbd.yaml`
- `configs/mono/tum/base_config.yaml`
- `configs/rgbd/replica/base_config.yaml`
- `configs/rgbd/tum/base_config.yaml`
- `configs/stereo/euroc/base_config.yaml`

新增配置包括：

- `render_guided_densification.enabled: False`
- `use_opacity_hole`
- `use_rgb_error`
- `use_depth_error`
- `opacity_threshold`
- `rgb_error_threshold`
- `depth_error_ratio_threshold`
- `max_new_points_per_kf`
- `max_new_points_per_update`
- `min_depth`
- `trigger_every`
- `trigger_offset`
- `downsample`
- `initial_opacity`
- `forgetting_regularization.enabled: False`
- `update_every_kf`
- `lambda_color`
- `lambda_scaling`
- `lambda_xyz`
- `lambda_opacity`
- `max_history_views`
- `normalize_importance`

## 3. 已验证内容

已完成静态验证：

1. Python 语法检查通过：

```text
python -m py_compile gaussian_splatting\scene\gaussian_model.py utils\slam_backend.py
```

2. Git diff 空白检查通过：

```text
git diff --check
```

3. 人工检查过关键开关语义：

- `render_guided_densification.enabled=False` 时，不会进入 render-guided 补点调用。
- `forgetting_regularization.enabled=False` 时，防遗忘统计函数会立即 return，不执行 tensor append / prune / snapshot / importance 更新。
- 普通 `adaptive_pointsize` 路径恢复原生 median 行为，valid-depth median 只用于 render-guided mask 补点。

未完成完整运行验证：

- 未运行完整 SLAM 数据集实验。
- 未验证 CUDA rasterizer、Open3D 反投影和实际训练中的显存曲线。
- 未生成 ATE / PSNR / SSIM / LPIPS 消融结果。

## 4. 剩余风险

1. 实验风险：render-guided 补点的阈值和预算仍需要实测调参，过低的 RGB / opacity 阈值可能导致高斯数量增长过快。
2. 几何风险：monocular 模式下没有 GT depth，render-guided 补点会使用当前渲染深度，几何可靠性弱于 RGB-D。
3. depth error 风险：`use_depth_error=True` 仅在 `viewpoint.depth is not None` 时生效，monocular 下会自动跳过。
4. 正则强度风险：`lambda_color` / `lambda_scaling` 过大可能压制当前帧收敛，需要从较小值开始做消融。
5. 运行时风险：虽然已对 visibility shape mismatch 做 warning 并跳过 importance 更新，但完整 prune / densify / sync 流程仍需长序列验证。
6. 配置兼容风险：如果用户启用 render-guided 或 forgetting 但配置字段非法，后端会在 `set_hyperparams()` 阶段抛 `ValueError`，这是有意的早失败行为。
