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

---

## 5. 本轮运行问题修复总结（2026-05-19）

### 5.1 ATE 评估阶段 `SVD did not converge`

运行过程中在 `evo` 轨迹对齐阶段出现：

```text
numpy.linalg.LinAlgError: SVD did not converge
```

该问题发生在 ATE 评估流程，而不是 tracking、mapping 或 uncertainty 模块本身。直接原因是传入 `evo` 的关键帧轨迹中存在非法位姿，常见形式包括：

- 位姿矩阵中包含 `NaN` 或 `Inf`；
- 有效位姿数量不足；
- 轨迹没有有效平移运动，导致 Umeyama 对齐退化。

已在 `utils/eval_utils.py` 中做了防护：

- 新增 `_is_valid_pose(...)`，过滤非 4x4 或含 `NaN/Inf` 的位姿矩阵；
- 新增 `_has_nonzero_motion(...)`，避免对退化轨迹继续做 `evo` 对齐；
- `eval_ate(...)` 在无关键帧时直接跳过；
- 对非法关键帧位姿做跳过处理，并记录 `skipped_invalid_ids`；
- `evaluate_evo(...)` 捕获 `np.linalg.LinAlgError` 和 `ValueError`，输出清晰日志并返回 `nan`，避免主流程崩溃；
- 修复原有日志字符串 `RMSE ATE \[m]` 的 Python 转义警告。

修复后的行为是：ATE 评估不会因为少量异常关键帧直接中断 SLAM 流程，但日志会提示被跳过的关键帧 id。例如：

```text
Eval: Skipped 3 invalid pose pair(s) for ATE; first ids: [504, 509, 514]
```

这类日志说明评估阶段已安全跳过异常位姿，但也提示前面的 tracking 或 backend BA 可能已经在对应帧附近出现不稳定，需要进一步排查。

### 5.2 建图效果差的初步判断

本轮日志中出现：

```text
Render-guided densification added 14 Gaussians
Render-guided densification added 6 Gaussians
Render-guided densification added 5 Gaussians
```

同时最终渲染指标较低：

```text
mean psnr: 17.98, ssim: 0.60, lpips: 0.45
```

初步判断主要问题不是 uncertainty 模块，因为当前配置中：

```yaml
Training:
  uncertainty:
    enabled: False
```

更可疑的是 `render_guided_densification` 实际补点数量太少。原逻辑中 render-guided 补点最终复用 `GaussianModel.extend_from_pcd_seq(...)`，而该路径默认使用：

```yaml
Dataset:
  pcd_downsample: 64
```

这会导致即使误差 mask 选出了较多候选像素，Open3D 点云生成后仍被随机下采样到约 `1/64`，最终新增高斯数量可能只剩个位数或十几个，难以改善地图结构。

### 5.3 Render-guided 诊断日志与独立下采样参数

按照“先做诊断日志和独立 `pcd_downsample`，不重构不相关代码”的原则，已做以下小范围修改。

#### `utils/slam_backend.py`

- 在 `_load_render_guided_config()` 中增加 `Training.render_guided_densification.pcd_downsample` 校验；
- 该字段缺省为 `4`，避免破坏已有未显式配置该字段的配置文件；
- 在 `add_render_guided_gaussians(...)` 中增加诊断日志，统计：
  - opacity hole 候选像素数；
  - RGB error 候选像素数；
  - depth error 候选像素数；
  - RGB/depth 有效区域过滤后的候选数量；
  - pixel downsample 后数量；
  - budget 限制后数量；
  - Open3D 点云下采样前后点数；
  - 最终新增 Gaussians 数。

诊断日志格式示例：

```text
Render-guided diagnostics kf=504: opacity=..., rgb=..., depth=..., candidate=..., valid_rgb=..., valid_depth=..., after_valid=..., after_pixel_downsample=..., after_budget=..., pcd_downsample=4, pcd_before=..., pcd_after=..., added=...
```

#### `gaussian_splatting/scene/gaussian_model.py`

只扩展现有函数参数，默认行为保持不变：

- `create_pcd_from_image(...)` 增加可选参数：
  - `pcd_downsample_factor`
  - `return_point_count`
- `create_pcd_from_image_and_depth(...)` 增加同名可选参数；
- `extend_from_pcd_seq(...)` 增加同名可选参数；
- 未传入 `pcd_downsample_factor` 时，仍按原逻辑使用：
  - 初始化帧：`Dataset.pcd_downsample_init`
  - 普通关键帧：`Dataset.pcd_downsample`
- 仅 render-guided 补点路径会显式传入独立的 `pcd_downsample_factor`。

#### `configs/mono/tum/base_config.yaml`

在 `render_guided_densification` 配置块中新增：

```yaml
pcd_downsample: 4
```

当前 TUM 单目实验中 `render_guided_densification.enabled: True` 是已有工作区修改；本轮新增的是独立补点下采样参数和诊断日志。

### 5.4 已验证内容

已完成 Python 语法检查：

```powershell
python -m py_compile utils\eval_utils.py
python -m py_compile utils\slam_backend.py gaussian_splatting\scene\gaussian_model.py
```

检查结果通过。

### 5.5 后续建议

下一次运行时重点观察诊断日志中的几个字段：

- 如果 `candidate` 很大但 `pcd_after` 很小，说明点云下采样仍然过强；
- 如果 `candidate` 本身很小，说明 `opacity_threshold` 或 `rgb_error_threshold` 太严格；
- 如果 `after_valid` 明显变小，说明 RGB/depth 有效区域过滤过强；
- 如果 `added` 增多后又出现 invalid pose，说明补点可能引入了错误几何，需要降低 `max_new_points_per_update` 或提高误差阈值。

建议先以如下配置作为保守起点继续实验：

```yaml
render_guided_densification:
  enabled: True
  downsample: 1
  pcd_downsample: 4
  rgb_error_threshold: 0.25
  opacity_threshold: 0.5
  max_new_points_per_update: 3000
  max_new_points_per_kf: 3000
```
