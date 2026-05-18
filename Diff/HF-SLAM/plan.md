# 将 HF-SLAM 可迁移模块嵌入 MonoGS 的修改计划

本文档基于 `compare_HF-SLAM.md` 中“可迁移模块分析”，并结合 HF-SLAM 源码中 `scripts/slam.py`、`utils/slam_external.py` 的实现，规划将以下三个模块嵌入 MonoGS：

1. Opacity 空洞检测
2. RGB 渲染误差引导致密化
3. 防遗忘正则化

当前阶段只制定计划，不修改 MonoGS 源码。

---

## 1. 总体迁移思路

HF-SLAM 与 MonoGS 都使用 3D Gaussian 作为地图表达，并通过可微渲染进行 tracking 和 mapping。三类模块不需要替换 MonoGS 的多进程框架，只需要作为后端 mapping 的增强逻辑接入。

建议分两阶段实现：

| 阶段 | 目标模块 | 原因 |
|---|---|---|
| 第一阶段 | Opacity 空洞检测 + RGB 渲染误差引导致密化 | 工程风险低，直接服务建图质量，适合先做消融验证 |
| 第二阶段 | 防遗忘正则化 | 需要维护高斯点历史统计量，涉及 GaussianModel 的增删同步，复杂度更高 |

第一阶段优先保持 MonoGS 现有 `densify_and_prune` 梯度致密化逻辑不变，在其外侧增加“基于像素误差的补点入口”；第二阶段再扩展 GaussianModel 的统计字段和 mapping loss。

---

## 2. 参考 HF-SLAM 实现

### 2.1 Opacity / RGB / Depth 引导新增点

HF-SLAM 中对应逻辑主要在：

```text
HF-SLAM-main/scripts/slam.py
- add_new_gaussians(...)
```

核心流程：

1. 对当前帧渲染 RGB、depth 和 opacity。
2. 计算像素级误差：
   - `rgb_loss = abs(gt_rgb - render_rgb).mean(dim=0)`
   - `depth_loss = abs(gt_depth - render_depth)`
3. 构造补点 mask：
   - `opacity < sil_thres`
   - `rgb_loss > 0.6`
   - `depth_diff_ratio > 0.1`
4. 只在有效深度区域从 RGB-D 反投影生成新点。
5. 将新高斯拼接到现有参数和辅助变量中。

本次计划只迁移 opacity 与 RGB 两部分；depth error 可作为后续 RGB-D 扩展项。

### 2.2 防遗忘正则化

HF-SLAM 中对应逻辑主要在：

```text
HF-SLAM-main/scripts/slam.py
- initialize_params(...)
- get_loss_mapping(...)
- update_importance_weights(...)

HF-SLAM-main/utils/slam_external.py
- remove_points(...)
- remove_points_wo_optimizer(...)
```

核心变量：

| 变量 | 作用 |
|---|---|
| `seen_times` | 每个高斯被有效观测到的次数 |
| `last_rgb_colors` | 上一轮快照中的颜色参数 |
| `rgb_colors_importance_weights` | 颜色重要性权重 |
| `last_depth` | 上一轮快照中的深度 / 几何参考 |
| `depth_importance_weights` | 几何重要性权重 |
| `log_scale_last_frame` | 上一轮快照中的尺度参数 |
| `scale_importance_weights` | 尺度重要性权重 |

正则项形式：

```text
color_reg = importance_color * abs(current_color - last_color)
depth_reg = importance_depth * abs(current_depth - last_depth)
scale_reg = importance_scale * abs(current_scale - last_scale)
```

重要性权重通过当前视角反向传播得到的参数梯度累计，并按 `seen_times` 求平均。

---

## 3. MonoGS 当前嵌入点

### 3.1 后端 mapping 主流程

主要文件：

```text
utils/slam_backend.py
```

关键位置：

| 位置 | 现有职责 | 迁移用途 |
|---|---|---|
| `BackEnd.map(...)` | 遍历滑窗关键帧、渲染、累计 mapping loss | 获取 `image/depth/opacity`，统计 RGB error 和 opacity hole |
| `update_gaussian` 分支 | 固定频率调用 `gaussians.densify_and_prune(...)` | 在同一时机触发 rendering-guided densification |
| `add_next_kf(...)` | 新关键帧加入时从 RGB-D/估计深度初始化高斯 | 可复用其反投影能力，按 mask 局部补点 |
| `prune` 分支 | 根据可见次数修剪高斯 | 同步维护防遗忘统计量 |

### 3.2 损失函数

主要文件：

```text
utils/slam_utils.py
```

关键函数：

| 函数 | 现有职责 | 迁移用途 |
|---|---|---|
| `get_loss_mapping(...)` | 根据输入类型分发 mapping loss | 增加正则项入口参数 |
| `get_loss_mapping_rgb(...)` | RGB mapping loss | 可输出或复用像素级 RGB error |
| `get_loss_mapping_rgbd(...)` | RGB-D mapping loss | RGB-D 模式可后续加入 depth error densification |

### 3.3 GaussianModel 参数管理

主要文件：

```text
gaussian_splatting/scene/gaussian_model.py
```

关键位置：

| 位置 | 现有职责 | 迁移用途 |
|---|---|---|
| `create_pcd_from_image_and_depth(...)` | 从整帧 RGB-D 生成点云 | 扩展为支持像素 mask 的局部反投影 |
| `extend_from_pcd(...)` / `densification_postfix(...)` | 追加新高斯并更新 optimizer | rendering-guided densification 的最终追加入口 |
| `prune_points(...)` | 删除高斯并同步基础统计量 | 同步删除防遗忘统计量 |
| `densify_and_clone(...)` / `densify_and_split(...)` | 梯度致密化 | 新增点后同步防遗忘统计量 |

---

## 4. 第一阶段：Opacity 空洞检测与 RGB 渲染误差引导致密化

### 4.1 新增配置项

建议在 `Training` 下增加独立配置块，默认关闭，避免影响 baseline：

```yaml
render_guided_densification:
  enabled: false
  use_opacity_hole: true
  use_rgb_error: true
  use_depth_error: false
  opacity_threshold: 0.5
  rgb_error_threshold: 0.25
  depth_error_ratio_threshold: 0.1
  max_new_points_per_kf: 3000
  min_depth: 0.01
  trigger_every: 150
  trigger_offset: 50
  downsample: 1
```

初始阈值说明：

| 参数 | 建议初值 | 说明 |
|---|---:|---|
| `opacity_threshold` | `0.5` | HF-SLAM 使用 `sil_thres` 思路；MonoGS opacity 需结合实际渲染范围调参 |
| `rgb_error_threshold` | `0.25` | MonoGS 图像通常归一化到 `[0, 1]`，不建议直接照搬 HF-SLAM 的 `0.6` |
| `max_new_points_per_kf` | `3000` | 防止一次补点过多导致显存膨胀 |
| `trigger_every/offset` | 与 `gaussian_update_every/offset` 对齐 | 便于和原始梯度致密化做公平比较 |

### 4.2 计算补点 mask

在 `BackEnd.map(...)` 的每个 keyframe render 后，记录：

```text
image_ab = exp(exposure_a) * image + exposure_b
rgb_error_map = abs(image_ab - viewpoint.original_image).mean(dim=0)
opacity_map = opacity.squeeze()
```

构造 mask：

```text
hole_mask = opacity_map < opacity_threshold
rgb_mask = rgb_error_map > rgb_error_threshold
densify_mask = hole_mask | rgb_mask
```

再叠加有效区域约束：

1. RGB 边界 mask：沿用 `rgb_boundary_threshold` 过滤无效黑边。
2. RGB-D 模式：优先要求 `viewpoint.depth > min_depth`。
3. 单目模式：不直接使用真实 depth，可先仅在当前 `depth` 渲染有效区域或初始化深度图有效区域补点。
4. 对 mask 做采样上限控制：若超过 `max_new_points_per_kf`，随机采样或按 RGB error top-k 采样。

### 4.3 从 mask 生成新高斯

推荐方案：

1. 在 `GaussianModel.create_pcd_from_image(...)` 或新增辅助函数中支持 `mask` 参数。
2. 对 RGB-D 模式，使用 `viewpoint.depth` 和 mask 直接反投影生成局部点云。
3. 对 monocular 模式，使用传入的 `depthmap` 或当前渲染深度作为近似深度，但第一阶段建议先在 RGB-D 配置上验证。
4. 复用 `extend_from_pcd(...)` 和 `densification_postfix(...)` 追加新高斯，避免重复实现 optimizer 拼接逻辑。

### 4.4 与现有梯度致密化的关系

不替换现有：

```text
gaussians.add_densification_stats(...)
gaussians.densify_and_prune(...)
```

建议执行顺序：

1. 先完成当前 mapping loss 的反向传播。
2. 累计原始梯度致密化统计。
3. 在 `update_gaussian` 为真时：
   - 先执行 rendering-guided add new gaussians；
   - 再执行原始 `densify_and_prune`；
   - 最后执行 pruning / opacity reset。

原因：rendering-guided 补点负责填补像素级不足，原始梯度致密化继续负责已有高斯的 clone/split 细化。

### 4.5 第一阶段验收标准

最小消融：

| 实验 | 设置 |
|---|---|
| E0 | MonoGS baseline |
| E1 | baseline + opacity hole densification |
| E2 | baseline + RGB error densification |
| E3 | baseline + opacity + RGB error densification |

记录指标：

1. ATE RMSE
2. PSNR / SSIM / LPIPS
3. 每帧新增高斯数
4. 最终高斯总数
5. mapping 时间和显存峰值

---

## 5. 第二阶段：防遗忘正则化

### 5.1 新增 GaussianModel 状态

建议在 `GaussianModel` 中增加以下成员，默认空 tensor，并与现有高斯数量保持一致：

```text
seen_times
last_xyz
last_features_dc
last_scaling
last_opacity
xyz_importance_weights
features_dc_importance_weights
scaling_importance_weights
opacity_importance_weights
```

注意：MonoGS 已有 `n_obs`，可用于可见次数统计，但它主要服务 pruning。防遗忘模块建议单独维护 `seen_times`，避免改变原 pruning 语义。

### 5.2 参数快照策略

在每次 keyframe mapping 完成后，保存当前重要参数快照：

```text
last_xyz = _xyz.detach().clone()
last_features_dc = _features_dc.detach().clone()
last_scaling = _scaling.detach().clone()
last_opacity = _opacity.detach().clone()
```

第一版建议只约束：

1. `features_dc`：直接对应颜色一致性，风险最低。
2. `scaling`：抑制高斯形状剧烈变化。

暂不优先约束：

1. `xyz`：过强约束可能影响新区域几何收敛。
2. `opacity`：MonoGS 已有 opacity reset，过强约束可能和重置策略冲突。

### 5.3 重要性权重更新

参考 HF-SLAM `update_importance_weights(...)`：

1. 选择当前关键帧和少量历史关键帧。
2. 渲染并计算 RGB mapping loss。
3. 单独 backward 一次，用参数梯度近似重要性。
4. 对可见高斯累计：
   - `features_dc_importance_sum += abs(_features_dc.grad)`
   - `scaling_importance_sum += abs(_scaling.grad)`
   - `seen_times += 1`
5. 计算平均重要性：
   - `importance = importance_sum / clamp(seen_times, min=1)`
6. 清空梯度，避免污染正式 mapping optimizer step。

实现时要注意：这一步应与正式优化解耦，避免重复 step 或累积错误梯度。

### 5.4 正则项加入 mapping loss

在 `utils/slam_utils.py` 中为 `get_loss_mapping(...)` 增加可选正则输入，或在 `BackEnd.map(...)` 中计算后叠加：

```text
loss_reg_color = mean(features_dc_importance_weights * abs(_features_dc - last_features_dc))
loss_reg_scale = mean(scaling_importance_weights * abs(_scaling - last_scaling))
loss_mapping += lambda_color_reg * loss_reg_color
loss_mapping += lambda_scale_reg * loss_reg_scale
```

建议配置：

```yaml
forgetting_regularization:
  enabled: false
  update_every_kf: 1
  lambda_color: 0.01
  lambda_scaling: 0.01
  lambda_xyz: 0.0
  lambda_opacity: 0.0
  max_history_views: 3
  normalize_importance: true
```

### 5.5 与增删高斯同步

防遗忘统计量必须跟随高斯增删变化：

| 操作 | 同步动作 |
|---|---|
| `densification_postfix(...)` 新增点 | 为新增点追加 0 权重，并用当前参数作为快照 |
| `densify_and_clone(...)` 克隆点 | 克隆对应统计量或初始化为 0；第一版建议初始化为 0 |
| `densify_and_split(...)` 分裂点 | 子点继承父点部分统计量或初始化为 0；第一版建议初始化为 0 |
| `prune_points(...)` 删除点 | 所有统计量使用同一 mask 删除 |
| `reset(...)` 清空地图 | 清空所有防遗忘统计量 |

这一部分是第二阶段最大的工程风险，必须优先写单元级 shape 检查。

### 5.6 第二阶段验收标准

消融：

| 实验 | 设置 |
|---|---|
| E4 | baseline + color regularization |
| E5 | baseline + scale regularization |
| E6 | baseline + color + scale regularization |
| E7 | 第一阶段最佳配置 + 防遗忘正则化 |

额外验证：

1. 历史关键帧 PSNR 是否下降更少。
2. 当前帧 PSNR 是否因正则过强而下降。
3. 高斯数量变化后所有统计 tensor shape 是否一致。
4. pruning 后是否出现 optimizer state 与参数 shape 不一致。

---

## 6. 推荐实现顺序

1. 新增配置项，默认关闭。
2. 增加一个只计算 mask、不新增点的 debug 路径，保存或打印：
   - hole pixel ratio
   - rgb high-error pixel ratio
   - selected new point count
3. 实现 masked point cloud 生成与 `extend_from_pcd` 复用。
4. 接入 `BackEnd.map(...)` 的 `update_gaussian` 分支。
5. 跑 E0/E1/E2/E3 消融，调阈值和新增点上限。
6. 增加 GaussianModel 防遗忘统计量。
7. 增加正则 loss，仅先启用 `features_dc` 与 `scaling`。
8. 跑 E4/E5/E6/E7 消融。

---

## 7. 风险与规避

| 风险 | 表现 | 规避方式 |
|---|---|---|
| RGB error 阈值过低 | 高斯数量暴涨，速度下降 | 使用 top-k / max_new_points_per_kf 限制 |
| opacity 阈值过高 | 在已建模区域重复补点 | 叠加 RGB 边界、有效深度、最小误差条件 |
| 单目深度不可靠 | 新点几何错误，ATE 变差 | 第一阶段先在 RGB-D 上验证，单目只开 RGB/opacity 或使用保守深度 |
| 正则过强 | 新区域收敛慢，当前帧质量下降 | 从 `lambda=0.01` 或更低开始，分参数消融 |
| 统计量 shape 不一致 | prune/densify 后运行时报错 | 所有增删函数统一调用同步工具函数 |
| 与 opacity reset 冲突 | 旧区域 opacity 被错误保持或重置 | 第二阶段默认不约束 opacity |

---

## 8. 建议最终对外描述

本计划将 HF-SLAM 的渲染质量感知思想迁移到 MonoGS 后端，在保留 MonoGS 原有梯度致密化和滑窗优化框架的基础上，引入由 opacity 空洞和 RGB 渲染误差共同驱动的补点机制；进一步通过基于历史梯度重要性的正则项约束关键高斯参数，减少在线建图中新帧优化对历史视角重建质量的破坏。

预期第一阶段主要提升 PSNR、SSIM、LPIPS；第二阶段主要提升历史视角稳定性，并可能间接改善 ATE RMSE。
