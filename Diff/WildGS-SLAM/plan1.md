# 将 WildGS-SLAM 不确定性图处理嵌入 MonoGS 的修改计划

## Summary

将当前 `plan.md` 中“读取预计算 uncertainty 图”的方案替换为 WildGS-SLAM 风格的不确定性建模：每帧先获取 DINO/FiT3D 图像特征，再用一个轻量 MLP 在线预测像素级 uncertainty map；mapping 阶段训练该 MLP，并用 `0.5 / uncertainty^2` 权重降低动态区域对 RGB/depth loss 的影响；tracking 阶段使用同一权重降低动态区域对位姿优化的影响。

为保持对 MonoGS 低侵入，不迁移 WildGS-SLAM 的 DROID-SLAM Dense BA、Metric3D 深度先验、loop closure，只迁移“不确定性图生成、训练、转权重、用于 tracking/mapping loss”这条链路。默认采用“预提取 DINO 特征 + MonoGS 运行时训练 uncertainty MLP”，避免每次 SLAM 运行都调用特征网络。

## Key Changes

### 1. 新增 uncertainty 模块

- 从 WildGS-SLAM 迁移 `MLPNetwork / generate_uncertainty_mlp`，输入每个 patch 的 DINO feature，输出正 uncertainty。
- 迁移必要的 mapping loss helper：SSIM components、median filter、`compute_mapping_loss_components`、`compute_dino_regularization_loss`。
- 在 MonoGS 中新增统一函数把 uncertainty 转为权重：

```python
u = clamp(u, min=0.1) + 1e-3
w = clamp(0.5 / u**2, 0, 1)
```

- 可选把 `w < 0.1` 置零，用于直接忽略高不确定性区域。

### 2. 数据与 Camera 接口

- Dataset 支持读取每帧 DINO feature，推荐路径为 `mono_priors/features/{frame_idx:05d}.npy`，shape 为 `H_feat x W_feat x 384`。
- `Camera` 新增 `features`、`uncertainty` 或 `uncertainty_weight` 字段；前端当前帧和后端关键帧都通过 `Camera` 携带 feature。
- 若未启用 uncertainty 或 feature 缺失，所有 loss 自动退化为原 MonoGS。

### 3. SLAM 初始化

- 在 `slam.py` 创建 `uncer_network = generate_uncertainty_mlp(feature_dim)`。
- 给后端持有 `uncer_network` 和 `uncer_optimizer`，学习率参考 WildGS-SLAM：`lr=0.0004`，`weight_decay=1e-5`。
- 前端 tracking 只读取当前 MLP 推理出的权重，不更新 MLP；后端 mapping 负责训练 MLP。

### 4. Tracking loss

- 在 `utils/slam_utils.py` 的 `get_loss_tracking_rgb` 中，对现有 `opacity * RGB L1 * rgb_pixel_mask * grad_mask` 再乘 uncertainty weight。
- RGB-D tracking 的 depth loss 默认不加权，避免早期 depth 约束被过度削弱；通过配置可开启 depth 加权。
- 权重由当前帧 `viewpoint.features` 经 `uncer_network` 推理得到，并 resize 到 RGB 尺寸。

### 5. Mapping loss

- 增加 `get_loss_mapping_uncertainty`，结构参考 WildGS-SLAM：用 `uncer_network(viewpoint.features)` 得到 uncertainty；计算 RGB L1、depth L1、SSIM-derived uncertainty loss；用 uncertainty weight 加权 RGB/depth loss。
- 在后端 `initialize_map` 和 `map` 中，当 uncertainty 启用且当前关键帧有 features 时调用 uncertainty-aware mapping loss。
- mapping 反向传播同时更新 Gaussians、关键帧 pose/exposure，以及 `uncer_network`。
- 加入 DINO feature regularization：对相似 feature 的 uncertainty 做一致性约束，参考 WildGS-SLAM 的 `reg_mult=0.5`、`reg_stride=2`。

### 6. 配置新增

```yaml
Training:
  uncertainty:
    enabled: true
    feature_dim: 384
    feature_type: "dino"
    min_uncertainty: 0.1
    eps: 0.001
    min_weight_threshold: 0.1
    train_frac_fix: 0.3
    ssim_window_size: 7
    ssim_median_filter_size: 5
    opacity_th_for_uncer_loss: 0.9
    reg_stride: 2
    reg_mult: 0.5
    ssim_mult: 0.5
    uncer_depth_mult: 0.2
    lr: 0.0004
    weight_decay: 0.00001
    apply_to_tracking_rgb: true
    apply_to_tracking_depth: false
    apply_to_mapping_rgb: true
    apply_to_mapping_depth: true
    apply_during_init: true

Dataset:
  feature_path: ""
  feature_format: "npy"
```

## Implementation Plan

### 1. 替换计划文件

- 将 `Diff/WildGS-SLAM/plan.md` 内容替换为本计划。
- 明确新方案不再依赖预计算 uncertainty 图，而是依赖预计算 DINO feature 和在线训练 MLP。

### 2. 新增 `utils/dyn_uncertainty/`

- `uncertainty_model.py`：迁移 WildGS-SLAM 的 MLP。
- `mapping_utils.py`：迁移 uncertainty loss、SSIM component、DINO regularization 相关 helper。
- `median_filter.py`：迁移 WildGS-SLAM median pooling。

### 3. 扩展 `utils/dataset.py`

- 增加 feature 读取函数，按帧编号加载 `.npy`。
- `MonocularDataset.__getitem__` 在启用 uncertainty 时返回 `image, depth, pose, features`。
- 不启用时保持原行为兼容。

### 4. 扩展 `utils/camera_utils.py`

- `Camera.__init__` 保存 `features=None`。
- `Camera.init_from_dataset` 兼容三元组和四元组返回值。
- `Camera.clean` 清理 features，避免内存增长。

### 5. 扩展 `utils/slam_utils.py`

- 增加 uncertainty 推理、resize、权重转换 helper。
- 增加 `get_loss_mapping_uncertainty`。
- 修改 tracking loss，在配置开启时乘 uncertainty weight。

### 6. 扩展 `utils/slam_backend.py`

- 后端初始化 `uncer_network`、`uncer_optimizer`。
- mapping loss 选择原始 loss 或 uncertainty-aware loss。
- optimizer step 后同步 step/zero `uncer_optimizer`。

### 7. 扩展 `utils/slam_frontend.py`

- 前端 tracking 调用 loss 时传入或访问共享的 `uncer_network`。
- tracking 只对 uncertainty MLP 做 no-grad 推理，不更新网络。

### 8. 扩展 `slam.py`

- 创建 uncertainty MLP。
- 将网络传给前端和后端，或至少传给后端并让前端通过同步后的模型/权重推理。
- 保存 config 时包含 uncertainty 参数，便于实验复现。

## Test Plan

### 1. 兼容性测试

- `Training.uncertainty.enabled=false` 时，MonoGS 能按原配置运行。
- feature 文件缺失时，打印明确 warning 并退化到原始 loss。
- 全零/全常数 feature 不应导致 NaN，uncertainty 输出应为正。

### 2. 单元级验证

- MLP 输入 `H_feat x W_feat x 384`，输出 `H_feat x W_feat`。
- uncertainty weight resize 到 RGB 尺寸后 shape 为 `1 x H x W`。
- `0.5 / u^2` 权重范围被 clamp 到 `[0,1]`。

### 3. 训练路径验证

- mapping backward 后 `uncer_network` 参数梯度非空。
- tracking backward 不更新 `uncer_network`。
- `uncer_loss`、RGB loss、depth loss 均为有限值。

### 4. 消融实验

- MonoGS baseline。
- MonoGS + uncertainty mapping only。
- MonoGS + uncertainty tracking only。
- MonoGS + uncertainty tracking + mapping。

### 5. 指标

- Tracking 使用 ATE RMSE。
- Rendering 使用 PSNR、SSIM、LPIPS。
- 额外记录 mean uncertainty、mean weight、低权重像素比例，用于分析动态区域是否被降权。

## Assumptions

- 默认采用预提取 DINO/FiT3D feature，而不是运行时在线提取 feature；这保留 WildGS-SLAM 的 uncertainty 学习机制，同时减少 MonoGS 主循环依赖和显存压力。
- 第一版只支持 monocular/RGB-D 的 RGB 图像特征；Realsense 实时模式暂不接入。
- 不迁移 WildGS-SLAM 的 DROID-SLAM BA、Metric3D、深度视频缓存、loop closure。
- `plan.md` 应被完整覆盖为本计划；不删除任何文件或目录。
