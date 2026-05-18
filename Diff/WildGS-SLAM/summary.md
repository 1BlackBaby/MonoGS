# WildGS-SLAM uncertainty 接入总结

## 1. 当前目标

本轮改动的目标是把 WildGS-SLAM 风格的像素级 uncertainty 机制接入 MonoGS，用 DINO/FiT3D 图像特征经过轻量 MLP 预测 uncertainty map，并把它用于 tracking 与 mapping loss 的动态区域降权。

当前实现的总开关是：

```yaml
Training:
  uncertainty:
    enabled: true
```

要求是：当 `Training.uncertainty.enabled: false` 时，系统应回退到原始 MonoGS 行为，不创建 uncertainty MLP、不读取或在线提取 feature、不进入 uncertainty-aware loss、不优化 uncertainty 网络。

## 2. 总开关检查结论

结论：核心创新点都在 `Training.uncertainty.enabled` 总开关之下，未发现关闭开关后仍会参与 loss、读取 feature 或训练 MLP 的实质性漏网路径。

具体检查结果：

- `slam.py`：只有 `Training.uncertainty.enabled: true` 时才调用 `generate_uncertainty_mlp()` 创建 `uncer_network`。关闭时前后端拿到的都是 `None`。
- `utils/dataset.py`：`BaseDataset.use_uncertainty` 直接来自 `Training.uncertainty.enabled`。关闭时 `load_features()` 立即返回 `None`，不会读取 `.npy`，也不会在线提取 DINO feature；dataset 返回三元组 `(image, depth, pose)`。
- `utils/camera_utils.py`：`Camera.features` 是兼容字段。关闭 uncertainty 时该字段为 `None`，`Camera.init_from_dataset()` 兼容三元组和四元组返回。
- `utils/slam_utils.py`：tracking RGB/depth 加权、mapping uncertainty loss、SSIM-derived uncertainty loss、DINO regularization 都需要 `enabled=True` 且 `uncertainty_network` 非空；mapping 还要求当前 viewpoint 有 features，否则回退到原始 MonoGS loss。
- `utils/slam_backend.py`：`uncer_optimizer` 只有在 `enabled=True` 且 `uncer_network` 非空时创建；optimizer step/zero_grad 都受 `uncer_optimizer is not None` 保护。
- `utils/slam_frontend.py`：tracking 使用 uncertainty 还要满足 tracking 开关、MLP 非空、warmup 完成。tracking 阶段通过 no-grad 推理，不更新 MLP。

存在少量常驻但惰性的兼容管线：`uncer_network=None` 字段、`Camera.features=None` 字段、队列消息里附带 `uncer_state=None`、结束时调用统计函数。这些不会改变关闭 uncertainty 时的 loss 与训练路径。

## 3. 当前实现内容

新增 `utils/dyn_uncertainty/`：

- `uncertainty_model.py`：轻量 MLP，输入 feature_dim 默认 384，输出正的 uncertainty，输出层后使用 Softplus。
- `median_filter.py`：median pooling，用于平滑 SSIM-derived uncertainty supervision。
- `mapping_utils.py`：SSIM components、mapping loss components、DINO feature regularization。

扩展 `utils/dataset.py`：

- 支持 `.npy` feature 读取。
- 默认 feature 路径为 `Dataset.dataset_path/mono_priors/features/{frame_idx:05d}.npy`。
- 支持在线提取 feature：`extract_features_online: True` 时走 `utils/mono_priors/img_feature_extractors.py`。
- 在线提取结果保存到 `feature_output_root/<scene>/mono_priors/features/`。
- 增加 feature 文件缺失、损坏、shape 错误、dtype 错误、feature_dim 不匹配、NaN/Inf 的防护。
- feature 保存在 CPU，只有 loss 计算时转到对应 device。

扩展 `utils/slam_utils.py`：

- 增加 uncertainty 预测、resize、weight 转换 helper。
- weight 形式为 `0.5 / (uncertainty + eps)^2`，再 clamp 到 `[0, 1]`，并支持 `min_weight_threshold`。
- tracking RGB loss 默认启用 uncertainty weight。
- tracking depth 默认不启用，可通过 `apply_to_tracking_depth` 打开。
- mapping loss 支持 uncertainty-aware 路径：RGB/depth loss 加权，叠加 SSIM-derived uncertainty loss。
- DINO regularization 使用局部窗口 keyframes 的 feature，并通过 `reg_stride` 与 `reg_max_samples` 控制计算量。
- 增加 `tracking_used / tracking_fallback / mapping_used / mapping_fallback` 统计。

扩展 `utils/slam_backend.py`：

- 后端持有并训练 `uncer_network`。
- mapping 与 initialization 调用 `get_loss_mapping(..., uncertainty_network=self.uncer_network)`。
- mapping 阶段把局部 keyframe window 传给 DINO regularization。
- 后端同步 `uncer_network.state_dict()` 给前端。
- `prune=True` 提前返回前会清理 `uncer_optimizer` 梯度，避免梯度残留。
- stop 时输出 backend uncertainty loss 统计。

扩展 `utils/slam_frontend.py`：

- 前端接收后端同步的 MLP state_dict，并设置 eval 模式。
- tracking 使用 `get_tracking_uncertainty_network()` 控制 warmup。
- 当前配置要求 `tracking_warmup_keyframes: 25` 且 `tracking_warmup_backend_syncs: 1` 后才启用 tracking uncertainty。
- tracking 使用 no-grad uncertainty 推理，不反向更新 MLP。
- 结束时输出 feature 读取统计与 frontend uncertainty loss 统计。

扩展配置：

- `configs/mono/tum/base_config.yaml` 当前已开启 uncertainty：
  - `enabled: True`
  - `extract_features_online: True`
  - `feature_extractor: "dinov2_reg_small_fine"`
  - `feature_output_root: "output"`
  - `apply_during_init: False`
  - `tracking_warmup_keyframes: 25`
  - `tracking_warmup_backend_syncs: 1`
- `configs/mono/tum/base_config.yaml` 同时补上了 `Dataset.single_thread: True`，使 Backend 与 Frontend 的单线程配置一致。
- 其他 RGB-D / Stereo / Replica base 配置保留 uncertainty 配置块，但默认 `enabled: False`。

## 4. 当前实测结果

这版代码在当前截图中的结果很好：

- ATE RMSE：`0.017598981007699743`
- color refinement 前：PSNR `18.031291275024415`，SSIM `0.6808279213309288`，LPIPS `0.38331055343151094`
- color refinement 后：PSNR `21.694127254486084`，SSIM `0.7236216893291252`，LPIPS `0.3538485154509544`
- frontend feature stats：`hit=593, missing=0, bad=0, total_checked=593`
- frontend loss stats：`tracking_used=38478, tracking_fallback=0, mapping_used=0, mapping_fallback=0`
- backend loss stats：`tracking_used=0, tracking_fallback=0, mapping_used=149331, mapping_fallback=2100`

这些统计说明：feature 读取完整，tracking uncertainty 已启用且没有 fallback；mapping 大量使用 uncertainty-aware loss，少量 fallback 主要来自初始化或缺少 features / apply 条件不满足的路径。

## 5. 仍需注意的点

- 当前 mono TUM 配置默认开启 uncertainty。如果要跑原始 MonoGS baseline，需要显式把 `Training.uncertainty.enabled` 改为 `False`。
- `utils/slam_backend.py` 读取 `Dataset.single_thread`，`utils/slam_frontend.py` 读取 `Training.single_thread`。mono TUM base 当前两个位置都为 `True`，这是稳定复现所需的配置。
- 即便单线程配置一致，源码仍存在随机初始化、随机采样和 CUDA 非完全确定性；严格逐次复现仍需要后续加入全局 seed 与 deterministic 设置。
- `color_refinement()` 仍使用原始 RGB + SSIM loss，没有接入 uncertainty。这是当前实现选择，不属于 tracking/mapping 主创新链路。
- 关闭 uncertainty 时仍会存在少量兼容字段和空状态传递，但不会进入 uncertainty loss 或 feature 读取路径。

## 6. 后续建议

如果这版结果要作为稳定实验版本，建议后续优先做三件事：

1. 加全局 seed 与子进程 seed，减少每次运行波动。
2. 统一 `single_thread` 的配置读取位置，避免 Dataset / Training 双字段不一致。
3. 增加一个短序列快速验证配置，用前 100 到 200 帧先筛掉明显跑崩的设置，再跑完整序列。
