# WildGS-SLAM uncertainty 接入任务总结

## 1. 目标

将 WildGS-SLAM 风格的 uncertainty 模块接入 MonoGS：使用预提取的 DINO/FiT3D feature，经轻量 MLP 预测 uncertainty map，并在 tracking/mapping loss 中对动态或高不确定区域降权。

关键要求是：`Training.uncertainty.enabled: false` 时，MonoGS 必须回退到原始行为，不应创建 uncertainty MLP、不应读取 feature、不应进入 uncertainty-aware loss。

## 2. 最终改动

- 新增 `utils/dyn_uncertainty/`：
  - `uncertainty_model.py`：轻量 MLP uncertainty 网络。
  - `median_filter.py`：median pooling。
  - `mapping_utils.py`：SSIM components、uncertainty mapping loss helper、DINO regularization helper。

- 扩展 `utils/dataset.py`：
  - 支持读取 `.npy` feature。
  - feature 默认路径为 `Dataset.dataset_path/mono_priors/features/{frame_idx:05d}.npy`。
  - 增加损坏文件、shape、dtype、NaN/Inf 校验。
  - feature 异常时 warning 并返回 `None`，回退原 MonoGS loss。
  - feature 保存在 CPU，避免关键帧长期占用 GPU 显存。

- 扩展 `utils/camera_utils.py`：
  - `Camera` 增加 `features` 字段。
  - `Camera.init_from_dataset()` 兼容三元组和四元组 dataset 返回值。
  - `Camera.clean()` 清理 `features`。

- 扩展 `utils/slam_utils.py`：
  - 新增 uncertainty 推理、resize、weight 转换 helper。
  - tracking RGB loss 可乘 uncertainty weight。
  - RGB-D tracking depth 默认不加权，可通过配置开启。
  - mapping loss 支持 uncertainty-aware 路径。
  - DINO regularization 改为局部窗口 feature + 随机采样，并增加 `reg_max_samples` 上限。

- 扩展 `utils/slam_backend.py`：
  - 后端持有并优化 `uncer_network`。
  - mapping 时传入局部窗口 keyframes 供 DINO regularization 采样。
  - 同步 MLP state_dict 给前端。
  - 修复 `prune=True` 提前返回前 uncertainty 梯度残留问题。

- 扩展 `utils/slam_frontend.py`：
  - 前端接收后端同步的 MLP state_dict。
  - tracking 只 no-grad 使用 uncertainty MLP，不更新 MLP。

- 扩展 `slam.py`：
  - 当 `Training.uncertainty.enabled: true` 时创建 uncertainty MLP。
  - 关闭时不创建新对象。

- 扩展配置：
  - 在 `configs/**/base_config.yaml` 和 `configs/live/*.yaml` 中新增 `Training.uncertainty` 默认配置。
  - 默认 `enabled: false`。
  - 新增 `Dataset.feature_path`、`Dataset.feature_format`。
  - 新增 `reg_max_samples: 4096`。

## 3. 已验证

- 执行 `python -m py_compile` 检查以下文件，语法通过：
  - `slam.py`
  - `utils/dataset.py`
  - `utils/camera_utils.py`
  - `utils/slam_utils.py`
  - `utils/slam_backend.py`
  - `utils/slam_frontend.py`
  - `utils/dyn_uncertainty/*.py`

- 执行 `git diff --check`，未发现空白错误。

- 静态确认各 base/live 配置默认 `Training.uncertainty.enabled: false`。

- 设计上确认 feature 缺失、损坏、shape 错误、dtype 错误、NaN/Inf 时会返回 `None`，从而回退到原始 MonoGS loss。

## 4. 仍有风险

- 当前 shell 环境没有可用 `torch`，因此没有完成真实张量级单测或完整 SLAM 运行测试。

- uncertainty 开启后 mapping 会变慢，原因包括：
  - 每帧额外执行 MLP 推理。
  - mapping loss 中多了 SSIM-derived uncertainty loss。
  - DINO regularization 仍需构建采样后的 feature 相似度矩阵。

- DINO regularization 已加入 `reg_max_samples` 限制，但在高分辨率 feature 或较大窗口时仍可能带来显存压力，需要实测调参。

- 前后端通过 MLP state_dict 同步，存在额外通信开销。

- feature 文件必须提前准备，且 shape 必须为 `H_feat x W_feat x feature_dim`。

- 工作区存在一个与本任务无关的 `AGENTS.md` 改动，未在本任务中回退。

## 5. 后续继续做的起点

1. 激活 MonoGS 的真实运行环境，确保 `torch`、CUDA、diff-gaussian-rasterization、simple-knn 可用。

2. 先运行 `Training.uncertainty.enabled: false` 的 baseline，确认原 MonoGS 行为未被破坏。

3. 准备少量 feature 测试用例：
   - 缺失文件。
   - 损坏 `.npy`。
   - shape 错误。
   - feature_dim 错误。
   - 含 NaN/Inf。
   - 合法 `H x W x 384` feature。

4. 开启 uncertainty，在短序列上验证：
   - loss 是否为 finite。
   - mapping backward 后 MLP 是否有梯度。
   - tracking 是否不更新 MLP。
   - 显存是否稳定。

5. 根据实测调整：
   - `reg_stride`
   - `reg_max_samples`
   - `reg_mult`
   - `ssim_mult`
   - `apply_to_tracking_rgb`
   - `apply_to_mapping_depth`
