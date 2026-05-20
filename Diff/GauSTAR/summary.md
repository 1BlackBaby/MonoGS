# GauSTAR Stage1 修 bug 与实验总结

## 1. 当前目标

本轮工作的目标是在 MonoGS 中以可开关、可消融的方式接入 GauSTAR Stage1 的三个创新点，并修正早期接入后导致指标变差的问题：

- Metric3D depth：作为单目 metric depth prior，用于在线预测/缓存、depth prior 构建、后端建图侧补点辅助。
- Optical flow pose initialization：用 RAFT 双向光流和上一帧深度建立 3D-2D 对应，通过 PnP 给前端 tracking 提供候选初值。
- Rendered-depth consistency filtering：用 Metric3D depth 与当前 rendered depth 做尺度对齐和一致性筛选，辅助过滤不可靠区域。

所有逻辑必须受 `gaustar_stage1.enabled` 总开关控制。关闭总开关时，应保持 MonoGS baseline 行为，不读取 prior、不预计算 flow、不替换 tracking 初值、不改变 keyframe depth 初始化、不改变 tracking loss。

## 2. 早期问题

最初嵌入后效果差，主要不是因为 Metric3D 或 RAFT 完全不能工作，而是接入策略过于激进：

- Metric3D depth 在早期/初始化阶段过早影响 keyframe depth，容易把局部尺度误差或边缘深度误差注入地图。
- flow pose init 只要 PnP 几何上成功就容易覆盖 previous-frame pose，但 MonoGS 的 tracking 最终优化目标是 photometric/render loss，PnP 几何重投影好不等于当前 Gaussian map 下的 tracking 初值更好。
- rendered-depth consistency filter 如果直接 hard mask tracking loss，会减少可用于纠正位姿的像素，特别是在早期地图还不稳定时容易让 tracking 变差。
- MonoGS 原工程没有内置 RAFT flow 计算，只有读取 flow prior 的逻辑，因此缺 flow 时 GauSTAR tracking init 无法验证。

## 3. 已完成的核心修复

### 3.1 Metric3D 在线预测与缓存

参考 WildGS-SLAM 的 Metric3D 接入方式，补齐了 MonoGS 中缺失的在线 depth 预测路径：

- 通过 `torch.hub.load("yvanyin/metric3d", model_name, pretrain=True)` 加载 Metric3D。
- 输入 resize 到配置的 canonical image size，做 ImageNet normalize 和 padding。
- 推理后去 padding、resize 回原始分辨率。
- 按 WildGS-SLAM 逻辑乘以 `fx / 1000.0` 做 canonical-to-real scale。
- 输出 depth clamp 到配置最大深度，并缓存到 `mono_priors/depths/{idx:05d}.npy` 或配置指定目录。

修复后，如果已有 Metric3D depth 文件会优先读取；缺失时才在线预测并写入缓存。预测行为由：

```yaml
gaustar_stage1:
  enabled: True
  use_metric3d_depth: True
  cache_metric3d_depth: True
```

控制。

### 3.2 RAFT flow 预计算

参考 GauSTAR 的 `data_process/RAFT/demo_GauSTAR.py`，补齐了 MonoGS 运行前/运行开始时的双向 flow 预计算：

- forward flow：`{idx:05d}_f.npz`，表示 frame `idx -> idx+1`。
- backward flow：`{idx:05d}_b.npz`，表示 frame `idx+1 -> idx`。
- `.npz` 内保存 key 为 `flow` 的 `H x W x 2` 光流，通道为 `(dx, dy)`。
- 输出目录默认是 `dataset_path/mono_priors/flow_bi`，也可由 `Dataset.flow_path` 指定。
- 已有完整缓存时跳过预计算。

配置入口：

```yaml
gaustar_stage1:
  precompute_flow: True
  raft_root: "/home/xushengchao/MonoGS/RAFT"
  raft_checkpoint: "/home/xushengchao/MonoGS/RAFT/models/raft-things.pth"
  flow_iters: 20
  flow_skip_existing: True
```

修复过的一个实际 bug：

```text
argument of type 'types.SimpleNamespace' is not iterable
```

原因是 RAFT 源码里会判断类似 `'dropout' not in self.args`，普通 `SimpleNamespace` 不支持 `in`。修复方式是给 RAFT args 增加 `__contains__` 兼容。

### 3.3 Flow pose init 安全门控

flow pose init 已经不是简单“PnP 成功就覆盖位姿”，而是作为候选 tracking 初值，必须通过多级检查：

- flow/depth prior 必须存在。
- 双向 flow consistency 必须通过。
- depth edge / depth forward-backward consistency 必须通过。
- PnP RANSAC 内点数和重投影误差必须达标。
- PnP 相对 previous pose 的旋转和平移不能超过阈值。
- 如果 `flow_pose_compare_init_loss=True`，还要比较 PnP 初值和 previous-frame 初值的 tracking photometric loss。

实际日志中出现：

```text
[GauSTAR Stage1] flow pose initialization enabled (inliers=4911, reproj=0.714)
[GauSTAR Stage1] flow pose initialization rejected (pnp_loss=0.070468, prev_loss=0.066800); using previous-frame pose.
```

这说明 flow pose init 已经接入并且 PnP 几何计算成功，但由于 PnP 初值在当前 Gaussian map 下的 photometric loss 更差，所以被安全门控拒绝。这个现象不是 flow 文件缺失，也不是 RAFT 没跑，而是当前序列上 previous-frame pose 更适合作为 MonoGS tracking 初值。

当前推荐保留：

```yaml
gaustar_stage1:
  use_flow_pose_init: True
  flow_pose_after_init: True
  flow_pose_compare_init_loss: True
  flow_pose_loss_improvement: 0.0
  max_flow_pose_translation_ratio: 0.5
  max_flow_pose_rotation_deg: 20.0
```

### 3.4 Metric3D keyframe depth 保护

为了避免 Metric3D depth 过早污染初始化地图，keyframe depth 的使用被改成保守策略：

- 默认 `apply_filter_to_keyframe_depth: False`。
- 即使后续打开 keyframe depth filter，也要求 `require_rendered_depth_for_keyframe_depth: True`，即必须有 rendered depth / opacity 支撑一致性检查后再使用。
- 如果尺度对齐或一致性检查失败，回退原 MonoGS 行为。

因此当前 best 配置中，Metric3D 不是强行替换 keyframe depth，而是作为 prior 和建图侧辅助信息参与。

### 3.5 Rendered-depth filter 作用范围收敛

实验表明，rendered-depth consistency filter 直接作用到 tracking loss 会变差。因此当前推荐的有效用法是：

- 开启 `use_rendered_depth_filter: True`。
- 开启 `apply_filter_to_render_guided_densification: True`。
- 关闭 `apply_filter_to_tracking: False`。
- 关闭 `apply_filter_to_keyframe_depth: False`。

这样 filter 主要作用在建图侧 densification / 补点筛选，而不是直接干扰前端 tracking loss 或关键帧深度初始化。

## 4. 三组实验结论

### 4.1 第一组：baseline

配置：

```yaml
gaustar_stage1:
  enabled: False
```

完整结果：

```text
ATE RMSE: 0.026723635989911457
mean PSNR: 21.3454039428337
mean SSIM: 0.7171386917752605
mean LPIPS: 0.3628916009779895
```

### 4.2 第二组：当前 best config

核心配置：

```yaml
gaustar_stage1:
  enabled: True
  use_metric3d_depth: True
  use_flow_pose_init: True
  use_rendered_depth_filter: True

  apply_filter_to_tracking: False
  apply_filter_to_keyframe_depth: False
  apply_filter_to_render_guided_densification: True

  require_rendered_depth_for_keyframe_depth: True
  flow_pose_after_init: True
  flow_pose_compare_init_loss: True
  flow_pose_loss_improvement: 0.0
  max_flow_pose_translation_ratio: 0.5
  max_flow_pose_rotation_deg: 20.0
  use_depth_scale_for_flow_pose: True

  precompute_flow: True
  raft_root: "/home/xushengchao/MonoGS/RAFT"
  raft_checkpoint: "/home/xushengchao/MonoGS/RAFT/models/raft-things.pth"
  flow_iters: 20
  flow_skip_existing: True
```

完整结果：

```text
ATE RMSE: 0.01848144646487302
mean PSNR: 21.8647140653859
mean SSIM: 0.7303702727608059
mean LPIPS: 0.3330022098253603
```

相对 baseline：

```text
ATE:   0.02672 -> 0.01848，约降低 30.8%
PSNR:  21.345 -> 21.865，提升约 0.52 dB
SSIM:  0.7171 -> 0.7304，提升约 0.013
LPIPS: 0.3629 -> 0.3330，降低约 0.030
```

第二组嵌入了三项创新点，但生效层级不同：

- Metric3D depth：已启用，参与 prior 构建和建图侧辅助，但没有直接替换 keyframe depth。
- Optical flow pose initialization：已启用并尝试，RAFT+PnP 计算成功，但本序列上被 photometric loss gate 拒绝，没有覆盖 tracking 初值。
- Rendered-depth consistency filtering：已启用，主要作用于 render-guided densification，是当前最可能的主要正收益来源。

### 4.3 第三组：打开 tracking soft filter

在第二组基础上打开：

```yaml
gaustar_stage1:
  apply_filter_to_tracking: True
  apply_filter_after_init: True
  tracking_filter_soft_weight: 0.3
```

完整结果：

```text
ATE RMSE: 0.02421929620374918
mean PSNR: 21.609293073018392
mean SSIM: 0.7229199063777924
mean LPIPS: 0.34789724071820577
```

第三组仍优于 baseline，但明显劣于第二组。因此当前不建议让 rendered-depth filter 参与 tracking loss，即使是 soft weight 也会干扰位姿优化。

## 5. 当前结论

当前最优配置是第二组：

```yaml
gaustar_stage1:
  enabled: True
  use_metric3d_depth: True
  use_flow_pose_init: True
  use_rendered_depth_filter: True

  apply_filter_to_tracking: False
  apply_filter_to_keyframe_depth: False
  apply_filter_to_render_guided_densification: True

  flow_pose_after_init: True
  flow_pose_compare_init_loss: True
  precompute_flow: True
```

这组配置的关键不是“所有模块强行作用到 tracking”，而是：

- Metric3D depth 以 prior 形式温和接入。
- RAFT flow pose init 作为候选初值接入，但必须通过 photometric loss gate。
- rendered-depth consistency filter 主要作用于建图侧 densification，而不是直接裁剪 tracking loss。

因此可以表述为：

```text
三项 GauSTAR Stage1 创新点都已可开关地嵌入 MonoGS。
当前 fr1_desk 上真正产生正收益的是 Metric3D prior + rendered-depth consistency guided densification。
flow pose init 已接入，但在该序列上被安全门控拒绝；换到运动更明显、previous pose 初值更弱的序列时，有可能被接受并发挥作用。
```

## 6. 后续建议

短期不建议再打开 `apply_filter_to_tracking`，因为第三组结果已经说明 tracking filter 会降低指标。

如果还要继续提升，下一步更合理的消融是只测试 keyframe depth filter，且仍关闭 tracking filter：

```yaml
gaustar_stage1:
  apply_filter_to_tracking: False
  apply_filter_to_keyframe_depth: True
  require_rendered_depth_for_keyframe_depth: True
  apply_filter_to_render_guided_densification: True
```

这组实验的目的不是让 Metric3D 全面替代 MonoGS 深度初始化，而是只在 rendered-depth 一致区域使用它，验证是否能进一步改善 keyframe 初始化和后端建图质量。

另一个建议是换不同序列验证 flow pose init。当前 `fr1_desk` 中 previous-frame pose 初值已经很强，PnP 候选虽然几何重投影好，但 photometric loss 不占优。运动更明显、转角更大、previous pose 初值更弱的序列更可能真正接受 flow pose init。

## 7. 已做的静态验证

最近一次代码静态验证通过：

```powershell
python -m py_compile utils\mono_priors\gaustar_stage1.py utils\slam_utils.py utils\slam_frontend.py slam.py utils\mono_priors\flow_precompute.py utils\dataset.py
```

当前仍需注意：

- 具体配置文件可能处于实验状态，提交前需要确认 `configs/mono/tum/base_config.yaml` 是否应保留当前开关。
- `Diff/GauSTAR/summary.md` 是本总结文件，用来记录当前修 bug 后的真实结论；旧的“尚未跑完整序列实验”等表述已被替换。
