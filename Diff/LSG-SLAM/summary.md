# LSG-SLAM 嵌入 MonoGS 阶段总结

## 阶段 1：多模态位姿初值

### 目标

在 MonoGS 前端 `tracking()` 优化 `cam_rot_delta` / `cam_trans_delta` 之前，为当前帧提供一个比单纯复制上一帧位姿更稳的初值。实现策略为：

1. 使用最近关键帧作为参考帧。
2. 使用 LSG-SLAM 中的 SuperPoint + LightGlue 提取和匹配局部特征。
3. 使用参考关键帧深度将匹配点反投影为 3D 世界点。
4. 使用当前帧 2D 匹配点执行 PnP/RANSAC，得到当前帧 W2C 初值。
5. PnP 失败或质量检查不通过时，不中断流程，继续回退到已有 GauSTAR flow pose init 或原生 MonoGS previous-pose 初始化。

### 总开关与默认行为

新增能力受统一开关控制：

```yaml
Training:
  lsg_slam:
    enabled: false
  lsg_pose_init:
    enabled: false
```

只有 `lsg_slam.enabled=true` 且 `lsg_pose_init.enabled=true` 时，阶段 1 才会生效。默认全部关闭，因此默认运行路径保持原生 MonoGS 行为。

### 修改文件

- `utils/slam_frontend.py`
  - 在 `initialize_tracking_pose()` 中接入 LSG PnP 位姿初值候选。
  - 新增最近关键帧选择、参考深度读取、匹配调用、PnP 调用、日志输出和 fallback 流程。
  - PnP 成功后只调用 `viewpoint.update_RT(R, T)` 写入当前帧初值，不直接修改 `cam_rot_delta` / `cam_trans_delta`。
  - 关键帧创建时保存 `lsg_keyframe_depth`，供后续帧 PnP 反投影使用。

- `utils/lsg_pose_init.py`
  - 新增阶段 1 的几何核心逻辑。
  - 包含配置合并、总开关判断、匹配点反投影、2D-3D PnP/RANSAC、重投影误差检查、inlier 检查和位姿运动幅度检查。

- `utils/lsg_features.py`
  - 新增 SuperPoint/LightGlue 懒加载封装。
  - 支持从当前仓库 `sp_lg/`、`LSG_SLAM_ROOT` 或 `feature_model_root` 查找模型文件。
  - 缓存每帧局部特征，避免同一关键帧重复提取。
  - LightGlue 匹配后可选使用 EssentialMat/RANSAC 做第一层几何过滤。

- `utils/camera_utils.py`
  - `Camera.clean()` 中清理新增的 `lsg_local_features`，避免非关键帧特征缓存造成长序列内存增长。

- `configs/live/realsense.yaml`
- `configs/live/realsense_rgbd.yaml`
- `configs/mono/tum/base_config.yaml`
- `configs/rgbd/replica/base_config.yaml`
- `configs/rgbd/tum/base_config.yaml`
- `configs/stereo/euroc/base_config.yaml`
  - 增加默认关闭的 `lsg_slam` 和 `lsg_pose_init` 配置段。

- `sp_lg/`
  - 从 LSG-SLAM 复制 SuperPoint / LightGlue 相关源码。
  - 修改 `sp_lg/superpoint.py` 中硬编码的 `cuda:0`，改为跟随输入图像所在 device。

- `tests/test_lsg_pose_init.py`
  - 新增阶段 1 的基础单元测试。

### 质量检查与拒绝逻辑

PnP 结果必须通过以下检查，否则回退：

- 匹配数量达到 `min_matches`。
- 2D-3D 点数量达到 `min_pnp_points`。
- PnP inliers 达到 `min_pnp_inliers`。
- inlier ratio 达到 `min_inlier_ratio`。
- 平均重投影误差不超过 `max_reproj_error`。
- 相对上一帧旋转不超过 `max_rotation_deg`。
- 相对上一帧平移不超过 `max_translation_ratio * median_depth`。
- 如果 `compare_init_loss=true`，PnP 初值的 tracking init loss 必须优于 fallback 初值。

每帧会输出：

```text
[LSG PoseInit] frame=..., ref=..., matches=..., inliers=..., accepted=..., reason=...
```

### 验收结果

已执行：

```powershell
D:\Postgraduate\Research\Python310\python.exe -m unittest tests.test_lsg_pose_init
D:\Postgraduate\Research\Python310\python.exe -m compileall -f utils\lsg_pose_init.py utils\lsg_features.py utils\slam_frontend.py utils\camera_utils.py sp_lg\superpoint.py
git diff --check
```

结果：

- 单元测试通过：3 个测试中 2 个通过，1 个因当前 Python 环境缺少 `numpy/cv2` 被跳过。
- 编译检查通过。
- `git diff --check` 通过。

### 当前限制与风险

- 当前可用 Python 环境缺少完整 SLAM 依赖，因此还未跑真实序列验证 ATE、tracking lost 次数和 FPS。
- `sp_lg/*.pth` 权重文件已复制到本机，但由于 `.gitignore` 忽略 `*.pth`，后续如果需要提交权重，需要显式 `git add -f sp_lg/*.pth`，或配置 `Training.lsg_pose_init.feature_model_root` 指向 LSG-SLAM 根目录。
- PnP 初值错误会直接影响 tracking 稳定性，虽然已加多层 reject，但阈值仍需在 RGB-D、stereo 和 monocular 序列上分别调参。
- monocular 场景依赖关键帧保存的初始化深度或 metric depth，尺度不稳时 PnP 平移可能不可靠，应优先在 RGB-D/stereo 或有 metric depth prior 的配置上验收。

## 阶段 2：Feature Warping Tracking Loss

### 目标

在 MonoGS 前端 tracking 优化过程中，除了原有 RGB/depth photometric loss 之外，引入稀疏特征几何约束。核心思想来自 LSG-SLAM 的 `feature_matching.py::get_loss_from_match()`：利用当前帧与参考关键帧的 SuperPoint/LightGlue 匹配点，通过参考帧深度反投影到 3D，再投影到当前帧，计算重投影 keypoint 的 SmoothL1 loss。

当前嵌入 MonoGS 的版本只实现最小可验收版本：

1. 只使用 keypoint 2D reprojection SmoothL1。
2. 不引入 descriptor/feature patch SmoothL1。
3. loss 必须对 `cam_rot_delta` / `cam_trans_delta` 可导。
4. PnP 初值关闭时，warping loss 可以独立打开。
5. 所有能力必须受 `lsg_slam` 和 `lsg_feature_warping` 开关控制。

### 总开关与默认行为

新增配置：

```yaml
Training:
  lsg_slam:
    enabled: true
  lsg_feature_warping:
    enabled: false
    weight_warp: 0.05
    min_matches_for_loss: 20
    robust_beta: 0.1
    normalize_pixels: true
    warp_after_init: true
    require_pose_init_accept: true
```

实际生效条件为：

```text
lsg_slam.enabled == true
and lsg_feature_warping.enabled == true
```

其中：

- `warp_after_init=true`：SLAM 初始化完成前不启用 warping loss，避免早期初始化阶段被稀疏几何约束干扰。
- `require_pose_init_accept=true`：当 `lsg_pose_init.enabled=true` 时，只复用 PnP accepted 后的匹配点；PnP rejected 或 init loss rejected 的匹配不会进入 warping loss。
- `normalize_pixels=true`：重投影误差按图像宽高归一化，避免像素级 SmoothL1 量级过大，压过原 MonoGS photometric loss。
- `weight_warp=0.05`：当前实验中比 `1.0` 稳定，作为保守默认值。

如果做 Warp only 消融，应使用：

```yaml
Training:
  lsg_slam:
    enabled: true
  lsg_pose_init:
    enabled: false
  lsg_feature_warping:
    enabled: true
```

### 修改文件

- `utils/lsg_warp_loss.py`
  - 新增阶段 2 的核心 loss 实现。
  - `build_lsg_warp_match_data()` 保存当前帧/参考帧匹配点、参考深度、参考帧位姿和内参。
  - `current_w2c_with_pose_delta()` 使用 `SE3_exp(cam_trans_delta, cam_rot_delta)` 构造临时可导 W2C，不调用 `update_pose()`。
  - `compute_lsg_warp_loss()` 完成参考点反投影、世界坐标转换、当前帧投影、有效点过滤和 SmoothL1 计算。
  - `should_prepare_lsg_warp_match_data()` 统一判断 warping 是否允许准备匹配数据。

- `utils/slam_frontend.py`
  - tracking 开始前调用 `prepare_lsg_warp_match_data()` 准备 `viewpoint.lsg_match_data`。
  - 如果 PnP accepted，则复用 PnP 阶段已有 LightGlue 匹配，避免重复提取/匹配。
  - 如果 PnP disabled，则 warping loss 可独立执行匹配，支持 Warp only 消融。
  - 每帧初始化 tracking pose 前清空旧的 `lsg_match_data` / `lsg_warp_stats` / `lsg_pose_init_accepted`，避免残留状态污染当前帧。

- `utils/slam_utils.py`
  - `get_loss_tracking()` 保持原函数签名。
  - 在原 RGB/RGB-D tracking loss 之后，根据 `get_lsg_feature_warping_config()` 判断是否叠加 `warp_loss`。

- `utils/camera_utils.py`
  - `Camera.clean()` 清理 `lsg_match_data`、`lsg_warp_stats` 和 `lsg_pose_init_accepted`。

- `tests/test_lsg_warp_loss.py`
  - 测试 warping loss 可反传到 pose delta。
  - 测试开关关闭时返回 0 loss。
  - 测试默认 `warp_after_init=true` 会阻止初始化前启用 warping。
  - 测试 PnP 开启时默认要求 pose init accepted。

### 已修复的问题

最初版本存在三个关键问题：

1. **初始化前误启用 warping loss**

   日志中曾出现：

   ```text
   [LSG WarpLoss] frame=1, used_matches=307, reason=ok
   MonoGS: Keyframes lacks sufficient overlap to initialize the map, resetting.
   ```

   这说明 warping loss 在地图初始化前已经影响 tracking，容易导致初始化关键帧 overlap 失败。修复方式为新增 `warp_after_init=true`，初始化前直接跳过。

2. **PnP rejected 的匹配仍被 warping 使用**

   旧逻辑在 PnP 质量检查之前就保存 `lsg_match_data`。这会导致 PnP 被 `init loss rejected` 后，错误或不稳定的匹配仍进入 warping loss。修复方式为：只有 PnP accepted 后才保存 match data；如果 PnP disabled，才允许 Warp only 自行准备匹配。

3. **像素坐标 loss 量级过大**

   旧配置 `normalize_pixels=false` 且 `weight_warp=1.0`，像素级 SmoothL1 很容易达到 `1~5` 量级，直接压过 MonoGS 原本的图像均值 L1 loss。修复方式为默认开启像素归一化，并将默认权重降为 `0.05`。

### 实验结果对比

当前对 `fr1_desk` 序列的实验结果如下：

| 实验组 | 配置说明 | ATE/RMSE | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| A | LSG-PnP + Metric3D depth | 2.22 | 21.70 | 0.73 | 0.33 |
| Warp only | 只开 feature warping tracking loss | **1.88** | 21.38 | 0.72 | 0.34 |
| B | A + Warp, `weight_warp=0.01`, `normalize_pixels=true` | 1.92 | 21.92 | 0.73 | **0.32** |
| C | A + Warp, `weight_warp=0.05`, `normalize_pixels=true` | 2.06 | **22.00** | **0.74** | **0.32** |
| D | A + Warp, `weight_warp=0.1`, `normalize_pixels=true` | 2.10 | 21.98 | 0.73 | 0.33 |

结论：

- Warp only 的 ATE 最低，说明 feature warping loss 对 tracking 有独立收益。
- `A + Warp 0.05` 的 PSNR/SSIM/LPIPS 综合最好，说明 LSG-PnP + Metric3D 与 warping loss 联合后对建图/渲染质量更有利。
- `weight_warp=0.1` 相比 `0.05` 已出现轻微退化，继续增大权重没有必要。
- 如果论文主指标强调 ATE，可重点报告 `weight_warp=0.01` 或 Warp only；如果强调 tracking + rendering 综合效果，建议使用 `A + Warp 0.05` 作为主方法。

### 为什么 Warp only 的 ATE 优于 LSG-PnP + Metric3D + Warp

从结果看，Warp only 的 ATE 为 `1.88`，优于联合方案的 `1.92 / 2.06 / 2.10`。主要原因不是 PnP 无效，而是二者对 tracking 的作用方式不同：

1. **PnP 是硬初始化，Warp 是软约束**

   LSG-PnP accepted 后会直接调用：

   ```python
   viewpoint.update_RT(candidate_R, candidate_T)
   ```

   这会把当前帧初值硬切到 PnP 解。若 Metric3D 深度或匹配点存在局部误差，PnP 解会给当前帧引入一次性 pose 偏移。Warp only 不修改初值，而是在原 MonoGS tracking 过程中以 loss 形式提供软几何约束，轨迹连续性更强。

2. **Metric3D 深度误差会直接影响 PnP 平移**

   PnP 的 3D 点来自参考帧深度。Metric3D 虽然提供了 monocular metric prior，但仍可能存在边缘误差、局部尺度误差和渲染深度对齐误差。PnP 对深度非常敏感，深度误差会直接反映到平移估计中。Warp loss 则是在优化过程中逐步约束重投影误差，对局部深度误差更温和。

3. **PnP accepted 不等于 tracking 最优**

   PnP accepted 只说明满足 inlier、inlier ratio、reprojection error 和运动幅度阈值，不代表该初值一定优于上一帧 pose 或 photometric optimum。实验日志中存在 accepted 但 reprojection error 较高的帧，这类帧可能对后续 tracking 产生轻微扰动。

4. **联合方案中的 warping 生效帧可能少于 Warp only**

   修复后默认：

   ```yaml
   require_pose_init_accept: true
   ```

   因此在 `LSG-PnP + Metric3D + Warp` 中，只有 PnP accepted 的帧会使用 warping loss；PnP rejected 或 init loss rejected 的帧不会使用该帧匹配。Warp only 没有 PnP 门控，初始化完成后只要匹配和参考深度有效，warping loss 就可以持续生效。这可能让 Warp only 的 ATE 更低。

5. **Tracking 与 rendering 的最优点不完全一致**

   Warp only 更偏向短期位姿跟踪，因此 ATE 最低；但其 PSNR/SSIM/LPIPS 不如联合方案。联合方案虽然 ATE 略高，但关键帧初值、Metric3D 深度和建图尺度更稳定，最终渲染质量更好。

### 后续建议

1. 统计每组实验中 `warp used_matches > 0` 的帧比例，确认联合方案是否因 `require_pose_init_accept=true` 导致 warping 覆盖率低。
2. 统计 PnP accepted / rejected / init loss rejected 的帧数，分析 PnP 门控对 warp 生效频率的影响。
3. 额外跑一组：

   ```yaml
   lsg_feature_warping:
     enabled: true
     weight_warp: 0.01
     normalize_pixels: true
     warp_after_init: true
     require_pose_init_accept: false
   ```

   如果该组 ATE 接近 Warp only，同时保持较好的 PSNR/SSIM/LPIPS，则说明主要瓶颈是 PnP accepted 门控降低了 warping 覆盖率。
4. 暂不建议引入 descriptor/feature patch loss。当前 keypoint reprojection loss 已经有效，下一步应先完成生效帧比例、loss 量级和 PnP accepted 质量的统计。
