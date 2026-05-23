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
