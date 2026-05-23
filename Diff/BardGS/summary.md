# BardGS Blur-aware Tracking 嵌入任务总结

## 1. 目标是什么

本任务目标是参考 BARD-GS 中“曝光时间内多虚拟相机渲染并平均”的思路，在 MonoGS 前端 tracking 阶段加入一个可开关控制的 blur-aware tracking 实验模块。

核心约束是：

- 不破坏 MonoGS 原有在线 SLAM 架构。
- 不修改无关代码，不重构后端、GUI、ATE 评估等逻辑。
- 所有新增创新点必须由 `Training.blur_aware_tracking.enabled` 总开关控制。
- 默认关闭时应尽量回到原生 MonoGS 的单次 render tracking 行为。

## 2. 最终改了什么

最终实现了一个默认关闭的 blur-aware tracking 最小版本：

- 在配置中新增 `Training.blur_aware_tracking`：
  - `enabled: False`
  - `num_virtual_views: 3`
  - `shutter_ratio: 1.0`
  - `motion_l2: 0.001`
  - `start_after_keyframes: 2`
- 新增 `utils/blur_tracking.py`：
  - 解析并校验 blur-aware tracking 配置。
  - 延迟创建当前帧的 `blur_rot_delta` / `blur_trans_delta` 参数。
  - 生成曝光窗口内的虚拟采样系数。
  - 对多个虚拟位姿分别 render RGB，并对 RGB 求平均。
  - depth、opacity、visibility 等仍使用中心位姿 render 结果。
  - 提供 blur motion L2 正则项。
- 修改 `utils/slam_frontend.py`：
  - 在 `FrontEnd.tracking()` 中接入 blur-aware tracking 分支。
  - 只有总开关打开且关键帧 warmup 满足时，才启用 blur 参数和多虚拟位姿 render。
  - 将 `blur_rot_delta` / `blur_trans_delta` 加入当前帧 tracking optimizer。
  - loss 中加入 `motion_l2 * (||blur_rot_delta||^2 + ||blur_trans_delta||^2)`。
  - tracking 结束或异常时通过 `finally` 清零 blur motion 参数。
- 修改 `utils/camera_utils.py`：
  - 给 `Camera` 增加 `blur_rot_delta` / `blur_trans_delta` 属性，但默认是 `None`，不再无条件创建 `nn.Parameter`。
- 修改 `utils/pose_utils.py`：
  - 增加 W2C 构造、W2C 写回、虚拟相机 delta 组合、blur motion reset 等辅助函数。
  - `update_pose()` 仍只写回中心位姿，不把 blur motion 保存为最终相机位姿。
- 修改 `gaussian_splatting/gaussian_renderer/__init__.py`：
  - 给 `render()` 增加显式可选参数 `pose_delta_override`。
  - 普通调用不传该参数时，仍使用原生 `cam_rot_delta` / `cam_trans_delta`。
  - blur-aware tracking 分支显式传入虚拟位姿 delta，避免隐式全局属性 hook 影响关闭开关后的行为。

涉及文件：

- `utils/blur_tracking.py`
- `utils/slam_frontend.py`
- `utils/camera_utils.py`
- `utils/pose_utils.py`
- `gaussian_splatting/gaussian_renderer/__init__.py`
- `configs/mono/tum/base_config.yaml`
- `configs/rgbd/tum/base_config.yaml`
- `configs/rgbd/replica/base_config.yaml`
- `configs/stereo/euroc/base_config.yaml`
- `configs/live/realsense.yaml`
- `configs/live/realsense_rgbd.yaml`

## 3. 已验证什么

已完成的轻量验证：

- 对修改过的 Python 文件执行了静态编译检查：

```powershell
python -m py_compile utils\camera_utils.py utils\pose_utils.py utils\blur_tracking.py utils\slam_frontend.py gaussian_splatting\gaussian_renderer\__init__.py
```

结果：通过。

- 检查确认代码中不再存在 `_render_rot_delta` / `_render_trans_delta` 这种隐式 renderer hook。
- 检查确认普通 `render()` 调用点不需要改动；新增 `pose_delta_override` 是可选参数。
- 检查确认 `blur_aware_tracking.enabled` 默认配置为 `False`。
- 检查确认 blur motion 参数只有在 blur-aware tracking 启用且 warmup 满足后才延迟创建并加入 optimizer。

未完成的验证：

- 没有跑完整 MonoGS 数据集实验。
- 没有做 ATE RMSE、tracking loss、每帧耗时、reset/lost 次数等实验指标验证。
- 当前 shell 可用 Python 环境缺少项目运行所需依赖，无法在该环境中执行完整 SLAM 流程。

## 4. 还有什么风险

主要风险如下：

- 性能风险：`num_virtual_views=3` 时，tracking 阶段渲染开销大约增加到原来的 3 倍；实时性可能下降。
- 数值/效果风险：当前虚拟位姿通过 MonoGS rasterizer 的 `theta/rho` delta 接口表达，用于保留相机 delta 的梯度路径；它是对计划中严格 `Exp(s * xi_blur) * P_mid` 的近似实现。
- 清晰序列退化风险：开启后新增 blur motion 自由度，可能吸收真实位姿误差；需要依赖 `motion_l2` 和 `start_after_keyframes` warmup 控制。
- 配置风险：`enabled` 已严格要求 bool 类型；如果实验配置中写成字符串 `"False"` 会直接报错，需要改成 YAML bool `False`。
- 实验风险：目前只完成代码级轻量验证，仍需要在真实 CUDA/PyTorch 环境中跑 monocular/RGB-D 序列，确认开启和关闭两种路径的行为。
- 语义风险：关闭总开关后，tracking 逻辑不进入 blur-aware 分支；但 `Camera` 仍多了两个默认 `None` 属性，属于对象结构层面的轻微差异，不应影响原生优化路径。

## 5. 后续第一阶段增强与实测记录

在最小版 blur-aware tracking 能跑通后，又针对“blur 参数是否真正有效学习”和实际运行中暴露的问题做了第一阶段增强。

### 5.1 新增的第一阶段增强

新增配置项均位于 `Training.blur_aware_tracking` 下，只有在 `enabled=True` 且 keyframe warmup 满足后才会生效：

- `init_motion_from_previous: True`：使用上一段相机运动初始化当前帧的 `blur_rot_delta` / `blur_trans_delta`，避免对称采样加零初始化导致 blur motion 一阶梯度容易抵消。
- `init_motion_scale: 1.0`：控制常速度 blur motion 初值缩放系数。
- `min_tracking_iterations: 15`：blur-aware tracking 启用后，即使中心位姿增量很小，也至少优化若干轮，避免 blur 参数还没来得及更新就提前退出。
- `debug: False`：控制逐帧打印 blur motion norm、迭代次数和 loss。正式跑指标时建议关闭，减少日志和 IO 干扰。

代码层面对应修改：

- `utils/blur_tracking.py`
  - 新增 `initialize_blur_motion_from_previous_frame()`，从 `prev_prev -> prev` 的相对运动估计当前帧曝光期初始运动。
  - 新增 `_so3_log()`，用于从相对旋转矩阵得到旋转向量初值。
  - 新增 `should_stop_blur_tracking()`，实现 blur 分支的最小迭代约束。
  - 新增 `blur_motion_norms()`，用于记录和汇总 blur motion 大小。
- `utils/slam_frontend.py`
  - 在 `use_blur_tracking=True` 后才初始化 blur motion、加入 blur 参数优化、应用最小迭代停止逻辑。
  - 新增 `record_blur_tracking_debug()` 和 `log_blur_tracking_debug_stats()`，输出 `frames / initialized / avg_rot_norm / avg_trans_norm / max_rot_norm / max_trans_norm`。
  - 逐帧 debug 只在 `Training.blur_aware_tracking.debug=True` 时打印；最终 summary 在实际使用过 blur 分支时打印。
- `utils/pose_utils.py`
  - 修复 `update_pose()` 中 `SE3_exp(tau)` 与相机 `R/T` dtype 不一致导致的 `expected scalar type Float but found Double`。当前做法是把 `camera_w2c(camera)` 对齐到 `tau.device/tau.dtype`。
- `gaussian_splatting/gaussian_renderer/__init__.py`
  - 运行环境必须同步到包含 `pose_delta_override` 的版本；否则 blur 分支调用 `render(..., pose_delta_override=...)` 会报 `unexpected keyword argument 'pose_delta_override'`。

补充了轻量静态测试：

- `tests/test_blur_tracking_static.py`
  - 检查第一阶段配置 key 是否存在。
  - 检查新增 helper 是否存在。
  - 检查 frontend 是否接入 blur 初始化、最小迭代和 debug 统计。
  - 检查 `update_pose()` 是否做了 W2C dtype/device 对齐。

已执行过的轻量验证命令：

```powershell
python -m unittest tests.test_blur_tracking_static
python -m py_compile utils\pose_utils.py utils\blur_tracking.py utils\slam_frontend.py tests\test_blur_tracking_static.py
```

### 5.2 运行中遇到的问题与修复

1. `RuntimeError: expected scalar type Float but found Double`

发生位置：`utils/pose_utils.py:update_pose()`。

原因：数据集读入的相机 `R/T` 可能是 `Double`，而优化增量 `cam_rot_delta/cam_trans_delta` 是 `Float`，导致 `SE3_exp(tau) @ T_w2c` dtype 不一致。

修复：`T_w2c = camera_w2c(camera).to(device=tau.device, dtype=tau.dtype)`。

2. `TypeError: render() got an unexpected keyword argument 'pose_delta_override'`

原因：Linux 实验环境只同步了 frontend/blur tracking 修改，但没有同步 renderer 接口修改。

修复：同步 `gaussian_splatting/gaussian_renderer/__init__.py`，确保 `render()` 签名包含 `pose_delta_override=None`，并在 rasterizer 调用前根据该参数选择 `theta/rho`。

### 5.3 fr1_desk 初步实验观察

实验配置示例：

```yaml
inherit_from: "configs/mono/tum/fr3_office.yaml"

Training:
  blur_aware_tracking:
    enabled: True
    num_virtual_views: 3
    shutter_ratio: 0.5
    motion_l2: 0.01
    start_after_keyframes: 5
    init_motion_from_previous: True
    init_motion_scale: 1.0
    min_tracking_iterations: 15
    debug: True
```

该配置下完整跑通，日志中可见：

- `[BlurTrack] enabled`
- `init_motion=True`
- `rot_norm/trans_norm` 非零，说明常速度初始化路径确实生效。

对 `init_motion_scale` 做了初步消融：

| 配置 | Final ATE RMSE | 观察 |
| --- | ---: | --- |
| baseline | `0.021063` | 当前参考基线 |
| `init_motion_scale=0.5` | `0.029282` | 明显退化，`avg_trans_norm` 很小，blur motion 贡献不足但仍引入额外自由度 |
| `init_motion_scale=1.0` | `0.020552` | 略优于 baseline，约 2.4% 改善，但幅度较小，需要多次重复确认 |
| `init_motion_scale=1.5` | `0.025191` | 退化，`avg_trans_norm` 偏大，可能吸收真实中心位姿误差 |

初步判断：

- `init_motion_scale=1.0` 是当前三组中最合理的默认实验值。
- `0.5` 过弱，`1.5` 偏强。
- `fr1_desk` 不一定是强模糊序列，blur-aware tracking 对所有 warmup 后帧启用，可能会在清晰帧上引入不必要自由度。

### 5.4 当前建议配置与下一步

正式统计时建议关闭逐帧 debug：

```yaml
Training:
  blur_aware_tracking:
    enabled: True
    num_virtual_views: 3
    shutter_ratio: 0.5
    motion_l2: 0.01
    start_after_keyframes: 5
    init_motion_from_previous: True
    init_motion_scale: 1.0
    min_tracking_iterations: 15
    debug: False
```

后续优先消融：

- 固定 `init_motion_scale=1.0`，测试 `motion_l2=0.02` 和 `motion_l2=0.05`。
- 如果仍不能稳定优于 baseline，不建议继续盲调 scale，应进入第二阶段：加入 blur gating，只在高 residual、大运动或低清晰度帧启用 blur-aware tracking。
