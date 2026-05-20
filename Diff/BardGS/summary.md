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
