# MonoGS 嵌入相机运动模糊感知跟踪核心计划

## 1. 目标

把 BARD-GS 的“曝光时间内多虚拟相机渲染并平均”的思想，轻量嵌入 MonoGS 前端 tracking。

当前 MonoGS 每帧只优化一个相机位姿，并用单次渲染图和输入图计算 tracking loss。改进后，对疑似模糊帧在曝光窗口内采样少量虚拟位姿，分别渲染，再平均成预测模糊图，与输入模糊图计算 loss。

目标优先级：

1. 主要提升模糊、快速运动、低光照场景下的 ATE RMSE。
2. 不破坏 MonoGS 在线 SLAM 结构。
3. 默认关闭，作为可开关模块做消融实验。

## 2. BARD-GS 可借鉴点

BARD-GS 源码中相关位置：

- `BARD_GS/bad_camera_optimizer.py`
- `BARD_GS/spline_functor.py`
- `BARD_GS/BARD_GS.py`

核心做法：

1. 为每张模糊图像建立曝光期间的相机轨迹。
2. 在线性模式下，用曝光起点和终点两个 SE(3) 控制点插值。
3. 在轨迹上均匀采样多个虚拟相机。
4. 对每个虚拟相机渲染一张瞬时图。
5. 将多张瞬时图 `mean()` 成预测模糊图。
6. 用预测模糊图和真实模糊输入计算渲染 loss。

MonoGS 不建议完整照搬 BARD-GS 的离线优化方式。BARD-GS 默认 `num_virtual_views=10`，且是离线训练；MonoGS 前端应先使用 `3` 个虚拟位姿，最多做 `5` 个用于实验。

## 3. MonoGS 推荐设计

推荐使用：

```text
中心位姿 P_mid + 曝光运动增量 xi_blur
```

其中：

- `P_mid` 是 MonoGS 当前帧最终要保存和传给后端的相机位姿。
- `cam_rot_delta/cam_trans_delta` 继续优化中心位姿。
- 新增 `blur_rot_delta/blur_trans_delta` 表示曝光期间的相对运动。

虚拟位姿采样：

```text
s_j = shutter_ratio * linspace(-0.5, 0.5, N)
P_j = Exp(s_j * xi_blur) * P_mid
```

建议默认：

```text
N = 3
s = [-0.5, 0.0, 0.5]
```

这样中间虚拟位姿就是中心位姿，tracking 结束后仍然只保存中心位姿，不需要修改后端、GUI 和 ATE 评估逻辑。

## 4. 曝光时间处理

第一版不强依赖真实曝光时间，使用归一化参数：

```yaml
Training:
  blur_aware_tracking:
    enabled: false
    num_virtual_views: 3
    shutter_ratio: 1.0
    motion_l2: 0.001
    start_after_keyframes: 2
```

说明：

- 如果数据集有真实 exposure metadata，后续可用 `exposure_time / frame_interval` 得到 `shutter_ratio`。
- 如果没有 metadata，就把 `shutter_ratio` 当成曝光窗口宽度的归一化尺度。
- 默认 `enabled=false`，确保 baseline 行为不变。

## 5. 主要修改文件

### 5.1 `utils/camera_utils.py`

给 `Camera` 增加曝光运动参数：

```python
blur_rot_delta
blur_trans_delta
```

它们只用于当前帧 tracking loss，不写入最终轨迹。

### 5.2 `utils/pose_utils.py`

复用现有 `SE3_exp()`，新增辅助函数：

- 构造当前 W2C 矩阵。
- 从 W2C 写回 `viewpoint.R/T`。
- 根据 `blur_*_delta` 和采样系数生成虚拟位姿。

不引入 PyPose，减少依赖风险。

### 5.3 `utils/blur_tracking.py`

建议新增文件，封装多虚拟位姿渲染逻辑：

1. 保存中心位姿。
2. 生成 `N` 个采样系数。
3. 临时设置 `viewpoint.R/T` 为虚拟位姿。
4. 调用现有 `render()`。
5. 平均多张 RGB 渲染图。
6. 恢复中心位姿。
7. 返回：

```python
{
    "render": image_blur,
    "depth": center_depth,
    "opacity": center_opacity,
}
```

深度和 opacity 第一版建议使用中心位姿结果，避免平均 depth 破坏 RGB-D tracking 和关键帧判断。

### 5.4 `utils/slam_frontend.py`

在 `FrontEnd.tracking()` 中增加分支：

- `enabled=false`：完全走原始单次 render tracking。
- `enabled=true` 且 warmup 满足：调用 blur-aware render。

优化器中加入：

```python
viewpoint.blur_rot_delta
viewpoint.blur_trans_delta
```

loss 中加入正则：

```text
motion_l2 * (||blur_rot_delta||^2 + ||blur_trans_delta||^2)
```

tracking 完成后：

- `update_pose(viewpoint)` 只更新中心位姿。
- `blur_*_delta` 不参与后端同步，不作为最终相机位姿。

### 5.5 `utils/slam_utils.py`

第一版可以不改。

因为 `get_loss_tracking()` 可以直接接收平均后的 `image_blur`，并继续复用现有：

- RGB boundary mask
- `grad_mask`
- rendered depth consistency mask
- uncertainty weight

后续如果要做动态区域或高残差区域降权，再扩展可选 mask 参数。

## 6. 实施阶段

### 阶段 A：最小版本

实现可开关的 blur-aware tracking。

验收标准：

1. `enabled=false` 时 baseline 行为不变。
2. `num_virtual_views=1` 时近似等价于原 tracking。
3. `num_virtual_views=3` 能正常运行，无 shape/device 错误。

### 阶段 B：稳定版本

加入：

- `motion_l2` 正则。
- `start_after_keyframes` warmup。
- debug 输出，例如虚拟位姿数量和 blur motion norm。

目标是防止清晰帧因新增自由度退化。

### 阶段 C：按需启用

加入简单 gating，只在可能模糊的帧启用：

- 图像清晰度低。
- 帧间运动大。
- tracking residual 偏高。

这样避免在清晰静态序列上额外增加计算和自由度。

## 7. 实验建议

建议做以下消融：

| 实验 | 虚拟位姿数 | 正则 | gating |
|---|---:|---:|---|
| Baseline | 1 | 0 | off |
| BlurTrack-3 | 3 | 1e-3 | off |
| BlurTrack-5 | 5 | 1e-3 | off |
| BlurTrack-3-Gated | 3 | 1e-3 | on |

主要看：

- ATE RMSE
- tracking loss
- 每帧耗时
- reset/lost 次数
- 清晰序列是否退化

第一阶段不要把 PSNR/SSIM/LPIPS 作为主指标，因为这里只改 tracking，不改 mapping。

## 8. 关键风险

1. **速度下降**：`N=3` 约等于 tracking 渲染开销增加到 3 倍。
2. **清晰帧退化**：`blur_*_delta` 可能吸收真实位姿误差，需要正则和 gating。
3. **深度不物理**：平均 depth 不等于真实观测深度，第一版 depth 用中心位姿。
4. **位姿语义混乱**：系统最终只保存中心位姿，虚拟位姿只用于 loss。
5. **临时改写相机状态**：多 render 后必须恢复 `viewpoint.R/T`。

## 9. 最终推荐

先实现一个保守版本：

```text
enabled=false 默认关闭
num_virtual_views=3
中心位姿 + 可学习曝光运动增量
RGB 用多虚拟位姿平均图
depth/opacity 用中心位姿
加入 motion_l2 正则
关键帧数量不足时禁用
```

这个版本改动集中、风险可控，能直接验证 BARD-GS 的相机运动模糊建模是否能改善 MonoGS 的 tracking ATE。若有效，再继续扩展动态/不确定区域降权和 blur-aware mapping。
