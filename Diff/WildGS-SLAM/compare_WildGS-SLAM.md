# WildGS-SLAM 与 MonoGS Baseline 对比分析

## 1. 一句话概括

**MonoGS / Gaussian Splatting SLAM** 默认认为场景基本静止：系统把图像中的大部分像素都当作可靠信息，用它们来估计相机轨迹并优化 3D Gaussian 地图。

**WildGS-SLAM** 关注动态场景：真实视频中可能有人、车、球、阴影等动态干扰。如果这些区域直接参与跟踪和建图，会拉偏相机位姿，并把动态物体错误地建进地图。WildGS-SLAM 的核心思想是学习一张**像素级不确定性图**，让系统知道哪些区域不可信，从而主要依赖静态背景完成跟踪和建图。

通俗理解：MonoGS 是“看到什么都相信”；WildGS-SLAM 是“先判断哪些地方可能在动，再重点相信稳定区域”。

---

## 2. MonoGS Baseline 的策略

### 2.1 Tracking：前端跟踪

MonoGS 的前端负责估计当前帧相机位姿。

| 项目 | MonoGS 策略 |
|---|---|
| 输入 | Monocular RGB / RGB-D / Stereo |
| 场景表达 | 3D Gaussian Splatting 地图 |
| 跟踪方式 | 固定高斯地图参数，只优化当前相机位姿 |
| 优化变量 | 相机旋转、平移、曝光参数 |
| 损失函数 | 渲染图像与真实图像之间的 RGB L1 loss；RGB-D 模式下加入 depth loss |
| 关键帧选择 | 根据时间间隔、相机位移、可见高斯重叠度判断是否插入关键帧 |

核心流程可以理解为：

1. 用当前地图从估计位姿渲染一张图像。
2. 比较渲染图像和真实图像的差异。
3. 反向优化相机位姿，让渲染结果更接近真实图像。

问题在于：如果图像中有移动物体，MonoGS 仍会把这些区域当作可靠观测，动态像素会错误地影响位姿优化，使 ATE RMSE 变大。

### 2.2 Mapping：后端建图

MonoGS 的后端负责优化 3D Gaussian 地图和局部关键帧位姿。

| 项目 | MonoGS 策略 |
|---|---|
| 地图表达 | 一组可优化的 3D Gaussians |
| 优化窗口 | 最近若干关键帧组成的局部滑窗 |
| 优化内容 | 高斯位置、颜色、尺度、旋转、不透明度，以及部分关键帧位姿 |
| 地图维护 | Densification、Pruning、Opacity Reset |
| 损失函数 | RGB loss、Depth loss、SSIM loss、Isotropic regularization |

MonoGS 在静态场景中效果较好，但在动态场景中容易出现两个问题：

1. 动态物体被错误建进 3DGS 地图，造成重影、浮点和伪影。
2. 动态区域污染地图优化，降低 PSNR、SSIM，并提高 LPIPS。

---

## 3. WildGS-SLAM 的策略

### 3.1 输入与整体框架

WildGS-SLAM 的定位是 **monocular RGB SLAM**。虽然它在 RGB-D 数据集上评估，但方法本身只使用 RGB 图像作为输入，深度由单目深度网络 Metric3D v2 预测得到。

整体流程：

1. 输入连续 RGB 图像。
2. 用 DINOv2 提取图像特征。
3. 用浅层 MLP 在线预测每帧的不确定性图。
4. 用 Metric3D v2 预测单目 metric depth。
5. 在 tracking 中用不确定性图降低动态区域权重。
6. 在 mapping 中用不确定性图降低动态区域对 3DGS 地图优化的影响。

### 3.2 Tracking：不确定性感知跟踪

WildGS-SLAM 的 tracking 基于 DROID-SLAM 的光流和 Dense Bundle Adjustment，而不是直接沿用 MonoGS 的渲染误差位姿优化。

| 项目 | WildGS-SLAM 策略 |
|---|---|
| 输入 | Monocular RGB |
| 光流估计 | 使用 DROID-SLAM 风格的 recurrent optical flow |
| 位姿优化 | Dense Bundle Adjustment |
| 深度辅助 | 使用 Metric3D v2 预测 metric depth，稳定单目位姿估计 |
| 动态处理 | 使用不确定性图降低动态区域在 BA 中的权重 |
| 长序列优化 | 引入 loop closure 和 online global BA 减少漂移 |

核心思想：

- 静态背景区域通常跨帧一致，适合用来估计相机运动。
- 动态物体区域跨帧不稳定，不应强烈参与位姿优化。
- 不确定性图越高，说明该像素越不可靠，在 BA 中权重越低。

因此，WildGS-SLAM 能在动态干扰明显的场景中获得更稳定的相机轨迹。

### 3.3 Mapping：不确定性感知建图

WildGS-SLAM 的 mapping 仍然使用 3D Gaussian Splatting 表达场景，但目标是只重建静态部分。

| 项目 | WildGS-SLAM 策略 |
|---|---|
| 地图表达 | 3D Gaussians |
| 地图目标 | 重建静态场景，去除动态干扰物 |
| 地图扩展 | 使用 Metric3D 预测深度作为 proxy depth，类似 MonoGS 的 RGB-D 初始化策略 |
| 优化窗口 | 局部关键帧窗口 |
| 损失函数 | RGB L1 + SSIM + Depth loss + Isotropic loss |
| 关键改动 | 用不确定性图对 color/depth loss 加权 |

核心思想：

- 如果某个区域被预测为高不确定性，说明它可能是移动物体、遮挡、阴影或不稳定区域。
- 建图时降低这些区域的损失权重，避免把动态内容固化到 3DGS 地图里。
- 最终地图更干净，渲染时动态物体残影更少。

---

## 4. 数据集、输入和指标

### 4.1 数据集

| 数据集 | 数据类型 | WildGS-SLAM 使用方式 | 主要用途 |
|---|---|---|---|
| Wild-SLAM MoCap | 自建 RGB-D 动态数据集，有 OptiTrack 真值轨迹 | 方法只使用 RGB，真值用于评估 | Tracking + Rendering 定量评估 |
| Wild-SLAM iPhone | 自建 iPhone RGB 数据集，无真值轨迹 | RGB 输入 | 真实场景定性评估 |
| Bonn RGB-D Dynamic | 动态 RGB-D 数据集 | 方法按 monocular RGB 使用 | ATE RMSE 评估 |
| TUM RGB-D | 常用 RGB-D SLAM 数据集 | 方法按 monocular RGB 使用 | ATE RMSE 评估 |

需要强调：WildGS-SLAM 的方法输入是 **RGB**，不是依赖真实深度传感器的 RGB-D 方法。它使用的深度来自 Metric3D v2 的单目深度预测。

### 4.2 评估指标

| 指标 | 评价对象 | 越大/越小越好 | 含义 |
|---|---|---|---|
| ATE RMSE | Tracking | 越小越好 | 相机轨迹误差，衡量位姿估计是否准确 |
| PSNR | Rendering / Mapping | 越大越好 | 渲染图像与真实图像的像素级接近程度 |
| SSIM | Rendering / Mapping | 越大越好 | 图像结构相似度 |
| LPIPS | Rendering / Mapping | 越小越好 | 感知相似度，更接近人眼视觉评价 |

---

## 5. 与 MonoGS 的核心差异

| 维度 | MonoGS | WildGS-SLAM |
|---|---|---|
| 场景假设 | 主要面向静态场景 | 面向动态真实场景 |
| 输入 | Monocular / RGB-D / Stereo | Monocular RGB |
| 跟踪依据 | 渲染图像与真实图像的差异 | 光流 + Dense BA + 深度 + 不确定性 |
| 动态处理 | 基本没有显式处理 | 用不确定性图降低动态区域权重 |
| 建图目标 | 重建观测到的场景 | 重建静态场景并去除动态干扰 |
| 深度来源 | RGB-D 模式用传感器深度；单目模式依赖初始化/渲染深度 | Metric3D v2 单目预测深度 |
| 地图质量 | 动态场景中容易有残影和伪影 | 动态物体更不容易进入地图 |

---

## 6. 能否嵌入 MonoGS Baseline

结论：**可以嵌入，而且最值得优先尝试的是不确定性图对 tracking loss 和 mapping loss 的加权。**

### 6.1 可迁移模块优先级

| 可迁移点 | 是否适合嵌入 MonoGS | 对 ATE RMSE 的影响 | 对 PSNR/SSIM/LPIPS 的影响 | 实现难度 | 优先级 |
|---|---|---|---|---|---|
| Uncertainty-aware tracking loss | 适合 | 直接降低动态区域对位姿的干扰 | 间接提升 | 中 | 高 |
| Uncertainty-aware mapping loss | 适合 | 间接提升 | 直接减少动态伪影 | 中 | 高 |
| Metric3D 单目深度先验 | 适合 | 稳定单目跟踪和初始化 | 改善几何质量 | 中 | 中高 |
| DINOv2 + MLP 在线预测 uncertainty | 适合但复杂 | 动态场景提升明显 | 动态残影减少 | 高 | 中 |
| 完整 DROID-SLAM DBA tracking | 可行但改动很大 | 可能大幅提升 | 间接提升 | 高 | 低到中 |
| Loop closure / global BA | 长期可做 | 降低长序列漂移 | 有帮助 | 高 | 低 |

### 6.2 推荐嵌入路线

#### 第一阶段：低侵入改动

在 MonoGS 原有结构上加入 uncertainty mask，不重写系统架构。

1. 在 tracking loss 中加入不确定性权重。
   - 当前 MonoGS 通过渲染误差优化相机位姿。
   - 可以让动态区域的 RGB loss 权重变小。
   - 目标是降低 ATE RMSE。

2. 在 mapping loss 中加入不确定性权重。
   - 当前 MonoGS 用关键帧图像监督高斯地图。
   - 可以让动态区域对高斯参数优化贡献变小。
   - 目标是提升 PSNR、SSIM，降低 LPIPS。

#### 第二阶段：加入单目深度先验

使用 Metric3D v2 预测深度，辅助单目模式下的初始化和地图扩展。

预期收益：

- 单目尺度更稳定。
- 新高斯初始化更合理。
- 建图几何更干净。

#### 第三阶段：更换或增强 tracking 后端

如果前两阶段提升有限，再考虑引入 DROID-SLAM 风格的光流和 Dense BA。

该路线收益可能更大，但改动范围也最大，需要处理：

- 光流网络推理；
- frame graph；
- DBA 优化；
- 与 MonoGS 后端地图同步；
- 与现有关键帧逻辑兼容。

---

## 7. 为什么这些点能提升指标

### 7.1 对 ATE RMSE 的影响

ATE RMSE 衡量相机轨迹与真值轨迹之间的误差。

动态物体会导致系统误判相机运动。例如，一个人在画面中移动，系统可能把人的移动错误解释成相机移动。这样会直接导致相机位姿漂移。

加入不确定性感知 tracking 后：

- 静态背景权重更高；
- 动态物体权重更低；
- 位姿优化主要依赖稳定区域；
- 因此 ATE RMSE 有望下降。

### 7.2 对 PSNR、SSIM、LPIPS 的影响

PSNR、SSIM、LPIPS 衡量渲染图像质量。

动态物体如果被建进 3DGS 地图，会造成：

- 残影；
- 浮点；
- 模糊；
- 错误遮挡；
- 新视角渲染不一致。

加入不确定性感知 mapping 后：

- 动态区域不再强烈监督高斯地图；
- 地图更偏向稳定静态背景；
- 渲染伪影减少；
- PSNR、SSIM 有望上升，LPIPS 有望下降。

---

## 8. 可写进论文的创新点表述

可以将方法概括为：

> 本文在 Gaussian Splatting SLAM 框架中引入动态感知的不确定性建模，通过像素级置信权重同时约束前端位姿跟踪和后端高斯地图优化，使系统在动态场景中更关注稳定静态区域，从而提升相机轨迹精度和新视角渲染质量。

也可以写成更正式的英文方法名：

> Uncertainty-aware Gaussian Splatting SLAM for Dynamic Monocular Scenes

中文题目可写为：

> 面向动态场景的不确定性感知 3D Gaussian Splatting SLAM

---

## 9. 推荐实验设计

### 9.1 Baseline 与消融实验

| 实验名称 | 内容 | 目的 |
|---|---|---|
| MonoGS | 原始 baseline | 基准结果 |
| MonoGS + uncertainty tracking | 只在 tracking loss 加不确定性权重 | 验证对 ATE RMSE 的贡献 |
| MonoGS + uncertainty mapping | 只在 mapping loss 加不确定性权重 | 验证对渲染质量的贡献 |
| MonoGS + uncertainty tracking + mapping | tracking 和 mapping 同时加权 | 验证完整不确定性感知框架 |
| MonoGS + depth prior | 加 Metric3D 深度先验 | 验证单目深度对几何和跟踪的帮助 |
| Full method | uncertainty + depth prior | 最终方法 |

### 9.2 指标

| 模块 | 指标 |
|---|---|
| Tracking | ATE RMSE，越低越好 |
| Mapping / Rendering | PSNR、SSIM 越高越好，LPIPS 越低越好 |

### 9.3 预期结论

预期结果应当体现：

1. 只加 tracking uncertainty 时，ATE RMSE 明显下降。
2. 只加 mapping uncertainty 时，PSNR/SSIM 提升，LPIPS 下降。
3. 同时加入 tracking 和 mapping uncertainty 时，轨迹和渲染质量整体最优。
4. 加入 Metric3D 深度先验后，单目场景中的初始化、尺度和几何稳定性进一步提升。

---

## 10. 总结

WildGS-SLAM 对 MonoGS 最有价值的启发不是单纯替换 3DGS 表达，而是引入了一个关键判断：**哪些像素可信，哪些像素不该被系统认真相信。**

对你的 MonoGS baseline 来说，最合理的改进路线是：

1. 先在 tracking loss 中加入 uncertainty-aware 权重，目标优化 ATE RMSE。
2. 再在 mapping loss 中加入 uncertainty-aware 权重，目标优化 PSNR、SSIM、LPIPS。
3. 然后加入 Metric3D 单目深度先验，提高单目初始化和几何建图稳定性。
4. 最后再考虑是否引入完整的 DROID-SLAM DBA tracking 和 loop closure。

这条路线改动从小到大、实验逻辑清晰、论文叙事自然，也容易通过消融实验证明每个模块的有效性。
