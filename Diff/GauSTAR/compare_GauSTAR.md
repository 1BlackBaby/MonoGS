# GauSTAR 与 MonoGS Baseline 对比分析

本文整理 Zheng 等人在 2025 年提出的 **GauSTAR: Gaussian Surface Tracking and Reconstruction**，并分析其思想是否适合嵌入当前 baseline **Gaussian Splatting SLAM / MonoGS**，用于提升跟踪指标 **ATE RMSE** 和建图指标 **PSNR、SSIM、LPIPS**。

---

## 1. 一句话总结

GauSTAR 是一个面向 **动态场景表面跟踪与重建** 的 3DGS 方法。它不是传统 SLAM 系统，不主要估计相机轨迹，而是用多视角 RGB-D 输入重建和跟踪会运动、变形、拓扑变化的物体表面。

通俗理解：

| 方法 | 主要解决的问题 |
|---|---|
| MonoGS | 相机在哪里，静态场景地图怎么建 |
| GauSTAR | 动态物体表面怎么重建、怎么跨帧跟踪 |

GauSTAR 的核心思想是：**把 3D Gaussians 绑定到 mesh 面片上，用 mesh 保持几何和跨帧对应关系，用 Gaussians 提供高质量外观渲染；当表面拓扑变化时，让局部 Gaussians 从旧 mesh 上解绑，并重新生成新表面。**

---

## 2. MonoGS Baseline 策略

### 2.1 Tracking 策略

MonoGS 的前端负责相机跟踪：

1. 固定当前 3D Gaussian 地图；
2. 从当前估计位姿渲染 RGB / depth；
3. 将渲染结果与输入帧比较；
4. 反向优化相机旋转和平移；
5. 输出当前帧相机位姿，并决定是否插入关键帧。

MonoGS 的 tracking 目标是 **相机位姿准确**，评价指标通常是 **ATE RMSE**。

### 2.2 Mapping 策略

MonoGS 的后端负责建图：

1. 维护局部关键帧滑窗；
2. 优化关键帧位姿和高斯点参数；
3. 使用 RGB loss、depth loss、SSIM loss 等监督渲染结果；
4. 通过 densification 增加高斯点；
5. 通过 pruning 删除冗余高斯点；
6. 使用 isotropic loss 约束高斯形状。

MonoGS 的 mapping 目标是 **重建出可渲染的静态场景地图**，评价指标包括 **PSNR、SSIM、LPIPS**。

---

## 3. GauSTAR 的输入与任务定位

### 3.1 输入类型

GauSTAR 使用 **multi-view RGB-D videos**。

论文实验中使用：

| 输入 | 说明 |
|---|---|
| 多视角 RGB 视频 | 52 个 RGB 相机 |
| 多视角 IR 视频 | 52 个 IR 相机 |
| 深度图 | 由 IR 捕获生成 raw point clouds，再投影到相机视角 |
| 初始 mesh | 第一帧由多视角重建方法获得 |
| mask | 用于前景/表面约束 |

采集设置：

| 项目 | 数值 |
|---|---|
| RGB 相机数量 | 52 |
| IR 相机数量 | 52 |
| 分辨率 | 3004 × 4092 |
| 帧率 | 30 fps |
| 每个三角面绑定高斯数量 | N = 6 |

这和 MonoGS 的输入差异较大：

| 方法 | 输入 |
|---|---|
| MonoGS | 单目 / RGB-D / Stereo 序列 |
| GauSTAR | 多视角 RGB-D 视频 + 初始 mesh |

因此，GauSTAR 的完整系统不能直接迁移到普通单目 MonoGS 中。

---

## 4. GauSTAR 的 Tracking 策略

GauSTAR 的 tracking 不是相机跟踪，而是 **动态表面跟踪**。

它跟踪的是物体表面上的点、mesh 顶点或面片，而不是估计相机轨迹。

### 4.1 Gaussian Surface 表达

GauSTAR 将 Gaussians 绑定到 mesh 三角面上，形成 **Gaussian Surface**。

每个 Gaussian 的中心由三角面三个顶点的重心坐标决定：

```text
p = b1 * v1 + b2 * v2 + b3 * v3
```

含义是：

1. mesh 顶点移动时，绑定在面片上的 Gaussians 跟着移动；
2. mesh 提供稳定的几何结构；
3. Gaussians 提供高质量颜色和外观；
4. 同一个 mesh 面片跨帧存在时，可以自然获得表面对应关系。

通俗说：  
**mesh 负责“表面是谁、在哪里”，Gaussians 负责“看起来像什么”。**

### 4.2 Scene Flow Warping 初始化

动态场景中，相邻帧可能有大运动或快速变形。GauSTAR 用 **scene flow warping** 初始化当前帧表面位置。

流程：

1. 将上一帧 mesh 顶点投影到多个相机视角；
2. 使用 2D optical flow 找到下一帧对应像素；
3. 根据下一帧 depth 将 2D 像素反投影回 3D；
4. 多视角融合得到每个顶点的 3D scene flow；
5. 用 surface-aware smoothing 平滑邻接顶点运动；
6. 用该 scene flow 将上一帧表面变形到当前帧。

作用：

| 作用 | 解释 |
|---|---|
| 提供好初始化 | 避免优化从错误位置开始 |
| 处理快速运动 | 减少局部最优问题 |
| 提升表面 tracking | 保持跨帧表面一致性 |

### 4.3 Fixed-Topology Tracking

对于拓扑不变的区域，GauSTAR 保持 mesh 拓扑不变，只优化：

1. mesh 顶点位置；
2. Gaussian 颜色；
3. Gaussian 尺度、旋转、不透明度等参数。

监督信号包括：

| Loss | 作用 |
|---|---|
| RGB loss | 保证渲染颜色接近输入图像 |
| SSIM loss | 保证图像结构一致 |
| Depth loss | 保证几何深度正确 |
| Mask loss | 保证前景/表面区域正确 |
| Normal smoothing | 保证表面连续 |
| Area preservation | 防止面片面积异常变化 |
| SH consistency | 保持跨帧颜色一致 |

这种策略适合稳定变形表面，例如衣服、身体、手臂等连续运动。

### 4.4 Adaptive Gaussian Unbinding

固定拓扑无法处理新表面出现、表面分裂、遮挡后显露等情况。GauSTAR 用 **Gaussian unbinding** 处理这些拓扑变化。

它为每个 face 计算 unbinding weight：

```text
W(f) = position gradient + RGB error + depth error
```

含义是：

| 信号 | 说明 |
|---|---|
| positional gradient 大 | 说明该区域为了拟合当前帧需要大幅移动 |
| RGB error 大 | 说明外观重建不好 |
| depth error 大 | 说明几何重建不好 |

当某个 face 的 W 很高时，说明该区域很可能发生了拓扑变化。此时该 face 上的 Gaussians 可以从 mesh 解绑，独立优化位置和旋转。

通俗说：  
**如果旧 mesh 管不住新出现的表面，就先让局部 Gaussians 自由移动，找到真实表面位置。**

### 4.5 Re-meshing

解绑后的 Gaussians 会指示新表面位置。GauSTAR 根据这些 Gaussians 重新生成 mesh，并把新 mesh 与旧 mesh 边界连接。

作用：

1. 更新拓扑结构；
2. 生成新出现的表面；
3. 保留未变化区域的跨帧对应关系；
4. 让后续帧继续稳定跟踪。

---

## 5. GauSTAR 的 Mapping / Reconstruction 策略

GauSTAR 的 mapping 更准确地说是 **动态表面重建**，不是 SLAM 里的静态环境建图。

其重建流程为：

1. 第一帧使用多视角 RGB-D 重建得到初始 mesh；
2. 在每个三角面上绑定多个 Gaussians；
3. 每一帧先用 scene flow 从上一帧初始化；
4. 在固定拓扑假设下优化 Gaussian Surface；
5. 检测拓扑变化区域；
6. 对高权重区域进行 Gaussian unbinding；
7. 根据解绑 Gaussians 重新生成新 mesh；
8. 再进行一轮固定拓扑优化精修。

GauSTAR 的建图重点是：

| 目标 | 说明 |
|---|---|
| 高质量外观 | 通过 3DGS 渲染获得高 PSNR / SSIM / LPIPS |
| 高质量几何 | 通过 depth / mask / mesh regularization 获得稳定表面 |
| 跨帧一致性 | 通过 mesh topology 和 SH consistency 保持跟踪 |
| 拓扑适应 | 通过 unbinding + re-meshing 处理新表面 |

---

## 6. 使用的数据集与评价指标

### 6.1 数据集 / 采集数据

论文使用自建多视角 RGB-D 动态捕获数据。

关键信息：

| 项目 | 说明 |
|---|---|
| 采集系统 | 52 RGB cameras + 52 IR cameras |
| 输入形式 | multi-view RGB-D videos |
| 分辨率 | 3004 × 4092 |
| 帧率 | 30 fps |
| 外观评估 | 4 个序列，共 850 帧，5 个测试视角 |
| 跟踪评估 | 2 个带 AprilTags 的人体序列 |

论文没有使用 TUM RGB-D、Replica、EuRoC 等常见 SLAM 数据集。

### 6.2 评价指标

| 指标 | 评价内容 | 越大/越小 |
|---|---|---|
| PSNR | 渲染图像像素质量 | 越大越好 |
| SSIM | 图像结构相似度 | 越大越好 |
| LPIPS | 感知差异 | 越小越好 |
| CD | 几何 Chamfer Distance | 越小越好 |
| F-Score | 几何重建完整性和准确性 | 越大越好 |
| 3D ATE | 表面点 3D 轨迹误差 | 越小越好 |
| 2D ATE | 表面点投影轨迹误差 | 越小越好 |

需要注意：

> GauSTAR 的 ATE 是 AprilTag 表面点 tracking ATE，不是 MonoGS 中常见的相机轨迹 ATE RMSE。

因此，GauSTAR 的 tracking 指标不能直接等价为 SLAM 相机 tracking 指标。

### 6.3 主要结果

GauSTAR 在表 1 中报告：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 3D ATE ↓ | 2D ATE ↓ |
|---|---:|---:|---:|---:|---:|
| Dynamic 3D Gaussians | 27.61 | 0.905 | 0.214 | 3.15 | 13.84 |
| PhysAvatar-SMPLX | 24.50 | 0.908 | 0.193 | 8.98 | 39.61 |
| 2DGS | 30.17 | 0.938 | 0.155 | - | - |
| GauSTAR w/o IR input | 30.05 | 0.946 | 0.110 | 0.671 | 3.02 |
| GauSTAR | 31.87 | 0.952 | 0.102 | 0.452 | 2.03 |

消融实验表明：

| 去掉模块 | 影响 |
|---|---|
| w/o unbinding | 拓扑变化区域处理变差，PSNR / LPIPS / ATE 下降 |
| w/o re-meshing | 新表面拓扑不能正确更新 |
| w/o scene flow | tracking 明显变差，3D ATE 从 0.45 上升到 6.56 |

---

## 7. 能否嵌入 MonoGS

结论：**完整 GauSTAR 不适合直接嵌入 MonoGS，但部分思想值得借鉴。**

原因如下：

1. GauSTAR 依赖多视角 RGB-D 和初始 mesh，而 MonoGS 通常是单目 / RGB-D / Stereo SLAM；
2. GauSTAR 跟踪动态物体表面，MonoGS 跟踪相机位姿；
3. GauSTAR 是离线或多视角重建范式，MonoGS 是在线 SLAM 范式；
4. GauSTAR 的 ATE 是表面点轨迹误差，不是相机 ATE RMSE；
5. GauSTAR 的 mesh-Gaussian 表达会显著改变 MonoGS 的地图结构，工程改动很大。

因此，不建议把 GauSTAR 整体搬进 MonoGS。

---

## 8. 可迁移点分析

### 8.1 可迁移点一：Scene Flow / Optical Flow 初始化

**可迁移性：中等。**

GauSTAR 用 optical flow + depth 估计 3D scene flow，为下一帧表面位置提供初始化。MonoGS 可以借鉴这个思想，用于改进相机 tracking 的初值。

在 MonoGS 中可改为：

1. 使用 optical flow 找到相邻帧像素对应；
2. RGB-D 模式下用深度反投影为 3D 点；
3. 估计帧间相对运动或辅助 pose initialization；
4. 作为当前帧 tracking 的初始位姿。

预期影响：

| 指标 | 影响 |
|---|---|
| ATE RMSE | 可能下降，尤其是快速运动时 |
| PSNR / SSIM / LPIPS | 间接提升，因为位姿更准，建图更稳定 |

风险：

1. 单目模式没有真实深度，需要预测深度；
2. 光流在动态物体、遮挡、低纹理区域可能不稳定；
3. 需要和 MonoGS 现有恒速初始化策略融合。

推荐程度：**可作为 tracking 初始化增强模块尝试。**

#### 单目模式下的深度来源选择

由于 MonoGS 的单目模式没有真实深度，如果要迁移 GauSTAR 的 `optical flow + depth` 初始化思想，关键问题是：深度应该来自 **3DGS 当前地图的渲染深度**，还是来自 **Metric3D 等单目深度网络的预测深度**。

建议采用：

```text
Metric3D 预测深度 + Optical Flow 作为主初始化信号
MonoGS 渲染深度作为一致性校验和辅助过滤信号
```

原因如下：

| 方案 | 优点 | 主要问题 | 建议用途 |
|---|---|---|---|
| 渲染深度 + Optical Flow | 不引入额外模型，和 MonoGS 当前地图一致 | 渲染深度依赖当前地图和当前位姿；如果 tracking 已经偏移，渲染深度也会偏移，容易用错误地图强化错误位姿 | 适合作为辅助校验 |
| Metric3D 深度 + Optical Flow | 深度由当前 RGB 图像预测，不依赖已有地图；在初始化、快速运动、弱纹理场景中更能提供独立几何约束 | 预测深度存在尺度偏差和域偏差，需要尺度对齐与置信度过滤 | 更适合作为主要创新点 |

更推荐的技术路线：

1. 用 optical flow 建立相邻帧像素对应；
2. 用 Metric3D 预测深度把匹配像素反投影到 3D；
3. 通过几何一致性估计相邻帧相对位姿，作为 MonoGS tracking 的初始位姿；
4. 用 MonoGS 当前渲染深度做一致性检查，过滤动态区域、遮挡区域和明显不可靠匹配；
5. 后续仍使用 MonoGS 原有 photometric rendering loss 精修相机位姿。

这样做的逻辑是：**Metric3D 提供独立的外部深度先验，帮助 tracking 在优化开始前获得更好的初值；渲染深度不负责主初始化，而负责判断预测深度和当前地图是否一致。**

需要注意的是，Metric3D 深度不能直接当作真实深度强监督使用。更稳妥的做法是只用于初始化或弱约束，并加入：

1. 尺度对齐：将 Metric3D 深度与当前 3DGS 渲染深度或关键帧尺度对齐；
2. 置信度过滤：去掉光流不稳定、遮挡、动态物体和深度突变区域；
3. 一致性检查：若 Metric3D 深度与渲染深度差异过大，则降低该区域在位姿初始化中的权重。

对指标的预期影响：

| 指标 | 预期影响 |
|---|---|
| ATE RMSE | 最可能下降，因为该模块直接改善相机 tracking 初始值 |
| PSNR | 间接提升，位姿更准会让后端建图更稳定 |
| SSIM | 间接提升，结构对齐更好 |
| LPIPS | 可能下降，主要来自更稳定的关键帧位姿和更少的重影 |

论文中可以这样表述：

> 与 GauSTAR 利用 optical flow 和 depth 初始化动态表面跟踪类似，本文将该思想迁移到单目 Gaussian Splatting SLAM 的相机跟踪前端。考虑到单目输入缺少真实深度，本文引入 Metric3D 预测深度与光流匹配构造几何一致的位姿初始化，并利用当前 3DGS 地图的渲染深度进行一致性校验，从而缓解快速运动和弱纹理场景下纯光度优化初值不稳的问题。

### 8.2 可迁移点二：RGB / Depth Error + Gradient 的异常区域检测

**可迁移性：高。**

GauSTAR 用 position gradient、RGB error、depth error 判断拓扑变化区域。MonoGS 虽然不需要处理 mesh 拓扑变化，但可以借鉴这个思想判断“地图哪里建得不好”。

可迁移为 MonoGS 的建图策略：

```text
高 RGB error 区域：颜色重建不足，需要增加或优化高斯
高 depth error 区域：几何不准，需要补点或调整高斯
高 gradient 区域：当前地图对 loss 敏感，需要重点优化
```

预期影响：

| 指标 | 影响 |
|---|---|
| PSNR | 提升 |
| SSIM | 提升 |
| LPIPS | 下降 |
| ATE RMSE | 间接下降 |

推荐程度：**最值得迁移。**

这与 HF-SLAM 的 rendering-guided densification 思想也一致，可以作为论文中较稳妥的建图增强点。

### 8.3 可迁移点三：Temporal Appearance Consistency

**可迁移性：高。**

GauSTAR 使用 SH consistency 约束相邻帧 Gaussian 的颜色参数，避免动态重建中外观闪烁。

MonoGS 可以借鉴为：

1. 对稳定可见的高斯点增加颜色/SH 正则；
2. 防止新关键帧优化破坏旧视角外观；
3. 与已有 mapping loss 或防遗忘正则结合。

预期影响：

| 指标 | 影响 |
|---|---|
| PSNR | 可能提升 |
| SSIM | 可能提升 |
| LPIPS | 下降 |
| ATE RMSE | 间接帮助较小 |

推荐程度：**适合用于提升建图稳定性。**

### 8.4 可迁移点四：Surface-aware Regularization

**可迁移性：中等。**

GauSTAR 使用 normal smoothing 和 area preservation 保持 mesh 表面稳定。MonoGS 没有 mesh，因此不能直接用面法线和三角面面积。

但可以替换为 3DGS 中更合适的正则：

1. depth-normal consistency；
2. local smoothness；
3. isotropic loss；
4. scale regularization；
5. opacity regularization。

预期影响：

| 指标 | 影响 |
|---|---|
| PSNR / SSIM | 可能提升 |
| LPIPS | 可能下降 |
| ATE RMSE | 几何更稳时可能下降 |

推荐程度：**可借鉴思想，不建议照搬实现。**

### 8.5 不建议迁移点：Mesh-Gaussian 完整表达

**可迁移性：低。**

GauSTAR 的核心是 mesh + Gaussian 绑定。如果把它完整嵌入 MonoGS，需要：

1. 为 MonoGS 地图引入 mesh；
2. 将高斯绑定到 mesh face；
3. 维护 mesh 拓扑；
4. 实现 unbinding 和 re-meshing；
5. 改写 densification / pruning；
6. 改写渲染、优化和关键帧逻辑。

这会把 MonoGS 从 3DGS SLAM 改成动态多视角表面重建系统，代价过大，也偏离 baseline。

推荐结论：**不建议作为当前论文主线。**

---

## 9. 对 MonoGS 指标的预期影响

| GauSTAR 思想 | ATE RMSE | PSNR | SSIM | LPIPS | 推荐程度 |
|---|---|---|---|---|---|
| Metric3D depth + optical flow 初始化 | 最可能下降 | 间接提升 | 间接提升 | 间接下降 | 高 |
| rendered depth consistency check | 辅助下降 | 间接提升 | 间接提升 | 间接下降 | 中 |
| RGB / depth error 区域检测 | 间接下降 | 提升 | 提升 | 下降 | 中；与 HF-SLAM 思想接近 |
| Gradient-aware 重点优化 | 间接下降 | 提升 | 提升 | 下降 | 中；更适合作为辅助 |
| SH temporal consistency | 影响小 | 可能提升 | 可能提升 | 下降 | 高 |
| Surface-aware regularization | 可能下降 | 可能提升 | 可能提升 | 可能下降 | 中 |
| Mesh-Gaussian + re-meshing | 不确定 | 不确定 | 不确定 | 不确定 | 低 |

总体判断：

1. **当前更适合作为主创新的是 Metric3D depth + optical flow 辅助 tracking 初始化。**
2. **rendered depth 更适合作为一致性校验，而不是单目模式下的唯一深度来源。**
3. **最可能改善 LPIPS 的是 temporal appearance consistency。**
4. **RGB / depth error + gradient 的异常区域检测与 HF-SLAM 的 rendering-guided densification 思想接近，不建议重复作为主创新点。**
5. **完整 mesh-Gaussian 和 re-meshing 不适合直接嵌入 MonoGS。**

---

## 10. 推荐嵌入优先级

建议按以下顺序尝试：

| 优先级 | 模块 | 原因 |
|---|---|---|
| 1 | Metric3D depth + optical flow pose initialization | 与 HF-SLAM 的 rendering-guided densification 不重复，直接服务 tracking 的 ATE RMSE |
| 2 | rendered depth consistency check | 用 MonoGS 渲染深度过滤不可靠匹配，避免预测深度误导 tracking |
| 3 | SH / color temporal consistency | 提升跨关键帧外观稳定性，间接改善 LPIPS |
| 4 | RGB / depth error guided densification | 思想与 HF-SLAM rendering-guided densification 接近，若已采用 HF-SLAM 思路则不作为主要创新点 |
| 5 | gradient-aware mapping weight | 与异常区域检测和 rendering-guided densification 接近，可作为辅助项而非主创新 |
| 6 | surface / normal regularization 替代项 | 需要设计适配 3DGS 的版本 |
| 7 | mesh-Gaussian binding / re-meshing | 不建议优先做 |

第一阶段推荐实验：

```text
MonoGS baseline
+ Metric3D depth + optical flow pose initialization
+ rendered-depth consistency filtering
```

第二阶段再考虑：

```text
+ SH temporal consistency
+ RGB / depth error guided densification
+ gradient-aware mapping weight
```

不推荐第一阶段做：

```text
完整 mesh-Gaussian 表达
Gaussian unbinding
surface re-meshing
```

---

## 11. 推荐消融实验

| 实验 | 目的 |
|---|---|
| E0: MonoGS baseline | 原始对照 |
| E1: + Metric3D depth + optical flow initialization | 验证 ATE RMSE 是否下降 |
| E2: + rendered-depth consistency filtering | 验证渲染深度过滤是否能减少错误匹配 |
| E3: + scale alignment for Metric3D depth | 验证预测深度尺度对齐的必要性 |
| E4: + SH temporal consistency | 验证跨帧外观一致性 |
| E5: + RGB / depth error guided densification | 作为 HF-SLAM 类思想的辅助对照，不作为本文主创新 |

评价指标：

| 任务 | 指标 |
|---|---|
| Tracking | 相机 ATE RMSE |
| Mapping / Rendering | PSNR、SSIM、LPIPS |
| Efficiency | 每帧 tracking 时间、mapping 时间 |

实验数据建议：

1. 单目配置下重点测试 `Metric3D depth + optical flow initialization`；
2. 在 TUM-RGBD / Replica RGB-D 上可用真实深度做上界对照，判断 Metric3D 深度与真实深度之间的差距；
3. 应重点选择快速运动、弱纹理、视角变化较大的序列，因为这些场景最能体现 tracking 初始化的价值；
4. RGB / depth error + gradient 类模块可作为辅助对照，避免与 HF-SLAM 的 rendering-guided densification 重复包装为主创新。

---

## 12. 可写入论文的方法表述

可以这样描述 GauSTAR 对本文工作的启发：

> GauSTAR 通过 optical flow 与 depth 反投影估计 scene flow，为动态表面跟踪提供了鲁棒初始化。受此启发，本文将该思想迁移到单目 Gaussian Splatting SLAM 的相机跟踪前端，引入 Metric3D 预测深度与光流匹配构造几何一致的位姿初始化，并利用当前 3DGS 地图的渲染深度进行一致性校验，从而缓解快速运动和弱纹理场景下纯光度优化初值不稳的问题。

如果需要说明为什么不采用 error / gradient 异常区域检测作为主创新，可以写为：

> GauSTAR 中基于 RGB / depth error 与 gradient 的异常区域检测，与 HF-SLAM 的 rendering-guided densification 在思想上高度接近，均是利用渲染误差定位当前地图表达不足的区域。因此本文不将其重复作为主要创新点，而是将研究重点放在 tracking 前端的深度辅助光流初始化上。

---

## 13. 最终建议

GauSTAR 的完整系统不适合直接嵌入 MonoGS，因为它依赖多视角 RGB-D、初始 mesh、表面跟踪、Gaussian unbinding 和 re-meshing，任务目标也不是相机 SLAM。

但以下思想适合借鉴：

1. **Metric3D depth + optical flow 初始化**：作为当前最推荐的主迁移点，用于改善快速运动下 tracking 初值；
2. **rendered depth consistency check**：用 MonoGS 渲染深度辅助过滤错误匹配，而不是作为唯一深度来源；
3. **temporal SH / color consistency**：用于提升关键帧间外观稳定性；
4. **RGB / depth error + gradient 异常区域检测**：与 HF-SLAM 的 rendering-guided densification 思想接近，可作为辅助对照，不建议重复作为主创新。

最终结论：

> GauSTAR 对 MonoGS 的价值主要是提供“光流/深度辅助初始化”的启发，而不是提供可直接替换的 SLAM 模块。当前更建议优先探索 Metric3D 预测深度与 optical flow 结合的 tracking 初始化，并用 MonoGS 渲染深度做一致性校验；RGB / depth error + gradient 类方法因与 HF-SLAM 的 rendering-guided densification 接近，不建议重复作为主创新；不建议直接迁移 mesh-Gaussian binding、Gaussian unbinding 和 re-meshing。
