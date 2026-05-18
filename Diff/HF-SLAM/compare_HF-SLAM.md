# HF-SLAM 与 MonoGS Baseline 对比分析

本文整理 Sun 等人在 2024 年提出的 **High-Fidelity SLAM Using Gaussian Splatting with Rendering-Guided Densification and Regularized Optimization**，下文简称 **HF-SLAM**。分析目标是判断其中哪些思想可以迁移到当前 baseline **Gaussian Splatting SLAM / MonoGS** 中，用于提升跟踪指标 **ATE RMSE** 和建图指标 **PSNR、SSIM、LPIPS**。

---

## 1. 一句话总结

HF-SLAM 是一个基于 3D Gaussian Splatting 的 **RGB-D 稠密 SLAM** 系统。它的核心不是改变 3DGS 表达，而是改进两个关键环节：

1. **渲染误差引导的高斯致密化**：哪里渲染不好，就在哪里增加或细化高斯点。
2. **正则化优化**：防止在线建图时新帧把旧区域的重建质量破坏掉。

通俗理解：MonoGS 已经能用 3D 高斯点建图和跟踪；HF-SLAM 进一步告诉系统“哪些地方该补点、哪些旧点不能随便改”。

---

## 2. MonoGS Baseline 策略

### 2.1 Tracking 策略

MonoGS 的前端负责 Tracking。其基本思想是：

1. 固定当前 3D Gaussian 地图；
2. 给定当前帧 RGB 或 RGB-D 输入；
3. 从估计位姿渲染图像和深度；
4. 最小化渲染结果与真实输入之间的差异；
5. 优化相机位姿参数，包括旋转、平移和曝光补偿参数。

也就是说，MonoGS 通过“当前位姿下地图渲染出来的图像是否像真实图像”来判断位姿是否正确。

### 2.2 Mapping 策略

MonoGS 的后端负责 Mapping。主要策略包括：

| 模块 | 作用 |
|---|---|
| 局部滑窗优化 | 同时优化多个关键帧位姿和可见高斯点参数 |
| Densification | 在梯度较大区域增加高斯点，提升细节表达 |
| Pruning | 删除不透明度过低或尺寸异常的冗余高斯点 |
| Opacity reset | 防止部分区域过拟合或长期错误保留 |
| Isotropic loss | 约束高斯点不要变成极端拉伸的薄片 |

MonoGS 支持 **Monocular、RGB-D 和 Stereo** 输入，其中 RGB-D 模式可以直接使用真实深度，单目模式则主要依赖图像重渲染误差和系统自身估计。

---

## 3. HF-SLAM 策略

### 3.1 输入类型

HF-SLAM 使用 **RGB-D 输入**。

每一帧包含：

| 输入 | 作用 |
|---|---|
| RGB 图像 | 用于颜色渲染误差、视觉重建质量优化 |
| Depth 图像 | 用于深度渲染误差、几何约束和高斯点初始化 |

因此，HF-SLAM 更适合直接迁移到 MonoGS 的 RGB-D 设置。如果要迁移到 MonoGS 单目设置，需要额外引入预测深度，或者只迁移 RGB / opacity 相关部分。

### 3.2 Tracking 策略

HF-SLAM 的 Tracking 也是 **render-and-compare** 思路：

1. 使用恒速运动模型初始化当前帧相机位姿；
2. 基于当前 3D Gaussian 地图渲染 RGB 图和深度图；
3. 将渲染图与真实 RGB-D 输入比较；
4. 通过反向传播优化相机位姿。

跟踪损失为颜色误差和深度误差的组合：

| 损失项 | 作用 |
|---|---|
| Color re-rendering loss | 约束渲染颜色接近真实图像 |
| Depth re-rendering loss | 约束渲染深度接近真实深度 |
| LAB 空间去亮度通道 | 减弱光照变化对跟踪的影响 |

HF-SLAM 在跟踪时将 RGB 转到 LAB 色彩空间，并丢弃亮度通道，只比较颜色相关通道。这样可以减少曝光变化、亮度变化对位姿优化的干扰。

### 3.3 Mapping 策略

HF-SLAM 的建图有两个核心改进。

#### 3.3.1 Rendering-Guided Densification

HF-SLAM 不只根据梯度或可见性增加高斯点，而是直接看渲染结果哪里不好。

其致密化依据包括：

| 判断依据 | 含义 | 操作 |
|---|---|---|
| Opacity 低 | 该区域地图覆盖不足，可能是空洞 | 添加高斯点 |
| RGB 渲染误差大 | 颜色重建不好 | 添加或细化高斯点 |
| Depth 渲染误差大 | 几何重建不好 | 添加或细化高斯点 |

可以理解为：如果某个像素渲染出来和真实图像差很多，说明该区域的地图表达能力不足，需要补充高斯点。

#### 3.3.2 Regularized Optimization

在线建图时，系统会不断用新帧优化地图。问题是：同一个高斯点可能同时服务多个历史视角，如果只为了拟合最新帧而修改它，旧视角的渲染质量可能下降。

HF-SLAM 将这个问题称为 **forgetting problem**，即持续建图中的遗忘问题。

为缓解该问题，HF-SLAM 为高斯点维护额外统计量，例如：

| 统计量 | 含义 |
|---|---|
| seen times | 该高斯点被多少帧观察到 |
| 参数梯度累计 | 衡量该参数对重建损失的重要性 |
| importance weight | 判断某个参数是否应该被强约束 |

然后在 mapping loss 中加入正则项，限制重要高斯参数被过度修改。

简单说：新帧可以更新地图，但不能为了拟合新帧把旧地图细节破坏掉。

---

## 4. 使用的数据集与指标

### 4.1 数据集

HF-SLAM 主要使用两个数据集：

| 数据集 | 类型 | 特点 |
|---|---|---|
| Replica | 合成 RGB-D 数据集 | RGB 和深度质量高，适合评估重建上限 |
| TUM-RGBD | 真实 RGB-D 数据集 | 存在噪声、运动模糊、曝光变化，更接近真实场景 |

### 4.2 评价指标

| 指标 | 评价内容 | 越大/越小 |
|---|---|---|
| ATE RMSE | 相机轨迹误差，即跟踪精度 | 越小越好 |
| PSNR | 渲染图与真实图的像素级相似度 | 越大越好 |
| SSIM | 图像结构相似度 | 越大越好 |
| LPIPS | 感知相似度，越低说明视觉差异越小 | 越小越好 |
| Depth L1 | 深度重建误差 | 越小越好 |

本文重点关注用户指定指标：**ATE RMSE、PSNR、SSIM、LPIPS**。

---

## 5. HF-SLAM 与 MonoGS 的核心差异

| 对比项 | MonoGS | HF-SLAM |
|---|---|---|
| 输入 | Monocular / RGB-D / Stereo | RGB-D |
| 跟踪方式 | 渲染图像与输入图像对齐 | 渲染 RGB-D 与输入 RGB-D 对齐 |
| 光照处理 | 曝光补偿参数 | LAB 空间去亮度通道 |
| 高斯致密化 | 主要依赖梯度、可见性等策略 | 直接使用 opacity、RGB error、depth error |
| 持续建图稳定性 | 依靠滑窗、关键帧和正则项 | 额外引入防遗忘正则化 |
| 主要优势 | 通用输入、系统完整、支持单目 | 重建质量高，细节和旧视角保持更好 |

---

## 6. 能否嵌入 MonoGS

结论：**可以嵌入，而且优先推荐迁移局部模块，而不是整体替换 MonoGS 框架。**

原因如下：

1. MonoGS 和 HF-SLAM 都以 3D Gaussian 作为地图表达；
2. 二者都使用可微渲染进行跟踪和建图；
3. MonoGS 已经有 tracking loss、mapping loss、densification 和 pruning 机制；
4. HF-SLAM 的改进主要集中在 loss 和 densification 规则上，比较适合作为模块化增强。

---

## 7. 可迁移模块分析

### 7.1 Opacity 空洞检测

**可迁移性：高。**

MonoGS 渲染时可以得到可见性和不透明度相关信息。可以增加规则：当某些区域 opacity 低于阈值时，认为该区域地图覆盖不足，触发新增高斯点。

预期影响：

| 指标 | 影响 |
|---|---|
| PSNR | 提升 |
| SSIM | 提升 |
| LPIPS | 下降 |
| ATE RMSE | 间接下降 |

原因：地图空洞减少后，渲染结果更完整，跟踪时的重投影误差也更稳定。

### 7.2 RGB 渲染误差引导致密化

**可迁移性：高。**

MonoGS 已经计算渲染 RGB 与真实 RGB 的误差，可以在误差较大的区域增加高斯点。

适合用于：

1. 物体边缘；
2. 纹理复杂区域；
3. 当前高斯表达不足的区域；
4. 重访时仍然渲染不好的区域。

预期影响：

| 指标 | 影响 |
|---|---|
| PSNR | 明显提升 |
| SSIM | 提升 |
| LPIPS | 明显下降 |
| ATE RMSE | 可能下降 |

这是最推荐优先实现的模块。

### 7.3 Depth 渲染误差引导致密化

**可迁移性：RGB-D 模式高，单目模式中等。**

在 MonoGS RGB-D 模式下，可以直接比较渲染深度与真实深度。如果深度误差较大，则说明几何结构不准，可以增加或调整高斯点。

如果是单目模式，没有真实深度，则需要：

1. 使用预测深度；
2. 或只使用相对深度约束；
3. 或暂时不迁移 depth error densification。

预期影响：

| 指标 | 影响 |
|---|---|
| ATE RMSE | 可能明显下降 |
| PSNR | 提升 |
| SSIM | 提升 |
| LPIPS | 下降 |

原因：几何更准确后，位姿优化更稳定，渲染视角也更一致。

### 7.4 防遗忘正则化

**可迁移性：中等。**

MonoGS 已经有局部滑窗优化，但仍可能出现新帧优化破坏旧视角质量的问题。可以参考 HF-SLAM，为高斯点维护重要性权重，并在 mapping loss 中加入正则项。

可约束的参数包括：

| 参数 | 作用 |
|---|---|
| scaling | 防止高斯点形状剧烈变化 |
| color / SH features | 保持历史视角颜色一致性 |
| depth / position related term | 保持几何稳定性 |
| opacity | 防止旧区域被错误削弱 |

预期影响：

| 指标 | 影响 |
|---|---|
| PSNR | 提升，尤其是历史视角 |
| SSIM | 提升 |
| LPIPS | 下降 |
| ATE RMSE | 间接下降 |

缺点是需要额外维护高斯点统计量，工程复杂度高于前面的致密化策略。

### 7.5 LAB 去亮度跟踪损失

**可迁移性：高。**

HF-SLAM 在 tracking 中将 RGB 转为 LAB 空间，并去掉亮度通道，以减少光照和曝光变化影响。

MonoGS 已有曝光补偿，但 LAB loss 可以作为补充实验：

1. 原始 RGB tracking loss；
2. RGB + exposure compensation；
3. LAB 去亮度 tracking loss；
4. RGB 与 LAB 混合 tracking loss。

预期影响：

| 指标 | 影响 |
|---|---|
| ATE RMSE | 在曝光变化场景中可能下降 |
| PSNR | 不一定直接提升 |
| SSIM | 不一定直接提升 |
| LPIPS | 不一定直接提升 |

该模块主要服务 tracking，不是主要服务 rendering。

---

## 8. 推荐实现优先级

建议不要一次性复现完整 HF-SLAM，而是按风险从低到高逐步嵌入。

| 优先级 | 模块 | 原因 |
|---|---|---|
| 1 | RGB 渲染误差引导致密化 | 改动小，最可能提升 PSNR、SSIM、LPIPS |
| 2 | Opacity 空洞检测致密化 | 有助于补全地图空洞 |
| 3 | Depth 渲染误差引导致密化 | RGB-D 下收益明显，单目需深度先验 |
| 4 | LAB 去亮度 tracking loss | 可改善光照变化下的 ATE |
| 5 | 防遗忘正则化 | 有价值，但需要维护额外高斯统计量 |

推荐第一阶段先做：

```text
MonoGS baseline
+ RGB rendering-error densification
+ opacity hole densification
```

第二阶段再加入：

```text
+ depth rendering-error densification
+ regularized mapping
+ LAB tracking loss
```

---

## 9. 对指标的预期影响

| 改进点 | ATE RMSE | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| Opacity 空洞检测 | 间接下降 | 提升 | 提升 | 下降 |
| RGB error densification | 可能下降 | 明显提升 | 提升 | 明显下降 |
| Depth error densification | 下降 | 提升 | 提升 | 下降 |
| 防遗忘正则化 | 间接下降 | 提升 | 提升 | 下降 |
| LAB tracking loss | 下降，尤其在曝光变化场景 | 影响较小 | 影响较小 | 影响较小 |

总体判断：

1. **最可能提升建图指标的是 RGB / depth 渲染误差引导致密化。**
2. **最可能提升长期稳定性的是防遗忘正则化。**
3. **最可能提升复杂光照下跟踪的是 LAB 去亮度 tracking loss。**
4. **ATE RMSE 的提升通常依赖地图质量和几何稳定性，不一定由单个模块直接保证。**

---

## 10. 需要注意的风险

### 10.1 HF-SLAM 依赖 RGB-D

HF-SLAM 原文方法基于 RGB-D。若当前实验使用 MonoGS 单目模式，则 depth error densification 不能直接使用真实深度。

可选方案：

1. 只迁移 RGB 和 opacity 部分；
2. 使用单目深度估计网络提供伪深度；
3. 只在 RGB-D 实验中使用 depth 模块。

### 10.2 致密化过强会增加计算量

如果大量区域都被判定为高误差，系统可能添加过多高斯点，导致显存和速度压力增大。

需要设计：

1. 误差阈值；
2. 每帧最大新增点数；
3. 区域采样策略；
4. 与 pruning 的配合机制。

### 10.3 真实数据上的跟踪不一定稳定提升

HF-SLAM 论文中也指出，其在 TUM-RGBD 上的 tracking 不总是最优，原因包括：

1. 运动模糊；
2. 曝光变化；
3. 深度噪声；
4. 真实图像质量不稳定。

因此迁移时应同时报告 Replica 和 TUM-RGBD，避免只在合成数据上证明有效。

---

## 11. 推荐消融实验

| 实验编号 | 设置 | 目的 |
|---|---|---|
| E0 | MonoGS baseline | 原始对照 |
| E1 | baseline + opacity densification | 验证补洞是否有效 |
| E2 | baseline + RGB error densification | 验证颜色重建是否提升 |
| E3 | baseline + RGB + depth error densification | 验证几何约束是否提升跟踪 |
| E4 | baseline + regularized mapping | 验证是否缓解历史视角退化 |
| E5 | baseline + LAB tracking loss | 验证光照鲁棒性 |
| E6 | 所有模块组合 | 验证最终性能 |

评价指标：

| 任务 | 指标 |
|---|---|
| Tracking | ATE RMSE |
| Mapping / Rendering | PSNR、SSIM、LPIPS |

推荐数据集：

| 数据集 | 作用 |
|---|---|
| Replica | 验证重建质量上限 |
| TUM-RGBD | 验证真实场景鲁棒性 |

---

## 12. 可写入论文的方法描述

可以将改进方法表述为：

> 本文在 Gaussian Splatting SLAM baseline 的基础上，引入渲染误差引导的高斯致密化策略。不同于仅依据梯度或可见性进行高斯点增殖，本文同时利用 opacity、RGB 渲染误差和 depth 渲染误差判断地图中未充分建模的区域，从而在空洞区域补充新高斯点，并在重建质量较差的区域进行局部细化。进一步地，为缓解在线建图中高斯参数过度拟合最新帧而导致历史视角质量下降的问题，本文引入参数正则约束，使地图在持续更新过程中保持跨视角一致性。

---

## 13. 可写入论文的创新点

可以概括为三点：

1. **误差感知的高斯致密化**  
   根据 RGB / depth 渲染误差主动发现地图表达不足区域，而不是被动依赖固定采样或梯度阈值。

2. **面向在线建图的稳定性约束**  
   通过高斯参数重要性正则，减少连续优化过程中对历史视角的破坏。

3. **面向鲁棒跟踪的光照弱敏感损失**  
   在 tracking 中引入 LAB 去亮度约束，降低曝光变化对位姿估计的影响。

---

## 14. 最终建议

HF-SLAM 中最值得迁移到 MonoGS 的不是完整框架，而是以下三个模块：

1. **RGB / depth rendering-error densification**：优先用于提升 PSNR、SSIM、LPIPS；
2. **opacity-based hole filling**：用于减少地图空洞；
3. **regularized mapping**：用于提升持续建图稳定性，防止旧视角质量下降。

如果当前目标是快速提升论文实验结果，建议优先实现：

```text
RGB 渲染误差致密化
+ opacity 空洞检测
+ depth 渲染误差致密化（RGB-D 模式）
```

如果后续目标是增强方法创新性，再加入：

```text
防遗忘正则化
+ LAB 去亮度 tracking loss
```

总体上，HF-SLAM 的思想与 MonoGS baseline 兼容度较高，尤其适合作为 MonoGS 后端建图质量增强模块。其对建图指标 PSNR、SSIM、LPIPS 的提升预期较明确；对跟踪指标 ATE RMSE 的提升更依赖几何质量、深度可靠性和真实场景噪声控制，需要通过消融实验验证。
