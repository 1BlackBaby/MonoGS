# LSG-SLAM 与 Gaussian Splatting SLAM Baseline 对比分析

## 1. 一句话结论

LSG-SLAM 是面向大规模场景的双目 3D Gaussian Splatting SLAM。它最有价值的思想不是单纯换一个 3DGS 表达，而是把传统视觉 SLAM 中可靠的几何约束重新接入 3DGS：先用特征匹配、PnP 和点云配准给相机位姿一个更可靠的初值，再用 3DGS 渲染损失和特征对齐 warping 损失联合优化位姿；建图时用连续 GS 子地图解决大场景显存问题，并用回环和结构精修提高全局一致性与渲染质量。

对 MonoGS / Gaussian Splatting SLAM baseline 来说，最值得迁移的是三类模块：

1. 多模态位姿先验，用于降低 ATE RMSE。
2. 渲染损失 + 特征 warping 损失联合跟踪，用于降低大视角变化、弱纹理和重复纹理下的跟踪漂移。
3. 子地图、回环、球形到椭球形结构精修，用于提升大场景建图稳定性和 PSNR / SSIM / LPIPS。

## 2. LSG-SLAM 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | Large-Scale Gaussian Splatting SLAM |
| 作者 | Xin 等，2025 |
| 系统类型 | 在线视觉 SLAM + 3DGS 建图 |
| 目标场景 | 大规模室外/室内外混合场景，尤其是 KITTI 这类道路场景 |
| 输入 | 双目 RGB 图像 |
| 是否 RGB-D | 不是直接使用 RGB-D 传感器；它用双目视差估计深度，论文中使用 IGEV 预训练模型生成 stereo depth |
| 地图表示 | 3D Gaussian points，前期使用各向同性球形高斯，后期结构精修时转成各向异性椭球高斯 |
| 是否有回环 | 有。通过 place recognition 在不同 GS 子地图之间检测回环 |
| 跟踪指标 | ATE RMSE |
| 建图/渲染指标 | PSNR、SSIM、LPIPS |

通俗理解：MonoGS 更像“边走边用当前地图渲染图片，然后调相机姿态让渲染图对上真实图”；LSG-SLAM 在此基础上加了“先用特征点和几何关系猜一个更靠谱的位置，再用渲染和特征一起微调”，所以在大场景和大视角变化下更稳。

## 3. 跟踪策略

LSG-SLAM 的 tracking 目标是：固定 3DGS 地图，只优化当前帧相机位姿。

### 3.1 位姿初值：多模态先验

MonoGS / SplaTAM 主要依赖 uniform motion model，也就是假设相机延续上一帧运动趋势。这个假设在室内小场景、帧率较高时通常够用，但在 KITTI 这种大范围运动、低帧率或大视角变化中容易漂。

LSG-SLAM 改为多模态位姿先验：

1. 用 SuperPoint 提取局部特征。
2. 用 LightGlue 匹配当前帧和最近关键帧。
3. 通过 2D-3D 对应关系，用 PnP + RANSAC 求当前帧初始位姿。
4. 如果 2D-3D 内点太少，再使用点云配准作为 3D 几何 fallback。

通俗理解：不要只根据“上一帧怎么动”猜当前位置，而是先找图像里的稳定地标，再用几何方法算出相机大概在哪里。

### 3.2 位姿优化：渲染损失 + 特征 warping 损失

LSG-SLAM 仍然使用 3DGS 可微渲染来优化相机位姿。给定当前位姿和高斯地图，可以渲染出颜色图和深度图，再与真实图像和双目深度对齐：

- RGB rendering loss：渲染颜色和真实颜色的 L1 误差。
- Depth rendering loss：渲染深度和双目深度的 L1 误差。
- Silhouette mask：只在当前地图可靠覆盖的像素上计算跟踪损失。

论文认为，仅靠渲染损失在大场景里不够稳。原因是道路、天空、墙面这类区域外观相似，渲染图即使对错位置也可能看起来差不多，梯度下降容易陷入错误局部最优。

因此它加入 feature-alignment warping constraints：

1. 根据关键帧深度和当前位姿，把关键帧特征点投影到当前帧。
2. 让投影位置接近当前帧匹配到的特征点位置。
3. 不只约束特征点坐标，还约束特征描述子/特征图的一致性。
4. 对深度过大的不稳定点做 mask，减少远处错误深度的干扰。

总跟踪损失为：

```text
Tracking Loss = Rendering Loss + Feature Warping Loss
```

通俗理解：渲染损失负责“整张图像看起来像不像”，特征 warping 负责“关键地标是不是落在正确位置”。二者互补：渲染损失密集但可能被相似外观误导，特征约束稀疏但几何指向更明确。

## 4. 建图策略

LSG-SLAM 的 mapping 目标是：固定相机位姿，优化 3DGS 地图参数。

### 4.1 关键帧选择

LSG-SLAM 只用关键帧参与建图，并选两类关键帧：

1. 与最新关键帧重叠最高的关键帧，用于优化新加入的高斯点。
2. 从历史关键帧中随机采样的关键帧，用于防止旧地图被遗忘。

通俗理解：一部分训练样本看“最近这一片区域”，保证新地图建好；另一部分回看“以前见过的区域”，防止旧地图被新数据覆盖。

### 4.2 建图损失

建图损失包括：

- RGB rendering loss。
- Depth rendering loss。
- SSIM loss，用于保持图像结构和感知质量。

论文中 mapping loss 可以概括为：

```text
Mapping Loss = RGB/Depth Rendering Loss + SSIM Loss
```

与 tracking 不同，mapping 不使用 silhouette mask，因为建图阶段需要让高斯地图主动解释更多观测区域。

### 4.3 连续 GS 子地图

大规模场景不能把所有高斯点一直塞进一个全局地图，否则显存和优化成本会快速失控。LSG-SLAM 按轨迹长度把场景分成多个连续 GS submaps：

1. 每个子地图只维护一段轨迹附近的高斯点。
2. 当前子地图最后一帧同时作为下一个子地图第一帧，保证连续性。
3. 每个高斯点记录它由哪个关键帧创建，后续全局优化后可以根据关键帧变换调整高斯点位置。

通俗理解：不是一次性画一张超大地图，而是把长路线切成多段小地图，最后再通过回环和全局优化把它们拼准。

### 4.4 回环闭合

LSG-SLAM 有明确 loop closure 模块：

1. 对每个关键帧提取 global feature。
2. 在不同 GS 子地图之间做 place recognition，找相似地点。
3. 用局部特征匹配筛掉假回环。
4. 对回环帧之间的相对位姿，再用渲染损失和 feature warping 损失优化。
5. 构建 keyframe pose graph，优化相邻边和回环边。
6. 根据关键帧位姿变化同步调整所有关联高斯点。

关键点是：回环时不需要存储所有原始图像，可以用 GS 子地图即时渲染 color/depth 来辅助匹配和位姿优化。

### 4.5 结构精修

LSG-SLAM 前期用各向同性球形高斯，原因是球形高斯更稳定，不容易在早期优化中形成漂浮物。全局位姿和点云调整完成后，再进行 structure refinement：

1. 把每个球形高斯的半径转换成三轴 scale，也就是椭球高斯。
2. 随机采样关键帧继续优化高斯属性。
3. 加入 scale regularization，让椭球更像物体表面的薄片，从而更好表达道路、墙面、天空等表面。

通俗理解：先用“圆点”稳稳地搭出地图骨架，等相机轨迹和地图整体对齐后，再把圆点拉伸成更贴合表面的“小贴片”，提升细节和渲染质量。

## 5. 使用的数据集与实验设置

| 数据集 | 类型 | 用途 | 特点 |
|---|---|---|---|
| EuRoC MAV | 双目序列，室内/室内外混合 | 跟踪与渲染评估 | 有剧烈视角变化和光照变化 |
| KITTI | 双目道路场景 | 大规模跟踪与渲染评估 | 包含城市、乡村、高速等长轨迹室外场景 |

实验设置要点：

1. 输入是双目 RGB，深度由双目视差估计得到。
2. 论文使用 IGEV 预训练模型估计 stereo disparity。
3. 为验证大视角变化鲁棒性，LSG-SLAM 降低输入频率：EuRoC 间隔 5 帧，KITTI 间隔 2 帧。
4. tracking 每帧优化 50 次。
5. mapping 和 loop optimization 每个关键帧优化 100 次。
6. 每张图最多提取 512 个 SuperPoint keypoints。

注意：论文为了公平比较，把 SplaTAM 和 MonoGS 的深度获取方式也改成了 stereo disparity。这说明 LSG-SLAM 的实验主线是 stereo SLAM，而不是纯 monocular SLAM。

## 6. 主要实验结果

### 6.1 EuRoC 跟踪结果

| 方法 | 是否回环 | ATE RMSE 平均值 |
|---|---:|---:|
| SplaTAM | 否 | 1.61 m |
| MonoGS | 否 | 0.77 m |
| LSG-SLAM | 否 | 0.17 m |
| ORB-SLAM3 | 是 | 0.03 m，V203 跟踪失败 |
| LSG-SLAM | 是 | 0.06 m |

结论：不加回环时，LSG-SLAM 已明显优于 MonoGS；加回环后，轨迹精度接近传统强基线 ORB-SLAM3，并且在挑战序列中重建成功率更高。

### 6.2 EuRoC 渲染结果

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| MonoGS，无 color refine | 16.50 | 0.65 | 0.46 |
| MonoGS，有 color refine | 22.36 | 0.80 | 0.30 |
| LSG-SLAM，无 structure refinement | 26.02 | 0.95 | 0.07 |
| LSG-SLAM，有 structure refinement | 31.38 | 0.98 | 0.05 |

结论：LSG-SLAM 的渲染质量提升来自两部分：更准确的位姿减少地图错位，结构精修进一步提升纹理和表面细节。

### 6.3 KITTI 跟踪结果

| 方法 | 是否回环 | ATE RMSE 平均值 |
|---|---:|---:|
| ORB-SLAM3 | 否 | 26.48 m |
| DROID-SLAM | 否 | 18.56 m |
| LSG-SLAM | 否 | 10.01 m |
| ORB-SLAM3 | 是 | 9.46 m |
| LSG-SLAM | 是 | 3.85 m |

结论：在大规模道路场景中，子地图 + 回环对 ATE RMSE 很关键。LSG-SLAM 加回环后明显优于 ORB-SLAM3。

### 6.4 KITTI 渲染结果

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| 3DGS，使用 GT pose/depth 初始化 | 25.15 | 0.94 | 0.13 |
| LSG-SLAM，无 structure refinement | 21.10 | 0.87 | 0.19 |
| LSG-SLAM，有 structure refinement | 26.58 | 0.97 | 0.07 |

结论：结构精修后，LSG-SLAM 在 KITTI 上甚至超过使用 GT pose/depth 初始化的原始 3DGS 渲染质量，说明“先球形稳定建图，再椭球精修”的策略很有效。

## 7. 与 MonoGS Baseline 的核心差异

| 维度 | MonoGS / Gaussian Splatting SLAM baseline | LSG-SLAM |
|---|---|---|
| 输入 | 支持 monocular 和 RGB-D；具体配置决定是否有深度 | 双目 RGB，深度由 stereo disparity 得到 |
| 位姿初值 | 主要依赖 uniform motion model | SuperPoint + LightGlue + PnP/RANSAC，必要时点云配准 |
| 跟踪损失 | 渲染 RGB/Depth loss，优化当前帧位姿和曝光 | 渲染 RGB/Depth loss + feature point / feature map warping loss |
| 建图范围 | 维护滑窗，局部 BA，优化关键帧和高斯 | 连续 GS 子地图，适合大规模轨迹 |
| 回环 | baseline 总结中没有显式 loop closure | 有 place recognition、回环相对位姿优化和 pose graph |
| 高斯形状 | MonoGS 中强调 isotropic loss，抑制极端拉伸 | 前期球形稳定，后期转椭球并加 scale regularization 精修 |
| 大规模能力 | 单全局地图容易受显存和累计漂移限制 | 子地图 + 回环更适合 KITTI 这类长序列 |

## 8. 能否嵌入 Baseline 来提升指标

可以，但建议分层迁移，不建议一次性照搬完整 LSG-SLAM。原因是 LSG-SLAM 是完整 stereo 大规模系统，依赖 SuperPoint、LightGlue、IGEV、place recognition、点云配准和 pose graph；全部迁入会明显增加工程复杂度和运行成本。更合理的路线是先迁移对指标最直接的模块。

### 8.1 优先级最高：多模态位姿初值

迁移位置：

- `utils/slam_frontend.py` 的 tracking 初始化阶段。
- 当前 baseline 中相当于在优化 `cam_rot_delta` 和 `cam_trans_delta` 之前，为当前帧设置更可靠的初始 `R, T`。

改造思路：

1. 对最近关键帧和当前帧提取 SuperPoint 特征。
2. 用 LightGlue 建立匹配。
3. 从关键帧深度或高斯渲染深度得到 2D-3D 对应。
4. 用 PnP + RANSAC 求初始位姿。
5. 若内点不足，则退回原本的 motion model；如果有 RGB-D / stereo depth，可进一步接入点云配准。

预期影响：

| 指标 | 影响 |
|---|---|
| ATE RMSE | 最直接，尤其是低帧率、大视角变化、快速运动场景 |
| PSNR / SSIM / LPIPS | 间接提升。位姿更准后，高斯地图对齐更好，重影和错位减少 |

风险：

1. 弱纹理、运动模糊或动态物体多时，特征匹配可能失败。
2. 需要维护 fallback，否则错误 PnP 初值会把 tracking 拉偏。
3. 如果 baseline 运行在 monocular 模式，没有可靠尺度深度，PnP 的 3D 点质量会成为瓶颈。

### 8.2 优先级最高：Feature Warping Tracking Loss

迁移位置：

- `utils/slam_utils.py` 的 `get_loss_tracking`。
- `utils/slam_frontend.py` 中 tracking loop 的 loss 组合处。

改造思路：

1. 保存当前帧和参考关键帧之间的 inlier matches。
2. 用参考关键帧深度把特征点反投影到 3D。
3. 根据当前待优化位姿投影到当前帧。
4. 计算投影点与匹配点的 2D 距离。
5. 如果有特征图，则增加 feature map distance。
6. 将该 loss 与原来的 RGB/Depth rendering loss 加权相加。

预期影响：

| 指标 | 影响 |
|---|---|
| ATE RMSE | 明显有潜力下降，因为优化过程不只看颜色相似，还看几何对应 |
| PSNR / SSIM / LPIPS | 通过减少位姿漂移间接提升 |

这部分比只加 PnP 更重要的一点是：PnP 只给初值，warping loss 会在后续梯度优化过程中持续约束相机姿态。

### 8.3 推荐：SSIM Mapping Loss

迁移位置：

- `utils/slam_utils.py` 的 `get_loss_mapping_rgb` / `get_loss_mapping_rgbd`。

改造思路：

1. 在 mapping loss 中保留 RGB L1 和 depth L1。
2. 额外加入 SSIM loss。
3. 对 RGB-D 模式可保持原有 RGB/depth 权重，再给 SSIM 一个较小权重。

预期影响：

| 指标 | 影响 |
|---|---|
| PSNR | 可能提升，但不一定总是最大 |
| SSIM | 通常更直接提升 |
| LPIPS | 有机会下降，因为结构更自然 |
| ATE RMSE | 基本不直接影响，除非地图质量提升后反哺 tracking |

实现成本较低，适合作为第一阶段建图改造。

### 8.4 推荐：球形建图 + 椭球结构精修

迁移位置：

- `gaussian_model.py` 中高斯 scale/rotation 参数管理。
- `utils/slam_backend.py` 的 map refinement 阶段。

改造思路：

1. 在线 tracking/mapping 阶段继续保持高斯尽量各向同性，保证稳定。
2. 在轨迹优化完成或某段子地图完成后，开启离线/半离线 refinement。
3. 将球形高斯扩展为三轴 scale 的椭球高斯。
4. 用随机关键帧继续优化颜色、位置、opacity、scale、rotation。
5. 加 scale regularization，让椭球趋向物体表面的薄片。

预期影响：

| 指标 | 影响 |
|---|---|
| PSNR | 高概率提升，尤其是纹理和表面细节丰富场景 |
| SSIM | 高概率提升 |
| LPIPS | 高概率下降 |
| ATE RMSE | 不直接提升，因为它主要发生在后处理或建图精修阶段 |

注意：这更适合“提升最终重建质量”，不适合放进每帧实时 tracking。

### 8.5 中高优先级：连续 GS 子地图

迁移位置：

- `utils/slam_backend.py` 的地图管理。
- `slam.py` 的前后端通信协议。
- 可能需要新增 submap manager。

改造思路：

1. 按轨迹长度、关键帧数量或高斯数量切分子地图。
2. 每个子地图维护自己的高斯点、关键帧列表和优化器。
3. 相邻子地图共享边界关键帧。
4. tracking 时只加载当前相关子地图，减少显存压力。
5. 全局评估前再做子地图对齐和合并。

预期影响：

| 指标 | 影响 |
|---|---|
| ATE RMSE | 对长序列有帮助，主要通过降低累计漂移和支持回环优化 |
| PSNR / SSIM / LPIPS | 对大场景有帮助，避免单地图过大导致优化质量下降 |

如果你的实验主要是小型室内 RGB-D 数据集，子地图收益可能不明显；如果要做 KITTI、TUM outdoor 或自采长轨迹，子地图非常关键。

### 8.6 高收益但高成本：回环闭合

迁移位置：

- 需要新增 place recognition、loop candidate verification、pose graph optimization。
- 需要让 Gaussian points 记录所属关键帧，方便全局位姿更新后同步调整地图。

改造思路：

1. 为关键帧提取 global descriptor。
2. 在非相邻关键帧或不同子地图中检索候选回环。
3. 用局部特征匹配验证。
4. 用 3DGS 渲染 color/depth + feature warping 优化回环相对位姿。
5. 做 pose graph optimization。
6. 根据关键帧位姿变化更新高斯点。

预期影响：

| 指标 | 影响 |
|---|---|
| ATE RMSE | 对长序列最明显，尤其是有回到旧区域的轨迹 |
| PSNR / SSIM / LPIPS | 全局地图对齐更好后，重影和断裂减少 |

这是论文级贡献点，但工程量大，建议放在多模态 tracking 稳定后再做。

## 9. 建议的论文创新路线

如果你的 baseline 是 Gaussian Splatting SLAM，可以把论文主线设计为：

> 面向大视角变化和大规模场景的几何增强 Gaussian Splatting SLAM。

建议分三阶段组织方法：

| 模块 | 核心思想 | 主攻指标 | 难度 |
|---|---|---|---|
| Geometry-guided Pose Initialization | 用特征匹配、PnP 和可选点云配准替代单纯 motion model | ATE RMSE | 中 |
| Feature-warping Tracking | 在渲染损失之外加入稀疏但可靠的几何 warping 约束 | ATE RMSE | 中 |
| Structure-aware Mapping Refinement | 在线阶段稳定建图，后处理阶段椭球精修和 scale regularization | PSNR / SSIM / LPIPS | 中 |
| Submap + Loop Closure | 面向长序列的大规模地图管理和全局一致性 | ATE RMSE、渲染质量 | 高 |

最稳妥的实验路线：

1. 先实现多模态位姿初值，验证 ATE RMSE 是否下降。
2. 再加入 feature warping tracking loss，验证大视角变化序列是否进一步稳定。
3. 加入 SSIM mapping loss 和结构精修，验证 PSNR / SSIM / LPIPS。
4. 如果数据集包含长轨迹，再做子地图和回环。

## 10. 最终建议

LSG-SLAM 中最适合嵌入 baseline 的点是“几何增强跟踪”，也就是 SuperPoint/LightGlue + PnP 初值和 feature warping tracking loss。这两项直接针对 MonoGS 依赖渲染损失和 motion model 的弱点，最可能降低 ATE RMSE。

建图指标方面，优先考虑 SSIM mapping loss 和球形到椭球形的 structure refinement。它们对 PSNR、SSIM、LPIPS 的提升路径清晰，而且不需要彻底重写前端。子地图和回环更适合长序列论文实验，是高价值但高工程量的后续增强。

如果只做一个最小可行论文版本，建议主线写成：

```text
Geometry-guided Tracking + Structure-refined Gaussian Mapping
```

这样既吸收了 LSG-SLAM 的核心优势，又不会把 baseline 一次性改成完全不同的大系统。
