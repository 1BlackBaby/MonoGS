# FlashGS 与 MonoGS Baseline 对比分析

# **不合适嵌入**

本文整理 Feng 等人在 2025 年提出的 **FlashGS: Efficient 3D Gaussian Splatting for Large-scale and High-resolution Rendering**，并分析其是否适合嵌入当前 baseline **Gaussian Splatting SLAM / MonoGS**，用于改善跟踪指标 **ATE RMSE** 和建图指标 **PSNR、SSIM、LPIPS**。

---

## 1. 一句话总结

FlashGS 不是一个 SLAM 系统，而是一个面向 3D Gaussian Splatting 的 **高效 CUDA 渲染器**。

它不负责相机跟踪，也不负责地图优化，而是让已有 3D Gaussian 场景渲染得更快。

通俗理解：

| 系统 | 主要解决的问题 |
|---|---|
| MonoGS | 相机在哪里、地图怎么建 |
| FlashGS | 已有高斯地图怎么更快渲染 |

因此，FlashGS 不能直接替代 MonoGS 的 tracking 或 mapping，但可以作为底层 rasterizer 加速模块，为跟踪和建图释放更多计算预算。

---

## 2. MonoGS Baseline 策略

### 2.1 Tracking 策略

MonoGS 的前端负责 Tracking，核心流程是：

1. 固定当前 3D Gaussian 地图；
2. 根据当前相机位姿渲染 RGB 图像或深度图；
3. 将渲染结果与真实输入帧比较；
4. 通过反向传播优化相机位姿；
5. 输出当前帧相机姿态，并判断是否插入关键帧。

MonoGS 的 tracking 本质上依赖反复渲染。如果渲染更快，同样时间内就可能进行更多位姿优化迭代。

### 2.2 Mapping 策略

MonoGS 的后端负责 Mapping，核心流程是：

1. 维护局部关键帧滑窗；
2. 同时优化关键帧位姿和可见高斯点参数；
3. 使用 RGB loss、depth loss、SSIM loss 等监督渲染结果；
4. 通过 densification 增加高斯点；
5. 通过 pruning 删除冗余或异常高斯点；
6. 使用 isotropic loss 约束高斯形状稳定。

MonoGS 的 mapping 同样依赖大量 differentiable rendering。因此，渲染器效率会直接影响系统速度和可承载的优化迭代数。

---

## 3. FlashGS 的核心策略

FlashGS 的研究重点是 **3DGS rasterization 加速**，不是 SLAM。

### 3.1 是否包含 Tracking / Mapping

| 模块 | FlashGS 是否包含 | 说明 |
|---|---|---|
| Tracking | 否 | 不估计相机位姿，不计算 ATE |
| Mapping | 否 | 不优化高斯点，不进行 SLAM 建图 |
| Rendering | 是 | 优化 3DGS 前向渲染速度 |

所以分析 FlashGS 时，不能把它理解为“新的 SLAM 策略”，而应理解为“可以服务 SLAM 的底层高效渲染模块”。

### 3.2 主要技术点

FlashGS 针对原始 3DGS tile-based rasterization 中的冗余计算和负载不均衡进行优化。

| 技术点 | 含义 | 作用 |
|---|---|---|
| Opacity-aware AABB | 根据高斯不透明度确定更紧的有效范围 | 减少无效 tile 候选 |
| Precise Gaussian-tile intersection | 精确判断投影椭圆是否真的覆盖 tile | 删除冗余 Gaussian-tile pair |
| Size-aware adaptive scheduling | 根据高斯大小动态分配 GPU 线程 | 缓解大小高斯导致的负载不均衡 |
| Pipelined tile rendering | 将内存读取和 alpha blending 流水线化 | 降低数据依赖造成的等待 |

简单理解：原始 3DGS 渲染时会检查很多“可能覆盖 tile、但实际没有贡献”的高斯；FlashGS 用更精确的判断和更合理的 GPU 调度减少这些无效计算。

---

## 4. 输入类型

FlashGS 的输入不是 SLAM 中常见的 RGB 或 RGB-D 视频流。

它的输入更准确地说是：

| 输入 | 说明 |
|---|---|
| 已训练好的 3D Gaussian 模型 | 场景已经由高斯点表示 |
| 给定相机位姿 | 用于指定从哪个视角渲染 |
| 渲染分辨率 | 如 1080p、4K 或更高 |

因此：

| 方法 | 输入类型 |
|---|---|
| MonoGS | Monocular / RGB-D / Stereo 序列 |
| FlashGS | Gaussian map + camera pose |

FlashGS 不从 RGB 或 RGB-D 数据中估计相机位姿，也不从输入序列中增量建图。

---

## 5. 使用的数据集

FlashGS 使用的是 3DGS / novel view rendering 数据集，而不是 SLAM 轨迹数据集。

| 数据集 | 类型 | 用途 |
|---|---|---|
| Truck | Tanks and Temples 室外场景 | 常规 3DGS 渲染评估 |
| Train | Tanks and Temples 室外场景 | 常规 3DGS 渲染评估 |
| Playroom | DeepBlending 室内场景 | 室内高质量渲染评估 |
| DrJohnson | DeepBlending 室内场景 | 室内高质量渲染评估 |
| MatrixCity | 大规模城市级场景 | 大规模、高分辨率渲染评估 |
| Rubble | 大规模高分辨率场景 | 极端渲染压力测试 |

这些数据集主要用于验证渲染速度和图像质量，不用于评估 SLAM 轨迹精度。

---

## 6. 评价指标

FlashGS 的主要指标是渲染速度和渲染质量。

| 指标 | 含义 | 越大/越小 |
|---|---|---|
| FPS | 平均渲染帧率 | 越大越好 |
| MinFPS | 最慢帧帧率 | 越大越好 |
| PSNR | 渲染图像与真值图像的像素相似度 | 越大越好 |
| SSIM | 结构相似度 | 越大越好 |
| LPIPS | 感知差异 | 越小越好 |

FlashGS 不报告：

| 指标 | 原因 |
|---|---|
| ATE RMSE | 不做相机跟踪 |
| SLAM 成功率 | 不运行 SLAM pipeline |
| 建图误差 | 不优化地图 |

论文结果表明，FlashGS 在保持接近原始 3DGS 渲染质量的同时，大幅提升 FPS，尤其适合大规模和高分辨率场景。

---

## 7. 与 MonoGS 的核心差异

| 对比项 | MonoGS | FlashGS |
|---|---|---|
| 方法类型 | 3DGS SLAM 系统 | 3DGS 高效渲染器 |
| 是否估计位姿 | 是 | 否 |
| 是否在线建图 | 是 | 否 |
| 是否优化高斯点 | 是 | 否 |
| 是否需要 RGB / RGB-D 输入 | 是 | 否 |
| 主要指标 | ATE、PSNR、SSIM、LPIPS | FPS、PSNR、SSIM、LPIPS |
| 主要贡献 | Tracking + Mapping | Rasterization 加速 |

FlashGS 的价值不在于提出新的 tracking loss 或 mapping loss，而在于让 MonoGS 中反复调用的渲染过程更快。

---

## 8. 能否嵌入 MonoGS

结论：**可以嵌入，但主要提升速度，不直接提升精度。**

MonoGS 中多个环节依赖渲染：

1. Tracking：渲染当前地图，与当前帧比较，优化相机位姿；
2. Mapping：渲染关键帧视角，与真实图像和深度比较，优化高斯点；
3. Evaluation：渲染测试视角，计算 PSNR、SSIM、LPIPS；
4. GUI：实时显示重建结果。

因此，如果 FlashGS 能兼容 MonoGS 的渲染接口，就可以降低这些环节的渲染耗时。

---

## 9. 对指标的预期影响

### 9.1 对 ATE RMSE 的影响

FlashGS 不直接优化相机位姿，因此不会直接降低 ATE RMSE。

但它可能通过以下方式间接改善 ATE：

1. 渲染更快，同样时间内可以进行更多 tracking 迭代；
2. tracking 可以使用更高分辨率或更多像素；
3. 实时系统中前端等待后端的时间减少，地图更新更及时；
4. 大场景下渲染瓶颈减轻，位姿优化更稳定。

因此，对 ATE RMSE 的判断是：

| 影响 | 结论 |
|---|---|
| 直接影响 | 很弱 |
| 间接影响 | 可能有帮助 |
| 是否适合作为主要 ATE 创新点 | 不适合 |

### 9.2 对 PSNR、SSIM、LPIPS 的影响

FlashGS 的目标是更快地渲染同一个高斯模型，而不是改变地图参数。

所以如果只替换渲染器：

| 指标 | 预期变化 |
|---|---|
| PSNR | 基本不变 |
| SSIM | 基本不变 |
| LPIPS | 基本不变 |

但如果利用节省出的时间做更多 mapping 优化，则可能间接提升：

| 间接方式 | 可能收益 |
|---|---|
| 增加 mapping 迭代次数 | 提升 PSNR、SSIM，降低 LPIPS |
| 使用更高分辨率监督 | 提升细节重建 |
| 更频繁后端更新 | 改善地图及时性 |
| 支持更大规模场景 | 减少渲染瓶颈 |

因此，对建图指标的判断是：

| 影响 | 结论 |
|---|---|
| 直接影响 | 较弱 |
| 间接影响 | 有潜力 |
| 是否适合作为主要建图质量创新点 | 不适合单独使用 |

---

## 10. 结合 FlashGS 源码对三点风险的解答

源码位置：

```text
D:\Postgraduate\Research\3DGS\Paper\FlashGS Efficient 3D Gaussian Splatting for Large scale and High resolution  Rendering\FlashGS-main
```

重点检查文件：

| 文件 | 作用 |
|---|---|
| `example.py` | Python 使用示例，封装 `Rasterizer.forward()` |
| `csrc/pybind.cpp` | PyTorch C++ 扩展绑定 |
| `csrc/ops.h` | C++ / CUDA 函数声明 |
| `csrc/cuda_rasterizer/preprocess.cu` | 高斯预处理、投影、tile key-value 生成 |
| `csrc/cuda_rasterizer/render.cu` | tile-based 前向颜色渲染 |

源码结论：**当前 FlashGS-main 不能直接替换 MonoGS 的 tracking / mapping rasterizer。**  
它可以作为前向 RGB 渲染加速器使用，但若要进入 MonoGS 的核心优化循环，需要补充 backward、pose gradient、depth / opacity / visibility 等接口。

### 10.1 风险一：可微性是否支持

**结论：当前源码没有提供可微反向传播接口。**

依据如下：

1. `csrc/pybind.cpp` 只绑定了：

```text
loadPly
preprocess
sort_gaussian
get_sort_buffer_size
render_16x16
render_32x16
render_32x32
```

2. 源码中没有发现 `backward`、`grad`、`torch::autograd::Function` 形式的反向传播封装。
3. `example.py` 中的 `Rasterizer.forward()` 只是按顺序调用 `preprocess -> sort_gaussian -> render_16x16`，没有构建 PyTorch autograd 计算图。
4. `render.cu` 最终写入的是 `uchar3* out_color`，Python 侧对应 `torch.int8` 输出，不适合直接作为 L1 / SSIM loss 的可微输入。

对 MonoGS 的影响：

| MonoGS 环节 | 是否能直接替换 | 原因 |
|---|---|---|
| Tracking loss | 不能 | 需要对相机位姿反向传播 |
| Mapping loss | 不能 | 需要对高斯位置、尺度、旋转、颜色、不透明度反向传播 |
| GUI 显示 | 可以考虑 | 只需要前向渲染 |
| Evaluation 渲染 | 可以考虑 | 只需要渲染图像计算指标 |

简明判断：  
**当前 FlashGS 是 forward renderer，不是 differentiable renderer。**

### 10.2 风险二：Pose gradient 是否支持

**结论：当前源码不支持 MonoGS tracking 所需的 camera pose gradient。**

MonoGS 当前渲染器调用中显式传入：

```text
theta = viewpoint_camera.cam_rot_delta
rho   = viewpoint_camera.cam_trans_delta
```

这说明 MonoGS tracking 会直接优化相机旋转和平移增量。也就是说，渲染器必须能把图像误差反传到相机位姿。

FlashGS 源码中相机参数的处理方式不同：

1. `example.py` 的 `Camera` 从 `cameras.json` 读取固定 `position` 和 `rotation`；
2. `pybind.cpp` 将 `position` 和 `rotation` 取出为普通 float 数据；
3. `preprocess.cu` 中用这些 float 构造 `view_matrix` 和 `proj_matrix`；
4. CUDA kernel 只做前向投影和排序，没有返回任何关于相机位姿的梯度。

因此，FlashGS 当前只能使用给定相机位姿渲染，不能根据 loss 自动修正相机位姿。

对 ATE RMSE 的影响：

| 使用方式 | 对 ATE RMSE 的影响 |
|---|---|
| 直接替换 MonoGS tracking renderer | 不可行 |
| 用于 GUI / evaluation | 不影响 ATE |
| 二次开发 pose gradient 后用于 tracking | 可能通过更多迭代间接降低 ATE |

简明判断：  
**当前 FlashGS 不能用于 MonoGS 前端位姿优化。**

### 10.3 风险三：输出接口是否兼容 MonoGS

**结论：当前输出接口与 MonoGS 核心需求不兼容，只满足 RGB 前向显示的一部分需求。**

MonoGS 当前 `gaussian_splatting/gaussian_renderer/__init__.py` 的 `render()` 返回：

| MonoGS 输出 | 用途 |
|---|---|
| `render` | RGB loss、PSNR、SSIM、LPIPS |
| `viewspace_points` | 屏幕空间梯度，用于 densification |
| `visibility_filter` | 判断可见高斯 |
| `radii` | 判断高斯屏幕半径，用于 densification / pruning |
| `depth` | RGB-D tracking、mapping depth loss |
| `opacity` | mask、tracking、可视化、建图判断 |
| `n_touched` | 可见性和统计信息 |

FlashGS 当前 `example.py` 和 C++ 接口表现为：

| FlashGS 输出 | 源码表现 | 是否满足 MonoGS |
|---|---|---|
| RGB image | `out_color`，类型为 `uchar3 / int8` | 只适合显示，不适合可微 loss |
| Depth image | `rgb_depth.w` 内部保存了 view-space depth，但 `render.cu` 未输出 depth map | 不满足 |
| Opacity map | 内部有 `conic_opacity`，但未输出 per-pixel opacity | 不满足 |
| Visibility filter | key-value 和 ranges 内部使用，未按 MonoGS 格式返回 | 不满足 |
| Radii | 未返回 | 不满足 |
| Viewspace points gradient | 未返回可微 tensor | 不满足 |

特别注意：`preprocess.cu` 中确实将 `rgb_depth[idx] = {r, g, b, p_view.z}`，说明每个高斯缓存了深度；但 `render.cu` 的 `get_gaussian_features()` 只取 RGB 和 conic / opacity 参与颜色混合，最终只调用 `write_color()` 写 RGB，没有生成 MonoGS 需要的 per-pixel depth / opacity。

简明判断：  
**当前 FlashGS 的接口更接近“高速 RGB 图像渲染器”，而不是 MonoGS 所需的“可微 RGB-D-Opacity-Visibility rasterizer”。**

### 10.4 三点风险的最终判定

| 风险 | 源码确认结果 | 对 MonoGS 的结论 |
|---|---|---|
| 可微性风险 | 未提供 backward / autograd | 不能直接用于 tracking / mapping loss |
| Pose gradient 风险 | 相机位姿只作为普通前向参数使用 | 不能直接优化 ATE |
| 输出接口兼容风险 | 只输出 RGB `uchar3`，不输出 MonoGS 所需 depth / opacity / radii 等 | 不能直接替换核心 renderer |

### 10.5 可行嵌入边界

当前源码条件下，FlashGS 可以优先用于：

1. **GUI 前向显示加速**：只要将 MonoGS 的高斯参数转换为 FlashGS 需要的 `position / shs / opacity / cov3d` 格式；
2. **Evaluation 前向渲染加速**：适合只计算 RGB 渲染速度或视觉展示；
3. **大场景离线可视化**：适合高分辨率快速渲染；
4. **作为可微 rasterizer 优化的参考实现**：借鉴 precise intersection、size-aware scheduling、pipelined rendering。

当前源码条件下，不建议直接用于：

1. MonoGS tracking loss；
2. MonoGS RGB-D tracking；
3. MonoGS mapping loss；
4. densification / pruning 的核心统计；
5. 任何需要反向传播的训练或位姿优化环节。

### 10.6 如果要真正接入 MonoGS 核心，需要补充什么

若后续要把 FlashGS 变成 MonoGS 可用的核心 renderer，至少需要二次开发：

| 需要补充 | 目的 |
|---|---|
| float RGB 输出 | 替代当前 `uchar3 / int8` 输出，支持 loss 计算 |
| depth map 输出 | 支持 RGB-D tracking 和 mapping depth loss |
| opacity map 输出 | 支持 mask、可视化和建图判断 |
| radii / visibility 输出 | 支持 densification 和 pruning |
| Gaussian 参数 backward | 支持优化位置、尺度、旋转、颜色、不透明度 |
| camera pose backward | 支持优化 `cam_rot_delta` 和 `cam_trans_delta` |
| PyTorch autograd 封装 | 让 FlashGS 能进入 MonoGS 的 loss.backward() 流程 |

因此，论文中更稳妥的表述应为：

> 基于 FlashGS 源码分析，其当前实现主要面向高效前向 RGB 渲染，尚不具备 MonoGS tracking / mapping 所需的可微反向传播、位姿梯度以及 RGB-D-Opacity-Visibility 输出接口。因此，FlashGS 更适合作为 GUI、evaluation 和大规模可视化的渲染加速模块；若要用于 SLAM 核心优化，需要在其 CUDA rasterizer 上扩展 backward、depth、opacity、radii 和 pose gradient 支持。

---

## 11. 推荐迁移优先级

建议不要一开始就替换 MonoGS 的核心 differentiable renderer，而是分阶段迁移。

| 优先级 | 迁移内容 | 原因 |
|---|---|---|
| 1 | GUI / evaluation 前向渲染加速 | 风险最低，不影响优化正确性 |
| 2 | 借鉴 precise intersection 思想优化现有 rasterizer | 可减少冗余计算 |
| 3 | 扩展 float RGB / depth / opacity 输出 | 补齐 MonoGS 基础渲染接口 |
| 4 | 二次开发 Gaussian gradient | 之后才可能进入 mapping loss |
| 5 | 二次开发 pose gradient | 之后才可能进入 tracking loss |

推荐第一阶段目标：

```text
不改变 MonoGS 原有 tracking / mapping loss
只将 FlashGS 用于 GUI 或 evaluation rendering
验证图像质量是否一致、速度是否提升
```

第二阶段再考虑：

```text
将 FlashGS 的 precise intersection / scheduling 思想迁移到可微 rasterizer
保持 MonoGS 需要的 backward、depth、visibility 输出
```

源码确认后的实际建议：

```text
当前 FlashGS-main 不直接用于 MonoGS tracking / mapping
优先作为前向显示和 evaluation 加速参考
若要进入核心优化，需要先补 backward 和输出接口
```

---

## 12. 适合写入论文的方法定位

可以这样描述 FlashGS 对本文工作的启发：

> FlashGS 并非完整 SLAM 系统，而是面向大规模、高分辨率 3D Gaussian Splatting 的高效 rasterization 框架。其通过精确 Gaussian-tile 相交判断、自适应 GPU 任务调度和流水线式 tile rendering，显著减少渲染过程中的冗余计算。该思想可作为 Gaussian Splatting SLAM 的底层渲染加速模块，用于降低 tracking 和 mapping 中反复 differentiable rendering 的计算开销。

如果要强调与 MonoGS 的结合，可以写为：

> 在 Gaussian Splatting SLAM 中，前端位姿跟踪和后端局部建图都需要频繁调用 differentiable rendering。受 FlashGS 启发，本文认为减少 Gaussian-tile 冗余交互和改善 GPU 任务负载均衡，可以为 SLAM 系统释放更多优化预算，从而支持更多 tracking / mapping 迭代或更高分辨率监督。

---

## 13. 可设计的实验

如果后续要验证 FlashGS 思想对 MonoGS 的帮助，建议实验不要只看精度，也要看速度。

| 实验 | 目的 |
|---|---|
| MonoGS baseline | 原始对照 |
| MonoGS + FlashGS 前向渲染 | 测 GUI / evaluation 加速 |
| MonoGS + 加速 renderer + 相同迭代数 | 验证图像质量是否保持 |
| MonoGS + FlashGS 思想改造后的可微 renderer | 验证 backward、depth、opacity、visibility 是否正确 |
| MonoGS + 可微加速 renderer + 更多 tracking 迭代 | 验证 ATE 是否下降 |
| MonoGS + 可微加速 renderer + 更多 mapping 迭代 | 验证 PSNR、SSIM、LPIPS 是否提升 |

建议同时报告：

| 类别 | 指标 |
|---|---|
| Tracking | ATE RMSE |
| Mapping / Rendering | PSNR、SSIM、LPIPS |
| Efficiency | FPS、每帧 tracking 时间、每轮 mapping 时间、显存 |

---

## 14. 最终建议

FlashGS 适合作为 MonoGS 的 **系统效率增强点**，但不适合作为主要精度创新点。

如果论文目标是提升：

| 目标 | FlashGS 适配度 |
|---|---|
| ATE RMSE | 间接帮助，不能单独保证 |
| PSNR / SSIM / LPIPS | 间接帮助，依赖更多优化预算 |
| 运行速度 | 高度适合 |
| 实时性 | 高度适合 |
| 大规模场景能力 | 高度适合 |

最合理的论文定位是：

1. FlashGS 作为底层渲染加速启发；
2. 真正提升精度的部分仍应来自 tracking loss、mapping loss、densification、depth prior、uncertainty weighting 或 regularization；
3. 当前 FlashGS-main 可优先用于前向显示、evaluation 和大规模可视化；
4. 若要让 FlashGS 负责降低 tracking / mapping 的核心优化成本，需要先完成可微接口扩展。

最终结论：

> 结合 FlashGS 源码可确认，当前 FlashGS-main 主要是高效前向 RGB 渲染库，不具备 MonoGS tracking / mapping 所需的 backward、camera pose gradient、depth map、opacity map、radii 和 visibility 输出。因此它目前不能直接改善 ATE RMSE、PSNR、SSIM 和 LPIPS；更稳妥的使用方式是先用于 GUI、evaluation 和大规模前向渲染加速。若后续补齐可微反向传播和 MonoGS 所需输出接口，FlashGS 的 precise intersection、size-aware scheduling 和 pipelined rendering 才有机会通过增加 tracking / mapping 优化预算，间接提升上述指标。
