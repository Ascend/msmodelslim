# SVDQuant：基于低秩残差重构的扩散模型后训练量化算法说明

## 简介

- **概述**：SVDQuant是一种针对扩散模型的后训练量化技术，通过三阶段流水线——**离群值迁移**、**低秩分解**、**残差量化**——将权重和激活中的离群值吸收到低秩分量中，从而缓解量化难度并提升模型性能。本工具基于 SVDQuant 的核心思想，实现了 `离群值抑制 → svd_res → linear_quant` 三阶段量化流水线。
- **核心思想**：
  1. **离群值迁移**（以 Iterative Smooth 为例）：将激活中的离群值通过数学等价变换迁移到权重中，使激活分布更均匀，降低激活量化难度；
  2. **低秩分解**（SVDResidual）：对迁移后的权重进行 SVD 低秩分解，将权重的主体结构提取为低秩分量（高精度保留），残差部分则更适合量化；
  3. **残差量化**（Linear Quant）：对残差权重进行低比特量化（如 4-bit），低秩分支以高精度（如 FP16）运行，从而在保持视觉质量的同时减少量化误差。

## 使用前准备

安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../../install_guide/install_guide.md)》。

## 原理和实现

### 原理

**核心思想：**

1. **离群值迁移**（以 Iterative Smooth 为例）：将激活中的离群值通过数学等价变换迁移到权重中，使激活分布更均匀，降低激活量化难度；
2. **低秩分解**（SVDResidual）：对迁移后的权重进行 SVD 低秩分解，将权重的主体结构提取为低秩分量（高精度保留），残差部分则更适合量化；
3. **残差量化**（Linear Quant）：对残差权重进行低比特量化（如 4-bit），低秩分支以高精度（如 FP16）运行，从而在保持视觉质量的同时减少量化误差。

**三阶段流水线：**

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  iter_smooth    │ ──► │    svd_res      │ ──► │  linear_quant   │
│  离群值迁移      │     │  低秩残差分解    │     │  残差量化        │
│  (需校准数据)    │     │  (data-free)    │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**阶段一：离群值迁移**

扩散模型的激活中存在显著的离群值，少数通道的极大值会压缩其余通道的量化精度。离群值抑制算法通过数学等价变换，将激活中的离群值"迁移"到权重中。以 Iterative Smooth 为例，其核心变换为：

$$y = x W + b = \underbrace{(x \cdot \text{diag}(s)^{-1})}_{\text{平滑后的激活}} \cdot \underbrace{(\text{diag}(s) \cdot W)}_{\text{吸收离群值的权重}} + b$$

其中缩放因子 $s$ 由激活和权重的统计信息联合计算：

$$s = \left(\frac{A_\text{scale}^\alpha}{W_\text{scale}^{1-\alpha}}\right), \quad s \geq s_\text{min}$$

- $A_\text{scale}$：激活值每通道的绝对最大值
- $W_\text{scale}$：权重每列的最大值
- $\alpha$：平衡参数，控制迁移强度（$\alpha$ 越大，更多离群值迁移到权重）

迁移后，激活分布更均匀（易于量化），但权重吸收了离群信息后变得更加难以量化——这正是下一阶段 SVD 低秩分解要解决的问题。

**阶段二：低秩残差分解**

经过离群值迁移后，权重吸收了激活的离群值，其分布中出现明显的低秩结构（离群值集中在少数通道上）。对迁移后的权重 $W$ 进行 SVD 低秩分解，提取低秩分量：

1. 执行 SVD 分解：$W \approx (U \cdot S) \cdot V^\top$
2. 计算残差：$R = W - (U \cdot S) \cdot V^\top$
3. 将权重替换为残差 $R$，低秩分量以参数形式保留

权重的主体结构（包括离群值引起的低秩成分）被提取到低秩分支中，以高精度（如 FP16）运行；残差部分分布更均匀，更适合低比特量化。

**阶段三：残差量化**

对残差权重 $R$ 和对应激活进行低比特量化（如 W4A4 MXFP4）。残差主通路以 4-bit 量化运行，低秩旁路以高精度运行，两者相加恢复原始输出精度。

**三阶段协作的数学等价性**

综合三个阶段，推理时的计算为：

$$
\begin{aligned}
\text{out} &= \underbrace{Q(x \cdot \text{diag}(s)^{-1}) \cdot Q(R)}_{\text{量化后的残差主通路}} + \underbrace{(x \cdot \text{diag}(s)^{-1} \cdot V) \cdot (U \cdot S)^\top}_{\text{高精度低秩旁路}} \\
           &\approx x \cdot \text{diag}(s)^{-1} \cdot W + b \\
           &= x W + b
\end{aligned}
$$

其中 $Q(\cdot)$ 表示量化操作。在量化精度足够时，输出近似等价于原始线性层。

### SVDResidual 原理详解

#### SVD 分解过程

对线性层权重 $W \in \mathbb{R}^{\text{out} \times \text{in}}$ 进行低秩分解：

1. 将权重转换为 `float32`，以保证 SVD 计算的数值稳定性：
   - `weight_float = weight.float()`
2. 设定秩：`rank = config.rank`
3. 执行分解：
   - `U, S, V = torch.svd_lowrank(weight_float, q=rank)`
   - 返回值满足 $W \approx U \cdot \text{diag}(S) \cdot V^\top$
   - `U [out_dim, rank]`，`S [rank]`，`V [in_dim, rank]`
4. 构建低秩参数：
   - `svd_lowrank_l1 = V[:, :rank].t()`          → $V^\top$，形状 `[rank, in_dim]`
   - `svd_lowrank_l2 = U[:, :rank] * S[:rank]`   → $U \cdot S$，形状 `[out_dim, rank]`
5. 重构低秩近似：
   - `reconstructed = svd_lowrank_l2 @ svd_lowrank_l1`，即 $(U \cdot S) \cdot V^\top$
6. 计算残差权重：
   - `residual = original_weight - reconstructed`，即 $R = W - (U \cdot S) \cdot V^\top$

处理后：

- 将 `weight` 替换为 `residual`；
- 将 `svd_lowrank_l1`、`svd_lowrank_l2` 以参数形式保存在对应的 `Linear` 模块中；
- 通过 Hook IR 将层包装为 `SVDResidualWrapper`，前向时执行"残差主通路 + 低秩旁路"并相加。

#### 前向等价关系

以单个线性层为例，原始线性层计算为：

$$y = x W^\top + b$$

经过 SVD 残差分解后：

- 原始权重：$W$
- 低秩近似：$W_\text{low} = (U \cdot S) \cdot V^\top$
- 残差权重：$R = W - W_\text{low}$

前向计算分为双通路：

1. **主通路（残差）**：
   - `residual_out = wrapped_module(x)`，即 $x R^\top + b$
   - 此时 `wrapped_module` 内部权重已被替换为 $R$，bias 保持不变

2. **旁路（低秩两段线性）**：

   > **关键**：`F.linear(x, W)` 的数学含义为 $x W^\top$（注意转置），因此：

   - `lowrank_hidden = F.linear(x, svd_lowrank_l1, bias=None)`
     - 权重参数为 $V^\top$，实际计算 $x (V^\top)^\top = x V$
   - `lowrank_out = F.linear(lowrank_hidden, svd_lowrank_l2, bias=None)`
     - 权重参数为 $U \cdot S$，实际计算 $(x V) (U \cdot S)^\top$

3. **汇总**：
   - `out = residual_out + lowrank_out`
   - 数学推导：

$$
\begin{aligned}
\text{out} &= x R^\top + b + (x V)(U \cdot S)^\top \\
           &= x R^\top + b + x V (U \cdot S)^\top \\
           &= x \big(R^\top + V (U \cdot S)^\top\big) + b \\
           &= x \big(R + (U \cdot S) V^\top\big)^\top + b \\
           &= x W^\top + b
\end{aligned}
$$

因此，双通路输出之和等价于使用原始权重 $W$ 的线性变换。

#### 处理范围

满足以下条件的 `nn.Linear` 会被处理：

- `isinstance(submodule, nn.Linear)`
- 模块名命中 `include`
- 模块名不命中 `exclude`

`post_run` 阶段会对未命中（且非 `"*"`）的模式给出告警。

### 实现

SVDQuant 的三阶段流水线由三个 Processor 依次协作完成：

1. **离群值迁移**：基于校准数据收集激活统计信息，计算缩放因子，将激活离群值迁移到权重中（以 Iterative Smooth 为例，计算每通道缩放因子 $s$，对权重乘以 $s$、对输入除以 $s$）。详见 《[Iterative Smooth 算法说明](../outlier_suppression_algorithms/iterative_smooth.md)》。

2. **低秩残差分解**（`svd_res`）：对迁移后的权重执行 SVD 低秩分解 $W \approx (U \cdot S) \cdot V^\top$，将权重替换为残差 $R = W - (U \cdot S) \cdot V^\top$，低秩分量 $V^\top$ 和 $U \cdot S$ 以参数形式保留。前向时双通路并行：残差主通路计算 $x R^\top + b$，低秩旁路计算 $(x V)(U \cdot S)^\top$，两者相加等价于原始线性变换。

3. **残差量化**（`linear_quant`）：对残差权重和激活进行低比特量化。详见 《[线性量化算法说明](linear_quant.md)》。

## 适用要求

- **低比特量化**：适合 W4A4 等极低比特量化场景，尤其适用于扩散模型。
- **离群值结构**：模型激活中存在显著的离群值，且离群值在权重中呈现低秩结构，适合通过低秩分支吸收。
- **计算资源**：`svd_res` 为 data-free 算法，计算开销较低；`iter_smooth` 需要校准数据收集激活统计信息。
- **使用限制**：
  - 目标层必须为标准 `torch.nn.Linear`，且可通过 `model.named_modules()` 获取模块名。
  - 三个阶段的 `include`/`exclude` 应保持一致，确保同一组 Linear 层依次经历离群值迁移 → 低秩分解 → 量化。

## 功能介绍

### YAML配置示例

以下为 SVDQuant 三阶段完整流水线配置示例（以 Wan2.2 T2V W4A4 量化为例）：

```yaml
spec:
  process:
    # 阶段一：离群值迁移
    - type: "iter_smooth"
      alpha: 0.25                        # 平衡参数，控制离群值迁移强度
      include: ["*"]                     # 包含的层
      exclude: ["*blocks.0.*"]

    # 阶段二：低秩残差分解
    - type: "svd_res"
      rank: 32                           # 低秩分解的秩
      include: ["*"]                     # 与 iter_smooth 保持一致
      exclude: ["*blocks.0.*"]

    # 阶段三：残差量化（W4A4 MXFP4）
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"
          dtype: "mxfp4"
          symmetric: True
          method: "minmax"
        weight:
          scope: "per_block"
          dtype: "mxfp4"
          symmetric: True
          method: "minmax"
      include: ["*"]
      exclude: ["*blocks.0.*"]
```

> [!NOTE]
>
> 三个阶段的 `include`/`exclude` 应保持一致，确保同一组 Linear 层依次经历离群值迁移 → 低秩分解 → 量化。

### YAML配置字段详解

SVDQuant 涉及三个处理器的配置字段，其中 `iter_smooth` 和 `linear_quant` 的字段详见各自文档，此处仅介绍 `svd_res` 的专属字段：

| 字段名  | 作用           | 说明 |
|---------|----------------|------|
| type    | 处理器类型标识 | 固定值 `"svd_res"`，用于标识这是一个 SVD 残差处理器。 |
| rank    | 低秩分解的秩   | 大于 0 的整数，控制近似的秩，默认为 32。受算子实现限制，建议不超过 128。 |
| include | 包含的层       | 字符串列表，支持通配符匹配，匹配的是 `model.named_modules()` 返回的完整模块路径。 |
| exclude | 排除的层       | 字符串列表，支持通配符匹配，用于显式排除不希望进行 SVD 分解的模块。 |

其他处理器的配置字段详见 [Iterative Smooth YAML配置字段详解](../outlier_suppression_algorithms/iterative_smooth.md#yaml配置字段详解)、[线性量化 YAML配置字段详解](linear_quant.md#yaml配置字段详解)。

## FAQ

### 1. 为什么需要先做离群值迁移再做 SVD？

- 扩散模型的激活中存在显著的离群值，直接量化会导致精度严重下降。
- 离群值迁移将离群值从激活迁移到权重，使激活更易量化。
- 但迁移后权重变得更难量化——幸运的是，离群值在权重中呈现低秩结构，恰好适合 SVD 低秩分解提取。
- 这种"迁移 → 分解"的配合是 SVDQuant 的核心创新：离群值先被迁移到权重，再被低秩分支吸收，从而同时解决激活和权重的量化难题。

### 2. rank 怎么选？

- 较大的 `rank` 能更好地拟合原始权重，但低秩分支的计算和存储开销更大；
- 较小的 `rank` 能够获得更强的压缩效果，但残差中保留更多未捕获的信息，可能影响量化精度；
- 可以根据模型规模与精度要求进行网格搜索或经验选取（如 8、16、32、64 等）。

### 3. 为什么先转 float32 再 SVD？

- 对于 `float16` / `bfloat16` 等低精度权重，直接执行 SVD 易产生数值不稳定；
- 转换为 `float32` 能使 SVD 计算在数值上更加稳定；
- 分解完成后，再将得到的低秩矩阵转换回原始 `dtype` 与 `device`，保证兼容训练/推理环境。

### 4. 如何确认哪些层被分解？

- 查看日志中关于未匹配模式的告警信息；
- 在代码中枚举模型的 `named_modules()`，确认名称是否与 `include` / `exclude` 模式一致；
- 运行 Processor 后检查目标 `Linear` 层是否多出了 `svd_lowrank_l1` / `svd_lowrank_l2` 参数。

### 5. Smooth 的 alpha 参数对 SVDQuant 有什么影响？

- `alpha` 控制离群值从激活迁移到权重的强度：$\alpha$ 越大，迁移越激进；
- 在 SVDQuant 流水线中，`alpha` 需要与 `rank` 协调：较大的 `alpha` 使更多离群值进入权重，可能需要较大的 `rank` 来充分吸收这些低秩成分。
