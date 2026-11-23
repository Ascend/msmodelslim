# HiF4_Sweep Lv-1 Scale搜索量化算法

> **词条类别**：[量化算法](../README.md#2-量化算法)<br>
> **英文名称**：hif4_sweep<br>
> **应用领域**：HiF4 W4A4 权重量化、低比特量化精度优化<br>

---

## 1. 概述

HiF4_Sweep 是一种面向 HiF4 W4A4 per-block 量化的 E6M2 偏移搜索算法。它在 E6M2 编码空间中对每个 block 的一级缩放因子进行 offset 遍历搜索，比较各候选的量化重构 MSE，选择误差最小的 scale_factor；核心特征是逐块 E6M2 整数偏移搜索、Weight 离线一次性搜索，适合在 HiF4 W4A4 量化中改善因单一 AbsMax 尺度选择带来的精度损失。

---

## 2. 词条介绍

HiF4 W4A4 量化中，标准量化过程使用 $scale = \text{absmax} / 7$ 作为一级缩放因子，虽然能保证最大绝对值的量化误差最小，但对于整个 HiF4 Block（64 个元素）并不一定是最优量化。HiF4_Sweep 针对 HiF4 中 E6M2 编码的 lv-1 scale 选择问题，引入逐块 offset 搜索，提供对权重的最优量化区间，可有效降低量化误差。

从量化流程中的定位看，该算法解决的是"如何把连续浮点值映射到受限数值集合，同时尽量保留模型输出"的问题。与只按极值直接计算尺度的基础方法相比，它通常会利用更细的统计信息、优化目标或结构约束来控制误差，因此更适合对精度有明确要求的量化场景。

### 2.1 核心思想

HiF4_Sweep 的核心思想是针对 hif4 中 E6M2 编码的一级 scale 选择问题，提供对权重的最优量化区间，在区间内搜索最优 lv-1 scale，可有效降低量化误差。

对每个 64 元素 block，标准 HiF4 通过 AbsMax 计算一级缩放因子 $scale_0$ 并依次计算lv-2、lv-3 scale 和 S1P2 完成量化；HiF4_Sweep 则以 $scale_0$ 的 E6M2 编码为中心，在理论最优范围内遍历E6M2 offset，对每个候选 scale 分别计算量化 MSE，选择 MSE 最小的结果。

**核心要点：**

1. **最优化范围**：HiF4 的最优 E6M2 位于初始 E6M2 的偏移 $[-3, 4]$ 范围内，不同的 E6M2 使 Block 量化分布与原始数值分布取得不同的误差。
2. **权重多路寻优**：权重 HiF4 离线量化时，对每个 64 长度数据块进行 $[-3, 4]$ 的多路量化，取量化后 MSE 最小的一路结果。

### 2.2 工作机制

**原始 hif4 的 scale 选择问题：**

对每个 64 元素 block，计算一级缩放因子：
$$scale_0 = \text{bf16\_to\_e6m2}\left(\frac{\max(|x|)}{7.0}\right)$$

由于一个 HiF4 block 有 64 个数，仅通过其中 AbsMax 值来确定 E6M2 的量化路径，其量化结果未必是最优的。

**HiF4_Sweep 的改进：**

对于 HiF4 的三级缩放因子结构，推导并证明了必定存在全局最优的 scale，其满足：
$$ \frac{7}{10}scale_0 \leq scale_{best} \leq \frac{7}{4}scale_0 $$

结合实际 bf16_to_e6m2 采用的舍入方式，上述 $scale_{best}$ 所在区间转变为 $[E6M2(scale_{0})-3, E6M2(scale_{0})+4]$。

当对权重进行离线量化时，可以先以 AbsMax 值计算出原始的 $scale_{0}$，然后在其 E6M2 编码的周边 $[-3, 4]$ 偏移范围内取得各候选 E6M2 值分别进行量化，取得量化 MSE 误差最小的结果。

**E6M2 编码格式：**

| 字段 | 位数 | 说明 |
|------|------|------|
| Exponent (E) | 6 bit | 实际指数 = E - 48 |
| Mantissa (M) | 2 bit | 实际尾数 = 1 + M/4 |
| 合计 | 8 bit | 无符号，值 = (1 + M/4) × 2^(E-48) |

E6M2 编码范围：[1, 254]（0 和 0xFF 为特殊值），对应的 float 范围约 $[3.55 \times 10^{-15}, 49152.0]$。

### 2.3 数学描述

标准 HiF4 一级缩放因子：
$$
scale_0 = \text{E6M2\_round}\left(\frac{\max_{i}|x_i|}{7.0}\right)
$$

其中 $\text{E6M2\_round}$ 表示 BF16 截断 → 7-bit 尾数舍入 → 2-bit 尾数舍入（E6M2 编码）的完整流程。

HiF4_Sweep 的候选缩放因子：
$$
scale_{candidate}(k) = \text{E6M2\_to\_float}\left(\text{clamp}\left(E6M2(scale_0) + k,\ 1,\ 254\right)\right), \quad k \in [k_{lower}, k_{upper}]
$$

- $k$：E6M2 整数偏移（offset）
- $k_{lower}, k_{upper}$：搜索范围，Weight 默认 $[-3, 4]$

对每个候选 $k$，计算 per-block 重构 MSE：
$$
\text{MSE}(k) = \sum_{i=1}^{64}\left(|x_i| - \hat{x}_i(k)\right)^2
$$

其中 $\hat{x}_i(k)$ 是使用 $scale_{candidate}(k)$ 经过完整三级 HiF4 量化-反量化后的重构值。

选择 MSE 最小的 offset：
$$
k^* = \arg\min_{k \in [k_{lower}, k_{upper}]} \text{MSE}(k)
$$

最终 scale_factor 为 $scale_{candidate}(k^*)$，后续的 scale_lv2、scale_lv3 计算自动使用该 scale_factor，无需额外修改。

**Weight 的搜索范围：**

- Weight：$k \in [-3, +4]$（8 个候选），覆盖较大范围以适配权重分布多样性

### 2.4 关键性质

- **E6M2 偏移搜索**：在 E6M2 整数编码空间搜索最优 scale，而非连续 float 空间。
- **Weight 离线搜索**：权重仅在量化阶段一次性搜索，覆盖理论最优范围，best_offsets 存入 QParam.ext，推理时无额外开销。
- **零影响回退**：当 best_offsets=None 时等价于标准 minmax，对现有代码零影响。
- **量化公式不变**：仅改变 scale_factor 的选择策略，量化/反量化公式与标准 HiF4 完全相同。

从误差与适用边界看，HiF4_Sweep 的优势是搜索空间小且离散（8 个候选），每个候选只需一次量化-反量化 + MSE 计算。激活量化使用标准 minmax 方法，无额外推理开销。

### 2.5 适用场景

- 对 HiF4 W4A4 量化精度有更高要求的场景。
- 权重分布范围较大、单一 AbsMax 尺度导致块内量化误差偏大的模型层。

更具体地说，是否适用主要取决于目标位宽、模型结构和部署后端三点。若目标部署链已经明确支持该算法对应的量化格式，并且校准数据能够覆盖主要业务分布，通常可以优先从该算法的推荐配置建立基线，再根据精度结果决定是否增加更复杂的优化。

### 2.6 使用限制

- 仅支持 HiF4 格式的 per_block 对称量化（W4A4）。
- 仅支持权重量化搜索，激活量化使用标准 minmax 方法。
- Weight sweep 为离线一次性搜索，`w_sweep_lower` 和 `w_sweep_upper` 取值范围为 $[-3, 4]$。

这些限制应在调参前确认，而不是等精度异常后再排查。尤其是数据类型、张量维度、分组大小和后端算子支持等硬约束，一旦不满足，继续调整算法参数通常无法解决问题；应先回到受支持的配置组合。

---

## 3. 关联词条

可以从"同类方法、前后处理关系和应用对象"三个方向理解本词条与其他算法的关系。下面的关联项既用于横向比较不同技术路线，也用于帮助定位该算法在完整量化方案中的位置。

- [线性量化](../linear_quant/term_linear_quant.md)：应用对象，HiF4_Sweep 作为线性量化的权重量化方法使用。
- [Ceil_X](../ceil_x/term_ceil_x.md)：同类算法，同为 per-block 缩放策略优化算法（针对 MXFP4）。
- [FouroverSix](../fouroversix/term_fouroversix.md)：同类算法，同为 per-block 多候选 MSE 择优的量化算法。
- [MinMax](../minmax/term_minmax.md)：对比算法，HiF4_Sweep 是 HiF4 MinMax 权重量化的 E6M2 offset 搜索优化，激活量化沿用 minmax。

---

## 4. 参考文档

1. 《[HiF4_Sweep 参数配置流程指南](./usage_hif4_sweep.md)》
