# MSE_Round 权重量化算法 量化术语百科词条

> **词条类别**：[量化算法](../README.md#2-量化算法)<br>
> **英文名称**：mse_round<br>
> **应用领域**：MXFP8 权重量化、低比特量化精度优化<br>

---

## 1. 概述

MSE_Round 是一种面向 MXFP8 per-block 权重量化的 shared exponent 选择算法。它对每个块分别尝试相邻的指数舍入候选，并通过实际量化-反量化的均方误差选择更优尺度，以降低固定舍入规则带来的块级重构误差；核心特征是逐块候选比较、MSE 决策和无需额外训练的尺度优化。

---

## 2. 词条介绍

MXFP8 per-block 量化中，传统 minmax 方法固定使用 $s = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}$ 计算 shared exponent，缩放后 block 内最大值的量级落在 $[8, 16)$。当 block 内最大值接近上界时，缩放结果可能超出 MXFP8 的表示上限（$448$），引发截断并引入较大量化误差。MSE_Round 通过 per-block 比较 ceil 与 floor 两档的实际量化误差，在避免截断与保持精度之间取得更优平衡。

从量化流程中的定位看，该算法解决的是“如何把连续浮点值映射到受限数值集合，同时尽量保留模型输出”的问题。与只按极值直接计算尺度的基础方法相比，它通常会利用更细的统计信息、优化目标或结构约束来控制误差，因此更适合对精度有明确要求的量化场景。

### 2.1 核心思想

MSE_Round 的核心思想是“per-block 两档候选 MSE 择优”：对每个 block 同时计算 ceil 与 floor 两档候选 shared exponent，分别完成量化-反量化并计算 block 内 MSE，选择 MSE 更小的 shared exponent 作为最终量化参数。ceil 档可避免大值截断，floor 档在分布均匀时可能获得更优的整体 MSE。

逐块比较能自适应不同分布：尖峰块可能更需要 ceil 的范围裕量，主体密集、最大值不极端的块则可能从 floor 的更细步长获益。

### 2.2 工作机制

块浮点量化的 shared exponent 只能取离散整数。若由块内最大值反推得到一个连续的理想指数，它通常位于 floor 与 ceil 两个相邻整数之间。向下取整对应更小的 2 的幂尺度，量化步长更细但更容易让大值超过格式可表示范围；向上取整有更大的动态范围裕量，却会让所有值的量化间隔变粗。MSE_Round 不预设哪一边一定更好，而是同时构造两种候选。

对每个 block，算法分别用 floor exponent 与 ceil exponent 完整执行量化-反量化，并计算该 block 的重构 MSE；只要候选指数在格式允许范围内，就选择误差较小者。因为比较发生在真实 FP 格点和真实饱和规则之后，它能够把“步长变粗”和“截断减少”这两类相反影响直接纳入决策。与使用一个全局经验边界的方法相比，它是逐块的局部二选一。

### 2.3 数学描述

传统 floor 缩放：

$$
s_{\text{floor}} = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}
$$

MSE_Round 同时计算 ceil 与 floor 两档候选：

$$
s_{\text{ceil}} = \lceil \log_2(\max(|x|)) \rceil - e_{\text{max}}
$$

$$
s_{\text{floor}} = \lfloor \log_2(\max(|x|)) \rfloor - e_{\text{max}}
$$

分别计算 block 内 MSE：

$$
\text{MSE}_{\text{ceil}} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i(s_{\text{ceil}}))^2, \quad \text{MSE}_{\text{floor}} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i(s_{\text{floor}}))^2
$$

最终 per-block 选择：

$$
s^* = \begin{cases} s_{\text{ceil}} & \text{if } \text{MSE}_{\text{ceil}} < \text{MSE}_{\text{floor}} \\ s_{\text{floor}} & \text{otherwise} \end{cases}
$$

- $x$：block 内的权重值
- $\hat{x}(s)$：使用 shared exponent $s$ 量化-反量化后的值
- $N$：block 内元素个数
- $e_{\text{max}}$：指数偏置，MXFP8 E4M3 格式下 $e_{\text{max}} = 2^{e_{\text{bits}}-1} = 8$

当某一候选的 shared exponent 超出 E8M0 表示范围（被标记为 NaN）时，自动回退至另一有效候选。

若连续对数尺度为 $r=\log_2 s_{ideal}$，整数 shared exponent 候选为 $e_f=\lfloor r\rfloor$、$e_c=\lceil r\rceil$，则：

$$
\hat x^{(e)}=2^e\,\mathcal F(x/2^e),\qquad e\in\{e_f,e_c\},
$$

$$
e^*=\arg\min_{e\in\{e_f,e_c\}}\sum_i(x_i-\hat x_i^{(e)})^2.
$$

这里 $\mathcal F$ 是块内低比特浮点码表投影。由于码表非均匀且可能饱和，不能仅根据 $r$ 距 floor/ceil 哪个更近来判断 MSE，必须比较真实 QDQ。

### 2.4 关键性质

- **双候选缩放**：per-block 在 ceil/floor 两档间 MSE 择优。
- **避免截断**：ceil 档将缩放后范围压缩至 $(4, 8]$，避免大值截断。
- **有限候选搜索**：只比较相邻 floor/ceil exponent，不需要连续尺度优化。
- **计算可控**：每个 block 执行两次量化-反量化评估，计算量约为标准 minmax 的2倍。
- **逐块离散决策**：每个 block 独立在 floor/ceil 两个相邻 shared exponent 中选择更小 MSE。

若最终后端的 exponent 取值或 FP 格式与模拟不一致，搜索结论也会失效。

### 2.5 适用场景

- 对 MXFP8 权重量化精度有更高要求的场景。
- block 内最大值分布不均、floor 缩放导致大值截断的模型层。

更具体地说，是否适用主要取决于目标位宽、模型结构和部署后端三点。若目标部署链已经明确支持该算法对应的量化格式，并且校准数据能够覆盖主要业务分布，通常可以优先从该算法的推荐配置建立基线，再根据精度结果决定是否增加更复杂的优化。

### 2.6 使用限制

- 当前仅注册于 `mxfp8_per_block_sym` 权重量化方案，仅支持权重量化。
- 激活值量化请继续使用 `minmax` 等已有方法。

这些限制应在调参前确认，而不是等精度异常后再排查。尤其是数据类型、张量维度、分组大小和后端算子支持等硬约束，一旦不满足，继续调整算法参数通常无法解决问题；应先回到受支持的配置组合。

---

## 3. 关联词条

可以从“同类方法、前后处理关系和应用对象”三个方向理解本词条与其他算法的关系。下面的关联项既用于横向比较不同技术路线，也用于帮助定位该算法在完整量化方案中的位置。

- [线性量化](../linear_quant/term_linear_quant.md)：应用对象，MSE_Round 作为线性量化的权重量化方法使用。
- [Ceil_X](../ceil_x/term_ceil_x.md)：同类算法，同为 MXFP 格式的 shared exponent 优化算法（针对 MXFP4）。
- [FouroverSix](../fouroversix/term_fouroversix.md)：同类算法，同为 per-block 双候选择优的量化算法（针对 MXFP4）。
- [MinMax](../minmax/term_minmax.md)：对比算法，MSE_Round 是 MinMax MXFP8 量化的精度优化。

---

## 4. 参考文档

参考文档优先列出算法原始论文或权威出处，并补充仓库内对应使用指南。需要进一步理解参数选择时，可先阅读使用指南，再回到原论文核对算法假设和推导。

1. 《[MSE_Round 参数配置流程指南](./usage_mse_round.md)》
