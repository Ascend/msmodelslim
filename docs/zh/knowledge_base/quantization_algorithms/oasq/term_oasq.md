# OASQ 离群感知平滑量化算法 量化术语百科词条

> **词条类别**：[离群值抑制算法](../README.md#1-离群值抑制算法)<br>
> **英文名称**：oasq<br>
> **应用领域**：大语言模型量化压缩、推理加速<br>

---

## 1. 概述

OASQ（Outlier-Aware Smooth Quantization）是一种面向激活离群通道的平滑量化预处理算法。它先用通道级 z-score 识别离群通道，再对正常通道与离群通道分别采用不同尺度公式，并把尺度融合到相邻可等价变换的子图中；核心特征是离群感知、迭代阈值搜索和多子图融合，可与后续 [Linear Quant](../linear_quant/term_linear_quant.md)、[Trainable Linear Quant](../trainable_linear_quant/term_trainable_linear_quant.md) 等量化步骤组合使用。

---

## 2. 词条介绍

传统 [SmoothQuant](../smooth_quant/term_smooth_quant.md) 对所有通道使用统一指数缩放，难以区分“主体分布”与“少数极端通道”。OASQ 在平滑前先根据激活通道统计估计离群比例，并在目标离群比例区间内迭代调整 z 阈值，使正常通道走更接近几何均值平衡的尺度、离群通道走更缓和的对数型尺度，从而降低极端通道对整体量化尺度的支配。

从量化流程中的定位看，该算法属于量化前的分布整形步骤：先降低离群值对量化尺度的支配，再由后续量化算法完成离散化。其价值在于不必简单扩大位宽，而是通过通道自适应的等价缩放，提高有限量化区间对主体数据分布的利用率。

### 2.1 核心思想

OASQ 的核心思想是“先识别离群通道，再分别设计平滑尺度”：用激活通道均值与标准差构造 z-score，将通道划分为正常集合与离群集合；正常通道更强调激活与权重幅值的平衡，离群通道则使用更保守的对数型迁移，避免把过大难度一次性压到权重侧。实现还会迭代调整 z 阈值，使离群比例落入预设目标区间。

收益来自把“少数通道拉高全局尺度”的问题，转化为按通道类型分流的尺度分配问题。

### 2.2 工作机制

校准阶段对目标线性层输入做 MinMax 统计，得到通道级激活幅值（对称模式下为绝对值最大值；非对称且子图支持时先按 shift 居中再取绝对值最大值）。随后 `OASQScaleCalculator` 在 `max_iters` 次内调整 z 阈值：离群比例过高则提高阈值，过低则降低阈值，直到比例落入目标区间或迭代结束。

得到逐通道尺度 $s$ 后，按子图类型做等价融合：`norm-linear`、`linear-linear`、`ov`、`up-down` 等将尺度吸收到相邻 norm/权重（及可选 shift）；无法前置融合的独立线性层可通过非融合 hook 在前向时对输入做乘性缩放。非对称 shift 当前主要在 `norm-linear` 路径启用，其他子图会关闭 shift。

### 2.3 数学描述

设通道激活统计为 $a_j$、权重输入通道统计为 $w_j$，通道均值与标准差为 $\mu$、$\sigma$。定义：

$$
z_j=\frac{a_j-\mu}{\sigma+\varepsilon},\qquad
\mathcal O=\{\,j:\lvert z_j\rvert>z_{\mathrm{thr}}\,\}.
$$

- $z_j$：第 $j$ 个通道的 z-score
- $z_{\mathrm{thr}}$：离群判定阈值（实现中从基准值出发迭代调整）
- $\mathcal O$：被判定为离群的通道集合
- $\varepsilon$：数值稳定项

对正常通道与离群通道分别构造尺度（实现中还会对尺度做均值归一）：

$$
s_j=
\begin{cases}
\sqrt{\dfrac{w_j}{\sqrt{a_j}+\varepsilon}+\varepsilon}, & j\notin\mathcal O\\[1.2em]
\log\!\big(1+\dfrac{w_j}{\log(1+a_j)+\varepsilon}+\varepsilon\big), & j\in\mathcal O
\end{cases}
$$

- $s_j$：第 $j$ 个通道的平滑尺度
- $a_j$：激活通道幅值统计
- $w_j$：权重对应输入通道幅值统计

浮点域仍可通过 $X'=X/s$、$W'=Ws$ 保持 $X'W'^T=XW^T$。迭代通过调整 $z_{\mathrm{thr}}$ 使离群比例落入目标区间 $[r_{\min},r_{\max}]$，从而在“过度平滑”与“几乎不识别离群”之间取得折中。

### 2.4 关键性质

- **离群感知**：尺度公式按通道是否离群分流，而不是全通道共用同一指数。
- **阈值可搜索**：用有限次迭代把离群比例约束到目标区间，降低手工设定固定阈值的难度。
- **多子图融合**：支持 `norm-linear`、`linear-linear`、`ov`、`up-down` 等可等价吸收尺度的结构。
- **对称 / 非对称路径**：`symmetric` 影响统计与是否启用 shift；非对称 shift 主要覆盖 `norm-linear`。
- **分布式友好**：统计收集支持按模块共享关系做分布式同步。

从误差边界看，阈值过严会使几乎所有通道走正常公式，收益接近普通平滑；阈值过松会把过多通道标为离群，对数型尺度可能迁移不足。尺度最终仍会被后续量化非线性截断，因此端到端精度取决于与后续量化器的组合。

### 2.5 适用场景

- 激活存在明显通道级离群、统一指数平滑收益有限的场景。
- 作为 W8A8、W4A8 等方案的前置离群值抑制步骤，为后续线性量化创造更均匀的激活分布。

更具体地说，这类算法适合“量化误差主要由少数大幅值通道拉高尺度”的情况。若问题来源并不是离群值，而是模型整体对低比特表示敏感，则单独增强平滑通常收益有限，应结合更高精度量化或敏感层回退。

### 2.6 使用限制

- 目标结构需要能够识别为可保持计算等价的子图；非融合路径仅适合乘性缩放。
- 非对称 `shift` 当前主要支持 `norm-linear`；在其他子图上会被关闭。
- 需要代表性校准数据收集激活统计；未激活的 MoE expert 等可能因无统计而被跳过。

这些限制应在调参前确认。结构不兼容时，应优先收窄 `enable_subgraph_type` 或 `exclude`，而不是盲目增大 `max_iters`。

---

## 3. 关联流程

- [《OASQ 参数配置流程指南》](./usage_oasq.md)：通过本流程可将 OASQ 写入量化任务并完成适配与调参。

---

## 4. 关联词条

可以从“同类方法、前后处理关系和应用对象”三个方向理解本词条与其他算法的关系。下面的关联项既用于横向比较不同技术路线，也用于帮助定位该算法在完整量化方案中的位置。

- [SmoothQuant](../smooth_quant/term_smooth_quant.md)：上位概念，本算法同属激活-权重协同平滑路线，但按离群通道分流尺度。
- [Iterative Smooth](../iterative_smooth/term_iterative_smooth.md)：同类算法，同样支持多子图平滑融合。
- [Flex Smooth Quant](../flex_smooth_quant/term_flex_smooth_quant.md)：同类算法，通过搜索 alpha/beta 决定平滑强度。
- [AWQ](../awq_smooth/term_awq_smooth.md)：对比算法，基于激活重要性搜索缩放因子。
- [Trainable Linear Quant](../trainable_linear_quant/term_trainable_linear_quant.md)：配套术语，低比特可训练量化前可先用本算法抑制离群。
- [DAOS](../daos/term_daos.md)：配套术语，多模态生成联合低比特方案，以本算法为离群抑制阶段并再接 TLQ。

---

## 5. 参考文档

参考文档优先列出算法原始论文或权威出处，并补充仓库内对应使用指南。需要进一步理解参数选择时，可先阅读使用指南，再回到原论文核对算法假设和推导。

1. Xiao G et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML 2023. https://arxiv.org/abs/2211.10438
2. 《[OASQ 参数配置流程指南](./usage_oasq.md)》
