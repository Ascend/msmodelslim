# HQQ 权重量化算法 量化术语百科词条

> **词条类别**：[量化算法](../README.md#2-量化算法)<br>
> **英文名称**：hqq<br>
> **应用领域**：大语言模型免训练、免校准的权重量化压缩<br>

---

## 1. 概述

HQQ 是一种免训练、免校准的权重量化算法。它在 [MinMax](../minmax/term_minmax.md) 计算出初始 scale/offset 后，通过半二次分裂迭代优化量化 offset，以最小化量化误差。核心特征是无需任何校准数据即可完成权重量化、通过 Lp 范数提升对离群值的鲁棒性、在指定粒度上贪心更新保证收敛质量，常用于高精度低比特权重量化场景。

---

## 2. 词条介绍

传统免校准量化（如 [MinMax](../minmax/term_minmax.md)）直接依据权重极值确定量化范围，当权重分布不均匀时量化误差较大。HQQ 观察到，量化误差可以通过迭代调整 offset 来显著降低：在固定 scale 的前提下，offset 的最优值可通过对量化残差施加 Lp 范数收缩后取均值得到，从而在无需数据的前提下逼近指定粒度上的最优量化参数。

从量化流程中的定位看，该算法解决的是“如何把连续浮点值映射到受限数值集合，同时尽量保留模型输出”的问题。与只按极值直接计算尺度的基础方法相比，HQQ 以降低量化误差为优化目标，迭代求解最优 offset，因此更适合对精度有明确要求的量化场景。

### 2.1 核心思想

HQQ 的核心思想是“半二次分裂迭代优化 offset”：将权重量化误差最小化问题拆解为两个可交替求解的子问题，在固定 scale 的情况下，通过迭代交替执行“残差计算”与“offset 更新”，在指定粒度上保留更优参数，直至收敛。由于整个过程仅依赖权重本身，因此是 data-free 的。

HQQ 的优点是量化过程完全由权重决定，不需要收集激活或校准数据；当校准数据难以获取且量化耗时不敏感时，它是 MinMax 之上低成本的精度改进方案。

### 2.2 工作机制

HQQ 先用 MinMax 观察器统计权重极值，计算初始 scale 与 offset。随后进入半二次分裂迭代：固定 scale 不变，每次迭代先计算当前伪量化残差 $u=W-W_r$，对残差计算广义软阈值得到收缩后的残差，再据此在指定粒度上更新最优 offset；对每个粒度独立比较新旧参数的量化误差，只保留更优者，并依据相对下降与绝对变化双阈值判断是否提前收敛。

由于 scale 全程固定，HQQ 的搜索维度被压缩为只优化 offset，配合在指定粒度上贪心更新，在免校准前提下以较低的计算成本逼近更优的量化参数。量化和反量化遵循框架约定 $q=\operatorname{round}(w/\text{scale}+\text{offset})$、$w=(q-\text{offset})\cdot\text{scale}$。

### 2.3 数学描述

HQQ 以最小化权重量化引入的 Lp 范数误差为目标：

$$
\min_{q,\, \text{offset}} \| W - \operatorname{dequant}(q) \|_p
$$

其中 $\operatorname{dequant}(q) = (q - \text{offset}) \cdot \text{scale}$。迭代过程如下：

- 计算量化残差 $u = W - W_r$（$W_r$ 为当前伪量化后的权重）。
- 对残差计算广义软阈值：

$$
W_e = \operatorname{sign}(u) \cdot \operatorname{relu}\left(|u| - \frac{|u|^{p-1}}{\beta}\right)
$$

- 更新当前最优 offset：

$$
\text{offset}_{\text{new}} = \mathbb{E}\left[q - \frac{W - W_e}{\text{scale}}\right]
$$

- 在指定粒度上贪心保留更优参数，并依据相对/绝对下降阈值判断收敛，早停退出。

- $W$：原始权重矩阵
- $q$：量化后的权重
- $W_r$：量化后反量化得到的权重
- $u$：量化残差，$u = W - W_r$
- $W_e$：收缩（shrink）后的残差
- $\text{scale}$：缩放因子在 HQQ 中固定不更新
- $\text{offset}$：非对称量化零点
- $p$：Lp 范数阶数，取值范围(0,1]
- $\beta$：正则化系数，控制对量化误差的惩罚强度

收敛判断条件（所有粒度均满足时提前退出）：

$$
\frac{|\text{best\_pnorm} - \text{current\_pnorm}|}{\text{best\_pnorm} \cdot \text{clamp}} < \text{min\_scale} \quad \text{或} \quad |\text{best\_pnorm} - \text{current\_pnorm}| < \text{converge\_threshold}
$$

### 2.4 关键性质

- **免校准**：量化过程仅依赖权重，无需校准数据集。
- **固定 scale、优化 offset**：HQQ 不更新 scale，仅迭代优化 offset，降低搜索维度。
- **仅量化权重、不量化激活**：HQQ 通过迭代求解 offset，求解时间相对较长，不适用于激活值量化。
- **对称与非对称**：由于 HQQ 通过优化 offset 降低量化误差，因此 offset 为变量，仅可使用非对称量化。
- **离群值不敏感**：通过以残差的 Lp 范数作为优化目标，降低离群值的影响。

由于 HQQ 只优化 offset 而不改变 scale，其优化空间相对受限，未必能像 SSZ/GPTQ 那样取得同样收益；但当目标是“无需量化激活值，需要提升权重量化精度”时，HQQ 是较为稳定的选择。

### 2.5 适用场景

- 对精度要求较高的权重量化场景。
- 权重分布不均匀、传统 MinMax 量化误差较大的场景。

更具体地说，是否适用主要取决于目标位宽、模型结构和部署后端三点。若目标部署链已经明确支持该算法对应的量化格式，且无需或难以准备代表性校准数据，通常可以优先从该算法的推荐配置建立基线，再根据精度结果决定是否引入需要数据的更复杂优化。

### 2.6 使用限制

- 目前支持 int8 的 per_channel 非对称量化，暂不支持 int4 量化。
- per_tensor / per_group 量化粒度暂不支持。
- 权重必须为 2D 张量。

这些限制应在调参前确认，而不是等精度异常后再排查。尤其是数据类型、张量维度、分组大小和后端算子支持等硬约束，一旦不满足，继续调整算法参数通常无法解决问题；应先回到受支持的配置组合。

---

## 3. 关联词条

可以从“同类方法、前后处理关系和应用对象”三个方向理解本词条与其他算法的关系。下面的关联项既用于横向比较不同技术路线，也用于帮助定位该算法在完整量化方案中的位置。

- [MinMax](../minmax/term_minmax.md)：前置术语，HQQ 使用 MinMax 观察器初始化 scale/offset。
- [SSZ](../ssz/term_ssz.md)：对比算法，同为迭代优化量化参数的权重量化方法，但 SSZ 依赖校准数据。
- [GPTQ](../gptq/term_gptq.md)：对比算法，GPTQ 依赖校准数据与二阶信息，HQQ 为免校准方法。
- [线性量化](../linear_quant/term_linear_quant.md)：应用对象，HQQ 作为线性权重量化的高精度量化方法使用。

---

## 4. 参考文档

参考文档优先列出算法原始论文或权威出处，并补充仓库内对应使用指南。需要进一步理解参数选择时，可先阅读使用指南，再回到原论文核对算法假设和推导。

1. Badri H, Shaji A. Half-Quadratic Quantization of Large Machine Learning Models. https://dropbox.github.io/hqq_blog/
2. 《[HQQ 参数配置流程指南](./usage_hqq.md)》
