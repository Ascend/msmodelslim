# Trainable Linear Quant 可训练线性量化算法 量化术语百科词条

> **词条类别**：[量化算法](../README.md#2-量化算法)<br>
> **英文名称**：trainable_linear_quant<br>
> **应用领域**：大语言模型量化压缩、低比特量化精度优化<br>

---

## 1. 概述

Trainable Linear Quant（TLQ）是面向线性层的块级训练式量化优化算法。它以浮点块输出为教师信号，经可插拔 OP（如 `minmax_tune`、`round_tune`、`trainable_smooth`）用 SignSGD 学习量化参数与舍入；核心特征是块级训练与可组合 OP，可视为 [AutoRound](../autoround/term_autoround.md) 的可扩展实现，主要用于 W4A4 精度增强，并可作为 [DAOS](../daos/term_daos.md) 的第二阶段。

---

## 2. 词条介绍

传统一次标定的 [MinMax](../minmax/term_minmax.md) 或固定舍入往往在 4bit 及以下引入较大重构误差。[AutoRound](../autoround/term_autoround.md) 证明可学习舍入能显著改善低比特权重，但实现与配置相对固定。TLQ 把同类思路抽象为“块级 Trainer + OP 插件 + 策略 qconfig”：用户可按策略为不同模块指定位宽/粒度，并按需选用可训练 OP，在统一训练循环中最小化块输出重构损失。

从量化流程中的定位看，该算法解决的是“如何在受限数值集合上为线性层选择更优离散表示，同时尽量保留块输出”的问题。与只按极值直接计算尺度的基础方法相比，它利用校准前向与优化目标控制误差，因此更适合对精度有明确要求的低比特场景。

### 2.1 核心思想

TLQ 的核心思想是“把量化参数变成可训练变量，并在块级重构目标下联合优化”。对每个 decoder 块：先采集浮点教师输出；再用伪量化前向得到学生输出；按 `train_config` 指定的损失（如 L1）与最优快照策略更新各 OP 参数；训练结束后把学习结果折叠回量化 IR / 权重表示，推理阶段不再继续训练。

有效的关键在于：低比特误差在层输出上是相关的，按块输出重构比逐元素独立四舍五入更接近最终精度目标。

### 2.2 工作机制

配置侧由三部分组成：

1. **`strategies`**：按 `include/exclude` 为线性层匹配 `LinearQConfig`（激活与权重量化 dtype/scope/method）。
2. **`operations`**：定义训练管线中启用的 OP（如 `minmax_tune`、`round_tune`、`trainable_smooth`），按需选用。
3. **`train_config`**：控制块级迭代次数、学习率、梯度累加、损失类型与最优 checkpoint 选择（ema / min_loss / last）。

执行时按块包装线性层为可训练 wrapper，绑定 OP 参数；教师路径保持浮点，学生路径走伪量化内核（当前支持 INT 与 MXFP 族）。可选 `train_with_act_quant` 决定训练前向是否伪量化激活；`enable_quanted_input` 决定是否把本层量化输出作为后续块的旁路量化输入。优化器默认使用 SignSGD，适合高维舍入类变量的符号更新。

### 2.3 数学描述

对块输入 $X$ 与浮点权重 $W$，浮点教师输出记为 $Y=f(X,W)$。量化学生用可训练参数 $\theta$（范围缩放、舍入偏移、平滑尺度等）生成 $\hat Y=f_{\mathcal Q}(X;W,\theta)$。块级目标可写为：

$$
\mathcal L(\theta)=\ell\big(Y,\hat Y\big),
$$

其中 $\ell$ 默认可为逐元素 L1（实现中 `loss_type: l1`），也可选用对离群区域加权的自定义形式（`custom_outlier`）。

以权重量化为例，归一化后的可学习舍入可抽象为：

$$
q(V)=\operatorname{clip}\big(\lfloor u\rfloor+h(V),q_{\min},q_{\max}\big),\qquad u=W/s+z,
$$

- $u$：按尺度 $s$ 与零点 $z$ 归一化后的权重
- $V$：可学习舍入偏移（由 `round_tune` 优化）
- $h(V)$：训练阶段连续松弛、结束后固化的 0/1 舍入决策
- $q_{\min},q_{\max}$：量化码上下界

`minmax_tune` 则额外学习对称/非对称范围相关缩放，使尺度不再锁死在原始 MinMax 极值上。若启用 `trainable_smooth`，还会在子图上学习平滑尺度 $s_{\mathrm{sm}}$，并在 finalize 时写回融合权重或非融合 hook。

### 2.4 关键性质

- **可插拔 OP**：范围调参、舍入调参、可训练平滑可按需选用，而不是单一固定配方。
- **可扩展 OP**：训练管线可插拔；除内置 `minmax_tune` / `round_tune` / `trainable_smooth` 外，也可按同一接口自行扩展其他可训练算子。
- **块级训练**：以 decoder 块输出重构为目标，降低跨层误差累积对局部优化的误导。
- **策略化混合精度**：不同模块可用不同 `qconfig`，便于敏感层回退。
- **多数值格式**：伪量化内核覆盖 INT 与 MXFP 等已注册 dtype/method 组合。
- **推理无训练开销**：优化只发生在量化阶段，部署使用折叠后的量化参数。

从误差与适用边界看，低比特越低，舍入边界与裁剪越敏感，训练收益通常越明显，但也更容易过拟合少量校准样本。迭代不足可能尚未收敛；迭代过多或学习率不当可能只改善局部重构而不改善下游任务。

### 2.5 适用场景

- W4A4 等超低比特线性层量化，一次标定精度不足、可接受更高显存与耗时以换取重构精度。
- 需要在同一流水线中组合舍入优化、范围调参，甚至可训练平滑的场景。
- INT 与 MXFP 等 TLQ 内核已注册支持的数值格式。

更具体地说，是否适用主要取决于目标位宽、已注册 kernel 组合和算力预算。校准数据应能覆盖主要业务分布；资源紧张时可先缩小低比特策略覆盖范围，而不是关闭核心 OP。

### 2.6 使用限制

- 仅面向线性层；`strategies` 中的 dtype/method 组合必须有对应 TLQ kernel，否则配置校验失败。
- 包含块级训练，量化耗时与显存显著高于一次标定算法；当前实现不声明分布式训练支持。
- `trainable_smooth` 依赖适配器可识别的子图类型；子图不匹配时不应强行扩大 `enable_subgraph_type`。
- 低比特场景仍强烈依赖良好的前置离群值抑制，建议配合 [OASQ](../oasq/term_oasq.md)、[QuaRot](../quarot/term_quarot.md) 或 [Iterative Smooth](../iterative_smooth/term_iterative_smooth.md) 使用。

这些限制应在调参前确认。尤其是 dtype/scope/method 硬约束一旦不满足，继续增大 `iters` 通常无法解决问题。

---

## 3. 关联流程

- [《Trainable Linear Quant 参数配置流程指南》](./usage_trainable_linear_quant.md)：通过本流程可将 TLQ 写入量化任务并完成适配与调参。

---

## 4. 关联词条

可以从“同类方法、前后处理关系和应用对象”三个方向理解本词条与其他算法的关系。下面的关联项既用于横向比较不同技术路线，也用于帮助定位该算法在完整量化方案中的位置。

- [AutoRound](../autoround/term_autoround.md)：同类算法，同为基于 SignSGD 的可学习舍入优化；TLQ 将其思路扩展为可组合 OP 管线。
- [MinMax](../minmax/term_minmax.md)：对比算法，本算法在 MinMax 初值基础上进一步学习范围与舍入。
- [Linear Quant](../linear_quant/term_linear_quant.md)：应用对象，TLQ 作用于线性层量化配置。
- [OASQ](../oasq/term_oasq.md)：前置术语，可在可训练量化前抑制激活离群。
- [QuaRot](../quarot/term_quarot.md)：配套术语，低比特量化前常配合旋转类离群抑制。
- [DAOS](../daos/term_daos.md)：配套术语，多模态生成联合低比特方案，以 OASQ 为前置、本算法为第二阶段。

---

## 5. 参考文档

参考文档优先列出算法原始论文或权威出处，并补充仓库内对应使用指南。需要进一步理解参数选择时，可先阅读使用指南，再回到原论文核对算法假设和推导。

1. Cheng W, Zhang W, Shen H, et al. "Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs." Findings of EMNLP 2024. https://arxiv.org/abs/2309.05516
2. 《[Trainable Linear Quant 参数配置流程指南](./usage_trainable_linear_quant.md)》
