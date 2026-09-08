# DAOS 低比特量化方案 量化术语百科词条

> **词条类别**：[量化算法](../README.md#2-量化算法)<br>
> **英文名称**：daos<br>
> **应用领域**：多模态生成模型量化压缩、低比特量化精度优化<br>

---

## 1. 概述

DAOS 是一种面向多模态**生成**模型的 **W4A4 / MXFP** 联合低比特方案（非单一 processor）。它先用 [OASQ](../oasq/term_oasq.md) 对激活离群做通道级平滑抑制，再用 [Trainable Linear Quant](../trainable_linear_quant/term_trainable_linear_quant.md) 以块级训练优化量化范围与舍入；核心特征是**先 OASQ、后 TLQ**的分治串联。方案中还可配合 [FA3 Quant](../fa3_quant/term_fa3_quant.md) 对 Attention 激活做 FA 量化，以进一步加速推理。

---

## 2. 词条介绍

多模态生成模型（文生视频 / 图生视频等）在 W4A4、MXFP4 等极低比特下同样面临激活离群与舍入误差叠加的问题。DAOS 将离群值抑制与可训练线性量化组合：OASQ 先按通道 z-score 分流并平滑离群激活，TLQ 再在更友好的分布上对线性层做块级重构训练，从而在生成模型的低比特部署路径下恢复精度。

从量化流程定位看，DAOS 不是单一 processor，而是 `multimodal_sd_modelslim_v1` 任务下由 OASQ 与 TLQ 串联构成的推荐方案；具体参数与实践配方见《[DAOS 参数配置流程指南](./usage_daos.md)》。

### 2.1 核心思想

DAOS 的核心思想是**先抑制离群、再训练量化参数**：

1. **OASQ**：在校准激活上识别离群通道，对可融合子图（如 norm-linear）施加平滑尺度，降低后续低比特动态范围浪费。
2. **TLQ**：在 OASQ 之后，对线性层执行可训练量化（范围 / 舍入等 OP），以块输出重构损失优化伪量化参数，并折叠回最终权重量化表示。

两阶段互补：OASQ 改善激活条件，不单独完成最终低比特表示；TLQ 优化离散化与尺度，但对严重离群通道仍敏感。先平滑再训练，通常比单独开 TLQ 或单独开 OASQ 更稳。

### 2.2 工作机制

DAOS 将多模态生成低比特量化中的两类困难分开处理。第一阶段用 OASQ 按通道识别并平滑离群激活，使后续量化面对更均衡的动态范围；第二阶段在已平滑的分布上，用 TLQ 对线性层做块级训练，优化量化范围与舍入等参数，减小离散化带来的重构误差。

这两类操作互补：平滑主要缓解离群通道对量化步长的支配，却不能消除有限码点本身的舍入误差；可训练量化可在固定低比特网格上优化尺度与舍入，却仍受严重离群影响。先整形分布、再优化离散决策，通常比把全部压力交给单一步骤更稳健。

得益于方案的重建思想，DAOS 可在低比特下保持较高精度，从而能够进一步迭代叠加 `fa3_quant`，对 Attention 路径做 FA 量化以加速推理。

校准依赖 `multimodal_sd_config.dump_config` 浮点 dump（通常需 `enable_dump: true`），与 LLM 文本 jsonl 校准路径不同。

### 2.3 数学描述（概念）

OASQ 阶段在子图上学习平滑尺度 $s$，使激活与权重按 $X' = X \operatorname{diag}(s)^{-1}$、$W' = \operatorname{diag}(s) W$ 迁移离群能量，并保持 $X'W' = XW$。

- $s$：逐通道平滑尺度
- $X$、$W$：平滑前的激活与权重
- $X'$、$W'$：平滑后的激活与权重

TLQ 阶段在伪量化前向下最小化块级重构损失，例如：

$$
\min_{\theta}\ \| Y_{\mathrm{fp}} - Y_{q}(\theta) \|_{1}
$$

- $\theta$：可训练的尺度 / 舍入等 OP 参数
- $Y_{\mathrm{fp}}$：浮点块输出
- $Y_{q}(\theta)$：伪量化块输出

整体仍是分阶段近似：OASQ 最优尺度未必是考虑 TLQ 训练后的全局最优，但工程上可分解、可复现。

### 2.4 关键性质

- **联合方案而非单算法**：由 `oasq` + `trainable_linear_quant` 串联组成。
- **面向多模态生成**：任务 API 为 `multimodal_sd_modelslim_v1`；校准为 dump 路径，非 LLM 文本默认集。
- **可进一步叠加 FA 量化**：在低比特精度可接受的基础上，叠加 FA3 等对 Attention 做 FA 量化，进一步加速推理。
- **依赖适配器子图**：OASQ 需要适配器提供可用的 `get_adapter_config_for_subgraph`（推荐实现 `OASQInterface`）；否则可能回退默认探测并导致作用范围为空。

### 2.5 适用场景

- 多模态生成模型的 W4A4 / MXFP 等低比特量化。
- 激活通道离群明显、仅用 MinMax + 静态线性量化精度不足的场景。

### 2.6 使用限制

- 含 OASQ 平滑与 TLQ 块级训练，量化耗时显著长于一次标定的静态方案，正式跑数前需预留充足时间预算。
- 校准 dump 与块级训练会明显抬高显存占用，资源紧张时应提前评估设备容量，必要时缩小作用范围或降低训练并行度。
- OASQ 作用范围须与模型子图命名一致；适配器未正确暴露子图时 OASQ 可能空跑。
- 部署格式需与保存器一致；启用 FA3 等时另依赖对应适配器接口。

---

## 3. 关联流程

- [《DAOS 参数配置流程指南》](./usage_daos.md)：通过本流程可将 OASQ 与 TLQ（及可选 FA3）串联配置并落地 DAOS。

---

## 4. 关联词条

- [OASQ](../oasq/term_oasq.md)：前置术语，DAOS 的离群值抑制阶段。
- [Trainable Linear Quant](../trainable_linear_quant/term_trainable_linear_quant.md)：配套术语，DAOS 的可训练线性量化阶段。
- [LAOS](../laos/term_laos.md)：同类算法，面向 LLM 的联合低比特方案（Adapt Rotation + AutoRound）。
- [FA3 Quant](../fa3_quant/term_fa3_quant.md)：配套术语，Attention FA 量化，用于进一步加速推理。

---

## 5. 参考文档

1. 《[DAOS 参数配置流程指南](./usage_daos.md)》
2. 《[OASQ 参数配置流程指南](../oasq/usage_oasq.md)》
3. 《[Trainable Linear Quant 参数配置流程指南](../trainable_linear_quant/usage_trainable_linear_quant.md)》
4. 《[FA3 Quant 参数配置流程指南](../fa3_quant/usage_fa3_quant.md)》
