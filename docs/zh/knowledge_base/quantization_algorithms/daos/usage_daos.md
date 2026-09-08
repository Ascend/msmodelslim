# DAOS 参数配置流程指南

## 1. 适用范围

DAOS 是面向多模态**生成**模型的低比特联合方案。它通过串联 `oasq`（离群感知平滑）与 `trainable_linear_quant`（TLQ 可训练线性量化），用于 W4A4 / MXFP 等极低比特场景；需要进一步加速 Attention 推理时，可再配置 `fa3_quant` 做 FA 量化。

本指南面向需要**把 DAOS（OASQ + TLQ）配进量化任务**的用户：说明串联顺序、阶段目标与约束。各算法的适配、推荐起步配置与字段调参，请分别查阅《[OASQ 使用指南](../oasq/usage_oasq.md)》、《[TLQ 使用指南](../trainable_linear_quant/usage_trainable_linear_quant.md)》、《[FA3 Quant 使用指南](../fa3_quant/usage_fa3_quant.md)》，不在此重复展开。

- **适用**：需要 OASQ + TLQ 联合恢复低比特精度；任务协议为 `multimodal_sd_modelslim_v1`（或对应多模态生成任务协议）。
- **不适用**：一次标定已够用、无需 OASQ/TLQ；目标 dtype/scope/method 无对应 TLQ kernel；适配器无法提供 OASQ 可融合子图且默认探测也不适用；或设备显存/时长无法支撑块级训练。

> **资源提示**：DAOS 含 OASQ 平滑与 TLQ 块级训练，量化耗时与显存显著高于一次标定算法；显存是否够用通常要在启动量化后才能确认，见步骤 3。

如果目标模型已有完整且已验证的量化配方，应优先复用该配方。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 量化任务配置 | 用户 YAML 或 `lab_practice` 配方 | 符合 `multimodal_sd_modelslim_v1`（或对应任务协议），可在 `spec.process` 中串联写入 | 能被量化流程加载 |
| 交付件 | DAOS 配置片段 | 写入上述 YAML 的 `process` 项 | 先 `oasq` 后 `trainable_linear_quant`；按需追加 `fa3_quant`；各段字段分别符合对应算法配置说明 | 可随完整配置复现量化 |

> 模型适配、校准数据属于运行量化的前置条件，在步骤 2 核对；本表只列本指南直接改动的配置输入与交付件。

## 3. 流程总览

本指南的操作顺序如下（与第 4 章步骤一一对应）：

```mermaid
flowchart LR
    A[确定量化模式] --> B[适配与编写配置]
    B[适配与编写配置] --> C[启动量化]
    C[启动量化] --> D[测评与迭代调参]
```

## 4. 操作步骤

### 步骤 1：确定量化模式

**目标**：固定 DAOS 要落地的量化模式，作为后续适配与配置的前提。

**操作**：

1. 明确线性层 W/A 目标位宽与数值格式（如 MXFP4 / INT4 等）。
2. 明确是否需要对 Attention 做 FA 量化以加速推理（是否纳入 `fa3_quant`）。
3. 明确 TLQ 的 OP 路径倾向：通常使用 `minmax_tune` + `round_tune`；也可按 TLQ 使用指南**单独**启用 `trainable_smooth`（不可与前两者联用）。

> 所需显存与耗时难以在配置前准确预估，不必在本步做资源定量；以步骤 3 实际运行为准。

**输出**：一份明确的量化模式说明（W/A 目标、是否开 FA、TLQ OP 路径）。

### 步骤 2：完成模型适配与配置编写

**目标**：按对应算法文档完成适配，并在 `spec.process` 中写入可加载的 DAOS 串联配置。

**操作**：

1. 若目标模型已有含完整 DAOS 链路的 `lab_practice` 配方，优先复用。
2. 否则按 **OASQ → TLQ → 可选 FA3** 完成适配与配置编写：
   - **OASQ**：见《[OASQ 使用指南](../oasq/usage_oasq.md)》完成子图适配并写入 `oasq`。
   - **TLQ**：见《[TLQ 使用指南](../trainable_linear_quant/usage_trainable_linear_quant.md)》完成适配并写入 `trainable_linear_quant`；`operations` 按步骤 1 选定路径配置；`strategies[].qconfig` 与最终部署位宽一致。若选用 `trainable_smooth`，须按该指南确认子图能力。
   - **FA3**：若步骤 1 决定开启 FA，见《[FA3 Quant 使用指南](../fa3_quant/usage_fa3_quant.md)》完成适配，并在 TLQ 之后追加 `fa3_quant`。
3. 字段类型与约束见第 6 章接口文档列表。
4. 核对串联约束：
   - 顺序为 OASQ → TLQ → 可选 FA3。
   - OASQ 的 `symmetric` 应与后续量化对称性假设一致。
   - `minmax_tune` / `round_tune` 与 `trainable_smooth` 不要混用。

**输出**：已写入 YAML 的 DAOS `process` 串联项。

### 步骤 3：启动量化

**目标**：启动整链量化并跑通；若出现显存不足，优先调整下列资源相关参数后重试。

**操作**：

1. 使用步骤 2 的配置启动量化。
2. 若 OASQ 疑似空跑，先按《[OASQ 使用指南](../oasq/usage_oasq.md)》排查适配与 `include`/`exclude`，再继续后续训练类参数调整。
3. **显存不足时**，可优先尝试下列调整，每次只改一项，细则见《[TLQ 使用指南](../trainable_linear_quant/usage_trainable_linear_quant.md)》：
   - 增大 `train_config.gradient_accumulate_steps`，以降低峰值显存；该改动会改变等效更新频率。
   - 收窄 TLQ `strategies` 中需训练的低比特范围，例如增加 `exclude`，或把部分模块迁到更高位宽策略。
   - 必要时临时降低 `train_config.iters` 以先验证链路可跑通；正式精度跑数时再恢复合理迭代次数。
4. 跑通后得到量化产物，进入步骤 4 测评。

**输出**：可复现的量化运行结果；若曾因显存不足调整过配置，保留最终可跑通的配置。

### 步骤 4：测评与迭代调参

**目标**：完成量化后做端到端测评；不达标则单变量迭代调参，直至满足目标或明确回退策略。

**操作**：

1. 保持评测条件不变，对比量化前后（或与一次标定基线）的端到端指标。
2. 精度不达标时，按对应算法使用指南做单变量调整后重新量化，例如：
   - OASQ：作用范围、`enable_subgraph_type`、`max_iters` 等。
   - TLQ：`strategies` 混合位宽、`train_config`（`iters` / `lr` 等）、OP 相关开关。
   - FA3：qconfig 与 Attention 作用范围（若已启用）。
3. 上游 OASQ 或主目标位宽发生变化后，应重新执行整链，而不是拼接旧结果。
4. 每次只改一项，并保留相对基线的调整记录。

**输出**：可合入任务配置的最终 DAOS 片段，以及相对基线的调整记录。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| DAOS 低比特量化方案 | 方案定义、原理、性质与限制。 | 《[DAOS 低比特量化方案 量化术语百科词条](./term_daos.md)》 |
| OASQ | DAOS 前置离群抑制。 | 《[OASQ 词条](../oasq/term_oasq.md)》 |
| Trainable Linear Quant | DAOS 可训练线性量化阶段。 | 《[TLQ 词条](../trainable_linear_quant/term_trainable_linear_quant.md)》 |
| FA3 Quant | 可选的 Attention FA 量化。 | 《[FA3 Quant 词条](../fa3_quant/term_fa3_quant.md)》 |

## 6. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| oasq 配置说明 | 字段类型、默认值与约束。 | 《[oasq 配置说明](../../../api_reference/config/processor/oasq.md)》 |
| trainable_linear_quant 配置说明 | 字段类型、默认值与约束。 | 《[trainable_linear_quant 配置说明](../../../api_reference/config/processor/trainable_linear_quant.md)》 |
| fa3_quant 配置说明 | 字段类型、默认值与约束。 | 《[fa3_quant 配置说明](../../../api_reference/config/processor/fa3_quant.md)》 |
| multimodal_sd_modelslim_v1 | 多模态生成任务级配置。 | 《[multimodal_sd_modelslim_v1 配置说明](../../../api_reference/config/task/multimodal_sd_modelslim_v1.md)》 |
