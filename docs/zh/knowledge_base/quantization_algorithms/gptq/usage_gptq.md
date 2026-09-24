# GPTQ 参数配置流程指南

## 1. 适用范围

本指南面向需要使用 [GPTQ 二阶误差补偿权重量化算法](./term_gptq.md) 的用户。GPTQ 作为 `linear_quant` 处理器的权重量化方法，通过逐列量化与二阶误差补偿提升权重量化精度。

适用场景：

- 需要在 W8/W4 等权重量化场景下利用 Hessian 二阶信息补偿逐列量化误差；
- 需要在保留 per-channel/per-group 灵活粒度的同时，通过阻尼与分块参数稳定量化过程。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 校准数据集 | 工具内置 `lab_calib/` 或用户自定义路径 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | GPTQ 参数配置方案 | 用户量化 YAML 或任务配置 | 参数取值合法、作用范围明确；关键参数说明选择依据 | 可作为后续量化流程的算法配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

GPTQ 的整体使用流程如下：

```mermaid
flowchart LR
    A[确定权重量化粒度] --> B[收集 Hessian<br/>近似统计]
    B --> C[设置阻尼与分块]
    C --> D[逐列量化<br/>并补偿误差]
    D --> E[验证精度]
```

各阶段的关键细节如下：

- **前置条件：模型适配**：量化前需先完成模型适配，适配器实现 `PipelineInterface`（基础流水线接口）等接口，详见[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。模型适配必须先完成，才能执行量化流程。
- **确定权重量化粒度**：先锁定 `dtype + scope + symmetric + method` 组合；GPTQ 支持 `per_channel` 与 `per_group`，后者需配合 `group_size`。
- **收集 Hessian 近似统计**：通过校准数据（推荐 50 条）前向运行，估计每个 Linear 的输入二阶矩作为 Hessian 近似。
- **设置阻尼与分块**：`percdamp` 控制 Hessian 对角线阻尼比例，`block_size` 控制逐列处理的计算块大小。
- **逐列量化并补偿误差**：每个块内顺序量化并把误差传播到后续列，块之间也会做补偿。
- **验证精度**：对比量化前后的端到端精度；精度不足时优先回退局部或调整 scope/group_size，而不是直接提升全模型位宽。

## 4. 操作步骤

### 步骤 1：确认目标与约束

**操作**：先固定目标模型、最终量化格式/位宽、作用对象和评测基线，再确认当前版本支持的参数组合。目标模型已有 `lab_practice` 配方时，优先把该配方作为实践基线；没有模型专用配方时，再使用本指南给出的通用推荐起点。若算法依赖校准统计或优化数据，还应在调参前固定代表性数据，避免把数据分布变化误判为参数收益。

**输出**：一份明确的配置目标：目标位宽/格式、处理范围、校准条件、评测基线和部署约束。

### 步骤 2：建立推荐配置

**操作**：

下面配置用于建立**第一版可比较基线**。如果目标模型已有 `lab_practice` 配方，优先使用已验证配方，再参考本节理解每个参数为什么这样选。

```yaml
spec:
  process:
    - type: "linear_quant"
      qconfig:
        weight:
          scope: "per_channel"   # 量化范围
          dtype: "int8"          # 量化数据类型
          symmetric: true        # 是否对称量化
          method: "gptq"         # 量化算法-GPTQ
          ext: # 可选，扩展参数
            percdamp: 0.01       # 可选，阻尼系数，默认值0.01
            block_size: 128      # 可选，分块大小，默认值128
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `scope` | 当前 GPTQ 权重量化器支持 `per_channel` 和 `per_group`。per-channel 每个输出通道共享一套量化参数；per-group 进一步沿输入列分组，尺度更局部但会增加量化参数和布局约束。 | 无明确 group-wise 部署要求时先用 `per_channel`；目标低比特 group-wise 权重时用 `per_group`。 | 切换 scope 会改变尺度粒度和 GPTQ 更新过程，应重新校准。per-group 还必须配合 `group_size`，并要求输入通道数可整除该值。 |
| `dtype` | 当前 GPTQ 注册 INT8 与 INT4 权重，两种位宽都支持 per-channel/per-group。位宽越低，单个舍入误差更大，GPTQ 的二阶补偿价值通常越明显。 | 由最终部署目标决定；常规精度基线可先 INT8，低比特目标直接按 INT4 评估。 | 不要把 INT8 结果作为 INT4 的替代验证。换位宽后重新收集/使用校准数据并执行 GPTQ。 |
| `symmetric` | 当前 GPTQ 对 INT8/INT4 的 per-channel/per-group 都注册了对称与非对称实现。非对称多一个 offset，可表示偏移分布，但部署和元数据更复杂。 | 无明确偏移需求时优先 `true`；只有目标后端支持且权重分布明显非中心时再比较 `false`。 | 对称/非对称切换会改变初始 MinMax qparam 和后续误差补偿，不应只改导出参数；必须重新量化并评测。 |
| `method` | 固定选择 GPTQ。 | 固定 `gptq`。 | 切换 method 即切换权重优化算法。 |
| `ext.percdamp` | Hessian 阻尼比例，默认 `0.01`。源码把 `percdamp × mean(diag(H))` 加到 Hessian 对角线上，再做 Cholesky/求逆；因此 0.01 表示相对平均曲率的 1% 阻尼。阻尼能缓解病态/接近奇异的 Hessian，但过大也会抹平真实曲率差异。 | 先用默认 `0.01`。 | 若出现 Hessian 分解不稳定、数值条件差或量化结果对校准批次极度敏感，可逐步增大；若算法稳定且希望保留更强二阶信息则保持默认。不要把 percdamp 当成普通正则越大越好。 |
| `ext.block_size` | GPTQ 逐列处理时的计算块大小，默认 `128`。每块内顺序量化并把误差传播到后续列，同时块之间也会做补偿。它主要影响临时计算/执行粒度和数值行为。 | 先用 `128`。 | 显存/计算资源受限时可减小；希望减少循环块数量时可增大，但应重新验证数值稳定性和耗时。block_size 并不是量化 `scope`，不要与 group_size 混淆。 |
| `ext.group_size` | 仅 `per_group` GPTQ 使用，默认 `128`。它决定每组输入列共享量化参数的宽度；源码要求输入通道数能被 group_size 整除。 | 使用 `per_group` 时从 `128` 起步，并先检查每个目标 Linear 的输入维度可整除。per-channel 时无需配置。 | 减小 group_size 会让尺度更局部，通常有利于低比特重构但增加 scale/offset 元数据；增大则相反。选择时还必须满足部署算子支持与整除约束。 |

#### 参数组合与选择顺序

GPTQ 建议按 **位宽 → scope/group_size → percdamp → block_size** 的顺序选择。`group_size` 决定量化表示，`percdamp` 决定二阶数值稳定性，`block_size` 更多是执行粒度；不要反过来先调计算块。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：根据结果收敛参数方案

**操作**：

- 优先保持 `percdamp=0.01`，先调整层范围/分组粒度；只有出现数值不稳定或二阶估计不可靠时再动阻尼。
- **先跑推荐配置，再调单变量。** 不要同时修改位宽、粒度、算法参数和层范围，否则很难判断精度变化来自哪一项。
- **优先回退局部，而不是整体提高精度。** 如果只有少数层敏感，优先通过 `exclude` 或混合策略保留高精度，通常比整体升位宽更划算。
- **最终以模型实践配置和部署能力为准。** 入门推荐用于建立稳定起点；目标模型已有 `lab_practice` 配方时，应优先复用已验证组合。

**输出**：一份可进入后续量化流程的最终参数方案，并保留相对于推荐配置的调整记录。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| GPTQ 权重量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《GPTQ 权重量化算法 量化术语百科词条》](./term_gptq.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| linear_quant 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《linear_quant 配置说明》](../../../api_reference/config/processor/linear_quant.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/zh/user_guide/usage_weight_quantization.md) |
