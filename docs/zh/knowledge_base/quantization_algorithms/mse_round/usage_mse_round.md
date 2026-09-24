# MSE_Round 参数配置流程指南

## 1. 适用范围

本指南面向需要使用[MSE_Round 权重量化算法](./term_mse_round.md)的用户。MSE_Round 作为 `linear_quant` 处理器的权重量化方法，通过 per-block 比较 ceil/floor 两档 shared exponent 的 MSE，提升 MXFP8 权重量化精度。

适用场景：

- MXFP8 权重量化精度不达标，需要通过逐 block 舍入选择降低重构误差；
- 目标部署后端明确支持 MXFP8 per-block 权重格式，且需要更优的 shared exponent 舍入策略。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 校准数据集 | 工具内置 `lab_calib/` 或用户自定义路径 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 输入 | 配置约束与实践基线 | 本算法配置说明、目标模型已有 `lab_practice` 配方（如有） | 字段名、支持组合、作用范围和模型适配与当前版本一致 | 推荐起点能够追溯到当前配置定义或已验证实践 |
| 交付件 | MSE_Round 参数配置方案 | 用户量化 YAML 或任务配置 | 参数取值合法、作用范围明确；关键参数说明选择依据 | 可作为后续量化流程的算法配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

MSE_Round 的整体使用流程如下：

```mermaid
flowchart LR
    A[确定 MXFP8<br/>per-block 方案] --> B[计算两种<br/>shared exponent 候选]
    B --> C[比较重构 MSE]
    C --> D[选择较优舍入]
    D --> E[量化权重]
```

各阶段的关键细节如下：

- **前置条件：模型适配**：量化前需先完成模型适配，适配器实现 `PipelineInterface`（基础流水线接口）等接口，详见[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。模型适配必须先完成，才能执行量化流程。
- **确定 MXFP8 per-block 方案**：当前 MSE-Round 只注册 MXFP8 per-block 对称权重量化，用户侧主要工作是确认格式与 axes 布局。
- **计算两种 shared exponent 候选**：算法对每个 block 的 shared exponent 计算向上/向下两个候选，无需用户干预。
- **比较重构 MSE**：算法自动比较两个候选的重构 MSE，选择误差更小的舍入结果。
- **量化权重**：按选定舍入执行 per-block 权重量化，提升 MXFP8 权重精度。
- **必要时换更稳健算法**：若 MXFP8 per-block 格式本身不满足目标，应换 MXFP4 或其他已验证方案，而不是给 MSE_Round 增加额外搜索参数。

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
        act:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: true
          method: "minmax"
        weight:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: true
          method: "mse_round"
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `scope` | 当前 MSE-Round 只注册 MXFP8 per-block 权重量化。 | 固定 `per_block`。 | 不支持其他 scope。 |
| `dtype` | 当前只支持 `mxfp8`。 | 固定 `mxfp8`。 | 需要 MXFP4 应使用相应算法。 |
| `symmetric` | 当前注册为对称 MXFP8。 | 固定 `true`。 | 不调。 |
| `method` | 固定 `mse_round`。算法对每个 block 的 shared exponent 比较向上/向下两个候选，选择重构 MSE 更小的一个。 | 固定。 | 算法已自动做局部舍入选择，不需要再暴露“向上/向下比例”参数。 |
| `ext.axes` | 决定沿哪些维度划分 MXFP8 block，默认 `-1`。它改变 block 组成和 shared exponent 统计，是布局/格式的重要部分。 | 通常 `-1`。 | 只有目标模型或部署格式明确要求其他轴时修改；改轴会重定义每个 block，必须重新量化和验证。 |

### 参数组合与选择顺序

MSE-Round 的算法内部已经自动比较两种 exponent 舍入，因此用户侧主要工作是确认 MXFP8 per-block 格式和 axes；没有必要人为增加额外搜索旋钮。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：根据结果收敛参数方案

**操作**：

- **先跑推荐配置，再调单变量。** 不要同时修改位宽、粒度、算法参数和层范围，否则很难判断精度变化来自哪一项。
- **优先回退局部，而不是整体提高精度。** 如果只有少数层敏感，优先通过 `exclude` 或混合策略保留高精度，通常比整体升位宽更划算。
- **最终以模型实践配置和部署能力为准。** 入门推荐用于建立稳定起点；目标模型已有 `lab_practice` 配方时，应优先复用已验证组合。

**输出**：一份可进入后续量化流程的最终参数方案，并保留相对于推荐配置的调整记录。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| MSE_Round 权重量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《MSE_Round 权重量化算法 量化术语百科词条》](./term_mse_round.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| linear_quant 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《linear_quant 配置说明》](../../../api_reference/config/processor/linear_quant.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/zh/user_guide/usage_weight_quantization.md) |
