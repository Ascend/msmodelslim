# FourOverSix 参数配置流程指南

## 1. 适用范围

本指南面向需要使用 [FourOverSix 自适应块缩放量化算法](./term_fouroversix.md) 的用户。FourOverSix 作为 `linear_quant` 处理器的权重量化方法，通过 per-block 在 Scale-to-6 与 Scale-to-4 间 MSE 择优，提升 mxFP4 权重量化精度。

适用场景：

- 需要在 mxFP4 权重量化场景下逐块比较 Scale-to-4 与 Scale-to-6，按重构 MSE 选择误差更小的方案；
- 需要在使用 MXFP4 per-block 布局约束的同时，提升低比特权重量化精度。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 校准数据集 | 工具内置 `lab_calib/` 或用户自定义路径 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | FourOverSix 参数配置方案 | 用户量化 YAML 或任务配置 | 参数取值合法、作用范围明确；关键参数说明选择依据 | 可作为后续量化流程的算法配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

FourOverSix 的整体使用流程如下：

```mermaid
flowchart LR
    A[确定 MXFP4<br/>per-block 方案] --> B[逐块比较<br/>Scale-to-4/6]
    B --> C[选择误差更小尺度]
    C --> D[量化权重]
    D --> E[对比误差]
```

各阶段的关键细节如下：

- **前置条件：模型适配**：量化前需先完成模型适配，适配器实现 `PipelineInterface`（基础流水线接口）等接口，详见[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。模型适配必须先完成，才能执行量化流程。
- **确定 MXFP4 per-block 方案**：先锁定部署目标与当前量化器实际注册的 `dtype + scope + symmetric + method` 组合；当前实现只注册 mxFP4 per-block 对称量化。
- **逐块比较 Scale-to-4/6**：算法对每个 MXFP4 block 分别尝试 scale-to-6 与 scale-to-4 两种尺度方案。
- **选择误差更小尺度**：按重构 MSE 在每个 block 内选择误差更小的方案，是 FourOverSix 相对单一尺度的核心收益来源。
- **量化权重**：按选定尺度应用最终量化；通过 `ext.axes` 控制 MXFP4 block 的划分轴。
- **对比误差**：与基线配置比较端到端精度，决定是否保留 FourOverSix 或调整 block 布局。

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
          scope: "per_block"    # 量化范围
          dtype: "mxfp4"        # 量化数据类型
          symmetric: true       # 是否对称量化
          method: "fouroversix" # 量化算法-FourOverSix
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `scope` | 当前实现只注册 MXFP4 per-block 权重量化。 | 固定 `per_block`。 | 不支持时换算法。 |
| `dtype` | 当前只支持 `mxfp4`。 | 固定 `mxfp4`。 | 不是调参项。 |
| `symmetric` | 当前注册组合为对称。 | 固定 `true`。 | 不要改成非对称。 |
| `method` | 选择 FourOverSix：每个 MXFP4 block 分别尝试 scale-to-6 与 scale-to-4，并按重构 MSE 选更好方案。 | 固定 `fouroversix`。 | 切换即换算法。 |
| `ext.axes` | 指定沿哪些维度划分 MXFP4 block，默认 `-1`。当前实现还要求所选轴长度能被 MXFP4 block_size 整除，否则会直接报错。 | 线性权重优先保持 `-1`。 | 只有模型权重布局/部署格式明确要求时改轴；改前先检查对应维度是否整除 block_size。FourOverSix 本身没有额外的“4/6 阈值”可调，axes 主要是布局选择，不应当成日常精度旋钮。 |

#### 参数组合与选择顺序

FourOverSix 是自动逐块二选一算法，通常没有必要再人为搜索 4/6 参数。用户真正需要确认的是 MXFP4 格式、block 轴和后端布局是否一致。

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
| FourOverSix 自适应块缩放量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《FourOverSix 自适应块缩放量化算法 量化术语百科词条》](./term_fouroversix.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| linear_quant 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《linear_quant 配置说明》](../../../api_reference/config/processor/linear_quant.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md) |
