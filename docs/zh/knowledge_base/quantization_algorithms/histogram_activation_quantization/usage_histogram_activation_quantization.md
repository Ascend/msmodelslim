# Histogram 参数配置流程指南

## 1. 适用范围

本指南面向需要使用 [Histogram 直方图激活值量化算法](./term_histogram_activation_quantization.md) 的用户。Histogram 作为 `linear_quant` 处理器的激活值量化方法，通过直方图分析自动搜索最优截断区间，用于提升激活值静态量化的精度。

适用场景：

- 激活分布存在长尾或离群值，直接 MinMax 量化时主体区间分辨率不足；
- 需要为静态激活量化选择自动裁剪区间的参数估计方法。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 校准数据集 | 工具内置 `lab_calib/` 或用户自定义路径 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | Histogram 参数配置方案 | 用户量化 YAML 或任务配置 | 参数取值合法、作用范围明确；关键参数说明选择依据 | 可作为后续量化流程的算法配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

Histogram 的整体使用流程如下：

```mermaid
flowchart LR
    A[收集激活直方图] --> B[搜索有效截断区间]
    B --> C[确定量化范围]
    C --> D[执行激活量化]
    D --> E[对比离群值影响]
```

各阶段的关键细节如下：

- **前置条件：模型适配**：量化前需先完成模型适配，适配器实现 `PipelineInterface`（基础流水线接口）等接口，详见[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。模型适配必须先完成，才能执行量化流程。
- **收集激活直方图**：观察器通过校准数据（推荐 50 条）前向运行，为整个激活张量维护 2048-bin 直方图。
- **搜索有效截断区间**：在直方图上搜索候选裁剪边界，允许舍弃少量尾部以减小主体区间的量化步长。
- **确定量化范围**：根据是否对称生成最终裁剪区间与量化参数；对称与非对称下最优裁剪边界不同。
- **执行激活量化**：按裁剪区间计算 scale/zero-point 并写入量化参数，仅作用于激活侧；权重仍需另选已注册算法。
- **对比离群值影响**：评估舍弃尾部对端到端精度的影响，确认 clipping 是否带来收益。

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
          scope: "per_tensor"  # 目前只支持per_tensor
          dtype: "int8"        # 目前只支持int8
          symmetric: false     # 支持对称/非对称量化，取值分别为True/False
          method: "histogram"  # 配置为"histogram", 即启用直方图激活值量化
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "minmax"     # 不支持直方图权重量化，此处不应配置为"histogram"
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `qconfig.act.scope` | 当前 Histogram 激活量化器只注册 `per_tensor`。观察器为整个激活张量维护直方图并搜索裁剪边界，不支持用本方法直接切成 per-token/per-channel。 | 固定 `per_tensor`。 | 如果需要 per-token 等粒度，应改用已注册的 MinMax 或其他量化器，不能只改字段。 |
| `qconfig.act.dtype` | 当前 Histogram 激活量化器只注册 INT8。 | 固定 `int8`。 | 其他 dtype 不属于当前实现。 |
| `qconfig.act.symmetric` | 控制最终裁剪区间/量化参数使用对称还是非对称方案。直方图算法的价值在于允许舍弃少量尾部，以减小主体区间的量化步长。 | 没有明显单侧偏移时可先 `true`；当前示例使用 `false`，适合分布存在明显偏移且后端支持非对称 INT8 的情况。 | 应根据激活分布和部署支持比较。非对称能利用零点覆盖偏移分布，但增加 offset；对称更简单。两种方案的直方图最优裁剪边界不同，切换后必须重新校准。 |
| `qconfig.act.method` | 固定 `histogram`。当前观察器内部使用 2048-bin 直方图并通过搜索裁剪边界降低量化误差；这些内部搜索细节目前不是 QConfig 的常规用户参数。 | 固定 `histogram`。 | 如果激活没有明显长尾、MinMax 已满足精度，可优先 MinMax 以降低校准复杂度；只有需要通过 clipping 对抗离群值时再用 Histogram。 |
| `qconfig.weight` | Histogram 当前只实现激活量化，权重需要另外选择已注册的权重量化器。示例配合 INT8 per-channel 对称 MinMax，是仓库最常见的 W8A8 基线之一。 | 权重先用 `int8 + per_channel + symmetric + minmax`；低比特权重需求再按 SSZ/GPTQ 等指南单独选择。 | 不要把 `method: histogram` 写到 weight。若更换权重算法，激活 Histogram 的校准和最终模型误差也会变化，应重新做整套评测。 |

#### 参数组合与选择顺序

Histogram 的关键决策不是“bin 数调多少”，因为当前用户配置并未暴露这类旋钮；主要选择是 **是否值得用 clipping 代替纯 MinMax**，以及对称/非对称是否符合激活分布和部署能力。

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
| Histogram 直方图激活值量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《Histogram 直方图激活值量化算法 量化术语百科词条》](./term_histogram_activation_quantization.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| linear_quant 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《linear_quant 配置说明》](../../../api_reference/config/processor/linear_quant.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/quant/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/zh/user_guide/usage_weight_quantization.md) |
