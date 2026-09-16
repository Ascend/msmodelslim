# DualScale 参数配置流程指南

## 1. 适用范围

本指南面向需要使用 [DualScale 双尺度量化算法](./term_dual_scale.md) 的用户。DualScale 作为 `linear_quant` 处理器的量化方法，通过两级缩放递进结构缓解异常通道影响，用于 Qwen3 稠密系列模型的 W4A4 低比特量化。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` 分片 | 可被目标 Transformers 版本正常加载 |
| 输入 | 校准数据集 | 任务 `dataset`、校准集或模型实践配方 | JSONL 或 JSON 格式文本 Prompt，推荐 50 条 | 能代表真实输入分布，能被当前量化流程正常读取并编码为前向张量 |
| 输入 | 量化配置文件 | 本地 YAML 文件 | 符合 `modelslim_v1` 协议规范 | 通过模式校验（Schema Validation） |
| 交付件 | 量化权重目录 | `--save_path` 指定路径 | 含 `quant_model_description.json` 及 `*.safetensors` 分片 | 导出完整且推理冒烟测试通过 |

## 3. 流程总览

DualScale 的整体使用流程如下：

```mermaid
flowchart LR
    A[确定 MXFP4<br/>双尺度方案] --> B[设置外层<br/>block 大小]
    B --> C[计算两级尺度]
    C --> D[量化激活与权重]
    D --> E[对比误差]
```

各阶段的关键细节如下：

- **前置条件：模型适配**：量化前需先完成模型适配，适配器实现 `PipelineInterface`（基础流水线接口）等接口，详见[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。模型适配必须先完成，才能执行量化流程。
- **确定 MXFP4 双尺度方案**：DualScale 注册为 MXFP4 `dual_scale` 专用粒度，在普通 MXFP4 block scale 之外再引入更大粒度的第二层尺度。
- **设置外层 block 大小**：通过 `ext.dual_block_size` 设置第二层尺度覆盖的大块大小；大块越小越适应局部分布，但尺度数量和开销更高。
- **计算两级尺度**：内部先做 MXFP4 per-block MinMax 量化，再按 `dual_block_size` 计算/保存额外 scale。
- **量化激活与权重**：激活与权重两路均可使用 dual_scale；首次比较建议保持两边 block 大小一致。
- **对比误差**：在同一模型上比较不同 outer block 大小的重构/任务精度与尺度开销，单变量逐步调整。

## 4. 操作步骤

### 步骤 1：确认目标与约束

**操作**：先固定目标模型、最终量化格式/位宽、作用对象和评测基线，再确认当前版本支持的参数组合。目标模型已有 `lab_practice` 配方时，优先把该配方作为实践基线；没有模型专用配方时，再使用本指南给出的通用推荐起点。若算法依赖校准统计或优化数据，还应在调参前固定代表性数据（推荐 50 条），避免把数据分布变化误判为参数收益。

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
          scope: "dual_scale"
          dtype: "mxfp4"
          symmetric: True
          method: "dualscale"
          ext:
            dual_block_size: 512
        weight:
          scope: "dual_scale"
          dtype: "mxfp4"
          symmetric: True
          method: "dualscale"
          ext:
            dual_block_size: 512
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `scope` | 当前 DualScale 量化器注册的专用粒度，表示在普通 MXFP4 block scale 之外再引入更大粒度的第二层尺度。 | 固定 `dual_scale`。 | 其他 scope 不属于本算法。 |
| `dtype` | 当前注册格式为 `mxfp4`。内部仍会先使用 MXFP4 per-block MinMax，再乘第二层 scale。 | 固定 `mxfp4`。 | 部署格式改变时应选择对应算法。 |
| `symmetric` | 当前注册组合为对称 MXFP4 DualScale。 | 固定 `true`。 | 当前不建议修改。 |
| `method` | 选择 DualScale 实现。 | 固定 `dualscale`。 | 不是精度旋钮。 |
| `ext.axes` | 决定沿哪个维度进行 inner block 与 outer dual block 的 reshape，默认 `-1`。它影响尺度共享的方向。 | 通常保持 `-1`，除非实践配置/后端明确指定其他轴。 | 轴与张量布局绑定，不能只为了离线 MSE 改轴；修改后需要同时验证权重与激活两路的布局和部署解析。 |
| `ext.dual_block_size` | 第二层尺度覆盖的大块大小。内部先做 MXFP4 小块量化，再按 `dual_block_size` 计算/保存额外 scale；大块越小，第二层尺度越局部，适应局部分布更强，但尺度数量和处理开销更高。 | 仓库 Qwen-Image-Edit 实践对激活和权重都使用 `512`，可作为该模型族/格式的优先起点。 | 局部动态范围差异很大时可尝试减小；若尺度元数据或处理开销更重要可增大。调整时激活和权重不一定必须相同，但首次比较建议保持一致，避免两个方向同时变化。 |

#### 参数组合与选择顺序

DualScale 的核心旋钮是 `dual_block_size`。先固定 MXFP4 + dual_scale 格式和 axes；再在同一模型上比较不同 outer block 大小的重构/任务精度与尺度开销。

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
| DualScale 双尺度量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《DualScale 双尺度量化算法 量化术语百科词条》](./term_dual_scale.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| linear_quant 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《linear_quant 配置说明》](../../../api_reference/config/processor/linear_quant.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md) |
