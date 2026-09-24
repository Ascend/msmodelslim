# 浮点稀疏参数配置流程指南

## 1. 适用范围

本指南面向需要使用 [浮点稀疏（ADMM）算法](./term_float_sparse.md) 的用户。浮点稀疏（Float Sparse）通过 ADMM 算法对浮点权重进行稀疏化，结合硬件压缩单元实现高压缩率部署。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 目标模型与量化目标 | 待量化模型及部署/评测方案 | 明确目标数值格式或位宽、作用模块、精度/性能目标以及部署约束 | 能说明为什么选择本算法以及它在整体量化方案中的位置 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 配置约束与实践基线 | 本算法配置说明、目标模型已有 `lab_practice` 配方（如有） | 字段名、支持组合、作用范围和模型适配与当前版本一致 | 推荐起点能够追溯到当前配置定义或已验证实践 |
| 输入 | 校准数据（算法需要时） | 任务 `dataset`、校准集或模型实践配方 | 数据应能代表真实输入分布；多阶段算法尽量保持各阶段数据分布一致 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 交付件 | 浮点稀疏参数配置方案 | 用户量化 YAML 或任务配置 | 参数取值合法、作用范围明确；关键参数说明选择依据 | 可作为后续量化流程的算法配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

浮点稀疏的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线接口] --> B[确定目标稀疏率]
    B --> C[选择处理层范围]
    C --> D[优化<br/>稀疏掩码/权重]
    D --> E[应用稀疏化]
    E --> F[对比精度<br/>与压缩收益]
```

各阶段的关键细节如下：

- **适配模型流水线接口**：适配器需实现 `PipelineInterface`（基础流水线接口），量化调度器（Runner）与处理器（Processor）只通过标准流水线方法驱动模型。模型适配必须先完成，才能执行浮点稀疏。
- **确定目标稀疏率**：先从默认 `0.3` 建立中等稀疏率基线，在可观察压缩收益和精度风险之间保留调节空间。
- **选择处理层范围**：先通过 `include`/`exclude` 限定到适合稀疏化的线性层，再逐步提高比例。
- **优化稀疏掩码/权重**：算法用校准数据和二阶/重构信息做剪枝，优化稀疏掩码与剩余权重。
- **应用稀疏化**：将稀疏化结果应用到模型权重，得到零权重分布。
- **对比精度与压缩收益**：观察精度下降是否集中在少数层；同时确认部署后端是否真正利用稀疏格式，而不是继续提高稀疏率。

## 4. 操作步骤

### 步骤 1：适配模型流水线接口

**目标**：量化工具不能直接操作任意结构的模型，需要适配器将模型操作转换为标准方法。`float_sparse` 处理器通过标准流水线方法驱动模型，模型适配器实现基础流水线接口即可。模型适配必须先完成，才能执行浮点稀疏。

**操作**：模型适配代码需实现以下接口，相关接口由 `msmodelslim.model.interface_hub` 汇总提供：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。量化调度器（Runner）与处理器（Processor）只通过这些标准方法驱动模型，与模型内部结构无关。

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface):
    pass
```

之后量化命令中通过 `--model_type MyModelAdapter` 引用该适配器。

如需开发新模型适配代码，请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。请注意：模型适配必须先完成，才能执行后续量化流程。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 2：建立推荐配置

**操作**：

下面配置用于建立**第一版可比较基线**。如果目标模型已有 `lab_practice` 配方，优先使用已验证配方，再参考本节理解每个参数为什么这样选。

```yaml
spec:
  process:
    - type: "float_sparse"
      sparse_ratio: 0.3          # 稀疏比例，取值范围为 0.0~1.0，默认0.3。
      include: ["*"]             # 包含的层，支持通配符。
      exclude: ["*self_attn*"]   # 排除的层，支持通配符。
```

推荐从 `sparse_ratio: 0.3` 开始，是为了在可观察压缩收益和精度风险之间保留调节空间。`include`/`exclude` 与稀疏率同样重要：先限定到适合稀疏化的线性层，再逐步提高比例，比直接全模型使用高稀疏率更容易定位问题并保留精度。

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

ModelSlim 实现入口：
[查看对应实现目录](../../../../../msmodelslim/processor/sparse)

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定为 `float_sparse`。 | 固定。 | 不调。 |
| `sparse_ratio` | 目标置零权重比例，范围 0~1。算法用校准数据和二阶/重构信息做剪枝，比例越高代表压缩更激进，留下的非零权重更少，重构压力也更大。代码默认 `0.3`；Qwen3 W16A16S 实践使用 `0.4`。 | 新模型从默认 `0.3` 建立精度基线；只有同模型实践明确验证时可直接使用 `0.4`。 | 精度裕量充足、希望更高稀疏时逐步提高；任务指标明显下降时先降低 ratio，再考虑排除敏感层。不要把单个模型验证过的 0.4 当成所有模型通用值。 |
| `include` | 作用范围白名单，使用模块名模式决定哪些匹配到的模块进入当前算法。它只决定“在哪些模块做”，不会改变算法内部公式。 | 无模型专用配方时从 `["*"]` 开始；已有 `lab_practice` 时直接沿用其模块范围。 | 先保证范围覆盖预期模块，再看精度。范围过宽时，少数结构不兼容或敏感层会放大整体风险；范围过窄则可能让算法收益看不出来。收窄范围时优先按结构族或已知敏感层调整，不建议仅凭层号大面积删除。 |
| `exclude` | 作用范围黑名单，命中后从 `include` 的候选中排除，优先级高于 `include`。适合保护敏感层、首尾层或模型专用不兼容结构。 | 默认先保持空列表；只有实践配方、兼容性约束或敏感性结果给出明确证据时再加入。 | 局部精度问题优先通过 `exclude` 做小范围回退，比提高全模型位宽或关闭整个算法更容易保留收益。每次增加排除项后应确认通配符没有误伤相邻模块。 |

#### 参数组合与选择顺序

Float Sparse 中主要只有一个连续旋钮 `sparse_ratio`。因此更应把“比例”和“作用范围”分开调：先在稳定范围上找可接受比例，再决定是否扩大到更多层。每轮只改一个变量，并保持同一校准集和评测集；若调整后没有稳定收益，回到上一个基线。若少数层非常敏感，优先通过 `exclude` 局部保护，而不是整体降低所有层的稀疏率。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：浮点稀疏 W16A16S（推荐起点）

##### 配置文件：`float_sparse_w16a16s.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  process:
    - type: float_sparse            # ADMM 浮点稀疏化
      sparse_ratio: 0.3             # 稀疏比例，取值范围为 0.0~1.0
      include: ["*"]                # 包含的层，支持通配符
      exclude: ["*self_attn*"]      # 排除的层，支持通配符
  save:
    - type: ascendv1_saver          # 昇腾推理标准保存格式
  dataset: mix_calib.jsonl          # 内置混合校准集
```

##### 执行命令（单卡量化）

```bash
msmodelslim quant \
  --model_path <浮点模型目录> \
  --save_path <量化权重输出目录> \
  --model_type <模型适配器名称> \
  --config ./float_sparse_w16a16s.yaml \
  --device npu \
  --device_id 0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 浮点稀疏（ADMM）算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《浮点稀疏（ADMM）算法 量化术语百科词条》](./term_float_sparse.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` | 模型流水线接口，由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| float_sparse 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《float_sparse 配置说明》](../../../api_reference/config/processor/float_sparse.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/zh/user_guide/usage_weight_quantization.md) |
