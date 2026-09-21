# KVCache Quant 参数配置流程指南

## 1. 适用范围

本指南面向需要使用 [KVCache Quant 缓存量化算法](./term_kvcache_quant.md) 的用户。KVCache Quant 作为缓存量化处理器（`dynamic_cache`），对写入 KV Cache 的 Key/Value 状态进行 INT8 量化，用于降低缓存显存占用、提升长序列推理效率。

适用场景：

- 长序列推理中 KV Cache 显存占用成为瓶颈，需要将 K/V 状态从浮点降为 INT8；
- 需要按通道统计 K/V 范围并保留 per-channel 量化粒度，以适配部署端的缓存量化算子。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 目标模型与量化目标 | 待量化模型及部署/评测方案 | 明确目标数值格式或位宽、作用模块、精度/性能目标以及部署约束 | 能说明为什么选择本算法以及它在整体量化方案中的位置 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 配置约束与实践基线 | 本算法配置说明、目标模型已有 `lab_practice` 配方（如有） | 字段名、支持组合、作用范围和模型适配与当前版本一致 | 推荐起点能够追溯到当前配置定义或已验证实践 |
| 输入 | 校准数据（算法需要时） | 任务 `dataset`、校准集或模型实践配方 | 数据应能代表真实输入分布；多阶段算法尽量保持各阶段数据分布一致 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 交付件 | KVCache Quant 参数配置方案 | 用户量化 YAML 或任务配置 | 参数取值合法、作用范围明确；关键参数说明选择依据 | 可作为后续量化流程的算法配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

KVCache Quant 的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线接口] --> B[确定 C8/KV Cache<br/>目标]
    B --> C[按通道统计<br/>Key/Value 范围]
    C --> D[生成<br/>INT8 尺度]
    D --> E[量化写入<br/>缓存的数据]
    E --> F[验证<br/>长序列精度/显存]
```

各阶段的关键细节如下：

- **适配模型流水线接口**：适配器需实现 `PipelineInterface`（基础流水线接口），量化调度器（Runner）与处理器（Processor）只通过标准流水线方法驱动模型。模型适配必须先完成，才能执行 KVCache Quant。
- **确定 C8/KV Cache 目标**：锁定 `dtype + scope + symmetric + method` 组合；当前 `dynamic_cache` 处理器只接受 `per_channel` 粒度。
- **按通道统计 Key/Value 范围**：通过校准数据（推荐 50 条）前向运行，重排 KV 张量后按通道统计范围。
- **生成 INT8 尺度**：根据对称/非对称选择生成量化参数，INT8 是仓库实践的主流 KV Cache 位宽。
- **量化写入缓存的数据**：按 `include/exclude` 圈定的作用范围逐层写入量化 K/V，作用范围决定哪些模块进入缓存量化。
- **验证长序列精度/显存**：在长序列输入下比较精度与显存收益，必要时通过 KV Smooth 或层范围回退缓解精度损失。

## 4. 操作步骤

### 步骤 1：适配模型流水线接口

**目标**：量化工具不能直接操作任意结构的模型，需要适配器将模型操作转换为标准方法。`dynamic_cache` 处理器对 KV Cache 的校准与伪量化通过标准流水线方法驱动，模型适配器实现基础流水线接口即可。模型适配必须先完成，才能执行 KVCache Quant。

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
    - type: "dynamic_cache"
      qconfig:
        scope: "per_channel"
        dtype: "int8"
        symmetric: true
        method: "minmax"
      include: ["*"]
      exclude: []
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定 `dynamic_cache`，对 Transformers DynamicCache 的 K/V 状态进行校准与伪量化部署。 | 固定。 | 不调。 |
| `qconfig.scope` | 当前 DynamicCacheQuantProcessor 在源码中明确只接受 `per_channel`；处理时会重排 KV 张量后按通道统计。 | 固定 `per_channel`。 | 其他 scope 会直接不符合当前处理器约束。需要不同粒度时应使用其他 KV/FA 方案。 |
| `qconfig.dtype` | KV Cache 数值格式。当前仓库的两个 dynamic_cache 实践均使用 INT8。 | 优先 `int8`。 | 更低位宽/其他格式只有在对应量化器和部署链都支持时再尝试。KV Cache 误差会随序列长度持续进入注意力计算，因此不要仅按显存收益选择位宽。 |
| `qconfig.symmetric` | 是否对称量化 K/V 通道。仓库实践使用 `true`。 | 优先 `true`。 | 只有确认激活分布偏移明显、对应 per-channel 非对称激活量化器与部署端都支持时才比较 false；切换后重新校准。 |
| `qconfig.method` | 量化参数估计方法。仓库 dynamic_cache 实践使用 MinMax。 | 推荐 `minmax`。 | 先用简单稳定的 MinMax 建立 C8 基线。不要直接套用线性权重专用的 GPTQ/SSZ 等 method；合法性由 AutoActQuantizer 注册组合决定。 |
| `include` | 作用范围白名单，使用模块名模式决定哪些匹配到的模块进入当前算法。它只决定“在哪些模块做”，不会改变算法内部公式。 | 无模型专用配方时从 `["*"]` 开始；已有 `lab_practice` 时直接沿用其模块范围。 | 先保证范围覆盖预期模块，再看精度。范围过宽时，少数结构不兼容或敏感层会放大整体风险；范围过窄则可能让算法收益看不出来。收窄范围时优先按结构族或已知敏感层调整，不建议仅凭层号大面积删除。 |
| `exclude` | 作用范围黑名单，命中后从 `include` 的候选中排除，优先级高于 `include`。适合保护敏感层、首尾层或模型专用不兼容结构。 | 默认先保持空列表；只有实践配方、兼容性约束或敏感性结果给出明确证据时再加入。 | 局部精度问题优先通过 `exclude` 做小范围回退，比提高全模型位宽或关闭整个算法更容易保留收益。每次增加排除项后应确认通配符没有误伤相邻模块。 |

#### 参数组合与选择顺序

当前 DynamicCache 的自由度其实不大：**per-channel 是硬约束，仓库基线是 INT8 + symmetric + MinMax**。精度问题优先通过 KV Smooth、层范围回退或更合适的数据校准解决，而不是随意改成未验证粒度。每轮只改一个变量，并保持同一校准集和评测集；若调整后没有稳定收益，回到上一个基线。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：KVCache Quant C8 + W8A8 动态量化（推荐起点）

##### 配置文件：`kvcache_c8_w8a8.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  process:
    - type: dynamic_cache           # KV Cache INT8 per-channel 量化
      qconfig:
        scope: per_channel
        dtype: int8
        symmetric: true
        method: minmax
      include: ["*"]
    - type: linear_quant            # 衔接 W8A8 动态量化
      qconfig:
        act:
          dtype: int8
          scope: per_token
          symmetric: true
          method: minmax
        weight:
          dtype: int8
          scope: per_channel
          symmetric: true
          method: minmax
      include: ["*"]
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
  --config ./kvcache_c8_w8a8.yaml \
  --device npu \
  --device_id 0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| KVCache Quant 缓存量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《KVCache Quant 缓存量化算法 量化术语百科词条》](./term_kvcache_quant.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` | 模型流水线接口，由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| dynamic_cache 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《dynamic_cache 配置说明》](../../../api_reference/config/processor/dynamic_cache.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md) |
