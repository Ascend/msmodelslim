# FA3 Quant 参数配置流程指南

## 1. 适用范围

FA3 Quant 注意力激活量化算法。FA3 Quant 作为激活量化处理器，对注意力机制中的 Q、K、V 激活进行 per-head 量化，用于降低显存占用并提升推理效率。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 目标模型与量化目标 | 待量化模型及部署/评测方案 | 明确目标数值格式或位宽、作用模块、精度/性能目标以及部署约束 | 能说明为什么选择本算法以及它在整体量化方案中的位置 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 及 `FA3QuantAdapterInterface` 接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 配置约束与实践基线 | 本算法配置说明、目标模型已有 `lab_practice` 配方（如有） | 字段名、支持组合、作用范围和模型适配与当前版本一致 | 推荐起点能够追溯到当前配置定义或已验证实践 |
| 输入 | 校准数据（算法需要时） | 任务 `dataset`、校准集或模型实践配方 | 数据应能代表真实输入分布；多阶段算法尽量保持各阶段数据分布一致 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 交付件 | FA3 Quant 参数配置方案 | 用户量化 YAML 或任务配置 | 参数取值合法、作用范围明确；关键参数说明选择依据 | 可作为后续量化流程的算法配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

```mermaid
flowchart LR
    A[适配模型<br/>流水线/FA3 接口] --> B[确定 Attention<br/>量化目标]
    B --> C[选择统一或 Q/K/V<br/>独立配置]
    C --> D[统计量化尺度]
    D --> E[应用 Attention<br/>激活量化]
    E --> F[验证长序列精度]
```

## 4. 操作步骤

### 步骤 1：适配模型流水线与 FA3 接口

**目标**：量化工具不能直接操作任意结构的模型，需要适配器将模型操作转换为标准方法。对于 FA3 Quant，模型适配器除流水线接口外，还需实现 FA3 专用接口，在 Attention 指定位置注入量化占位模块。模型适配必须先完成，才能执行 FA3 量化。

**操作**：模型适配代码需实现以下接口，相关接口均由 `msmodelslim.model.interface_hub` 汇总提供：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。
2. **`FA3QuantAdapterInterface`**：FA3 专用接口，核心方法 `inject_fa3_placeholders(root_name, root_module, should_inject)`：在 Attention 模块内的指定插入点通过 `set_submodule` 安装 `FA3QuantPlaceHolder` 占位模块，是否注入由回调 `should_inject(模块全名)` 判定。占位模块后续由 FA3 处理器替换为监听器，用于收集 Q/K/V per-head 激活统计。

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
    # ---- FA3 算法适配接口（本步骤重点）----
    FA3QuantAdapterInterface,  # 算法适配：inject_fa3_placeholders 在 Attention 指定位置注入 FA3QuantPlaceHolder 占位模块
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface, FA3QuantAdapterInterface):
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
    - type: "fa3_quant"
      # 不设置 qconfig/details：使用默认 INT8 per-head 对称量化。
      include: ["*"]
      exclude: []
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

ModelSlim 实现入口：
[查看对应实现目录](../../../../../msmodelslim/processor/quant/fa3)

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定为 `fa3_quant`。 | 固定。 | 不作为调参项。 |
| `qconfig` | Q/K/V 共用的一套量化配置；与 `details` 互斥。两者都省略时，当前处理器构造默认 INT8、`per_head`、对称、MinMax 配置。 | 第一次验证通用 Attention/MLA 路径可省略，使用默认 INT8 per-head；已有模型实践时优先照实践显式使用 `fp8_e4m3 + per_token`、INT8 per-token 或 MXFP4 per-block。 | 统一配置能满足三路精度时不要使用 `details` 增加复杂度。改变 dtype/scope 后要重新校准/验证；尤其 `per_head` 与 `per_token/per_block` 的统计方式和数据依赖不同。 |
| `qconfig.dtype` | Q/K/V 的数值格式。仓库实践已出现 `int8`、`fp8_e4m3`、`mxfp4`，具体合法性还取决于 scope/method 注册和部署后端。 | 没有模型专用配方时先用默认 INT8；有实践配方且目标后端支持时直接沿用其 FP8/MXFP 格式。 | dtype 是部署目标，不是简单“精度旋钮”。切换格式时同时检查 scope、量化方法、硬件/算子支持，并重做量化评测。 |
| `qconfig.scope` | 决定 FA 张量尺度粒度。当前 FA3 后处理明确支持 `per_head`、`per_token`、`per_block`。`per_head` 为每个注意力头保存统计尺度；`per_token/per_block` 更偏运行时或格式化局部尺度。 | 通用默认为 `per_head`；仓库多个 FA3 实践使用 `per_token`，MXFP4 实践使用 `per_block`。 | `per_head` 需要校准统计并能捕获不同 Head 的范围差异；源码中仅当所有启用分支都是 per-token/per-block 时可走 data-free 路径。优先按模型实践和部署格式选，不要只依据粒度“越细越好”。 |
| `qconfig.symmetric` / `method` | 决定是否有 zero-point 以及如何估计尺度。现有默认与主要实践均使用对称 + MinMax。 | 优先 `symmetric: true`、`method: minmax`。 | 只有注册组合和部署端明确支持其他设置时才偏离；不要把不支持的通用 QConfig 枚举直接用于 FA3。 |
| `details` | 允许 `fa_q`、`fa_k`、`fa_v` 分别配置，适合三路敏感性或目标格式不同的场景；某一路配置为 `null` 时当前处理逻辑可跳过该分支。 | 入门不设置。只有统一 `qconfig` 无法兼顾三路时再拆分。 | 先用 Attention MSE 或量化评测定位是 Q、K 还是 V 更敏感，再只提高敏感分支精度/更换粒度。不要一开始就给三路都不同配置，否则很难归因。 |
| `include` | 作用范围白名单，使用模块名模式决定哪些匹配到的模块进入当前算法。它只决定“在哪些模块做”，不会改变算法内部公式。 | 无模型专用配方时从 `["*"]` 开始；已有 `lab_practice` 时直接沿用其模块范围。 | 先保证范围覆盖预期模块，再看精度。范围过宽时，少数结构不兼容或敏感层会放大整体风险；范围过窄则可能让算法收益看不出来。收窄范围时优先按结构族或已知敏感层调整，不建议仅凭层号大面积删除。 |
| `exclude` | 作用范围黑名单，命中后从 `include` 的候选中排除，优先级高于 `include`。适合保护敏感层、首尾层或模型专用不兼容结构。 | 默认先保持空列表；只有实践配方、兼容性约束或敏感性结果给出明确证据时再加入。 | 局部精度问题优先通过 `exclude` 做小范围回退，比提高全模型位宽或关闭整个算法更容易保留收益。每次增加排除项后应确认通配符没有误伤相邻模块。 |

#### 参数组合与选择顺序

FA3 的首要决策是 `qconfig` 统一配置还是 `details` 分路配置。先统一，再定位敏感分支；只有证据表明 Q/K/V 需求不同才拆分。每轮只改一个变量，并保持同一校准集和评测集；若调整后没有稳定收益，回到上一个基线。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：FA3 Quant + W8A8 动态量化（推荐起点）

##### 配置文件：`fa3_quant_w8a8.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  process:
    - type: fa3_quant               # Attention Q/K/V 激活量化，按注意力头统计尺度
      qconfig:
        dtype: int8
        scope: per_head
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
  --config ./fa3_quant_w8a8.yaml \
  --device npu \
  --device_id 0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| FA3 Quant 注意力激活量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | 《[FA3 Quant 注意力激活量化算法 量化术语百科词条](./term_fa3_quant.md)》 |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` / `FA3QuantAdapterInterface` | 模型流水线与 FA3 Quant 算法适配接口，均由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| fa3_quant 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《fa3_quant 配置说明》](../../../api_reference/config/processor/fa3_quant.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md) |
