# AWQ Smooth 参数配置流程指南

## 1. 适用范围

本指南面向需要使用 [AWQ 激活感知权重量化算法](./term_awq_smooth.md) 的用户。AWQ 作为离群值抑制算法，通常作为权重量化前的预处理步骤，通过激活感知搜索最优缩放因子，提升低比特量化的精度。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 目标模型与量化目标 | 待量化模型及部署/评测方案 | 明确目标数值格式或位宽、作用模块、精度/性能目标以及部署约束 | 能说明为什么选择本算法以及它在整体量化方案中的位置 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 及 `AWQInterface` 接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 配置约束与实践基线 | 本算法配置说明、目标模型已有 `lab_practice` 配方（如有） | 字段名、支持组合、作用范围和模型适配与当前版本一致 | 推荐起点能够追溯到当前配置定义或已验证实践 |
| 输入 | 校准数据（算法需要时） | 任务 `dataset`、校准集或模型实践配方 | 数据应能代表真实输入分布；多阶段算法尽量保持各阶段数据分布一致 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 交付件 | AWQ Smooth 参数配置方案 | 用户量化 YAML 或任务配置 | 参数取值合法、作用范围明确；关键参数说明选择依据 | 可作为后续量化流程的算法配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

AWQ 的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线/AWQ 接口] --> B[确定下游<br/>权重量化方式]
    B --> C[收集激活<br/>重要性统计]
    C --> D[网格搜索缩放因子]
    D --> E[融合缩放]
    E --> F[进入权重量化]
```

各阶段的关键细节如下：

- **适配模型流水线/AWQ 接口**：适配器需实现 `PipelineInterface`（基础流水线接口）及 `AWQInterface`（AWQ 专用接口），提供 AWQ 子图结构映射，是后续所有搜索与应用步骤的前提。模型适配必须先完成，才能执行 AWQ 算法。
- **确定下游权重量化方式**：先固定最终量化格式/位宽、作用对象和评测基线；AWQ 搜索缩放系数时用于模拟的权重量化配置必须与最终权重量化器保持一致。
- **收集激活重要性统计**：通过校准数据（推荐 50 条）统计激活重要性，作为缩放因子搜索的依据。
- **网格搜索缩放因子**：以激活感知方式网格搜索最优缩放因子，网格越密候选越细，搜索时间也相应增加。
- **融合缩放**：将搜索得到的缩放因子融合进权重，使激活分布更均匀、权重量化误差更小。
- **进入权重量化**：平滑后衔接最终权重量化步骤；建议保持后续量化配置不变，只调整 AWQ 参数以定位收益。

## 4. 操作步骤

### 步骤 1：适配模型流水线与 AWQ 接口

**目标**：量化工具不能直接操作任意结构的模型，需要适配器将模型操作转换为标准方法。对于 AWQ，模型适配器除流水线接口外，还需实现 AWQ 专用接口，提供 AWQ 子图结构映射。模型适配必须先完成，才能执行 AWQ 算法。

**操作**：模型适配代码需实现以下接口：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，由 `msmodelslim.model.interface_hub` 汇总提供，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。
2. **`AWQInterface`**：AWQ 专用接口，由 `msmodelslim.processor.anti_outlier.awq` 提供，核心方法 `get_adapter_config_for_subgraph()` 返回 AWQ 子图映射配置（子图类型如 `norm-linear`、`linear-linear`、`ov`、`up-down`）。AWQ 处理器强制要求适配器实现该接口。

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
)
from msmodelslim.processor.anti_outlier.awq import (
    # ---- AWQ 算法适配接口（本步骤重点）----
    AWQInterface,  # 算法适配：get_adapter_config_for_subgraph 返回 AWQ 子图映射
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface, AWQInterface):
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
    - type: "awq"
      weight_qconfig:
        scope: "per_channel"
        dtype: "int8"
        symmetric: true
        method: "minmax"
      n_grid: 20
      enable_subgraph_type:
        - "norm-linear"
        - "linear-linear"
        - "ov"
        - "up-down"
      include: ["*"]
      exclude: []
```

**输出**：一份可复现的推荐配置，后续所有参数调整均以此为比较对象。

### 步骤 3：选择并调整参数

**操作**：

ModelSlim 实现入口：
[查看对应实现目录](../../../../../msmodelslim/processor/anti_outlier/awq)

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定为 `awq`。 | 固定。 | 不作为精度旋钮。 |
| `weight_qconfig` | AWQ 搜索缩放系数时用于模拟最终权重量化误差的配置。搜索器会用它评价候选缩放，因此其 `dtype/scope/symmetric/method` 必须代表最终权重量化器。 | 必须与下游最终权重量化配置保持一致。INT4 权重目标就用 INT4 配置；只做 W8 权重量化则用 INT8。 | 最终权重量化方式一旦变化，应重新搜索 AWQ scale。不要用 INT8 的搜索结果直接服务 INT4，也不要搜索时用 per-channel、落地时改成另一种粒度。 |
| `n_grid` | AWQ 缩放系数的网格搜索密度，默认 `20`。网格越密，候选更细，搜索时间也相应增加。 | 先用默认 `20`。 | 当最优点对网格位置很敏感、重复试验显示粗网格错过明显更优区间时再增加；快速验证可减少。若精度问题来自错误 qconfig 或子图范围，增加网格数不会解决。 |
| `enable_subgraph_type` | 控制 AWQ 在哪些可融合结构上搜索/应用平滑，支持 `norm-linear`、`linear-linear`、`ov`、`up-down`。不同模型结构可匹配的子图不同。 | 没有模型配方时从默认四类建立覆盖基线；有实践配置时只启用实践中验证的结构。 | 如果某类结构不匹配、效果不稳定或部署融合不支持，就只关闭该类；不要为了规避单个问题把所有子图都关掉。新增子图类型后要重新校准，因为搜索数据和权重路径都变了。 |
| `include` | 作用范围白名单，使用模块名模式决定哪些匹配到的模块进入当前算法。它只决定“在哪些模块做”，不会改变算法内部公式。 | 无模型专用配方时从 `["*"]` 开始；已有 `lab_practice` 时直接沿用其模块范围。 | 先保证范围覆盖预期模块，再看精度。范围过宽时，少数结构不兼容或敏感层会放大整体风险；范围过窄则可能让算法收益看不出来。收窄范围时优先按结构族或已知敏感层调整，不建议仅凭层号大面积删除。 |
| `exclude` | 作用范围黑名单，命中后从 `include` 的候选中排除，优先级高于 `include`。适合保护敏感层、首尾层或模型专用不兼容结构。 | 默认先保持空列表；只有实践配方、兼容性约束或敏感性结果给出明确证据时再加入。 | 局部精度问题优先通过 `exclude` 做小范围回退，比提高全模型位宽或关闭整个算法更容易保留收益。每次增加排除项后应确认通配符没有误伤相邻模块。 |

#### 参数组合与选择顺序

AWQ 最关键的联动是 **`weight_qconfig` ↔ 最终权重量化配置**。先把目标 qconfig 定死，再决定子图范围，最后才用 `n_grid` 微调搜索精度。每轮只改一个变量，并保持同一校准集和评测集；若调整后没有稳定收益，回到上一个基线。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：AWQ + W4A8 动态量化（推荐起点）

##### 配置文件：`awq_w4a8.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  process:
    - type: awq                     # 激活感知搜索权重缩放因子
      weight_qconfig:               # 与最终权重量化配置保持一致
        scope: per_channel
        dtype: int4
        symmetric: true
        method: minmax
      n_grid: 20
      enable_subgraph_type:
        - "norm-linear"
        - "linear-linear"
        - "ov"
        - "up-down"
      include: ["*"]
    - type: linear_quant            # 衔接 W4A8 动态量化
      qconfig:
        act:
          dtype: int8
          scope: per_token
          symmetric: true
          method: minmax
        weight:
          dtype: int4
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
  --config_path ./awq_w4a8.yaml \
  --device npu:0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| AWQ 激活感知权重量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《AWQ 激活感知权重量化算法 量化术语百科词条》](./term_awq_smooth.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` | 模型流水线接口，由 `msmodelslim.model.interface_hub` 汇总导出；`AWQInterface` 由 `msmodelslim.processor.anti_outlier.awq` 提供。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| awq 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《awq 配置说明》](../../../api_reference/config/processor/awq.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md) |
