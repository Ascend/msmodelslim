# 线性量化参数配置流程指南

## 1. 适用范围

本指南面向需要使用[线性量化算法](./term_linear_quant.md)的用户。线性量化通过 `linear_quant` 处理器对模型的线性层（`nn.Linear`）权重与激活进行量化，是大多数量化方案的基础。

适用场景：

- 需要对模型线性层权重与激活执行基础量化，建立可比较的量化基线；
- 需要在基线之上通过调整激活/权重的位宽、粒度与参数估计算法定位精度或压缩率收益。

模型是否在官方预验证列表中请参考[《大模型支持矩阵》](../../model/README.md)；如需快速了解 CLI 基础用法可参阅[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 目标模型与量化目标 | 待量化模型及部署/评测方案 | 明确目标数值格式或位宽、作用模块、精度/性能目标以及部署约束 | 能说明为什么选择本算法以及它在整体量化方案中的位置 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 配置约束与实践基线 | 本算法配置说明、目标模型已有 `lab_practice` 配方（如有） | 字段名、支持组合、作用范围和模型适配与当前版本一致 | 推荐起点能够追溯到当前配置定义或已验证实践 |
| 输入 | 校准数据（算法需要时） | 任务 `dataset`、校准集或模型实践配方 | 数据应能代表真实输入分布；多阶段算法尽量保持各阶段数据分布一致 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 交付件 | 线性量化参数配置方案 | 用户量化 YAML 或任务配置 | 参数取值合法、作用范围明确；关键参数说明选择依据 | 可作为后续量化流程的算法配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

线性量化的整体使用流程如下：

```mermaid
flowchart LR
    A[适配模型<br/>流水线接口] --> B[确定<br/>W/A 位宽]
    B --> C[选择激活<br/>与权重量化粒度]
    C --> D[选择参数估计算法]
    D --> E[圈定处理层]
    E --> F[比较精度/性能]
```

各阶段的关键细节如下：

- **适配模型流水线接口**：适配器需实现 `PipelineInterface`（基础流水线接口），量化调度器（Runner）与处理器（Processor）只通过标准流水线方法驱动模型。模型适配必须先完成，才能执行线性量化。
- **确定 W/A 位宽**：先固定目标位宽，再确认当前版本支持的参数组合；位宽由产品和后端目标决定，不作为小幅调参。
- **选择激活与权重量化粒度**：粒度越局部越能适应动态范围变化，但运行时尺度计算与元数据通常更多；必须与 dtype/symmetric/method 存在已注册量化器。
- **选择参数估计算法**：先让 method 与 dtype/scope 匹配，再考虑复杂度；不要因为某算法“更高级”就替换已满足精度的方法。
- **圈定处理层**：通过 `include/exclude` 控制作用范围，范围过宽会放大不兼容或敏感层风险，过窄则可能让算法收益看不出来。
- **比较精度/性能**：每轮只调一个变量，与基线直接比较；效果不理想时优先回到最近一次修改的参数。

## 4. 操作步骤

### 步骤 1：适配模型流水线接口

**目标**：量化工具不能直接操作任意结构的模型，需要适配器将模型操作转换为标准方法。`linear_quant` 处理器通过标准流水线方法驱动模型，模型适配器实现基础流水线接口即可。模型适配必须先完成，才能执行线性量化。

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
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_token"
          dtype: "int8"
          symmetric: true
          method: "minmax"
        weight:
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

ModelSlim 实现入口：
[查看对应实现目录](../../../../../msmodelslim/processor/quant/linear.py)

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定 `linear_quant`。 | 固定。 | 不调。 |
| `qconfig.act.dtype` | 激活数值格式。`float` 表示不量化激活；当前通用 QConfig 还支持 INT8/INT4/MXFP8/MXFP4/FP8 E4M3，但能否用于 Linear 取决于实际注册组合。 | 通用 LLM W8A8 从 `int8` 开始；只做权重量化则用 `float`；MX/FP8 只按已验证模型/后端配方选择。 | dtype 由产品和后端目标决定，不作为小幅调参。位宽降低会直接增大量化噪声，应重新选择 scope/method 并完整评测。 |
| `qconfig.act.scope` | 激活尺度粒度。仓库最常见 W8A8 配方是 `per_token`；也存在 INT8 per-tensor 非对称、PDMIX、MX per-block、DualScale 等专用组合。粒度越局部越能适应动态范围变化，但运行时尺度计算/元数据通常更多。 | 普通 INT8 LLM 首选 `per_token`；有模型实践时按其 per-tensor/PDMIX/per-block 等组合。 | 不要从 QConfig 枚举里任意挑 scope，必须与 dtype/symmetric/method 有已注册量化器。若 per-token 精度已满足，不必为了“更细”切到更复杂的专用 scope。 |
| `qconfig.act.symmetric` | 决定激活是否使用 zero-point。对称量化围绕 0，计算简单；非对称可以适应明显偏移分布。 | INT8 per-token 基线用 `true`；仓库也有 per-tensor 非对称和 PDMIX `false` 实践。 | 以量化器支持和后端为前提。激活分布有明显偏移且 per-tensor 对称损失高时可考虑非对称；per-token 对称已经能局部适应时通常先保持简单方案。 |
| `qconfig.act.method` | 激活尺度/裁剪估计方法。MinMax 是仓库最常用基线；Histogram 目前专用于 INT8 per-tensor 激活，通过直方图搜索 clipping。某些特殊格式的 method 由专用量化器定义。 | 先 `minmax`；只有明确存在长尾 clipping 问题且组合受支持时再考虑 `histogram` 或专用方法。 | 方法改变会改变校准需求和量化误差定义。MinMax 已满足精度时没必要增加复杂算法；长尾明显时再引入 Histogram。 |
| `qconfig.weight.dtype` | 权重数值格式。仓库常见 INT8 per-channel；W4A8 实践使用 INT4 per-channel SSZ；MX 模型使用 MXFP4/8 per-block。 | 普通 W8A8 用 `int8`；低比特目标按模型实践选 `int4` 或 MXFP4。 | 权重位宽是压缩收益的重要来源，也是精度风险来源。INT8 基线正常后再尝试 INT4，便于判断退化来自位宽还是其他处理。 |
| `qconfig.weight.scope` | 权重尺度粒度。INT 权重常用 per-channel，GPTQ 还支持 per-group；MX 格式常用 per-block，DualScale 使用专用 scope。 | INT8/INT4 权重的通用起点为 `per_channel`。 | 需要更细 group-wise GPTQ 或 MX block 时按对应算法指南配置。粒度越细一般更能适应通道内差异，但尺度元数据与后端要求更高。 |
| `qconfig.weight.symmetric` | 权重是否对称。Transformer Linear 权重通常围绕 0，仓库主流 INT/MX 实践均使用对称量化。 | 优先 `true`。 | 只有目标量化器明确注册非对称组合（如 GPTQ）且后端支持时才比较 false；普通 MinMax INT4/INT8 权重当前主要注册对称 per-channel。 |
| `qconfig.weight.method` | 权重量化算法。`minmax` 计算简单；`ssz` 对低比特 per-channel 权重迭代优化 scale；`gptq` 用校准激活 Hessian 做二阶误差补偿；`mse_round/ceil_x/fouroversix/dualscale` 属于特定 MX 格式。 | W8A8 从 `minmax`；W4A8 仓库已有 `ssz` 实践；需要二阶低比特优化时按 GPTQ 指南；MX 格式按其专用 method。 | 先让 method 与 dtype/scope 匹配，再考虑复杂度。不要因为某算法“更高级”就替换已满足精度的 MinMax；高级算法通常增加校准/搜索成本。 |
| `qconfig.*.ext` | 量化器专用扩展参数，不同 method 含义完全不同，例如 GPTQ 的 `percdamp/block_size/group_size`、SSZ 的 `step`、MX 方法的 `axes`/搜索参数。 | 普通 MinMax 无特殊需求时保持空；只有选用对应 method 才按其指南填写。 | 不要跨算法复用 ext 字段。修改 method 后应清理不属于新量化器的 ext，以免得到无效或误导配置。 |
| `include` | 作用范围白名单，使用模块名模式决定哪些匹配到的模块进入当前算法。它只决定“在哪些模块做”，不会改变算法内部公式。 | 无模型专用配方时从 `["*"]` 开始；已有 `lab_practice` 时直接沿用其模块范围。 | 先保证范围覆盖预期模块，再看精度。范围过宽时，少数结构不兼容或敏感层会放大整体风险；范围过窄则可能让算法收益看不出来。收窄范围时优先按结构族或已知敏感层调整，不建议仅凭层号大面积删除。 |
| `exclude` | 作用范围黑名单，命中后从 `include` 的候选中排除，优先级高于 `include`。适合保护敏感层、首尾层或模型专用不兼容结构。 | 默认先保持空列表；只有实践配方、兼容性约束或敏感性结果给出明确证据时再加入。 | 局部精度问题优先通过 `exclude` 做小范围回退，比提高全模型位宽或关闭整个算法更容易保留收益。每次增加排除项后应确认通配符没有误伤相邻模块。 |

### 常用组合怎么选

| 目标 | 激活建议 | 权重建议 | 说明 |
| --- | --- | --- | --- |
| 通用 W8A8 基线 | `int8 / per_token / symmetric / minmax` | `int8 / per_channel / symmetric / minmax` | 仓库 `lab_practice` 中出现最频繁，适合先建立稳定基线。 |
| 仅权重量化 | `float` | `int8 / per_channel / symmetric / minmax` | 不量化激活，适合先验证权重量化风险。 |
| W4A8 | `int8 / per_token / symmetric / minmax` | `int4 / per_channel / symmetric / ssz` | 仓库已有 DeepSeek/GLM 类实践；SSZ 细节见对应指南。 |
| MXFP8 | `mxfp8 / per_block / symmetric / minmax` | `mxfp8 / per_block / symmetric / minmax` 或 `mse_round` | 只有目标后端支持 MX 格式时使用。 |

选择时先确定“最终部署格式”，再确定“量化粒度”，最后选择“参数估计算法”。`include/exclude` 用于局部回退，不建议通过随意混搭未注册的 QConfig 组合来试错。每轮只改一个变量，并保持同一校准集和评测集；若调整后没有稳定收益，回到上一个基线。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：线性量化 W8A8 动态量化（推荐起点）

##### 配置文件：`linear_quant_w8a8.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  process:
    - type: linear_quant            # 线性层 W8A8 动态量化
      qconfig:
        act:
          dtype: int8
          scope: per_token          # 动态激活量化，精度表现好
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
  --config_path ./linear_quant_w8a8.yaml \
  --device npu:0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 线性量化算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | [《线性量化算法 量化术语百科词条》](./term_linear_quant.md) |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` | 模型流水线接口，由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| linear_quant 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《linear_quant 配置说明》](../../../api_reference/config/processor/linear_quant.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md) |
