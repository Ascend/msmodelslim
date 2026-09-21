# Flex AWQ SSZ 参数配置流程指南

## 1. 适用范围

Flex AWQ SSZ 灵活激活感知权重量化平滑算法。Flex AWQ SSZ 作为离群值抑制算法，通常作为量化前的预处理步骤，通过真实量化器评估参数、自动搜索最优 `alpha` 抑制激活离群值，提升低比特量化的精度。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 目标模型与量化目标 | 待量化模型及部署/评测方案 | 明确目标数值格式或位宽、作用模块、精度/性能目标以及部署约束 | 能说明为什么选择本算法以及它在整体量化方案中的位置 |
| 输入 | 模型适配器 | 用户适配代码，通过 `--model_type` 调用 | 实现 `PipelineInterface` 及 `FlexSmoothQuantInterface` 接口 | 能被调度器（Runner）与处理器（Processor）正常驱动 |
| 输入 | 配置约束与实践基线 | 本算法配置说明、目标模型已有 `lab_practice` 配方（如有） | 字段名、支持组合、作用范围和模型适配与当前版本一致 | 推荐起点能够追溯到当前配置定义或已验证实践 |
| 输入 | 校准数据（算法需要时） | 任务 `dataset`、校准集或模型实践配方 | 数据应能代表真实输入分布；多阶段算法尽量保持各阶段数据分布一致 | 能被当前量化流程正常读取，并覆盖主要输入形态 |
| 交付件 | Flex AWQ SSZ 参数配置方案 | 用户量化 YAML 或任务配置 | 参数取值合法、作用范围明确；关键参数说明选择依据 | 可作为后续量化流程的算法配置输入，并可复现本指南中的基线选择 |

## 3. 流程总览

```mermaid
flowchart LR
    A[适配模型<br/>流水线/算法接口] --> B[确定 W4A8<br/>等目标格式]
    B --> C[用真实量化器<br/>评估候选缩放]
    C --> D[搜索 alpha]
    D --> E[融合缩放]
    E --> F[进入后续量化]
```

## 4. 操作步骤

### 步骤 1：适配模型流水线与算法接口

**目标**：量化工具不能直接操作任意结构的模型，需要适配器将模型操作转换为标准方法。Flex AWQ SSZ 与 Flex Smooth Quant 共用平滑处理器框架，模型适配器需要继承并实现流水线接口及灵活平滑接口，提供模型的平滑子图拓扑。模型适配必须先完成，才能执行 Flex AWQ SSZ 算法。

**操作**：模型适配代码需实现以下接口，相关接口均由 `msmodelslim.model.interface_hub` 汇总提供：

1. **`PipelineInterface`**（`ModelSlimPipelineInterfaceV1`）：基础流水线接口，负责模型加载（`init_model`）、数据预处理（`handle_dataset`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）等。
2. **`FlexSmoothQuantInterface`**：灵活平滑量化专用接口，由 `msmodelslim.model.interface_hub` 汇总导出，提供模型的平滑子图拓扑配置，描述 Norm 与后续 Linear、OV、up/down 等结构之间的连接关系，供平滑处理器定位待平滑模块。

对于基于 HuggingFace Transformers 实现的标准开源 LLM，建议通过组合继承构建适配器：

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
    # ---- Flex AWQ SSZ 算法适配接口（本步骤重点）----
    FlexSmoothQuantInterface,  # 算法适配：get_adapter_config_for_subgraph 返回平滑子图拓扑
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface, FlexSmoothQuantInterface):
    pass
```

各接口方法说明：

- **`PipelineInterface`** 的标准方法：加载数据（`handle_dataset`）、加载模型（`init_model`）、逐层遍历（`generate_model_visit`）、逐层前向（`generate_model_forward`）、控制 KV Cache（`enable_kv_cache`）。量化调度器（Runner）与处理器（Processor）只通过这些标准方法驱动模型，与模型内部结构无关。
- **`FlexSmoothQuantInterface`** 的核心方法：返回待平滑的子图拓扑配置（子图类型如 `norm-linear`、`linear-linear`、`ov`、`up-down`），处理器据此定位待平滑模块，并用真实量化器评估候选平滑尺度。

之后量化命令中通过 `--model_type MyModelAdapter` 引用该适配器。

如需开发新模型适配代码，请参考[《LLM 模型接入量化流程指南》](../../ptq/llm/integration_guide_large_language_model_quantization.md)。请注意：模型适配必须先完成，才能执行后续量化流程。

**输出**：已在框架中注册可用的模型适配器（通过 `--model_type` 调用）。

### 步骤 2：建立推荐配置

**操作**：

下面配置用于建立**第一版可比较基线**。如果目标模型已有 `lab_practice` 配方，优先使用已验证配方，再参考本节理解每个参数为什么这样选。

```yaml
spec:
  process:
    - type: "flex_awq_ssz"
      # alpha 省略时自动搜索，通常比手工固定值更稳妥。
      qconfig:
        act:
          scope: "per_token"
          dtype: "int8"
          symmetric: true
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int4"
          symmetric: true
          method: "ssz"
          ext:
            step: 10
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

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `type` | 处理器标识，固定为 `flex_awq_ssz`。 | 固定。 | 不调。 |
| `alpha` | 控制激活统计在平滑尺度中的指数权重。当前实现如果未提供 `alpha` 会用真实 `qconfig` 模拟量化误差搜索 alpha；因此省略它会把选择交给数据驱动搜索。 | 推荐省略 `alpha` 让算法搜索；有经过同模型同 qconfig 验证的固定值时才手工设置。 | 手工值只对当前校准分布、子图和 qconfig 有意义。最终权重位宽、SSZ 设置或子图范围变化后，应重新搜索而不是照搬旧 alpha。 |
| `beta` | 配置层面提供 0~1 的系数，但当前 FlexAWQSSZ 搜索/应用路径把 beta 固定为 `0`，主要优化 alpha；它与 FlexSmoothQuant 的双参数搜索行为不同。 | 通常省略/保持默认，不把 beta 当作当前版本的主要调参旋钮。 | 除非后续实现/模型配方明确使用 beta，否则不要根据 FlexSmoothQuant 的经验手动调 beta；当前算法的主要选择依据是 alpha + 真实 qconfig。 |
| `qconfig` | 搜索时直接构造最终线性量化器来评价候选平滑尺度，因此 qconfig 是 AWQ 搜索目标的一部分。DeepSeek/GLM W4A8 实践统一使用激活 INT8 per-token MinMax + 权重 INT4 per-channel SSZ，并在 SSZ 中配置 `step: 10`。 | W4A8 目标可优先使用仓库验证组合；其他目标必须改成真正的最终 qconfig。 | qconfig 任一关键项改变都应重新搜索 alpha。特别是权重从 MinMax 换 SSZ、INT8 换 INT4，会改变候选尺度的真实量化误差。 |
| `qconfig.weight.ext.step` | 传给 SSZ 的最大迭代次数。SSZ 内部默认最多 50 次并支持提前收敛，仓库 FlexAWQSSZ 实践使用 `10` 来平衡搜索代价。 | 复用 W4A8 实践时用 `10`；独立模型精度优先且时间允许时可省略，让 SSZ 使用默认最多 50 次。 | step 太小可能让每个 alpha 候选下的 SSZ 还未充分优化，进而影响 alpha 排序；但过大也会显著放大“外层 alpha 搜索 × 内层 SSZ”成本。先固定 qconfig，再在时间/收敛之间选择。 |
| `enable_subgraph_type` | 决定在哪类可融合子图上执行。虽然默认支持四类，但当前仓库 5 个 FlexAWQSSZ 实践均只启用 `["up-down"]`，说明模型配方常会有意收窄范围。 | 有实践配方时完全照其子图类型；无配方时不要机械照抄“默认四类”，先根据模型结构和目标层选择。 | 扩大子图类型会增加搜索与变换范围，也可能引入不适配结构。若算法本来只为 MLP up/down 低比特优化，先从 `up-down` 建立基线更容易解释。 |
| `include` | 作用范围白名单，使用模块名模式决定哪些匹配到的模块进入当前算法。它只决定“在哪些模块做”，不会改变算法内部公式。 | 无模型专用配方时从 `["*"]` 开始；已有 `lab_practice` 时直接沿用其模块范围。 | 先保证范围覆盖预期模块，再看精度。范围过宽时，少数结构不兼容或敏感层会放大整体风险；范围过窄则可能让算法收益看不出来。收窄范围时优先按结构族或已知敏感层调整，不建议仅凭层号大面积删除。 |
| `exclude` | 作用范围黑名单，命中后从 `include` 的候选中排除，优先级高于 `include`。适合保护敏感层、首尾层或模型专用不兼容结构。 | 默认先保持空列表；只有实践配方、兼容性约束或敏感性结果给出明确证据时再加入。 | 局部精度问题优先通过 `exclude` 做小范围回退，比提高全模型位宽或关闭整个算法更容易保留收益。每次增加排除项后应确认通配符没有误伤相邻模块。 |

#### 参数组合与选择顺序

FlexAWQSSZ 的参数耦合非常强：**先锁定最终 qconfig → 再锁定子图 → 最后搜索 alpha**。SSZ `step` 同时影响每个候选的评价质量和总搜索时间，因此不要和 alpha 手工值同时大幅修改。每轮只改一个变量，并保持同一校准集和评测集；若调整后没有稳定收益，回到上一个基线。

**输出**：一份完成单变量调整的算法参数方案，关键字段均有明确的选择依据和调整方向。

### 步骤 4：编写量化配置并执行命令

**目标**：整合上述步骤生成完整的 YAML 量化配置文件，并通过 CLI 启动量化流程。

#### 完整示例：Flex AWQ SSZ + W4A8 动态量化（推荐起点）

##### 配置文件：`flex_awq_ssz_w4a8.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto
  process:
    - type: flex_awq_ssz            # 用真实量化器评估候选平滑尺度并搜索 alpha
      qconfig:                      # 与最终量化目标一致：W4A8
        act:
          dtype: int8
          scope: per_token
          symmetric: true
          method: minmax
        weight:
          dtype: int4
          scope: per_channel
          symmetric: true
          method: ssz
          ext:
            step: 10                # SSZ 最大迭代次数，平衡搜索代价
      enable_subgraph_type:
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
          method: ssz
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
  --config ./flex_awq_ssz_w4a8.yaml \
  --device npu \
  --device_id 0
```

**输出**：在指定的 `--save_path` 目录下生成完整的量化权重文件与描述文件。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| Flex AWQ SSZ 灵活激活感知权重量化平滑算法 | 说明该算法的定义、核心原理、关键性质、适用场景与限制。 | 《[Flex AWQ SSZ 灵活激活感知权重量化平滑算法 量化术语百科词条](./term_flex_awq_ssz.md)》 |

## 6. 相关文档

| 接口或文档 | 简述 | 链接 |
| --- | --- | --- |
| `PipelineInterface` / `FlexSmoothQuantInterface` | 模型流水线与 Flex AWQ SSZ 算法适配接口，均由 `msmodelslim.model.interface_hub` 汇总导出。 | [接口汇总模块](../../../../../msmodelslim/model/interface_hub.py) |
| flex_awq_ssz 配置说明 | 字段类型、默认值、合法取值与完整配置约束。 | [《flex_awq_ssz 配置说明》](../../../api_reference/config/processor/flex_awq_ssz.md) |
| modelslim_v1 配置说明 | 需要继续探索 runner、prior、save、dataset 等任务级高级配置时查阅。 | [《modelslim_v1 配置说明》](../../../api_reference/config/task/modelslim_v1.md) |
| 权重量化使用指南 | 用户指南：量化命令参数与完整使用说明。 | [《权重量化使用指南》](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/user_guide/usage_weight_quantization.md) |
