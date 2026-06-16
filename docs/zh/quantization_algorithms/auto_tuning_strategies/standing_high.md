# Standing High 调优算法

## 简介

Standing High（摸高算法）是一种基于量化回退层选择和离群值抑制策略的自动调优算法。该算法通过逐步减少量化回退级别（disable level）和尝试不同的离群值抑制策略，找到在满足精度要求的前提下，尽可能量化更多层的量化配置方案。量化回退是指部分层不做量化而是回退到浮点精度。

## 原理与特点

### 算法原理

Standing High 算法的核心思想是“摸高”：算法首先使用配置文件中的初始配置进行评估，如果满足精度要求则直接返回。否则，使用二分搜索快速定位满足精度要求的最小回退级别，然后通过摸高过程逐步减少回退层数，结合不同的离群值抑制策略，寻找最优配置。算法采用自适应步长机制，在找到满足要求的配置时加快搜索速度。

**1. 初始化阶段**

算法首先加载模型和量化回退层选择器，然后使用配置文件中的校准数据集（`template.dataset`）运行层选择器，用于确定哪些层需要回退到浮点。层选择器基于敏感层分析实现，会根据模型结构和校准数据，分析各层对量化精度的敏感程度，将模型层分组，以便后续进行量化回退级别的选择。

**2. 初始评估**

算法使用配置文件中的初始配置（`template`），直接生成量化配置并评估。初始配置基于 `template` 中的量化配置，如果该配置满足精度要求，算法直接返回，结束调优。

**3. 二分搜索最小回退级别**

如果初始配置不满足精度要求，算法使用二分搜索找到满足精度要求的最小回退级别。回退级别由层选择器根据模型结构自动计算，级别越高，回退到浮点的层越多。算法在回退级别 1 到最大级别之间进行二分搜索，快速定位满足精度要求的最小回退级别。

**4. 摸高过程**

从找到的最小回退级别开始，算法执行摸高过程，逐步减少回退层数（量化更多层）：

1） **尝试减少回退级别**：从当前回退级别开始，尝试减少回退级别（减少回退到浮点的层数，量化更多层）。

2） **遍历离群值抑制策略**：对于每个回退级别，算法会遍历配置文件中定义的所有离群值抑制策略（`anti_outlier_strategies`），尝试不同的策略组合。

3） **评估配置**：对每个回退级别和离群值抑制策略的组合进行量化配置生成和精度评估。

4） **自适应步长**：如果找到满足精度要求的配置，算法会记录该配置，并将步长翻倍（最大不超过当前回退级别），以加快搜索速度。如果所有策略都不满足要求，则重置步长为1，继续尝试。

5） **终止条件**：出现以下任一情况时终止并返回当前记录的最佳配置：初始配置已满足精度要求；摸高过程中回退级别已到 0 且所有离群值抑制策略均已尝试；或步长已为 1 且所有离群值抑制策略均不满足精度要求。

**5. 离群值抑制策略遍历**

算法在遍历离群值抑制策略时具有记忆功能，从上次成功的策略开始遍历，以提高搜索效率。策略遍历顺序按照配置文件中 `anti_outlier_strategies` 的顺序进行。

### 算法特点

1. **高效搜索机制**：算法采用两种机制提高搜索效率：
   - **二分搜索**：在定位最小回退级别时，使用二分搜索快速缩小搜索范围。
   - **自适应步长**：在摸高过程中，如果找到满足要求的配置，步长会翻倍，加快搜索速度。

2. **策略记忆**：离群值抑制策略的遍历具有记忆功能，从上次成功的策略开始，提高搜索效率。

3. **逐步优化**：从满足精度要求的最小回退级别开始，逐步减少回退层数（量化更多层），找到最优配置。

## 适用要求

**推理引擎支持**：需要注意推理引擎对任意回退的支持情况。一般情况下，单算子模式下 vLLM-Ascend 支持任意回退，但使用混合算子时可能不支持任意回退。在使用 Standing High 算法时，请确保配置的推理引擎（如 vLLM-Ascend）能够支持量化回退功能。

**模型适配**：策略启动时会**自动运行敏感层分析**以生成回退候选。模型适配器须实现 **`ModelSlimPipelineInterfaceV1`**（即 `core/runner/pipeline_interface.py` 中的 `PipelineInterface`）。分析由 `PipelineAnalysisService` 经 `LayerWiseRunner` 调用适配器的 `init_model`、`handle_dataset` 及 visit/forward pipeline，**不会在策略侧预先调用 `load_model`**。与 《[敏感层分析使用说明](../../feature_guide/sensitive_layer_analysis/usage.md)》 及 CLI `msmodelslim analyze` 要求一致。详见 《[LLM 大模型接入指南](../../developer_guide/integrating_models.md#自动调优与敏感层分析)》。

## 功能介绍

### 使用说明

在自动调优流程中通过 `msmodelslim tune` 启动；在调优 YAML 的 `strategy` 字段将 `type` 设为 `standing_high`，并按下文配置其他字段即可。完整调优配置与命令参数见 《[自动调优使用说明](../../feature_guide/auto_precision_tuning/usage.md)》。

### YAML 配置示例

对应于自动调优配置文件中的 `strategy` 字段

```yaml
strategy:
  type: standing_high
  anti_outlier_strategies:
    - [ ]
    - - type: "iter_smooth"
        alpha: 0.5
  template:
    runner: auto
    process:
      - type: linear_quant
        qconfig:
          act:
            scope: per_tensor
            dtype: int8
            symmetric: false
            method: minmax
          weight:
            scope: per_channel
            dtype: int8
            symmetric: true
            method: minmax
        include: [ "*" ]
        exclude: [ ]
    save:
      - type: ascendv1_saver
        part_file_size: 4
    dataset: mix_calib.jsonl
  metadata:
    config_id: standing_high
    label:
      w_bit: 8
      a_bit: 8
      is_sparse: false
      kv_cache: false
```

### YAML 配置字段详解

#### type - 策略类型

**作用**: 指定调优算法的类型，对于 Standing High 算法，该字段应设置为 `standing_high`。

**类型**: `string`

**值**: `standing_high`

#### anti_outlier_strategies - 离群值抑制策略列表

离群值抑制策略列表，每个元素是一个处理器配置列表。算法会遍历这些策略组合，寻找满足精度要求的配置。

**配置说明**：

- 可以配置空列表 `[]`，表示不使用离群值抑制策略
- 可以配置单个离群值抑制策略，如 `iter_smooth`
- 可以配置多种离群值抑制策略的混合策略，将多个策略放在同一个列表中。

**配置示例**：

```yaml
# 不使用离群值抑制策略
anti_outlier_strategies: [ ]
```

```yaml
anti_outlier_strategies:
  - - type: "iter_smooth"  # 仅使用单个离群值抑制策略
      alpha: 0.5
  - - type: "iter_smooth"  # 使用混合策略，iter_smooth+quarot
      alpha: 0.5
    - type: "quarot"
```

#### template - 量化基础配置

量化基础配置模板，定义了量化处理的基础配置，包括量化调度器、处理器、保存器和数据集配置。该配置是开启调优的起点，基础配置的选择一定程度上会影响调优的迭代次数。

**配置协议**: template 字段的配置协议与一键量化配置协议中的 `spec` 字段保持一致，详细配置说明请参考《[一键量化使用说明](../../feature_guide/quick_quantization_v1/usage.md#量化配置协议详解)》。

**核心字段**:

- **runner**: 量化调度器类型，定义量化处理的调度方式（auto、layer_wise、dp_layer_wise、model_wise等）
- **process**: 处理器列表，定义量化处理的处理器配置（linear_quant、iter_smooth等）
- **save**: 保存器列表，定义量化结果的保存方式（ascendv1_saver等）
- **dataset**: 校准数据集配置，指定校准数据集文件名

**配置要求**：

- 必须包含至少一个 `linear_quant` 处理器
- 可以包含其他处理器（如离群值抑制处理器）

**配置示例**:

```yaml
template:
  runner: auto
  process:
    - type: linear_quant
      qconfig:
        act:
          scope: per_tensor
          dtype: int8
          symmetric: false
          method: minmax
        weight:
          scope: per_channel
          dtype: int8
          symmetric: true
          method: minmax
      include: [ "*" ]
      exclude: [ ]
  save:
    - type: ascendv1_saver
      part_file_size: 4
  dataset: mix_calib.jsonl
```

#### metadata - 策略元数据

**作用**: 定义策略的元数据信息，用于标识和分类量化配置（如量化位宽标签），便于在“最佳实践库”中被检索与复用。

当在启动自动调优前设置环境变量 `MSMODELSLIM_CUSTOM_PRACTICE_REPO` 时，调优在精度达标后会将最终量化配置写入该路径对应的实践库。后续执行一键量化（`msmodelslim quant`）时，只需在同一 `MSMODELSLIM_CUSTOM_PRACTICE_REPO` 环境下指定与调优一致的 `quant_type`，工具即可按一键量化的实践库检索规则，从该实践库中自动匹配并使用对应配置。

**字段说明**:

| 字段名 | 作用 | 类型 | 说明 |
|--------|------|------|------|
| config_id | 配置文件名称 | string | 量化配置的标识符 |
| label | 配置文件标签 | object | 量化配置的标签信息，包括量化位数、稀疏性等 |

**label 字段说明**:

| 字段名 | 作用 | 类型 | 说明 |
|--------|------|------|------|
| w_bit | 权重量化位数 | int | 权重量化的位数，如8表示8bit量化 |
| a_bit | 激活值量化位数 | int | 激活值量化的位数，如8表示8bit量化 |
| is_sparse | 是否使用稀疏量化 | bool | 是否为稀疏量化 |
| kv_cache | 是否量化KV缓存 | bool | 是否对KV缓存进行量化 |

**配置示例**:

```yaml
metadata:
  config_id: standing_high
  label:
    w_bit: 8
    a_bit: 8
    is_sparse: false
    kv_cache: false
```
