# AutoRound：低比特量化算法说明

## 简介

- **来源**：Intel 开发的一种基于 SignSGD 的低比特量化方法。
- **背景**：传统量化方法（如四舍五入）在权重量化中并非最优选择，往往会引入较大的量化误差，从而显著降低模型精度，尤其在低比特（如 4bit 及以下）量化场景中表现更为明显。
- **核心思想**：通过引入可学习的舍入偏移参数，结合SignSGD优化器自适应调整各权重的舍入方向，并利用温度调度策略逐步硬化舍入操作，有效降低量化重构误差，在超低比特条件下实现模型精度与压缩效率的最优平衡。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

## 原理和实现

### 原理

AutoRound 的核心思想是优化权重的舍入过程。该过程不是采用简单的四舍五入方式，而是基于 SignSGD（符号梯度下降）算法，自适应地学习每个权重的最佳舍入方向（向上或向下），并有针对性地调整缩放因子和零点。

**核心公式：**

在传统量化中，权重 W 的量化公式通常为：

Ŵ = s × clip(⌊W/s + zp⌉, n, m)

其中 s 是缩放因子，zp 是零点，n 和 m 是量化后的上下界。

AutoRound 在此基础之上引入了可学习的舍入偏移 V 和可选的缩放因子调整参数 α 和 β：

`Ŵ = s × clip(⌊W/s + zp + V⌉, n, m)`，`s = (max(W) × α - min(W) × β) / (2^bit - 1)`

其中 V 用于控制舍入的方向，α 和 β 用于调整缩放因子的范围。

**算法流程：**

该算法是一种逐层优化算法，对每一个decoder layer执行以下优化步骤：

1. **基准建立**：进行浮点前向传播，记录原始输出作为精度基准
2. **参数初始化**：对缩放因子、舍入偏移量进行初始化，并将其设置为可训练参数引入量化过程中
3. **量化重构**：执行量化-反量化操作，进行前向传播得到量化结果
4. **损失计算**：对比量化输出与浮点输出的差异，计算重构损失
5. **参数更新**：通过SignSGD优化器更新缩放因子和舍入偏移量
6. **迭代优化**：重复步骤1-5直至满足收敛条件或达到最大迭代次数
7. **最终量化**：应用迭代训练后的最优参数得到最终量化权重

### 实现

#### 代码实现

- 算法在 [msmodelslim/processor/quant/autoround.py](../../../../../msmodelslim/processor/quant/autoround.py) 中实现，核心类为 `AutoroundQuantProcessor`。

#### 初始化阶段

  * 层配置初始化：读取量化配置，并为每个网络层分配对应的量化配置方案
  * 参数预分配：初始化浮点输出、量化输出和最佳参数

#### pre_run阶段

  * 梯度冻结：关闭所有网络层的自动梯度计算，防止在训练过程中直接优化权重

#### preprocess阶段（逐层循环执行）

  * 基准输出采集：执行当前层浮点前向传播，并记录浮点输出结果作为优化基准
  * 线性层封装：对每个线性层进行包装处理，注入可训练的缩放因子和舍入偏移参数
  * 计算图构建：建立包含量化和反量化操作的可微计算图，支持梯度反向传播

#### process阶段（逐层循环执行）

  * 训练器初始化：设置学习率、迭代次数等参数，配置SignSGD优化器
  * 输入输出配置：使用上一层量化输出作为当前层输入，以浮点输出和量化输出的差距作为优化目标
  * 参数优化：通过多次迭代更新缩放因子和舍入偏移，最小化重构误差
  * 收敛监测：实时监测损失变化，达到收敛阈值或最大迭代次数时停止优化，得到最优参数

#### postprocess阶段（逐层循环执行）

  * 参数应用：将优化后的量化参数应用于对应层的权重量化
  * 解除封装：移除线性层的包装，恢复原始网络结构
  * 前向传播：执行当前层量化后的前向传播，作为下一层的输入

#### post_run阶段

  * 清理工作：移除所有模块的临时属性，完成量化流程的最终清理工作

## 适用要求

- **低比特量化**：适合极低比特量化场景中的4比特量化。
- **高精度需求**：在低比特条件下仍能保持较高的模型精度。
- **计算资源**：需要额外的优化过程，计算成本高于简单量化方法。
- **使用限制**：
  - 适用于llm中的线性层量化。
  - 需要足够的校准数据或训练迭代次数来优化参数。
  - **低比特量化极度依赖于良好的离群值抑制算法，建议用户配合[QuaRot](../outlier_suppression_algorithms/quarot.md)或[Iterative Smooth](../outlier_suppression_algorithms/iterative_smooth.md)等离群值抑制方法一起使用，不建议用户（尤其是缺乏量化调优经验的基础用户）单独使用AutoRound，否则可能导致模型精度严重下降、对话输出异常或其他不可预期的行为，相关风险由用户自行承担。**

## 功能介绍

### 昇腾AI处理器支持情况

| 产品系列 | 支持 |
|---------|------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | ✓ |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | ✓ |
| Atlas 推理系列产品 | ✗ |

**注：算法实现包含训练过程，对NPU显存有一定的要求，仅支持NPU显存>=64G的设备。**

### YAML配置示例

作为Processor使用，YAML配置示例如下：

```yaml
# AutoRound支持混合量化，即对不同的层使用不同的量化配置，这里以 W8A8 和 W4A4 混合量化为例
# W8A8 动态量化配置
default_w8a8_dynamic: &default_w8a8_dynamic
  weight:
    scope: "per_group"        # 权重量化范围
    dtype: "int8"             # 权重量化数据类型
    symmetric: True           # 是否启用对称量化
    method: "autoround"       # 权重量化方法：AutoRound算法，即包含参数训练的权重量化
    ext:
      group_size: 256         # 量化组大小，分组将在待量化nn.Linear的input_features维度进行，该值必须能够被其整除
      scale_dtype: "bfloat16" # 缩放因子数据类型
  act:
    scope: "per_token"        # 激活值量化范围
    dtype: "int8"             # 激活值量化数据类型
    symmetric: True           # 是否启用对称量化
    method: "minmax"          # 激活值量化方法：MinMax算法

# W4A4 动态量化配置模板
default_w4a4_dynamic: &default_w4a4_dynamic
  weight:
    scope: "per_group"
    dtype: "int4"
    symmetric: True
    method: "autoround"
    ext:
      group_size: 256
      scale_dtype: "bfloat16"
  act:
    scope: "per_token"
    dtype: "int4"
    symmetric: True
    method: "minmax"


spec:
  process:
    - type: "autoround_quant"     # 固定为 `autoround_quant`，用于指定 Processor 类型。
      iters: 400                  # 优化迭代次数
      enable_minmax_tuning: True  # 是否启用最小最大值调优
      enable_round_tuning: True   # 是否启用舍入调优
      strategies:
        # 策略1：除 up_proj、gate_proj 和 o_proj 层外，其余层均应用 W8A8 量化。
        - qconfig: *default_w8a8_dynamic
          exclude:
            - "*.up_proj"
            - "*.gate_proj"
            - "*.o_proj"
        # 策略2：对up_proj、gate_proj、o_proj层使用W4A4量化
        - qconfig: *default_w4a4_dynamic
          include:
            - "*.up_proj"
            - "*.gate_proj"
            - "*.o_proj"

```

### YAML配置字段详解

| 字段名 | 作用 | 类型 | 说明 | 默认值 |
|--------|------|------|------|--------|
| type | 处理器类型标识 | `string` | 固定值，用于标识这是一个AutoRound量化处理器 | `"autoround_quant"` |
| iters | 优化迭代次数 | `int` | 迭代次数，必须大于0，影响优化效果和计算时间 | `10` |
| enable_minmax_tuning | 是否启用最小最大值调优 | `bool` | 是否启用最小最大值调优，True表示启用，False表示不启用 | `True` |
| enable_round_tuning | 是否启用舍入调优 | `bool` | 是否启用舍入调优，True表示启用，False表示不启用 | `True` |
| strategies | 量化策略配置 | `array[object]` | 用于指定量化策略，支持int4和int8混合量化策略 | [见下方详细配置](#strategies-量化策略配置) |

#### strategies (量化策略配置)

**作用**: 配置不同层的量化策略，支持混合量化。

| 字段名 | 作用 | 类型 | 说明 | 示例值 |
|--------|------|------|------|--------|
| qconfig | 量化配置参数 | `object` | 包含激活值量化和权重量化的详细配置 | [激活值配置](#qconfigact-激活值量化配置)、[权重量化配置](#qconfigweight-权重量化配置) |
| include | 包含的层 | `array[string]` | 支持通配符匹配，指定要量化的层 | `["*"]`, `["*self_attn*"]` |
| exclude | 排除的层 | `array[string]` | 支持通配符匹配，优先级高于include | `["*down_proj*"]` |

#### qconfig.act (激活值量化配置)

**作用**: 配置激活值的量化参数。

| 参数名 | 作用 | 可选值 | 说明 | 默认值 |
|--------|------|--------|------|--------|
| scope | 量化范围 | `"per_token"` | 每个token独立参数（动态量化），AutoRound目前仅支持per_token | `"per_token"` |
| dtype | 量化数据类型 | `"int8"`, `"int4"` | 8位/4位整数量化 | `"int8"` |
| symmetric | 是否对称量化 | `True` | 对称量化，零点为0，AutoRound激活值量化仅支持对称量化 | `True` |
| method | 量化方法 | `"minmax"` | 激活值量化方法：MinMax算法 | `"minmax"` |

#### qconfig.weight (权重量化配置)

**作用**: 配置权重的量化参数。

| 参数名 | 作用 | 可选值 | 说明 | 默认值 |
|--------|------|--------|------|--------|
| scope | 量化范围 | `"per_channel"`, `"per_group"` | per_channel: 每个通道独立参数<br/>per_group: 每个组独立参数，AutoRound权重量化不支持per_tensor | `"per_group"` |
| dtype | 量化数据类型 | `"int8"`, `"int4"` | 8位/4位整数量化 | `"int8"` |
| symmetric | 是否对称量化 | `True`, `False` | True: 对称量化，零点为0<br/>False: 非对称量化，零点可调整 | `True` |
| method | 量化方法 | `"autoround"` | 权重量化方法：AutoRound算法，即包含参数训练的权重量化 | `"autoround"` |
| ext | 扩展配置 | `object` | 包含AutoRound特有的配置参数 | [见下方详细配置](#ext-autoround扩展配置) |

#### ext (AutoRound扩展配置)

**作用**: 配置AutoRound算法特有的参数。

| 参数名 | 作用 | 类型 | 说明 | 示例值 |
|--------|------|------|------|--------|
| group_size | 量化组大小 | `int` | 分组量化的大小，必须能被待量化nn.Linear层的input_features维度整除 | `256` |
| scale_dtype | 缩放因子数据类型 | `string` | 缩放因子的数据类型，影响精度和内存占用。可选值：`"float16"`、`"float32"`、`"bfloat16"` | `"bfloat16"` |

### 层过滤机制

层过滤机制用于指定哪些层需要量化，支持include和exclude模式匹配。详细的过滤规则、匹配模式、示例说明和常见层名模式请参考 [线性量化层过滤机制详解](linear_quant.md#层过滤机制详解)。

## FAQ

### 优化不收敛

**现象**：在优化过程中，浮点结果与量化结果之间的差距波动较大或不收敛。

**解决方案**：调整学习率或增加迭代次数。

### 精度下降明显

**现象**：量化后模型精度下降超过预期。

**解决方案**：增加优化步数，调整量化配置，减少使用`w4a4`量化的层数，或使用更多更优质的校准数据。

### group_size配置错误

**现象**：在量化过程中抛出了shape相关的异常如： shape '[-1, 257]' is invalid for input of size 512。

**原因**：`group_size`参数必须能够被待量化`nn.Linear`层的`input_features`维度整除，否则会导致分组量化失败。

**解决方案**：

  - 检查模型各层的`input_features`维度，确保`group_size`能够被其整除
  - 常见的`input_features`维度包括：4096、8192、11008等
  - 推荐的`group_size`值：128、256、512等，这些值通常能够被大多数层的`input_features`整除

### 层匹配告警

**现象**：在量化过程中，工具抛出了层匹配告警。

**解决方案**：检查模型定义，确保include/exclude模式匹配到正确的层。如果`include/exclude`未匹配到任何层时，工具会进行告警。详细的常见匹配失败原因和排查步骤请参考 [LinearQuantProcess层匹配告警](linear_quant.md#层匹配告警)。
