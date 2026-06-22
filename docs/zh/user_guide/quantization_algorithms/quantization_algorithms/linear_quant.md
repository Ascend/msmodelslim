# 线性量化算法说明

## 简介

线性量化（Linear Quantization）是深度学习模型压缩中最基础且广泛应用的算法类别。

在 msModelSlim 中，线性量化通过 `linear_quant` 处理器实现，专门用于对模型的线性层（`torch.nn.Linear` 模块）进行量化处理。它支持对线性层的权重（Weight）和激活值（Activation）进行灵活的量化配置。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

## 原理和实现

### 原理

线性量化的核心是将连续的浮点数值范围映射到离散的数值集合中。其基本公式为：

$$Q = \text{clamp}(\text{round}(\frac{V}{S}) + Z, Q_{min}, Q_{max})$$

其中：

- $V$：原始浮点值。
- $S$ (Scale)：缩放因子，决定了量化的步长。
- $Z$ (Zero-point)：偏移量（零点），用于处理非对称分布。
- $Q$：量化后的数值。

### 实现

在 msModelSlim 中，线性量化通过 `linear_quant` 处理器实现，支持对模型的线性层（Linear/Dense Layers）进行灵活的量化配置。算法在 [msmodelslim/processor/quant/linear.py](../../../../../msmodelslim/processor/quant/linear.py) 中实现。

### 算法分类

根据参数统计和计算时机的不同，线性量化主要分为以下几类：

1. **静态量化 (Static Quantization)**：
   - **原理**：在推理前，通过校准数据集（Calibration Dataset）统计激活值的分布，计算并固定 Scale 和 Zero-point。
   - **优势**：推理时无需动态计算量化参数，性能最优。
   - **适用场景**：对推理时延极度敏感的生产环境。

2. **动态量化 (Dynamic Quantization)**：
   - **原理**：在推理过程中，针对每一组输入（如每个 Token 或每个 Batch）动态计算激活值的 Scale 和 Zero-point。
   - **优势**：能够自适应输入数据的分布变化，精度通常显著高于静态量化。
   - **适用场景**：对精度要求较高，且模型激活值分布随输入剧烈变化的场景（如大语言模型）。

3. **混合量化 (Mixed/Hybrid Quantization)**：
   - **原理**：结合静态和动态量化的优点。例如 **PDMIX** 算法，在 Prefilling 阶段使用动态量化以保证首字精度，在 Decoding 阶段使用静态量化以提升生成速度。

## 功能介绍

### 支持的量化算法

在 `linear_quant` 处理器中，可以通过 `method` 参数选择不同的量化算法：

| 算法名称 | 适用对象 | 简介 | 详细说明 |
| :--- | :--- | :--- | :--- |
| **MinMax** | 权重 / 激活 | 最基础的量化算法，通过统计最大最小值确定量化范围。简单高效，INT8 场景首选。 | [MinMax 说明](minmax.md) |
| **Histogram** | 激活 | 通过直方图分析激活值分布，自动截断离群值以优化量化范围。精度通常优于 MinMax。 | [Histogram 说明](histogram_activation_quantization.md) |
| **SSZ** | 权重 | 通过迭代搜索最优缩放因子，最小化量化误差。专为 INT4 等低比特权重量化设计。 | [SSZ 说明](ssz.md) |
| **GPTQ** | 权重 | 通过逐列优化方式，将量化误差在后续未权重中进行补偿，进而达到最小化量化误差。| [GPTQ说明](gptq.md) |

### 静态量化与动态量化的核心差异

在一键量化中，通过 `qconfig.act.scope` 字段来区分 **静态量化** 与 **动态量化**：

- **静态量化 (`per_tensor`)**：在量化校准阶段统计并固定量化参数（scale和offset），推理时直接使用。**特点**：推理性能最优，计算开销最小，但在分布变化剧烈时精度可能受损。
- **动态量化 (`per_token`)**：在推理过程中，针对每个 token 实时计算量化参数。**特点**：量化粒度更细，能够更好地捕捉激活值的动态分布，**精度通常优于静态量化**，但会引入一定的实时计算开销。
- **PDMIX 混合量化 (`pd_mix`)**：Prefilling 阶段使用 `per_token`，Decoding 阶段使用 `per_tensor`。**特点**：旨在平衡精度和性能，特别适用于生成式模型的推理加速，参考[PDMIX：激活值阶段间混合量化算法说明](pdmix.md)。

### YAML配置示例

#### W8A8静态量化配置

```yaml
- type: "linear_quant"         # 处理器类型：线性层量化
  qconfig:
    act:                       # 激活值量化配置
      scope: "per_tensor"      # 静态量化标识：整个张量共用量化参数
      dtype: "int8"            # 数据类型：int8
      symmetric: false         # 是否对称量化：false（静态量化推荐非对称）
      method: "minmax"         # 量化方法：minmax
    weight:                    # 权重量化配置
      scope: "per_channel"     # 权重量化粒度：逐通道量化
      dtype: "int8"            # 数据类型：int8
      symmetric: true          # 是否对称量化：true
      method: "minmax"         # 量化方法：minmax
  include: [ "*" ]             # 包含的层。默认：["*"]
  exclude: [ "*down_proj*" ]   # 排除的层。默认：[]
```

#### W8A8动态量化配置

```yaml
- type: "linear_quant"         # 处理器类型：线性层量化
  qconfig:
    act:                       # 激活值量化配置
      scope: "per_token"       # 动态量化标识：每个 token 独立量化参数
      dtype: "int8"            # 数据类型：int8
      symmetric: false         # 是否对称量化：false
      method: "minmax"         # 量化方法：minmax
    weight:                    # 权重量化配置
      scope: "per_channel"     # 权重量化粒度：逐通道量化
      dtype: "int8"            # 数据类型：int8
      symmetric: true          # 对称量化：true
      method: "minmax"         # 量化方法：minmax
  include: [ "*mlp*" ]         # 仅包含包含 mlp 的层
```

#### W4A8动态量化配置

```yaml
- type: "linear_quant"         # 处理器类型：线性层量化
  qconfig:
    act:                       # 激活值量化配置
      scope: "per_token"       # 动态量化标识：每个 token 独立量化参数
      dtype: "int8"            # 数据类型：int8
      symmetric: true          # 是否对称量化：true
      method: "minmax"         # 量化方法：minmax
    weight:                    # 权重量化配置
      scope: "per_channel"     # 权重量化粒度：逐通道量化
      dtype: "int4"            # 数据类型：int4（低比特权重量化）
      symmetric: true          # 对称量化：true
      method: "ssz"            # 量化方法：ssz
  include: [ "*" ]             # 包含所有层
```

#### W8A8 PDMIX配置

```yaml
- type: "linear_quant"         # 处理器类型：线性层量化
  qconfig:
    act:                       # 激活值量化配置
      scope: "pd_mix"          # PDMIX量化标识：Prefilling 阶段使用 `per_token`，Decoding 阶段使用 `per_tensor`
      dtype: "int8"            # 数据类型：int8
      symmetric: false         # 是否对称量化：false
      method: "minmax"         # 量化方法：minmax
    weight:                    # 权重量化配置
      scope: "per_channel"     # 权重量化粒度：逐通道量化
      dtype: "int8"            # 数据类型：int8
      symmetric: true          # 对称量化：true
      method: "minmax"         # 量化方法：minmax
  include: [ "*mlp*" ]         # 仅包含包含 mlp 的层
```

### YAML配置字段详解

| 字段名 | 作用 | 类型 | 说明 | 示例值 |
|--------|------|------|------|--------|
| type | 处理器类型标识 | `string` | 固定值，用于标识这是一个线性层量化处理器 | `"linear_quant"` |
| qconfig | 量化配置参数 | `object` | 包含激活值量化和权重量化的详细配置 | 见下方详细配置 |
| include | 包含的层 | `array[string]` | 支持通配符匹配，指定要量化的层 | `["*"]`, `["*self_attn*"]` |
| exclude | 排除的层 | `array[string]` | 支持通配符匹配，优先级高于include | `["*down_proj*"]` |

#### qconfig.act (激活值量化配置)

**作用**: 配置激活值的量化参数。

| 参数名 | 作用 | 可选值                                       | 说明                                                                                                               | 默认值 |
|--------|------|-------------------------------------------|------------------------------------------------------------------------------------------------------------------|--------|
| scope | 量化范围 | `"per_tensor"`, `"per_token"`, `"pd_mix"` | per_tensor: 整个张量使用相同参数<br/>per_token: 每个token独立参数（动态量化）<br/>pd_mix: prefilling时per_token，decoding时per_tensor | `"per_tensor"` |
| dtype | 量化数据类型 | `"int8"`, `"int4"`, `"float"` | int8: 8位整数量化<br/>int4: 4位整数量化<br/>float: 16位浮点激活（此时需设置scope为"per_tensor"，symmetric为"true"，method为"none"） | `"int8"` |
| symmetric | 是否对称量化 | `true`, `false` | true: 对称量化，零点为0<br/>false: 非对称量化，零点可调整 | `false` |
| method | 量化方法 | `"minmax"`, `"histogram"` | minmax: 最小最大值量化<br/>histogram: 直方图量化 | `"minmax"` |

#### qconfig.weight (权重量化配置)

**作用**: 配置权重的量化参数。

| 参数名 | 作用 | 可选值 | 说明 | 默认值 |
|--------|------|--------|------|--------|
| scope | 量化范围 | `"per_tensor"`, `"per_channel"`, `"per_group"` | per_tensor: 整个张量使用相同参数<br/>per_channel: 每个通道独立参数<br/>per_group: 每个分组独立参数 | `"per_channel"` |
| dtype | 量化数据类型 | `"int8"`, `"int4"` | 8位/4位整数量化 | `"int8"` |
| symmetric | 是否对称量化 | `true`, `false` | true: 对称量化，零点为0<br/>false: 非对称量化，零点可调整 | `true` |
| method | 量化方法 | `"minmax"`, `"ssz"`, `"gptq"` | minmax: 最小最大值量化<br/>ssz: ssz权重量化<br/>gptq: gptq权重量化 | `"minmax"` |

### 层过滤机制详解

#### 过滤规则

1. **include规则**: 定义要包含的层，只有匹配include模式的层才会被处理。
2. **exclude规则**: 定义要排除的层，匹配exclude模式的层会被跳过。
3. **优先级**: exclude规则的优先级高于include规则。

#### 匹配模式（Unix通配符的匹配模式）

| 通配符 | 作用 | 示例 |
|--------|------|------|
| `*` | 匹配任意字符序列 | `*self_attn*` 匹配包含 "self_attn" 的任意字符串 |
| `?` | 匹配单个字符 | `layer?` 匹配 "layer1", "layerA" 等 |
| `[abc]` | 匹配字符集中的任意一个字符 | `layer[123]` 匹配 "layer1", "layer2", "layer3" |
| `[!abc]` | 匹配不在字符集中的任意字符 | `layer[!123]` 匹配除 "layer1", "layer2", "layer3" 外的字符串 |

#### 过滤顺序

1. **第一步**: 检查层名是否匹配include模式。
    - 如果include为空或未设置，默认包含所有层。
    - 如果层名不匹配任何include模式，该层被排除。
2. **第二步**: 检查层名是否匹配exclude模式。
    - 如果层名匹配任何exclude模式，该层被排除。
    - 即使该层在第一步中被include包含，也会被exclude排除。

#### 使用示例

**示例1**: 基础过滤

```yaml
include: [ "*" ]
exclude: [ "*down_proj*" ]
```

- **结果**: 包含所有层，但排除包含"down_proj"的层。

**示例2**: 选择性包含

```yaml
include: [ "*self_attn*", "*mlp*" ]
exclude: [ ]
```

- **结果**: 只包含包含"self_attn"或"mlp"的层。

**示例3**: 复杂过滤

```yaml
include: [ "*attention*", "*mlp*" ]
exclude: [ "*down_proj*", "*gate*" ]
```

- **结果**: 包含包含"attention"或"mlp"的层，但排除包含"down_proj"或"gate"的层。

**示例4**: 精确匹配

```yaml
include: [ "model.layers.*.self_attn.*" ]
exclude: [ "model.layers.*.self_attn.down_proj" ]
```

- **结果**: 只包含self_attn层，但排除其中的down_proj子层。

#### 常见层名模式

**Transformer架构常见层名**

| 模式 | 描述 | 典型用途 |
|------|------|----------|
| `*self_attn*` | 自注意力层 | 注意力机制相关的权重和偏置 |
| `*mlp*` | 多层感知机层 | 前馈网络层 |
| `*attention*` | 注意力相关层 | 广义的注意力相关组件 |
| `*ffn*` | 前馈网络层 | Feed Forward Network |
| `*gate*` | 门控层 | 门控机制相关层 |
| `*down_proj*` | 下投影层 | 降维投影层 |
| `*up_proj*` | 上投影层 | 升维投影层 |

**量化策略建议**

| 策略 | 配置 | 说明 |
|------|------|------|
| 全量量化 | `include: ["*"]` | 量化所有线性层 |
| 注意力层量化 | `include: ["*self_attn*"]` | 只量化自注意力相关层 |
| MLP层量化 | `include: ["*mlp*"]` | 只量化多层感知机层 |
| 排除敏感层 | `exclude: ["*down_proj*", "*gate*"]` | 排除对精度敏感的层 |

## FAQ

### 量化组合有效性

**现象**：并非所有配置组合都是有效的量化组合，当检测到无效的量化配置组合时，无效组合会抛出`UnsupportedError`异常。

**解决方案**：异常信息会详细说明具体的配置冲突原因，请根据异常信息调整配置参数。

### 层匹配告警

**现象**: 当include/exclude模式未匹配到任何层时，工具会进行告警，请务必关注这些告警信息。

**常见匹配失败原因**：

1. **层名不匹配**

   ```yaml
   # ❌ 可能匹配失败：层名中不包含"self_attn"
   include: ["*self_attn*"]
   # 实际层名可能是: "attention", "attn", "self_attention"
   ```

2. **路径层级错误**

   ```yaml
   # ❌ 可能匹配失败：路径层级不匹配
   include: ["model.layers.*.attention"]
   # 实际路径可能是: "layers.*.attention" 或 "transformer.layers.*.attention"
   ```

3. **大小写敏感**

   ```yaml
   # ❌ 可能匹配失败：大小写不匹配
   include: ["*SelfAttn*"]
   # 实际层名可能是: "*self_attn*"
   ```

4. **拼写错误**

   ```yaml
   # ❌ 可能匹配失败：拼写错误
   include: ["*self_atttn*"]  # 拼写错误：多了一个t
   # 实际层名可能是: "*self_attention*"

   include: ["*mlp*"]  # 可能匹配失败：不同模型使用不同命名
   # 实际层名可能是: "*ffn*" 或 "*feed_forward*"
   ```

5. **不是nn.Linear的路径**

   ```yaml
   # ❌ 可能匹配失败：匹配到非Linear层
   include: ["*gate.weight"]
   # 实际上是nn.Parameter的路径
   # LinearQuantProcess只处理nn.Linear层，其他层会被忽略
   ```

**解决方案**：

1. **检查层名**: 使用模型分析工具查看实际的层名结构。
2. **验证模式**: 使用简单的通配符模式进行测试。
3. **逐步调试**: 从`include: ["*"]`开始，逐步缩小范围。
4. **查看日志**: 关注工具输出的匹配结果和告警信息。
    - 匹配失败告警: `patterns are not matched any module, please ensure this is as expected`。
