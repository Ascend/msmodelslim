# AWQ：激活感知权重量化算法说明

## 简介

- **概述**：AWQ（Activation-aware Weight Quantization，激活感知权重量化）是一种用于大语言模型量化过程中抑制激活离群值的算法。该算法通过观察激活值的统计特征，自动搜索最优的缩放因子，在权重量化前对权重进行缩放，从而在保持模型精度的同时有效减少量化误差。AWQ 的核心理念是：并非所有权重对模型输出同等重要，通过激活值分布来识别重要权重通道并给予保护，可以在低比特量化场景下获得更优的精度表现。
- **核心思想**：AWQ 算法使用激活值的均值（mean）来度量各权重通道的重要性，通过网格搜索找到使量化结果与浮点基准结果之间均方误差（mean squared error，MSE）最小的缩放因子。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim 工具安装指南》](../../../install_guide/install_guide.md)。

## 原理和实现

### 原理

**算法公式：**

```python
scales = act_mean.pow(ratio).clamp(min=1e-4)
scales = scales / sqrt(scales.max() * scales.min())
```

其中：

- `act_mean`：激活值绝对值的逐通道均值（`mean(abs(act))`），反映各通道的重要性。
- `ratio`：缩放比例系数，在 `[0, 1)` 范围内以 `1 / n_grid` 为步长进行网格搜索。
- `n_grid`：网格搜索步数，默认值为 `20`，用于控制搜索精度。

**关键特性：**

1. **基于激活均值的重要性评估**：使用激活值绝对值的逐通道均值衡量权重通道的重要性，均值越大的通道在量化时会获得更多保护。
2. **网格搜索最优缩放**：在 `[0, 1)` 范围内遍历不同的 `ratio` 值，评估各候选缩放因子的量化效果。
3. **实际量化器评估**：使用真实的权重量化器对缩放后的权重进行量化，并基于量化结果与浮点基准结果的均方误差选择最优参数。
4. **块级误差评估**：通过自动发现目标模块的最低公共祖先（lowest common ancestor，LCA）并缓存其输入参数，在块级别评估量化误差，而不是只比较单个线性层的权重误差。

**缩放因子搜索流程：**

1. **初始化**：收集激活值均值和祖先模块输入参数缓存。
2. **基准推理**：使用原始浮点权重在祖先模块上执行推理，得到浮点基准输出。
3. **网格搜索**：在 `[0, 1)` 范围内遍历 `ratio` 值。
   - 计算缩放因子：`scales = act_mean.pow(ratio).clamp(min=1e-4)`。
   - 归一化缩放因子：`scales = scales / sqrt(scales.max() * scales.min())`。
   - 对目标线性层权重应用缩放。
   - 使用量化器量化缩放后的权重。
   - 对量化后的权重执行反向缩放。
   - 在祖先模块上执行推理并计算均方误差。
   - 恢复原始权重。
4. **选择最优参数**：选择均方误差最小的 `ratio` 对应的缩放因子。
5. **应用缩放**：通过 `SubgraphFusionFactory` 将最优缩放因子融合到子图权重中。

### 支持的子图类型

#### NormLinearSubgraph（归一化-线性子图）

适用于包含归一化层和多个线性层的结构，例如：

```python
x = norm(x)
y = torch.cat([linear(x) for linear in linears], dim=-1)
```

处理方式：

- 使用所有目标线性层进行缩放因子搜索。
- 通过自动发现的最低公共祖先模块进行块级误差评估。
- 搜索到最优缩放因子后，对归一化层应用反向缩放，对线性层应用正向缩放。

#### LinearLinearSubgraph（线性-线性子图）

适用于两个连续线性层的结构：

```python
y = linear2(linear1(x))
```

处理方式：

- 基于 `linear2` 的权重进行缩放因子搜索。
- 通过自动发现的最低公共祖先模块进行块级误差评估。
- 对 `linear2` 应用正向缩放，对 `linear1` 应用反向缩放。

#### OVSubgraph（注意力输出-值子图）

适用于注意力机制中的输出投影和值投影，支持以下结构：

- MHA（多头注意力）
- MQA（多查询注意力）
- GQA（分组查询注意力）

处理方式：

- 基于 `o_proj` 权重进行缩放因子搜索。
- 通过自动发现的最低公共祖先模块进行块级误差评估。
- 对 `o_proj` 应用正向缩放，对 `v_proj` 应用反向缩放。

#### UpDownSubgraph（上投影-下投影子图）

适用于 MLP 门控结构：

```python
y = down_proj(ReLU(gate_proj(x)) * up_proj(x))
```

处理方式：

- 基于 `down_proj` 权重进行缩放因子搜索。
- 通过自动发现的最低公共祖先模块进行块级误差评估。
- 对 `down_proj` 应用正向缩放，对 `up_proj` 应用反向缩放。

### 实现

#### 代码实现

AWQ 算法的代码组织在 [msmodelslim/processor/anti_outlier/awq/](../../../../../msmodelslim/processor/anti_outlier/awq/__init__.py) 目录下：

| 文件                                                                                                | 核心类/函数                                        | 职责                                   |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------- | -------------------------------------- |
| [processor.py](../../../../../msmodelslim/processor/anti_outlier/awq/processor.py)                     | `AWQProcessor`, `AWQProcessorConfig`               | Processor 入口，管理预处理和后处理流程 |
| [api.py](../../../../../msmodelslim/processor/anti_outlier/awq/api.py)                                 | `awq()`                                            | 各子图类型的 AWQ 分发与实现            |
| [best_scales_search.py](../../../../../msmodelslim/processor/anti_outlier/awq/best_scales_search.py)   | `AWQSearcher`, `AWQBestScalesSearcher`             | 缩放因子网格搜索逻辑                   |
| [awq_stats_collector.py](../../../../../msmodelslim/processor/anti_outlier/awq/awq_stats_collector.py) | `AWQStatsCollector`                                | 激活统计信息收集和中间参数缓存         |
| [common.py](../../../../../msmodelslim/processor/anti_outlier/awq/common.py)                           | `AWQConfig`, `AWQContext`, `offload()`, `onload()` | 算法配置、运行时上下文和张量迁移工具   |
| [interface.py](../../../../../msmodelslim/processor/anti_outlier/awq/interface.py)                     | `AWQInterface`                                     | 模型适配器需要实现的抽象接口           |

#### 预处理阶段

- 通过 `AWQInterface.get_adapter_config_for_subgraph()` 获取子图配置。
- 根据 `include` 和 `exclude` 过滤要处理的子图。
- 为目标线性层安装 forward hook，收集激活值绝对值的逐通道均值。
- 通过 LCA 自动发现块级评估用的祖先模块，并为其安装 forward pre-hook，缓存输入参数。

#### 后处理阶段

- 按优先级处理子图：`up-down`、`ov`、`norm-linear`、`linear-linear`。
- 从统计信息中构建 `AWQContext`，包括激活均值、祖先模块实例和输入参数缓存。
- 调用 `AWQBestScalesSearcher.search()` 搜索最优缩放因子。
- 通过 `SubgraphFusionFactory` 将最优缩放因子融合到子图中。
- 停止所有 hook 并清理统计信息。

## 适用要求

- **模型接口要求**：模型适配器需要实现 `AWQInterface` 接口。
- **模块命名要求**：配置中的模块名称必须与 `named_modules()` 返回的完整路径一致。
- **子图类型要求**：`enable_subgraph_type` 支持的取值为 `norm-linear`、`linear-linear`、`ov`、`up-down`。
- **模块属性要求**：目标模块必须存在且具备可写的 `weight`。
- **运行时要求**：AWQ 依赖 `ContextManager` 提供全局上下文，运行时会自动创建和管理相关上下文对象。

## 功能介绍

### YAML 配置示例

作为Processor使用，YAML配置示例如下：

```yaml
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
  include:
    - "*"
  exclude: []
```

### YAML 配置字段详解

| 字段名                 | 作用           | 说明                                                                                                                                                      |
| ---------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`                 | 处理器类型标识 | 固定值为 `"awq"`。                                                                                                                                        |
| `weight_qconfig`       | 权重量化配置   | AWQ 搜索阶段使用的权重量化配置。字段定义与 [线性量化算法说明](../quantization_algorithms/linear_quant.md#yaml配置字段详解) 中 `qconfig.weight` 保持一致。 |
| `n_grid`               | 网格搜索步数   | 正整数，默认值为 `20`，数值越大搜索越细致，但耗时也会增加。                                                                                               |
| `enable_subgraph_type` | 启用的子图类型 | 支持的取值包括 `norm-linear`、`linear-linear`、`ov`、`up-down`。                                                                                          |
| `include`              | 包含的层       | 支持通配符匹配。                                                                                                                                          |
| `exclude`              | 排除的层       | 支持通配符匹配，优先级高于 `include`。                                                                                                                    |

### 模型适配

#### 接口与数据结构

AWQ 模型适配依赖以下接口和数据结构：

```python
from dataclasses import dataclass
from typing import List, Optional
from abc import ABC, abstractmethod

@dataclass
class MappingConfig:
    targets: List[str]
    source: Optional[str] = None

@dataclass
class AdapterConfig:
    subgraph_type: str
    mapping: MappingConfig

class AWQInterface(ABC):
    @abstractmethod
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        ...
```

#### 适配步骤

1. 模型适配器继承 `AWQInterface` 并实现 `get_adapter_config_for_subgraph()`。
2. 为模型中的标准子图配置 `subgraph_type` 和 `mapping`。
3. 使用完整模块路径填写 `source` 和 `targets`。
4. 由框架自动完成块级评估所需的 LCA 发现和参数缓存。

参考实现请参见 [msmodelslim/model/qwen2/model_adapter.py](../../../../../msmodelslim/model/qwen2/model_adapter.py)。

#### 配置示例

以下示例展示了典型 Transformer 层的 AWQ 子图映射：

```python
def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
    adapter_config = []
    for layer_idx in range(self.config.num_hidden_layers):
        norm_linear_mapping_config1 = MappingConfig(
            source=f"model.layers.{layer_idx}.input_layernorm",
            targets=[
                f"model.layers.{layer_idx}.self_attn.k_proj",
                f"model.layers.{layer_idx}.self_attn.q_proj",
                f"model.layers.{layer_idx}.self_attn.v_proj",
            ],
        )

        norm_linear_mapping_config2 = MappingConfig(
            source=f"model.layers.{layer_idx}.post_attention_layernorm",
            targets=[
                f"model.layers.{layer_idx}.mlp.gate_proj",
                f"model.layers.{layer_idx}.mlp.up_proj",
            ],
        )

        ov_mapping_config = MappingConfig(
            source=f"model.layers.{layer_idx}.self_attn.v_proj",
            targets=[f"model.layers.{layer_idx}.self_attn.o_proj"],
        )

        up_down_mapping_config = MappingConfig(
            source=f"model.layers.{layer_idx}.mlp.up_proj",
            targets=[f"model.layers.{layer_idx}.mlp.down_proj"],
        )

        adapter_config.extend([
            AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config1),
            AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config2),
            AdapterConfig(subgraph_type="ov", mapping=ov_mapping_config),
            AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping_config),
        ])

    return adapter_config
```

### 与 Flex AWQ SSZ 的区别

| 特性         | AWQ                                   | [Flex AWQ SSZ](flex_awq_ssz.md)       |
| ------------ | ------------------------------------- | ------------------------------------- |
| 缩放因子计算 | `act_mean.pow(ratio)` 网格搜索        | `A_scale**alpha / W_scale**beta`      |
| 误差评估方式 | 块级评估                              | 使用量化器评估不同参数组合            |
| 参数搜索空间 | `ratio` ∈ `[0, 1)`，步长 `1 / n_grid` | `alpha` ∈ `[0, 1]`，`beta` 常设为 `0` |
| 配置接口     | `AWQInterface`                        | `FlexSmoothQuantInterface`            |
| 量化配置     | 仅需权重量化配置                      | 需要激活和权重量化配置                |

## FAQ

### 模块名不匹配

**现象**: `include/exclude` 未命中时，日志提示未匹配模式。

**解决方案**: 核对完整模块名是否与 `named_modules()` 返回的路径一致。

### 子图配置错误

**现象**: `get_adapter_config_for_subgraph()` 返回的配置不正确。

**解决方案**: 检查配置中的 `source` 和 `targets` 字段是否正确。

### 模块不存在

**现象**: 配置中指定的模块名称在模型中不存在。

**解决方案**: 通过 `model.named_modules()` 验证模块是否确实存在。

### 子图类型不支持

**现象**: 配置的子图类型不被支持。

**解决方案**: 建议按模型实际已适配的子图类型填写，支持的取值为 `norm-linear`、`linear-linear`、`ov`、`up-down`。如无特殊需求，可保持默认配置。

### 祖先模块未找到

**现象**: 日志提示 "No name found for inspect module of subgraph"，子图被跳过。

**解决方案**: 检查 `targets` 中的模块名称是否具有合理的共同路径前缀，确保其最低公共祖先模块在模型中存在。

### 激活统计信息缺失

**现象**: 日志提示 "No activation mean for target module"，子图被跳过。

**解决方案**: 确保校准数据（calibration data）足够且模型前向推理正常执行，使钩子能够正确收集激活统计信息。

### 中间参数缓存为空

**现象**: 日志提示 "No kwargs cache for parent module"，子图被跳过。

**解决方案**: 确保通过 LCA 自动发现的祖先模块在前向推理中被正确触发，检查 `targets` 中的模块路径是否正确。
