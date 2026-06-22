# Iterative Smooth：离群值抑制算法说明

## 简介

- **概述**：Iterative Smooth（迭代平滑）是一种用于大语言模型量化过程中抑制激活离群值的算法。该算法通过动态调整权重和激活的缩放因子，在保持模型精度的同时，有效减少量化误差。
- **核心思想**：Iterative Smooth算法的核心思想是通过在相邻层之间重新分配量化误差，使得激活值的分布更加均匀，从而减少离群值对量化精度的影响。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

## 原理和实现

### 原理

**算法核心：**

- 基于收集的激活统计信息计算每通道的缩放因子。
- 使用 `iter_smooth` 算法对子图进行迭代平滑优化。
- 支持可配置的平滑参数：`alpha`（平滑强度）、`scale_min`（最小缩放）、`symmetric`（对称量化）。

**算法公式：**

```text
scales = (A_scale**α / W_scale**(1-α)).clamp(min=scale_min)
```

其中：

- `A_scale`：激活值的缩放因子
- `W_scale`：权重的缩放因子（取每列的最大值）
- `α`：平衡参数，控制激活和权重的相对重要性（默认值：0.9）
- `scale_min`：缩放因子的最小值（默认值：1e-5）

### 支持的子图类型

#### NormLinearSubgraph（归一化-线性子图）

适用于包含归一化层和多个线性层的结构，如：

```python
x = norm(x)
y = torch.cat([linear(x) for linear in linears], dim=-1)
```

**处理方式：**

- 计算所有线性层权重的列最大值作为权重缩放因子
- 对每个线性层应用正向缩放
- 对归一化层应用反向缩放（1/scales）

#### LinearLinearSubgraph（线性-线性子图）

适用于两个连续线性层的结构：

```python
y = linear2(linear1(x))
```

**处理方式：**

- 基于linear2的权重计算缩放因子
- 对linear2应用正向缩放
- 对linear1应用反向缩放（1/scales）

#### OVSubgraph（注意力输出-值子图）

适用于注意力机制中的输出投影和值投影：

- 支持MHA（多头注意力）
- 支持MQA（多查询注意力）
- 支持GQA（分组查询注意力）

**处理方式：**

- 基于o_proj权重计算缩放因子
- 对o_proj应用正向缩放
- 对v_proj应用反向缩放（1/scales）

#### UpDownSubgraph（上投影-下投影子图）

适用于MLP门控机制：

```python
y = down_proj(ReLU(gate_proj(x)) * up_proj(x))
```

**处理方式：**

- 基于down_proj权重计算缩放因子
- 对down_proj应用正向缩放
- 对up_proj应用反向缩放（1/scales）

#### NonFusionSubgraph（非融合子图）

适用于**仅对若干线性层做平滑、且不融合到前置层**的场景。当映射中 `source` 为 `None`、仅提供 `targets` 时，处理器会走非融合分支，将 `targets` 中的线性层组成非融合子图进行平滑。

**典型场景：**

- 没有可融合的前置归一化/线性层，仅需对若干独立线性层做离群值抑制
- 结构特殊，无法归类为 norm-linear / linear-linear / ov / up-down，但仍希望对这些线性层做平滑

**处理方式：**

- 将 `targets` 中的多个线性层视为一组，基于其权重的列最大值与激活统计计算统一的每通道缩放因子
- 对每个线性层的权重应用正向缩放
- 在每个线性层上注册前向 pre-hook，在推理时对**输入**施加对应的反向缩放（1/scales），从而在数值上等价于“先对输入除以 scale 再乘回”，与权重缩放配合保持数值一致
- **不支持 shift**（偏置平移）；若配置了 shift，非融合路径会忽略并打日志提示

**配置方式：** 在 `get_adapter_config_for_subgraph()` 中返回的 `AdapterConfig` 中，将 `mapping` 设为 `MappingConfig(source=None, targets=[...])`，`targets` 为需要平滑的线性层完整路径列表；`subgraph_type` 可填 `"norm-linear"`（表示多目标时记录所有 linear 名称）或其他已支持类型，仅影响内部命名，不改变非融合行为。

### 实现

#### 代码实现

算法在 [msmodelslim/processor/anti_outlier/iter_smooth/processor.py](../../../../../msmodelslim/processor/anti_outlier/iter_smooth/processor.py) 中实现，处理流程分两阶段。

#### 预处理阶段

**子图发现与构建：**

- 通过 `SubgraphProcessor` 获取全局子图信息，识别四种类型的子图：`norm-linear`、`linear-linear`、`ov`、`up-down`。
- 根据配置的 `include/exclude` 模式过滤子图。

**统计信息收集：**

- 为所有子图中的线性模块安装前向钩子（forward hook）。
- 钩子在 `[batch, seq, hidden_dim]` 维度上收集激活值统计信息：
  - 每通道的最大值、最小值
  - 每通道的绝对最大值（用于平滑缩放计算）
  - 通道偏移量（用于对称量化）
- 支持分布式训练环境下的统计信息聚合。

#### 后处理阶段

**按优先级处理子图：**

- 按默认配置的优先级顺序处理：`up-down`（最高）→ `ov`（高）→ `norm-linear`（中）→ `linear-linear`（低）。
- 每种子图类型调用相应的平滑处理方法。

**子图平滑处理：**

- **Norm-Linear子图**：对归一化层和后续线性层应用平滑，支持RMSNorm偏置调整。
- **Linear-Linear子图**：对两个线性层应用平滑，调整权重和偏置。
- **OV子图**：处理注意力机制中的输出投影（Output projection）和值投影（Value projection）之间的连接关系，支持QKV融合模式。
- **Up-Down子图**：处理MLP门控机制，对上下投影层应用平滑。
- **非融合子图**：当 `mapping.source` 为 `None` 且 `mapping.targets` 非空时，将目标线性层组成 NonFusionSubgraph，仅对权重做缩放并在每层注册输入侧 scale 的 pre-hook，不融合到前置层；不支持 shift。

**资源清理：**

- 清理所有安装的统计钩子
- 释放统计信息内存
- 恢复模型原始状态

## 适用要求

- **模型架构要求**：模型必须支持 `IterSmoothInterface` 接口，并正确配置子图映射关系。
- **模块命名要求**：模块名称必须与 `named_modules()` 返回的完整路径完全一致。
- **子图类型支持**：目前支持四种标准子图类型：`norm-linear`、`linear-linear`、`ov`、`up-down`；另支持**非融合子图**（映射中 `source=None`、仅提供 `targets` 的线性层列表）。
- **模块属性要求**：目标模块必须存在且具备可写的 `weight`（以及可选 `bias`），其他自定义模块暂不支持。
- **模型结构假设**：算法基于标准的Transformer架构设计，对于非标准结构需要谨慎评估适用性。

## 功能介绍

### YAML配置示例

作为Processor使用，YAML配置示例如下：

```yaml
spec:
  process:
    - type: "iter_smooth"                  # 固定为 `iter_smooth`，用于指定 Processor。
      alpha: 0.9                           # 平衡参数，控制激活和权重的相对重要性，默认0.9。
      scale_min: 1e-5                      # 缩放因子的最小值，防止数值不稳定，默认1e-5。
      symmetric: True                      # 是否启用对称量化，默认True。
      enable_subgraph_type:                # 开启的子图类型。
        - 'norm-linear'
        - 'linear-linear'
        - 'ov'
        - 'up-down'
      include: ["*"]                       # 包含的层，支持通配符。
      exclude: ["*self_attn*"]             # 排除的层，支持通配符。
```

### YAML配置字段详解

| 字段名 | 作用      | 说明 |
|--------|---------|------|
| type | 处理器类型标识 | 固定值"iter_smooth"，用于标识这是一个迭代平滑处理器。|
| alpha | 平衡参数    | 大于0的浮点数，控制激活和权重的相对重要性，默认0.9。 |
| scale_min | 缩放因子最小值 | 大于0的浮点数，防止数值不稳定，默认1e-5。 |
| symmetric | 是否对称量化  | 布尔值，True为对称，False为非对称，默认True。 |
| enable_subgraph_type | 开启的子图类型 | 支持的子图类型列表，包括"norm-linear"、"linear-linear"、"ov"、"up-down"。 |
| include | 包含的层  | 支持通配符匹配。 |
| exclude | 排除的层  | 支持通配符匹配。|

## 模型适配

### 接口与数据结构

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class MappingConfig:
    """模块映射关系配置"""
    source: Optional[str]  # 源模块名称，如 "model.layers.0.input_layernorm"；为 None 时表示非融合子图，仅对 targets 中的线性层做平滑
    targets: List[str]  # 目标模块名称列表，如 ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"]

@dataclass
class FusionConfig:
    """融合配置，支持QKV融合等高级功能"""
    fusion_type: str = "none"  # 融合类型：none, qkv, custom等
    num_attention_heads: Optional[int] = None  # 注意力头数量
    num_key_value_heads: Optional[int] = None  # 键值头数量
    custom_config: Optional[Dict[str, Any]] = None  # 自定义配置

@dataclass
class AdapterConfig:
    """子图适配器配置"""
    subgraph_type: str  # 子图类型：norm-linear, linear-linear, ov, up-down（非融合时也可填 norm-linear 等，仅影响内部命名）
    mapping: Optional[MappingConfig] = None  # 模块映射关系
    fusion: FusionConfig = field(default_factory=lambda: FusionConfig())  # 融合配置

# 模型适配Smooth算法接口
class IterSmoothInterface(ABC):
    @abstractmethod
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """
        返回模型中所有可进行Smooth处理的子图配置

        Returns:
            List[AdapterConfig]: 子图配置列表，每个配置包含：
                - subgraph_type: 子图类型
                - mapping: 源模块到目标模块的映射关系
                - fusion: 融合配置（如QKV融合）
        """
        pass
```

### 适配步骤

**前置要求：**

- 模型需要继承 `IterSmoothInterface` 接口。
- 模块名称必须与 `named_modules()` 返回的完整路径一致。
- 支持的子图类型：`norm-linear`、`linear-linear`、`ov`、`up-down`；另支持**非融合子图**（`mapping.source=None`，仅提供 `targets`）。
- 配置中的`subgraph_type`、`mapping` 是必要参数。
- 当配置`FusionConfig`且`fusion_type`为qkv时，必须给出num_attention_heads和num_key_value_heads。

**步骤：**

1. **继承接口**：模型适配器继承 `IterSmoothInterface` 接口，实现 `get_adapter_config_for_subgraph()` 方法。
2. **配置子图映射**：为每层配置子图映射关系：
   - **Norm-Linear子图**：归一化层到后续线性层的映射（`source` 为 norm，`targets` 为后续线性层）
   - **OV子图**：注意力机制中V投影到O投影的映射
   - **Up-Down子图**：MLP门控机制中上投影到下投影的映射
   - **Linear-Linear子图**：连续线性层的映射
   - **非融合子图**：`source=None`，`targets` 为需要平滑的线性层路径列表（可不融合到前置层）
3. **指定模块路径**：使用完整的模块路径，如 `model.layers.{i}.self_attn.q_proj`。

**参考实现：** 可参考 [msmodelslim/model/qwen3/model_adapter.py](../../../../../msmodelslim/model/qwen3/model_adapter.py) 中的 `Qwen3ModelAdapter` 实现。

### 配置示例

以下是一个典型的Transformer层配置示例：

```python
def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
    adapter_config = []
    for layer_idx in range(self.config.num_hidden_layers):
        # 1. 输入层归一化到QKV投影的Norm-Linear映射
        norm_linear_config1 = AdapterConfig(
            subgraph_type="norm-linear",
            mapping=MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",
                targets=[
                    f"model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.layers.{layer_idx}.self_attn.v_proj"
                ]
            )
        )

        # 2. 后注意力层归一化到MLP投影的Norm-Linear映射
        norm_linear_config2 = AdapterConfig(
            subgraph_type="norm-linear",
            mapping=MappingConfig(
                source=f"model.layers.{layer_idx}.post_attention_layernorm",
                targets=[
                    f"model.layers.{layer_idx}.mlp.gate_proj",
                    f"model.layers.{layer_idx}.mlp.up_proj"
                ]
            )
        )

        # 3. 注意力机制中的OV映射
        ov_config = AdapterConfig(
            subgraph_type="ov",
            mapping=MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.v_proj",
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"]
            )
        )

        # 4. MLP门控机制的Up-Down映射
        up_down_config = AdapterConfig(
            subgraph_type="up-down",
            mapping=MappingConfig(
                source=f"model.layers.{layer_idx}.mlp.up_proj",
                targets=[f"model.layers.{layer_idx}.mlp.down_proj"]
            )
        )

        # 5. 非融合子图示例：仅对若干线性层做平滑，不融合到前置层（source 设为 None）
        # non_fusion_config = AdapterConfig(
        #     subgraph_type="norm-linear",
        #     mapping=MappingConfig(
        #         targets=[
        #             f"model.layers.{layer_idx}.some_module.linear_a",
        #             f"model.layers.{layer_idx}.some_module.linear_b",
        #         ]
        #     )
        # )
        # adapter_config.append(non_fusion_config)

        adapter_config.extend([norm_linear_config1, norm_linear_config2, ov_config, up_down_config])

    return adapter_config
```

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

**解决方案**: 确保配置的子图类型在 `ENABLE_SUBGRAPH_TYPES` 列表中。

### 映射关系错误

**现象**: `MappingConfig` 中的 `source` 和 `targets` 指向错误的模块。

**解决方案**: 检查 `MappingConfig` 中的 `source` 和 `targets` 是否指向正确的模块。

### 非融合子图不生效

**现象**: 已配置 `source=None` 和 `targets`，但未走非融合平滑。

**解决方案**: 确认 `mapping.source` 为 `None`（Python 中显式传入 `None`），且 `mapping.targets` 非空；确认这些目标模块在 `include` 范围内且未被 `exclude` 过滤掉。
