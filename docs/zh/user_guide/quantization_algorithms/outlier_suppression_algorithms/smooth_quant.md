# SmoothQuant：离群值抑制算法说明

## 简介

- **来源**：MIT 提出的 SmoothQuant 算法。
- **概述**：SmoothQuant 是一种用于大语言模型量化过程中抑制激活离群值的算法。该算法通过在归一化层和线性层之间协同缩放，实现将激活值中的离群值“平滑”到权重中，从而使激活值更易于量化。
- **核心思想**：SmoothQuant 的核心思想是利用数学等价变换，将激活值除以一个平滑因子，同时将权重乘以该因子，在不改变模型输出的前提下，使激活值的分布更加均匀，减少离群值对量化精度的影响。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

## 原理和实现

### 原理

**算法核心：**

- 使用 `smooth_quant` 算法对子图进行平滑处理。
- 支持可配置的平滑参数：`alpha`（平滑强度）、`symmetric`（对称量化）。
- 缩放因子下界固定为 `1e-5`。

**算法公式：**

SmoothQuant 算法基于以下数学等价变换：

```text
Y = XW = (X · diag(s)^(-1)) · (diag(s) · W) = X̂ · Ŵ
```

其中：

- `X`：激活值
- `W`：权重
- `s`：平滑缩放因子
- `X̂ = X · diag(s)^(-1)`：平滑后的激活值
- `Ŵ = diag(s) · W`：平滑后的权重

平滑缩放因子的计算公式：

```text
scales = (A_scale**α / W_scale**(1-α)).clamp(min=1e-5)
```

其中：

- `A_scale`：激活值每通道的绝对值最大值
- `W_scale`：权重每列的绝对值最大值
- `α`：平衡参数，控制激活和权重的相对重要性（默认值：0.5）
- `1e-5`：缩放因子的最小值，防止数值不稳定

### 支持的子图类型

#### NormLinearSubgraph

SmoothQuant 仅支持 NormLinearSubgraph（归一化-线性子图）类型。

适用于包含归一化层和多个线性层的结构，如：

```python
x = norm(x)
y = torch.cat([linear(x) for linear in linears], dim=-1)
```

**处理方式：**

- 计算所有线性层权重的列最大值作为权重缩放因子
- 对每个线性层执行正向缩放操作（权重乘以 scales）
- 对归一化层执行反向缩放操作（权重除以 scales）
- 如果启用非对称量化，还会计算并应用偏移量

### 实现

#### 代码实现

算法在 [msmodelslim/processor/anti_outlier/smooth_quant/](https://gitcode.com/Ascend/msmodelslim/tree/master/msmodelslim/processor/anti_outlier/smooth_quant) 中实现，处理流程分两阶段。

#### 预处理阶段

**子图发现与构建：**

- 通过模型适配器的 `get_adapter_config_for_subgraph()` 获取子图信息。
- 仅处理 `norm-linear` 类型的子图，其他类型会被自动过滤。
- 根据配置的 `include/exclude` 模式过滤子图。

**归一化层替换：**

- 将原始的 RMSNorm 模块替换为支持偏置的 RMSNormBias 模块（为了在非对称量化模式下能够正确处理偏移量）。

**统计信息收集：**

- 为所有子图中的线性模块安装前向钩子（forward hook）。
- 钩子在 `[batch, seq, hidden_dim]` 维度上收集激活值统计信息：
  - 每通道的绝对最大值（用于平滑缩放计算）
  - 通道偏移量（用于非对称量化）

#### 后处理阶段

**子图平滑处理：**

- 遍历所有 `norm-linear` 子图，依次应用平滑算法。
- 基于收集的激活统计信息和权重信息计算平滑缩放因子。
- 对归一化层和线性层分别应用反向/正向缩放。

**资源清理：**

- 清理所有安装的统计钩子
- 释放统计信息内存
- 恢复模型原始状态

## 适用要求

- **模型架构要求**：模型必须支持 `SmoothQuantInterface` 接口，并正确配置子图映射关系。
- **模块命名要求**：模块名称必须与 `named_modules()` 返回的完整路径完全一致。
- **子图类型支持**：SmoothQuant 仅支持 `norm-linear` 子图类型。
- **模块属性要求**：目标模块必须存在且具备可写的 `weight`（以及可选 `bias`）。
- **模型结构假设**：算法基于标准的Transformer架构设计，对于非标准结构需要谨慎评估适用性。

## 功能介绍

### YAML配置示例

作为Processor使用，YAML配置示例如下：

```yaml
spec:
  process:
    - type: "smooth_quant"                 # 固定为 `smooth_quant`，用于指定 Processor 类型。
      alpha: 0.5                           # 平衡参数，控制激活和权重的相对重要性，浮点数，0~1，默认0.5。
      symmetric: True                      # 是否启用对称量化，默认True，True为对称，False为非对称。
      include: ["*"]                       # 包含的层，支持通配符匹配，默认为["*"]（全量）。
      exclude: ["*self_attn*"]             # 排除的层，支持通配符匹配，默认为空。
```

**注意**：SmoothQuant 仅支持 `norm-linear` 子图类型，不支持其他子图类型（如 `ov`、`up-down`、`linear-linear`），因而不支持指定 `enable_subgraph_type` 字段。

### YAML配置字段详解

| 字段名 | 作用      | 说明 |
|--------|---------|------|
| type | 处理器类型标识 | 固定值"smooth_quant"，用于标识这是一个SmoothQuant处理器。|
| alpha | 平衡参数    | 0~1之间的浮点数，控制激活和权重的相对重要性，默认0.5。 |
| symmetric | 是否对称量化  | 布尔值，True为对称，False为非对称，默认True。 |
| include | 包含的层  | 字符串列表，支持通配符匹配，默认为["*"]（全量）。 |
| exclude | 排除的层  | 字符串列表，支持通配符匹配，默认为空。|

## 模型适配

### 接口与数据结构

```python
from dataclasses import dataclass, field
from typing import List, Optional
from abc import ABC, abstractmethod

@dataclass
class MappingConfig:
    """模块映射关系配置"""
    source: str  # 源模块名称，如 "model.layers.0.input_layernorm"
    targets: List[str]  # 目标模块名称列表，如 ["model.layers.0.self_attn.q_proj", ...]

@dataclass
class AdapterConfig:
    """子图适配器配置"""
    subgraph_type: str  # 子图类型，SmoothQuant仅支持 "norm-linear"
    mapping: Optional[MappingConfig] = None  # 模块映射关系

# 模型适配SmoothQuant算法接口
class SmoothQuantInterface(ABC):
    @abstractmethod
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """
        返回模型中所有可进行SmoothQuant处理的子图配置

        Returns:
            List[AdapterConfig]: 子图配置列表，每个配置包含：
                - subgraph_type: 子图类型（应为"norm-linear"）
                - mapping: 源模块到目标模块的映射关系
        """
        pass
```

### 适配步骤

**前置要求：**

- 模型需要继承 `SmoothQuantInterface` 接口。
- 模块名称必须与 `named_modules()` 返回的完整路径一致。
- SmoothQuant 仅支持 `norm-linear` 子图类型。
- 配置中的`subgraph_type`、`mapping` 是必要参数。

**步骤：**

1. **继承接口**：模型适配器继承 `SmoothQuantInterface` 接口，实现 `get_adapter_config_for_subgraph()` 方法。
2. **配置子图映射**：为每层配置 norm-linear 子图映射关系。
3. **指定模块路径**：使用完整的模块路径，如 `model.layers.{i}.input_layernorm`。

**参考实现：** 可参考 [msmodelslim/model/qwen3/model_adapter.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/qwen3/model_adapter.py) 中的 `Qwen3ModelAdapter` 实现。

### 配置示例

以下是一个典型的Transformer层配置示例：

```python
def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
    adapter_config = []
    for layer_idx in range(self.config.num_hidden_layers):
        # 1. 输入层归一化到QKV投影的norm-linear映射
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

        # 2. 后注意力层归一化到MLP投影的norm-linear映射
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

        adapter_config.extend([norm_linear_config1, norm_linear_config2])

    return adapter_config
```

## FAQ

### 模块名称不匹配

**现象**: `include/exclude` 未命中时，日志提示未匹配模式。

**解决方案**: 核对完整模块名称是否与 `named_modules()` 返回的路径一致。

### 子图配置错误

**现象**: `get_adapter_config_for_subgraph()` 返回的配置不正确。

**解决方案**: 检查配置中的 `source` 和 `targets` 字段是否正确。

### 模块不存在

**现象**: 配置中指定的模块名称在模型中不存在。

**解决方案**: 通过 `model.named_modules()` 验证模块是否确实存在。

### 映射关系错误

**现象**: `MappingConfig` 中的 `source` 和 `targets` 指向错误的模块。

**解决方案**: 检查 `MappingConfig` 中的 `source` 是否为归一化层，`targets` 是否为其后续的线性层。
