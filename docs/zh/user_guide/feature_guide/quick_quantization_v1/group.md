# 组合处理器 (Group Processor)

## 1. 简介

组合处理器（Group Processor）是 msModelSlim 中用于实现**精细化量化策略**的核心组件。它允许用户将多个处理器（如线性量化、平滑处理等）封装在一个逻辑组内，针对模型的不同层应用差异化的量化配置。

通过组合处理器，您可以轻松实现“混合量化”（Mixed Quantization），例如在同一个模型中对某些层使用静态量化以追求性能，而对另一些层使用动态量化以保证精度。

## 2. 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

## 3. 原理和实现

### 3.1 原理

在处理超大规模模型（如 LLM）时，不同模块（如 Attention 和 MLP）的激活值分布特征往往截然不同。单一的全局量化配置难以在所有层上同时达到最优的精度与性能平衡。

组合处理器的核心原理是**分而治之与过程共享**：

1. **逻辑分组**：通过 `include` 和 `exclude` 模式匹配，将模型层划分为不同的逻辑集合。
2. **局部最优**：为每个集合分配最适合其分布特征的处理器类型和量化参数。
3. **过程共享与加速**：这是使用组合处理器的核心优势。在量化校准阶段，每个独立的处理器（如 `linear_quant`）通常都需要对校准集执行完整的推理（Forward）过程。
   - **不使用 Group**：如果有 M 个独立的 `linear_quant` 处理器，且校准集大小为 N，则总共需要执行 M × N 次推理。
   - **使用 Group**：将多个处理器挂载到 Group 容器后，它们会共享推理过程，总共只需执行 N 次推理。
   因此，使用 Group 容器能显著提升量化速度。虽然不使用 Group 也能通过多个独立处理器实现混合量化，但效率较低，且由于处理时机不同，结果可能存在细微差别（通常不影响最终效果）。
4. **资源优化**：在逐层量化（Layer-wise）模式下，组合处理器可以确保同一层的多个处理操作在一次加载中完成，显著降低 I/O 开销。

### 3.2 实现

组合处理器在 [msmodelslim/processor/container/group.py](../../../../../msmodelslim/processor/container/group.py) 中实现。它作为一个容器，顺序调度其 `configs` 列表中的子处理器。

## 4. 功能介绍

### 4.1 适用场景

- **混合量化策略**：如 Attention 层使用 W8A8 静态量化，MLP 层使用 W8A8 动态量化。
- **敏感层保护**：对精度极度敏感的层（如 `down_proj` 或 `gate`）使用更高比特或特定的平滑算法。
- **多算法组合**：在同一组层上先后应用平滑处理（Smooth）和线性量化（Linear Quant）。

### 4.2 YAML配置示例

以下示例展示了如何使用组合处理器实现 **W8A8 静态+动态混合量化**：

```yaml
# 1. 定义静态量化模板 (Anchor)
default_w8a8_static: &w8a8_static
  act:
    scope: "per_tensor"        # 静态量化
    dtype: "int8"
    symmetric: false
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: true
    method: "minmax"

# 2. 定义动态量化模板 (Anchor)
default_w8a8_dynamic: &w8a8_dynamic
  act:
    scope: "per_token"         # 动态量化
    dtype: "int8"
    symmetric: true
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: true
    method: "minmax"

spec:
  process:
    - type: "group"            # 使用组合处理器
      configs:
        - type: "linear_quant" # 子处理器 1：针对 Attention 层
          qconfig: *w8a8_static
          include: ["*self_attn*"]

        - type: "linear_quant" # 子处理器 2：针对 MLP 层
          qconfig: *w8a8_dynamic
          include: ["*mlp*"]
          exclude: ["*gate*"]  # 排除门控层进行更精细控制
```

### 4.3 YAML配置字段详解

| 字段名 | 作用 | 类型 | 说明 |
|--------|------|------|------|
| type | 处理器类型标识 | `string` | 固定值 `"group"`。 |
| configs | 子处理器配置列表 | `list[object]` | 包含多个子处理器的配置。每个子处理器的字段（如 `type`, `qconfig`, `include`）与其独立使用时一致。 |

## 5. FAQ

**组合处理器中的执行顺序是怎样的？**

组合处理器会严格按照 `configs` 列表中的顺序依次对匹配到的层执行处理。

**如果一个层被多个子处理器匹配到了会怎样？**

如果一个层同时满足多个子处理器的 `include` 条件，它将先后接受这些处理器的处理。**注意**：通常不建议对同一层重复应用多次量化操作，这可能会导致精度异常或报错。建议通过 `exclude` 字段进行精确避让。

**为什么要优先使用 Group 容器而不是多个独立的处理器？**

虽然通过在 `process` 列表中配置多个独立的处理器也能实现混合量化，但 Group 容器通过 **共享推理过程（Forward）** 极大提升了效率。

如果不使用 Group 容器，每个独立的处理器都会对校准集调用 N 次推理；而将它们挂载到 Group 容器时，总共只需调用 N 次推理。这在处理大规模校准集或复杂模型时能显著节省量化时间。此外，使用 Group 容器在结果上与独立处理器可能存在细微差别，但通常不构成精度问题。
