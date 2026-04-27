# FA3量化：Flash Attention 3激活量化算法说明

## 简介

- **背景**：一方面，在长序列下，Attention 的中间激活 Q、K、V 张量在显存中占比高，对其进行量化将有效降低显存占用并提升计算效率；另一方面，Q、K、V 的激活动态范围大且分布高度不均，直接进行全局量化可能会导致精度损失严重。
- **核心思想**：Flash Attention 3（FA3）是一种针对注意力机制激活的 per-head（逐注意力头）量化算法，对注意力机制中的 Q、K、V 激活进行 per-head 粒度的 INT8 量化，在保持模型精度的前提下提升推理性能和降低显存占用。FA3 量化通常与[线性量化](linear_quant.md)配合使用，以实现全量化方案。

## 使用前准备

安装 msModelSlim 工具，详情请参见《[msModelSlim工具安装指南](../../getting_started/install_guide.md)》。

## 原理和实现

### 原理

**核心思想：**

1. **量化目标**：对注意力机制中的 Q、K、V 激活值进行 per-head 粒度的 INT8 量化。
2. **量化粒度**：per-head（每个注意力头独立量化参数），适应不同 head 的激活分布差异。
3. **量化时机**：在 Multi-head Latent Attention (MLA) 计算的关键位置插入量化节点
4. **量化策略**：per-head：对每个注意力头独立计算量化参数。

**算法流程：**

1. 收集每个 head 的激活统计数据：

    - 输入：激活张量 x，shape 为 `(B, H, S, D)`
    - 其中 B=batch_size, H=num_heads, S=seq_len, D=head_dim
    - 将 x reshape 为 `(H, N)`，`N = B * S * D`
    - 每个 head 独立收集 N 个数据点

2. 对每个 head 使用 Recall Window 算法找到最小量化范围：

    - 输入：head_data (N,), ratio (默认 0.9999)
    - 对 N 个数据点进行排序：sorted_data = sort(head_data)
    - 计算目标元素数量：target_num = int(ratio * N)
    - 滑动窗口搜索最小范围：

      - 遍历所有可能的窗口起点 i = 0 到 (N - target_num)
      - 窗口范围：[sorted_data[i], sorted_data[i + target_num - 1]]
      - 计算窗口长度：window_length = sorted_data[i + target_num - 1] - sorted_data[i]
      - 保留窗口长度最小的窗口

    - 输出：该 head 的 (min_val, max_val)

3. 跨批次累积统计：

    - 对每个校准批次，计算当前批次的 (min_val, max_val)
    - 更新累积统计值，确保量化范围覆盖所有校准数据：

      - min_values[h] = min(min_values[h], current_min[h])
      - max_values[h] = max(max_values[h], current_max[h])

4. 计算每个 head 的量化参数：

    - 对称量化公式：

      - abs_max[h] = max(abs(min_values[h]), abs(max_values[h]))
      - scale[h] = abs_max[h] / 127

    - 输出：量化参数 q_param

### 实现

#### 代码实现

- FA3 量化在 [processor.py](../../../../msmodelslim/processor/quant/fa3/processor.py) 中实现，处理流程分三阶段。

#### 注入阶段

- 阶段：`preprocess`。
- 调用模型适配器的 `inject_fa3_placeholders()` 方法。
- 适配器负责在 MLA 计算流程中的关键位置插入占位器 `FA3QuantPlaceHolder`。
- 支持通过 `include/exclude` 配置选择性注入。

#### 校准阶段

- 阶段：`process`。
- 占位符被替换为监听器 `_FA3PerheadObserver`。
- 校准数据流经注意力层时，监听器收集每个 head 的激活统计信息。
- 根据滑动窗口的思想找到包含指定比例数据的最小数值分布区间。

#### 伪量化部署阶段

- 阶段：`postprocess`。
- 从监听器提取每个 head 的 min/max 值。
- 调用 `calculate_qparam()` 计算对称量化参数。
- 创建 IR 替换监听器。

## 适用要求

- **模型结构要求**：
  - 必须有支持 FA3 的模型适配器实现 `FA3QuantAdapterInterface`。
  - 适用于基于 MLA 的注意力机制。
  - 需要明确的 Q、K、V 激活计算路径以插入量化节点。

- **量化方式限制**：
  - 当前仅支持 INT8 对称量化。
  - 量化参数在校准后固定，尚不支持动态调整。

## 功能介绍

### 模型支持

- DeepSeek-R1-0528
- DeepSeek-V3.1

### YAML配置示例

作为Processor使用，YAML配置示例如下：

```yaml
spec:
  process:
    - type: "fa3_quant"
      include: [ "*" ]                           # 包含的注意力层
      exclude: [ "model.layers.0.self_attn" ]   # 排除的注意力层
```

### YAML配置字段详解

| 字段名  | 作用               | 数据类型      | 默认值 | 说明                                               |
| ------- | ------------------ | ------------- | ------ | -------------------------------------------------- |
| type    | 处理器类型标识     | string        | -      | 固定值"fa3_quant"，用于标识该对象为FA3量化处理器。 |
| include | 包含的注意力层 | array[string] | ["*"]  | 支持通配符匹配，指定要执行FA3量化的注意力层。      |
| exclude | 排除的注意力层 | array[string] | []     | 支持通配符匹配，优先级高于 include。               |

## 模型适配

### 接口与数据结构

目前已支持 DeepSeek-V3 系列模型，其他基于 MLA 的模型需要实现相应的适配器。

```python
# 模型适配 FA3 量化接口
class ModelAdapter(FA3QuantAdapterInterface):
    def inject_fa3_placeholders(
            self,
            root_name: str,
            root_module: nn.Module,
            should_inject: Callable[[str], bool],
    ) -> None: ...
```

### 适配步骤

- **前置要求**：

  - 模型基于 Transformer 架构，包含明确的注意力层。
  - 注意力层的 Q、K、V 激活值在计算流程中可定位。
  - 适配器能够访问模型的注意力模块并修改其 forward 方法。

- **步骤**：

  可参考 DeepSeek 的 [model_adapter.py](../../../../msmodelslim/model/deepseek_v3/model_adapter.py) 的实现：

  1. 模型适配器继承 `FA3QuantAdapterInterface` 接口。
  2. 遍历模型，通过 `should_inject` 在注意力层中选择性注入占位器 `FA3QuantPlaceHolder` 作为子模块。
  3. 定位 Q、K、V 激活流向 Attention 计算的临界位置，该位置即为需要插入 FA3 量化的节点。
  4. 包裹注意力层的 `forward` 方法，在定位到的临界位置插入对 FA3 量化的调用。
