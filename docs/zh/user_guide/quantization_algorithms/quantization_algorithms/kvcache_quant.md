# KVCache量化：缓存量化算法说明

## 简介

- **简介**：KVCache量化机制。KV Cache 量化通常与[线性量化](linear_quant.md)配合使用，以实现全量化方案。
- **问题**：在大模型推理中，KVCache 存储的 Key/Value 状态占用大量显存，随序列长度线性增长，成为推理瓶颈。
- **目标**：对写入 KVCache 的 `key_states` 和 `value_states` 进行量化，在保持生成质量的前提下显著降低缓存内存占用。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

## 原理和实现

### 原理

**核心思想：**

1. **量化目标**：对注意力机制中写入 KVCache 的 `key_states` 和 `value_states` 进行 INT8 量化。
2. **量化时机**：在 `DynamicCache.update()` 调用时，拦截 Key/Value 状态应用量化校准。
3. **量化策略**：per_channel：按隐藏层维度计算量化参数，平衡精度和效率。
4. **内存优化**：量化后的缓存状态理论上可减少约 50% 的cache内存占用（FP16→INT8）。

### 实现

#### 代码实现

- 算法在 [msmodelslim/processor/quant/attention.py](../../../../../msmodelslim/processor/quant/attention.py) 中实现，处理流程分三阶段。

#### 检测阶段

  - 阶段：`pre_run`。
  - 自动检测模型中的注意力层，基于模块命名规则识别 `self_attn` 模块。
  - 为每个注意力层创建对应的 `DynamicCacheQuantizer`，配置量化参数。
  - 在目标注意力层的第一层安装触发钩子，检测推理开始。

#### 校准阶段

  - 阶段：`run`。
  - 通过钩子机制在 `DynamicCache.update()` 调用时拦截 Key/Value 状态。
  - 使用 `DynamicCacheQuantizer` 对缓存状态进行伪量化，收集量化统计信息。
  - 支持增量式校准，适应动态序列长度变化。

#### 伪量化部署阶段

  - 阶段：`postprocess`。
  - 将校准完成的量化器转换为推理优化的 `FakeQuantDynamicCache` IR。
  - 保持与原有缓存机制的兼容性，无需修改上层推理逻辑。

### 核心组件

#### 量化器实现

```python
# DynamicCacheQuantizer：校准阶段的量化器
class DynamicCacheQuantizer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 转置：(batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
        x = x.transpose(-2, -3)
        # 2. 重塑：适配量化算法的输入格式
        x = x.reshape(-1, x.shape[-1] * x.shape[-2])
        # 3. 量化：应用伪量化收集统计信息
        x = self.input_quantizer(x)
        # 4. 恢复：还原到原始形状
        return x.transpose(-2, -3)

# FakeQuantDynamicCache：部署阶段的伪量化IR
class FakeQuantDynamicCache(AutoFakeQuantDynamicCache):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 转置：(batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
        x = x.transpose(-2, -3)
        x_shape = x.shape
        # 2. 重塑：适配量化算法的输入格式
        x = x.reshape(-1, x.shape[-1] * x.shape[-2])
        # 3. 量化：使用固定的量化参数进行伪量化
        x_q_dq = fake_quantize(QStorage(QDType.FLOAT, x), self.x_q_param).value
        # 4. 恢复：还原到原始形状
        x_q_dq = x_q_dq.reshape(x_shape)
        x_q_dq = x_q_dq.transpose(-2, -3)
        return x_q_dq
```

## 适用要求

- **模型结构限制**：
  - 注意力模块forward函数必须接受一个DynamicCache对象并调用 `cache.update()`。
  - update接口需要正确传递 `layer_idx` 参数以区分不同层的量化器。
  - 目前基于模块类名称模式匹配（`*self_attn*`），自定义命名需要适配。

- **量化方式限制**：
  - 当前仅支持 INT8 量化。
  - 仅对 KVCache 状态量化，不影响 query_states 和注意力权重。

- **内存管理限制**：
  - 伪量化阶段仍需原精度内存，真实内存节省需要底层算子支持。
  - 量化参数本身占用少量额外内存。

- **缓存类型限制**：
  - **DynamicCache**：Transformers 标准动态缓存，完全支持。
  - **自定义Cache**：自定义模型内部需要实现 `update(key_states, value_states, layer_idx)` 接口，要求如下：
  1. **接口要求**：

      ```python
      class CustomCache:
          def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
                      layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
              # 返回更新后的 key_states 和 value_states
              pass
      ```

  2. **钩子注册**：

     - 缓存对象必须作为参数传递给注意力模块的 `forward` 方法。
     - 系统会自动检测 `cache.update` 调用并注册量化钩子。

  3. **参数传递**：

     - 注意力模块需要通过 `layer_idx` 参数指示当前层索引。
     - 支持嵌套调用和递归量化。

## 功能介绍

### 模型支持

- Qwen2.5系列
- Qwen3系列

### YAML配置示例

作为Processor使用，YAML配置示例如下：

```yaml
- type: "dynamic_cache" # 固定为 `dynamic_cache`，用于指定 Processor。
  qconfig: # 量化配置，支持 per_channel 量化
    scope: "per_channel" # 量化粒度：仅支持per_channel
    dtype: "int8" # 量化数据类型，目前支持 int8
    symmetric: True # 对称量化，默认 True
    method: "minmax" # 量化方法，目前支持 minmax
  include: # 字符串列表，参与量化的 attention 匹配模式（完整路径，支持 `*` 通配），默认全量。
    - "*"
  exclude: # 字符串列表，禁止量化的 attention 匹配模式（完整路径，支持 `*` 通配），默认为空。
    - "model.layers.0.self_attn"
```

### YAML配置字段详解

| 字段名 | 作用 | 数据类型 | 默认值 | 说明 |
|--------|------|----------|--------|------|
| type | 处理器类型标识 | string | - | 固定值"dynamic_cache"，用于标识该对象为KVCache量化处理器。 |
| qconfig | KVCache量化配置 | object | - | 包含KVCache的量化参数配置。 |
| scope | 量化粒度 | string | "per_channel" | 量化粒度设置，当前仅支持配置为"per_channel"，表示按隐藏层维度计算量化参数。 |
| dtype | 量化数据类型 | string | "int8" | 量化后的数据类型，当前仅支持配置为"int8"。 |
| symmetric | 对称量化开关 | boolean | true | 是否使用对称量化。true表示使用对称量化，false表示使用非对称量化。 |
| method | 量化方法 | string | "minmax" | 量化算法方法，当前仅支持"minmax"算法。 |
| include | 包含的注意力层 | array[string] | ["*"] | 支持通配符匹配，指定要执行KVCache量化的注意力层。 |
| exclude | 排除的注意力层 | array[string] | [] | 支持通配符匹配，优先级高于include。 |

## FAQ

### 缓存未被量化

**现象**：缓存未被量化。

**解决方案**：确认注意力前向接受了一个 `cache` 参数并正确调用 `cache.update()`。

### 新缓存类型不支持

**现象**：新缓存类型不支持。

**解决方案**：确认自定义缓存实现了标准的 `update` 接口，并正确处理返回值。
