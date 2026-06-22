# KVSmooth：KVCache量化离群值抑制算法说明

## 简介

- **问题**：在 KVCache 量化中，Key 的少量离群值会显著抬高量化尺度，导致大部分通道有效比特不足，从而使注意力打分退化、生成质量下降。
- **目标**：在不改变注意力打分 QK^T 期望值的前提下，压缩 K 的动态范围，使其更易量化，同时保持数值稳定与准确率。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

## 原理和实现

### 原理

**算法核心：**

- 平滑 KVCache 的激活值 `key_states` ，实现方式是把缩放系数 s 融合进 RoPE 之前的 Q/K 投影或归一化权重：
    - `K' = K / s`
    - `Q' = Q × s`
    - 有 `Q'K'^T = QK^T`，注意力分数保持不变，同时 K 的动态范围被压缩，量化更稳健。
- 离群值从 `key_states` 迁移到 `query_states`。由于推理时仅对写入 KVCache 的 `key_states` 做量化，而不量化 `query_states`
  ，该迁移是可接受的，不会引入额外的量化误差。
- RoPE 将通道成对旋转，通道维度呈两两配对关系。算法先在配对通道间取最大，之后再恢复到配对结构进行缩放。

### 实现

#### 代码实现

算法在 [msmodelslim/processor/kv_smooth](../../../../../msmodelslim/processor/kv_smooth/) 中实现，处理流程分两阶段。

#### 观察阶段

  - 阶段：`preprocess`。
  - 通过注入观察器封装 `past_key_values`，在注意力模块调用 `Cache.update()` 时捕获 `key_states`。
  - 使用观测器在维度 [batch, seq] 上聚合 min/max，得到每层每通道的绝对值的最大值，作为缩放的统计基准。

#### 平滑阶段

  - 阶段：`postprocess`。
  - 根据统计到的 `|key_states|` 最大值计算缩放向量，按融合方式重写位于 RoPE 之前的相应模块的 `weight`（和可选 `bias`），使 RoPE 之后写入 KVCache 的 key_states 被平滑；同时，query_states 则相应放大：
    - `state-rope-linear`：沿 `Linear → RoPE → KVCache` 的通路，将缩放折叠进 `k_proj`/`q_proj`。
    - `state-rope-norm`：沿 `Norm → RoPE → KVCache` 的通路，将缩放折叠进 `k_norm`/`q_norm`。

## 适用要求

- **校准集数据依赖**：需要推理标定以观测抑制缩放尺度，若校准集数据分布偏离实际业务，将影响效果。
- **模型实现限制**：注意力前向必须接受并使用 `past_key_values` 或  `past_key_value`，否则无法观测抑制缩放尺度。
- **融合点限制**：目前支持 `Linear/Norm → RoPE → KVCache` 两类通路的融合。
- **融合模块限制**：目标Linear或Norm子模块必须存在且具备可写的 `weight`（以及可选 `bias`），其他自定义模块暂不支持。
- **RoPE假设**：默认按 RoPE 成对通道规约/还原，非 RoPE 结构需谨慎评估与验证。
- **量化方式假设**：算法基于仅量化 KVCache 的 `key_states`/`value_states`，不量化 `query_states` 的假设，若对
  `query_states` 做量化，请谨慎评估该方法的适用性。

## 功能介绍

### YAML配置示例

作为Processor使用，YAML配置示例如下：

```yaml
spec:
  process:
    - type: "kv_smooth"
      smooth_factor: 1.0                    # 控制平滑激进程度，>0，越大平滑越激进
      include: ["*"]                        # 包含的层，支持通配符
      exclude: ["model.layers.0.self_attn"] # 排除的层，支持通配符
```

### YAML配置字段详解

| 参数名             | 作用        | 类型        | 默认值         | 说明              | 示例                             |
|-----------------|-----------|-----------|-------------|-----------------|--------------------------------|
| `type`          | 指定处理器类型   | str       | "kv_smooth" | 固定为 `kv_smooth` | `"kv_smooth"`                  |
| `smooth_factor` | 控制平滑激进程度  | float     | 1.0         | > 0，越大平滑越激进     | `1.5`                          |
| `include`       | 指定参与平滑的模块 | List[str] | ["*"]       | 支持通配符           | `["model.layers.*.self_attn"]` |
| `exclude`       | 指定禁止平滑的模块 | List[str] | []          | 支持通配符           | `["model.layers.0.self_attn"]` |

**注意**：

- `smooth_factor` 必须大于 0
- `include` 和 `exclude` 支持通配符匹配，如 `"model.layers.*.self_attn"`
- `exclude` 的优先级高于 `include`，即如果模块同时匹配 include 和 exclude，则会被排除

## 模型适配

### 接口与数据结构

```python
# 融合方式枚举
class KVSmoothFusedType(Enum):
    StateViaRopeToNorm = 'state-rope-norm'  # 支持 key_states/query_states → Norm 融合
    StateViaRopeToLinear = 'state-rope-linear'  # 支持 key_states/query_states → Linear 融合


# KVSmooth单元信息，描述模型子结构和融合方式
class KVSmoothFusedUnit(BaseModel):
    attention_name: str  # 完整模块名，如 "model.layers.0.self_attn"
    layer_idx: int  # 层索引，如 0
    fused_from_query_states_name: str  # RoPE 前 query_states 分支上的模块名，如 "q_proj" 或 "q_norm"
    fused_from_key_states_name: str  # RoPE 前 key_states 分支上的模块名，如 "k_proj" 或 "k_norm"
    fused_type: KVSmoothFusedType  # 融合类型


# 模型适配KVSmooth算法接口
class KVSmoothFusedInterface(ABC):
    # 模型中所有可进行KVSmooth的单元列表
    def get_kvsmooth_fused_subgraph(self) -> List[KVSmoothFusedUnit]: ...

    # 获取 head_dim 信息
    def get_head_dim(self) -> int: ...

    # 获取 num_key_value_groups 信息
    def get_num_key_value_groups(self) -> int: ...

    # 获取 num_key_value_heads 信息
    def get_num_key_value_heads(self) -> int: ...
```

### 适配步骤

- **前置要求**：
    - 注意力前向需通过 kwargs 接受 `past_key_values` 或 `past_key_value` 并在内部调用 `Cache.update()`，否则观察器无法工作。
    - 目标通路符合 `Linear/Norm → RoPE → KVCache` 的结构。
- **步骤**：
    1. 模型适配器继承`KVSmoothFusedInterface`接口，并实现所有方法， 可参考
       [msmodelslim/model/qwen3/model_adapter.py](../../../../../msmodelslim/model/qwen3/model_adapter.py)。
    2. 在 `get_kvsmooth_fused_subgraph()` 中，为每层返回 `KVSmoothFusedUnit`，指定：
        - `attention_name`：与 `named_modules()` 一致的完整路径（如 `model.layers.{i}.self_attn`）。
        - `layer_idx`：层索引， 用于 Cache.update()。
        - `fused_from_query_states_name`：RoPE 前 `query_states` 分支上的 `norm` 或 `linear` 子模块名，如 `q_proj`。
        - `fused_from_key_states_name`：RoPE 前 `key_states` 分支上的 `norm` 或 `linear` 子模块名，如 `k_proj`。
        - `fused_type`：融合方式枚举，StateViaRopeToNorm 或 StateViaRopeToLinear。
    3. 提供模型全局结构信息：`get_head_dim()`、`get_num_key_value_heads()`、`get_num_key_value_groups()`。

## FAQ

### 回退未命中

**现象**：告警日志中出现 `are not matched any module` 描述。

**解决方案**：核对完整模块名，是否填错 `include` 或 `exclude`。

### 头维度信息缺失

**现象**：抛出 `UnsupportedError`，指明 `get_head_dim`、`get_num_key_value_groups`、`get_num_key_value_heads` 缺失。

**解决方案**：对应模型适配器确保实现 `KVSmoothFusedInterface` 接口，否则模型不适用算法。

### 注意力不适用

**现象**：日志告警 `past_key_values and past_key_value both are None`。

**解决方案**：检查 `Transformers` 中的模型文件，确保 `Attention` 层 `forward` 传入 `past_key_values` 和
      `past_key_value`，否则模型不适用算法。

### 模块名不一致

**现象**：抛出 `ToDoError`，指明 `has no submodule`。

**解决方案**：检查模型适配器，确认 `fused_from_query_states_name` 和 `fused_from_key_states_name` 取值与实际融合子模块命名一致。
