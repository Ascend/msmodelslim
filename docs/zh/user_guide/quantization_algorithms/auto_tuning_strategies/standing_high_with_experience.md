# Standing High With Experience 调优算法

## 简介

Standing High With Experience（基于专家经验的摸高算法）是在 《[Standing High 调优算法](standing_high.md)》 摸高算法基础上，结合**专家经验**的自动调优策略。该策略基于专家经验自动生成完整的摸高配置，用户只需指定**量化类型**（`quant_type`）和**模型结构配置**（`structure_configs`），无需提供完整的量化配置，即可启动摸高调优。

算法内部将生成的完整配置委托给 Standing High 策略执行，因此摸高流程（二分搜索最小回退级别、摸高过程、离群值抑制策略遍历等）与 Standing High 一致，区别仅在于**配置来源**由专家经验自动填充。

## 原理与特点

### 算法原理

Standing High With Experience 的核心思想是**配置简化 + 专家经验填充**，流程分为以下三阶段：

1. **用户侧**：仅配置 `quant_type`（如 `w8a8`、`w4a8`）和 `structure_configs`（如 GQA、FFN 等结构类型及其 `include`/`exclude` 模式）。
2. **策略侧**：根据 `quant_type` 和 `structure_configs` 基于专家经验生成完整摸高配置，并组装为 `StandingHighStrategyConfig`。
3. **执行侧**：将生成的配置交给 `StandingHighStrategy` 执行摸高，后续与 《[Standing High 调优算法](standing_high.md)》 一致，迭代优化直至得到满足精度要求的最优配置。

### 算法特点

1. **开箱即用**：无需手动提供初始量化配置与离群值抑制策略，降低配置门槛，帮助用户快速上手。
2. **专家经验驱动**：量化配置与策略统一由预置的专家经验模板管理，便于维护与复用。
3. **与 Standing High 一致的效果**：执行阶段复用 Standing High 的二分搜索、摸高与策略遍历逻辑，在保证精度的前提下最大化量化层数。
4. **可扩展性好**：可通过修改 [专家经验配置](../../../../../msmodelslim/core/tune_strategy/common/config_builder/expert_experience/expert_experience.yaml) 扩展支持的模型结构类型与量化类型，无需改动策略代码；**仅建议对量化有较深入了解的用户**自行增改，否则易导致调优行为异常或精度不可预期。

## 适用要求

**推理引擎与回退支持**：与 Standing High 相同，需确保推理引擎（如 vLLM-Ascend）支持量化回退；使用混合算子时可能不支持任意回退，需根据实际环境确认。

**模型适配**：

- 须实现 **`ModelSlimPipelineInterfaceV1`**（自动敏感层分析，委托 Standing High 执行）及 **`StandingHighWithExperienceInterface`**（仅 `load_model`，用于过滤当前模型不支持的离群值抑制策略）。
- 常见写法：模型适配器同时继承 **`StandingHighWithExperienceInterface`**（`load_model` 探测）与 **`ModelSlimPipelineInterfaceV1`**（敏感层分析与量化 pipeline）。
- 敏感层分析不会在策略侧预先 `load_model`；详见《[自动调优配置协议说明](../../feature_guide/auto_precision_tuning/configuration_protocols.md#21-strategy---调优策略配置)》与《[LLM 大模型接入指南 — 自动调优与敏感层分析](../../../development_guide/integrating_models.md#自动调优与敏感层分析)》。

## 功能介绍

### 使用说明

在自动调优流程中通过 `msmodelslim tune` 启动；在调优 YAML 的 `strategy` 字段将 `type` 设为 `standing_high_with_experience`，并按下文配置其他字段即可。完整调优配置与命令参数见 《[自动调优使用说明](../../feature_guide/auto_precision_tuning/usage.md)》。

**与 Standing High 的对比**

| 维度           | Standing High | Standing High With Experience |
|----------------|---------------|--------------------------------|
| 配置复杂度     | 需手写初始量化配置、离群值抑制策略等 | 仅需量化类型和模型结构 |
| 摸高执行逻辑   | 一致           | 一致（委托同一套 StandingHighStrategy） |
| 适用场景       | 需要精细控制每项量化与策略时 | 希望开箱即用、按模型结构自动选策略时 |

### YAML 配置示例

对应于自动调优配置文件中的 `strategy` 字段

```yaml
strategy:
  type: standing_high_with_experience
  quant_type: w8a8
  structure_configs:
    - type: "GQA"
      include:
        - "*self_attn*"
    - type: "FFN"
      include:
        - "*mlp*"
```

### YAML 配置字段详解

#### type - 策略类型

**作用**：指定调优算法类型。使用基于专家经验的摸高算法时，应设置为 `standing_high_with_experience`。

**类型**：`string`

**值**：`standing_high_with_experience`

#### quant_type - 量化类型

**作用**：指定目标量化类型，须在专家经验 `expert_experience.yaml` 的 `supported_quant_types` 范围内。当前支持 `w8a8`、`w4a8`。

**类型**：`string`

**示例**：`w8a8`

**说明**：专家经验会据此选择对应的量化配置模板与离群值抑制策略列表。

#### structure_configs - 模型结构配置列表

**作用**：描述当前模型的结构类型及各自参与量化的模块范围（include/exclude），用于从专家经验中查找每种结构对应的量化配置。

**类型**：`list`，每个元素为包含以下字段的对象：

| 字段名   | 作用         | 类型  | 必选/可选 | 说明 |
|----------|--------------|-------|-----------|------|
| type     | 结构类型     | string | 必选 | 支持 `GQA`、`FFN`、`MHA`、`MoE`、`MLA`、`DSA`、`SWA`、`GatedDeltaNet` 等，须在 [专家经验配置](../../../../../msmodelslim/core/tune_strategy/common/config_builder/expert_experience/expert_experience.yaml) 中存在 |
| include  | 包含的模式   | list[string] | 必选 | 模块名匹配模式，不可为空列表且不能包含空字符串，如 `["*self_attn*"]`、`["*mlp*"]` |
| exclude  | 排除的模式   | list[string] | 可选 | 需要排除的模块名模式，如 `["*kv_b_proj*"]` |

**说明**：`include` 与 `exclude` 通过通配符（如 `*`）匹配**模型中的模块名**（即权重某线性层对应的完整名称，例如 `model.layers.0.self_attn.q_proj`）。**include** 表示“哪些模块属于当前结构类型”，只有被 include 匹配到的模块才会套用该类型对应的专家经验；**exclude** 表示“在这些模块里再排除掉哪些”，用于同一结构下对部分层单独处理。配置时可根据模型结构名称，确认各子结构的命名规律后填写，使 include/exclude 与模型实际结构类型保持一致即可。

**注意事项**：各条配置的 include/exclude 范围须**互不重叠**（正交），即每一层仅被一条配置覆盖；否则同一层可能被重复量化，导致结果异常。

**配置示例**：

```yaml
structure_configs:
  - type: "GQA"
    include:
      - "*self_attn*"
    exclude:
      - "*kv_b_proj*"
  - type: "FFN"
    include:
      - "*mlp*"
```
