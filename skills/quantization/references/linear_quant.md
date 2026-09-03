# linear_quant 参数参考

`linear_quant` 处理器的**完整字段 schema、类型与取值以 docs 为准**：《[linear_quant 处理器配置](../../../docs/zh/api_reference/config/processor/linear_quant.md)》、《[modelslim_v1 任务规格](../../../docs/zh/api_reference/config/task/modelslim_v1.md)》。本文只保留 Practice YAML 编写时的常用建议。

## 常用取值速查（非权威，取值集合以 docs 为准）

| 配置块 | 字段 | 常用值 |
|--------|------|--------|
| 处理器 | `type` | `"linear_quant"` |
| 处理器 | `include` | `["*"]`, `["*self_attn*"]` |
| 处理器 | `exclude` | `["*down_proj*"]`（优先级高于 `include`） |
| `qconfig.act` | `scope` | `"per_tensor"`（默认）/ `"per_token"` / `"pd_mix"` |
| `qconfig.act` | `dtype` | `"int8"`（默认） |
| `qconfig.act` | `symmetric` | `false`（默认） |
| `qconfig.act` | `method` | `"minmax"`（默认）/ `"histogram"` |
| `qconfig.weight` | `scope` | `"per_channel"`（默认）/ `"per_tensor"` / `"per_group"` |
| `qconfig.weight` | `dtype` | `"int8"`（默认）/ `"int4"` |
| `qconfig.weight` | `symmetric` | `true`（默认） |
| `qconfig.weight` | `method` | `"minmax"`（默认）/ `"ssz"` / `"gptq"` |

## 最小可用建议

- 优先从 `act.scope: "per_tensor"` + `weight.scope: "per_channel"` 开始。
- 优先使用 `method: "minmax"` 作为基础配置。
- 先用 `include: ["*"]` 验证流程，再按需增加 `exclude`。
- 对 MoE 模型，路由器 `gate` 模块一般不量化，建议 `exclude: ["*.gate"]`。
