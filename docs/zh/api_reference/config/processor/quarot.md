<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.quarot.offline_quarot.quarot.QuaRotProcessorConfig -->
# quarot 配置说明

## 1. 配置概述

QuaRot（离线旋转）处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `QuaRotProcessorConfig` |
| 源码 | [quarot.py](../../../../../msmodelslim/processor/quarot/offline_quarot/quarot.py) |

## 2. 参数列表

<h3 id="2-1-quarot">2.1 QuaRotProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `quarot` | `quarot` | 处理器类型，固定为 `quarot`。 | 无 |
| `online` | `bool` | 可选 | `false` | — | 是否在线旋转（默认离线）。 | 无 |
| `block_size` | `int` | 可选 | `-1` | — | 旋转块大小，-1 表示按 hidden_dim 整块旋转。 | 无 |
| `down_proj_online_layers` | `list[int]` | 可选 | `[]` | — | 需要在线旋转的 down_proj 层索引列表。 | 无 |
| `max_tp_size` | `int` | 可选 | `4` | — | 最大 TP（Tensor Parallelism，张量并行）并行度，必须为2的幂。 | 无 |
| `export_extra_info` | `bool` | 可选 | `true` | — | 是否导出 `optional.quarot.global_rotation` 旋转信息，用于下游部署。 | 无 |

**配置约束**

- 校验 max_tp_size：必须大于等于1且为2的幂
- 校验 block_size：取值范围为-1或2的非负整数次幂

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: quarot
    online: false
    block_size: -1
    down_proj_online_layers: []
    max_tp_size: 4
    export_extra_info: true
```
