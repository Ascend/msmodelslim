<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.quarot.online_quarot.online_quarot.OnlineQuaRotProcessorConfig -->
# online_quarot 配置说明

## 1. 配置概述

在线 QuaRot 旋转处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `OnlineQuaRotProcessorConfig` |
| 源码 | [online_quarot.py](../../../../../msmodelslim/processor/quarot/online_quarot/online_quarot.py) |

## 2. 参数列表

<h3 id="2-1-online-quarot">2.1 OnlineQuaRotProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `online_quarot` | `online_quarot` | 处理器类型，固定为 `online_quarot`。 | 无 |
| `include` | `list[string] / null` | 可选 | `null` | — | 包含的模块名称模式；不设置表示全部匹配。 | 无 |
| `exclude` | `list[string] / null` | 可选 | `null` | — | 排除的模块名称模式，优先级高于 `include`。 | 无 |
| `block_size` | `int` | 可选 | `-1` | — | 旋转块大小，-1 表示按 hidden_dim 整块旋转；可被 RotationConfig 覆盖。 | 无 |

**配置约束**

- 校验 block_size：取值范围为-1或2的非负整数次幂

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: online_quarot
    block_size: -1
```
