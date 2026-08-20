<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.processor.memory.load.LoadProcessorConfig -->
# load 配置说明

## 1. 配置概述

模块加载/卸载处理器配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `LoadProcessorConfig` |
| 源码 | [load.py](../../../../../msmodelslim/processor/memory/load.py) |

## 2. 参数列表

<h3 id="2-1-load">2.1 LoadProcessorConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `load` | `load` | 处理器类型，固定为 `load`。 | 无 |
| `device` | `string` | 可选 | `cpu` | — | 目标设备，如 `cpu`、`npu:0` | 无 |
| `non_blocking` | `bool` | 可选 | `false` | — | 是否非阻塞加载 | 无 |
| `mode` | `string` | 可选 | `load` | `load`、`offload` | 加载模式：`load` 加载到目标设备，`offload` 卸载到 CPU | 无 |
| `cleanup` | `bool` | 可选 | `false` | — | 是否清理缓存 | 无 |
| `post_offload` | `bool` | 可选 | `false` | — | 卸载后是否 offload 激活值 | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
apiversion: modelslim_v1
spec:
  process:
  - type: load
    device: cpu
    non_blocking: false
    mode: load
    cleanup: false
    post_offload: false
```
