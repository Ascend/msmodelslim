# compressed-tensors 格式说明

## 简介

compressed-tensors 是面向 HuggingFace 生态的量化权重格式，对齐 [vllm-project/compressed-tensors](https://github.com/vllm-project/compressed-tensors) schema **v0.13.0**。msModelSlim 在导出侧本地 fork 了 schema 模型（无运行时 pip 依赖），推理侧加载时需安装对应版本的 `compressed-tensors` 包。

**适用场景**：vLLM 等支持 HF `config.json` → `quantization_config` 的推理框架。

## YAML 配置

在一键量化 YAML 的 `spec.save` 中配置：

```yaml
spec:
  save:
    - type: "compressed_tensors"
      part_file_size: 4
```

### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | string | `"compressed_tensors"` | 保存器类型标识，固定值 |
| `part_file_size` | int | `4` | 权重分片大小（GB）；`>0` 时分片，`0` 表示不分片 |

## 导出产物结构

```
save_directory/
├── config.json                          # 注入 quantization_config 字段
├── model.safetensors                    # 或 model-00001-of-0000N.safetensors（分片）
├── model.safetensors.index.json         # 分片时生成
└── （从源模型复制的 HF 辅助文件）
    └── *.json / *.py / *.txt / *.jinja
```

**导出流程**：层间写入 safetensors 张量，收尾阶段从量化 QIR 模型反向推导 `quantization_config` 并写入 `config.json`。

## config.json → quantization_config 结构

`quantization_config` 由 `QuantizationConfig.to_quantization_config_dict()` 生成，典型结构如下：

```json
{
  "version": "0.13.0",
  "quant_method": "compressed-tensors",
  "sparsity_config": {},
  "transform_config": {},
  "config_groups": {
    "group_0": {
      "targets": ["Linear"],
      "weights": { "num_bits": 8, "type": "int", "strategy": "channel", "symmetric": true, "dynamic": false },
      "input_activations": { "num_bits": 8, "type": "int", "strategy": "tensor", "symmetric": false, "dynamic": false },
      "format": "int-quantized"
    }
  },
  "format": "int-quantized",
  "quantization_status": "compressed",
  "global_compression_ratio": null,
  "ignore": ["re:...(?![.\\w])"],
  "kv_cache_scheme": null
}
```

### 顶层字段说明

| 字段 | 说明 |
|------|------|
| `version` | Schema 版本，固定 `"0.13.0"` |
| `quant_method` | 固定 `"compressed-tensors"` |
| `config_groups` | 按唯一 scheme 分组，键名为 `group_0`、`group_1` 等 |
| `format` | 根格式，如 `int-quantized`、`mixed-precision` |
| `quantization_status` | 导出时为 `"compressed"` |
| `ignore` | 未量化但同类型层名的 regex 列表 |
| `kv_cache_scheme` | KV Cache 量化方案，**当前不支持**，恒为 `null` |
| `sparsity_config` / `transform_config` | 空对象占位 |

## QuantizationScheme（config_groups 内）

每个 `config_groups` 条目描述一组层的量化方案。**msModelSlim 当前仅支持线性层（`nn.Linear` / QIR FakeQuantLinear）量化**，因此 `targets` 固定为 `["Linear"]`。

| 字段 | 说明 |
|------|------|
| `targets` | 目标模块类型；**当前固定为 `["Linear"]`** |
| `weights` | 权重量化参数（`QuantizationArgs`） |
| `input_activations` | 输入激活量化参数；仅权重量化时为 `null` |
| `output_activations` | 输出激活量化参数 |
| `format` | 层压缩格式，如 `int-quantized` |

## QuantizationArgs 参数说明

| 参数 | 类型/默认 | 说明 |
|------|-----------|------|
| `num_bits` | int, 8 | 量化位宽 |
| `type` | `"int"` / `"float"` | 量化类型 |
| `symmetric` | bool, True | 是否对称量化 |
| `strategy` | enum | `tensor` / `channel` / `group` / `block` / `token` / `tensor_group` / `attn_head` |
| `group_size` | int, optional | group 策略的组大小；`-1` 表示 channel |
| `block_structure` | `[rows, cols]`, optional | block 策略专用 |
| `dynamic` | bool / `"local"` | 动态量化；token 策略必须 `dynamic=True` |
| `actorder` | enum/bool, optional | 激活排序（仅 weight group 策略） |
| `scale_dtype` / `zp_dtype` | torch dtype | scale / zero-point 数据类型 |
| `observer` | str, optional | 校准方法，如 `minmax` |
| `observer_kwargs` | dict | observer 额外参数 |

## 当前支持的量化 Preset

msModelSlim 当前仅对以下 QIR 模块实现了导出 handler：

| Preset | QIR 模块 | weights | input_activations |
|--------|----------|---------|-------------------|
| W8A8 Static | `W8A8StaticFakeQuantLinear` | int8, channel, symmetric, static | int8, tensor, asymmetric, static |
| W8A8 Dynamic | `W8A8DynamicPerChannelFakeQuantLinear` | int8, channel, symmetric, static | int8, token, symmetric, dynamic |

未量化的 `nn.Linear` 层直接写入原始浮点参数，并在 `ignore` 列表中生成对应 regex。

## Safetensors 张量命名

### W8A8 Static

| 张量名 | 数据类型 | 说明 |
|--------|----------|------|
| `{prefix}.weight` | int8 | 量化权重 |
| `{prefix}.weight_scale` | float32 | 权重量化 scale（unsqueeze 为 2D） |
| `{prefix}.input_scale` | float32 | 激活量化 scale |
| `{prefix}.input_zero_point` | - | 仅当 `input_offset != 0` 时写入 |
| `{prefix}.bias` | float32 | 可选 |

### W8A8 Dynamic Per-Channel

| 张量名 | 数据类型 | 说明 |
|--------|----------|------|
| `{prefix}.weight` | int8 | 量化权重 |
| `{prefix}.weight_scale` | float32 | 权重量化 scale（1D 时 unsqueeze 为 2D） |
| `{prefix}.bias` | float32 | 可选 |

> 动态激活的 scale/zero-point **不写入**权重文件，推理时 per-token 动态计算。

## 当前限制

- 不支持分布式导出（`support_distributed() = False`）
- KV Cache 量化暂不支持（`kv_cache_scheme = null`）
- 仅 W8A8 Static / W8A8 Dynamic 两种 QIR 有 handler
- 更多 preset（如 W4A8）在 schema 中预留常量，尚未实现导出 handler

## 相关文档

- 《[格式支持矩阵](README.md)》
- 《[量化格式接入指南](../developer_guide/iformat_integration_guide.md)》
- 《[AscendV1 格式说明](ascendv1.md)》
