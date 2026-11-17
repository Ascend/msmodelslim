# 量化数据类型 ops Python API

## 功能说明

`msmodelslim.core.dtype_ops` 提供张量级量化、反量化与参数计算接口，覆盖 INT8、INT4、FP8、MXFP8、MXFP4。公开入口是模块级函数，不导出类。命名、调用步骤见《[量化数据类型 ops 使用流程指南](../../knowledge_base/dtype_ops/usage_dtype_ops.md)》；概念词条见[量化数据类型](../../knowledge_base/dtype_ops/README.md)。函数名为 `{类型}_{操作}_{粒度}_{对称性}`。操作为 `calc_qparam`、`quantize`、`dequantize`、`quantize_dynamic`。MX 不加粒度后缀，只接 `_sym`。`per_channel` 同时用于权重和激活：激活侧习称 per-token，仍调用 `_per_channel_*`。入参和返回值相近的接口写在同一篇文档中，各接口的参数与取值见下方文档。

## 接口文档

### INT8

- [int8_calc_qparam](dtype_ops/int8_calc_qparam.md)
- [int8_quantize](dtype_ops/int8_quantize.md)
- [int8_dequantize](dtype_ops/int8_dequantize.md)
- [int8_quantize_dynamic](dtype_ops/int8_quantize_dynamic.md)

### INT4

- [int4_calc_qparam](dtype_ops/int4_calc_qparam.md)
- [int4_quantize](dtype_ops/int4_quantize.md)
- [int4_dequantize](dtype_ops/int4_dequantize.md)
- [int4_quantize_dynamic](dtype_ops/int4_quantize_dynamic.md)

### FP8

- [fp8_calc_qparam](dtype_ops/fp8_calc_qparam.md)
- [fp8_quantize](dtype_ops/fp8_quantize.md)
- [fp8_dequantize](dtype_ops/fp8_dequantize.md)
- [fp8_quantize_dynamic](dtype_ops/fp8_quantize_dynamic.md)

### MXFP8

- [mxfp8](dtype_ops/mxfp8.md)

### MXFP4

- [mxfp4](dtype_ops/mxfp4.md)
