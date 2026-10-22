# 量化数据格式 ops 使用流程指南

## 1. 适用范围

本指南面向需要直接调用 `msmodelslim.ir.qal.dtype_ops` 的开发者。该包按数据格式提供张量级量化、反量化与参数计算函数，不绑定线性层、KVCache 或 FA 结构。

适用场景：

- 离线把权重或校准激活量化到 [INT8](term_int8.md)、[INT4](term_int4.md)、[FP8](term_fp8.md)、[MXFP8](term_mxfp8.md) 或 [MXFP4](term_mxfp4.md)；
- 推理时按已有参数反量化，或在线动态计算参数；
- 训练伪量化：输出 float32 网格并反传。

不适用于：

- 选择整层量化模式或编写 YAML 配方，请改走《[一键量化使用指南](../../user_guide/usage_one_click_quantization.md)》与对应模式词条；
- 存储打包（两个 4bit 打进一个字节），本层不做 pack / unpack。

格式定义见[量化数据格式](README.md)下各词条；公式见[量化与反量化](term_quantization.md)。

## 2. 流程关系与前置条件

**上级流程**：在量化算法或模式实现中，需要对单个张量做格式级量化时进入本流程。

**前置条件**：

- 已安装与当前仓库匹配的 msModelSlim，并能 `import msmodelslim.ir.qal.dtype_ops`；
- 已确定目标格式、粒度、对称性，以及张量来自权重、激活还是 KV；
- 输入张量可转为 float32 参与计算。

**后续操作**：把返回的 `q` 与 scale / offset / shared_exp 交给上层模式或落盘逻辑；精度不达标则进入《[量化精度调优指南](../../user_guide/process_quantization_precision_tuning.md)》。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 待量化浮点张量 | 权重、激活或 KV 缓存 | 可在 float32 下统计与计算 | 形状已知，能沿选定轴归约 |
| 输入 | 格式与粒度选择 | 调用方根据部署目标确定 | INT8/INT4 可选 per_tensor、per_channel、per_group；FP8 仅 per_tensor/per_channel；MX 为块长 32 | 与目标模式一致，见步骤 1 |
| 交付件 | 量化网格与量化参数 | 函数返回值 | `q` 默认 float32 同形状网格；INT 对称返回 scale，非对称再加 offset；MX 返回 uint8 E8M0 | 能被对应 `dequantize` 还原，参数可广播 |
| 交付件 | 反量化浮点张量 | 调用 `dequantize` 的输出 | float32，形状与输入一致 | 与原张量同形状，可用于后续计算或损失 |

## 4. 流程总览

```mermaid
flowchart LR
  A[选择格式与粒度] --> B[计算量化参数]
  B --> C[量化]
  C --> D[反量化]
```

- **选择格式与粒度**：先定编码家族，再定函数名里的粒度与 `sym` / `asym`。
- **计算量化参数**：静态路径先 `calc_qparam`；动态路径由 `quantize_dynamic` 内部完成。
- **量化**：得到 `q` 与原样返回的参数；需要低精度存储时再传 `dtype`。
- **反量化**：用同一组粒度参数把 `q` 还原为 float32。

## 5. 操作步骤

### 步骤 1：选择数据格式、粒度与对称性

**目标**：确定要调用的函数名，避免格式、粒度、对称性与张量轴不一致。

**操作**：从 `msmodelslim.ir.qal.dtype_ops` 导入模块级函数，命名为 `{格式}_{操作}_{粒度}_{sym|asym}`。操作是 `calc_qparam`、`quantize`、`dequantize`、`quantize_dynamic`。类名不对外导出。

| 粒度 | 函数名后缀 | 额外入参 | 适用格式 |
| --- | --- | --- | --- |
| per-tensor | `_per_tensor_sym` / `_per_tensor_asym` | 无 `axis` | INT8、INT4、FP8；FP8 无 `_asym` |
| per-channel / per-token | `_per_channel_sym` / `_per_channel_asym` | `axis` | INT8、INT4、FP8；FP8 无 `_asym` |
| per-group | `_per_group_sym` / `_per_group_asym` | `axis`，`group_size` | INT8、INT4；该维长度须整除 `group_size`，否则报错 |
| MX 块 | `_sym`（不带 `per_block`） | `axis=-1`，或 `0`，或 `[0, 1]` | MXFP8、MXFP4；不是 32 的倍数时内部 pad，还原时裁回 |

同一条静态路径上，`calc_qparam`、`quantize`、`dequantize` 必须用同一粒度，`axis` / `group_size` 保持一致。

**输出**：一组确定的函数名与粒度参数。

### 步骤 2：计算量化参数

**目标**：得到可广播、可落盘的 scale / offset / shared_exp。

**操作**：

- INT 对称：`int8_calc_qparam_per_channel_sym` 等，返回 float32 scale。区间强制包含 0，`scale = amax / 127`（INT4 为 `/7`）。
- INT 非对称：返回 `(scale, offset)`，`offset` 为整数 zero-point。
- FP8：`fp8_calc_qparam_per_channel_sym(x, axis, fmt)`，`scale = amax / max_norm`。
- MX：`mxfp8_calc_qparam_sym` / `mxfp4_calc_qparam_sym` 直接返回可落盘的 `uint8` E8M0。

`fmt` 只出现在 FP8 / MXFP8，用来选元素编码，不是存储类型。默认 `"e4m3"`（上界 448），可选 `"e5m2"`（上界 57344）。其它字符串会报 `ValueError`。INT8、INT4、MXFP4 不传 `fmt`。`calc_qparam` 与后续 `quantize` 必须传同一个 `fmt`。

**输出**：float32 scale（及可选 offset），或 `uint8` E8M0。

### 步骤 3：执行量化与反量化

**目标**：得到与输入同形状的量化网格，并能还原为 float32。

**操作**：默认不传 `dtype` 时，`q` 是 float32 网格。需要低精度存储时再传 `dtype`：INT8 / INT4 / MXFP4 常用 `torch.int8`；FP8 / MXFP8 用对应 `float8`。都是逐元素转换，不打包。非 float32 的 `q` 不能反传。

```python
import torch
from msmodelslim.ir.qal.dtype_ops import (
    int8_calc_qparam_per_channel_sym, int8_quantize_per_channel_sym, int8_dequantize_per_channel_sym,
    int4_calc_qparam_per_group_sym, int4_quantize_per_group_sym, int4_dequantize_per_group_sym,
    fp8_quantize_dynamic_per_channel_sym, fp8_dequantize_per_channel_sym,
    mxfp8_quantize_dynamic_sym, mxfp8_dequantize_sym,
    mxfp4_calc_qparam_sym, mxfp4_quantize_sym, mxfp4_dequantize_sym,
)

scale = int8_calc_qparam_per_channel_sym(w, axis=0)
q, scale = int8_quantize_per_channel_sym(w, scale, axis=0, dtype=torch.int8)
w_hat = int8_dequantize_per_channel_sym(q, scale, axis=0)

scale4 = int4_calc_qparam_per_group_sym(w, axis=-1, group_size=128)
q4, scale4 = int4_quantize_per_group_sym(w, scale4, axis=-1, group_size=128, dtype=torch.int8)
w_hat4 = int4_dequantize_per_group_sym(q4, scale4, axis=-1, group_size=128)

q_fp8, scale_fp8 = fp8_quantize_dynamic_per_channel_sym(
    x, axis=-1, fmt="e4m3", dtype=torch.float8_e4m3fn
)
x_fp8 = fp8_dequantize_per_channel_sym(q_fp8, scale_fp8, axis=-1)

q_mx, se_mx = mxfp8_quantize_dynamic_sym(x, axis=-1, fmt="e4m3")
x_mx = mxfp8_dequantize_sym(q_mx, se_mx, axis=-1)

se = mxfp4_calc_qparam_sym(w, axis=-1)
q_mx4, se = mxfp4_quantize_sym(w, se, axis=-1)
w_mx = mxfp4_dequantize_sym(q_mx4, se, axis=-1)
```

**输出**：`(q, scale)` 或 `(q, scale, offset)` 或 `(q, shared_exp)`，以及 float32 还原张量。

### 步骤 4：按场景选择静态或动态

**目标**：静态参数由调用方准备，动态参数在函数内计算并脱离计算图。

**操作**：

| 场景 | 调用 | 参数 |
| --- | --- | --- |
| 离线权重量化 | 先 `calc_qparam` 再 `quantize` | 传入的 scale / offset / shared_exp 原样返回 |
| 推理反量化 | `dequantize` | 使用量化时返回的参数；`q` 可以是 float32 或已转换的低精度 |
| 在线 / 动态量化 | `quantize_dynamic` | 当场计算参数，`detach` 后和 `q` 一起返回 |

静态训练若要把参数当常数，在调用 `quantize` 前对参数 `detach`。

**输出**：与步骤 3 相同的返回值；动态路径的参数已 `detach`。

### 步骤 5：选择输出精度与反传

**目标**：分清 float32 网格（能反传）和低精度存储（不能反传）。

**操作**：`dtype` 决定 `q` 的类型。`ste_clamp` 只出现在 `quantize` / `quantize_dynamic`，默认 `True`：前向饱和、反向恒等，饱和位置梯度仍为 1；`False` 时饱和位置梯度为 0。`round_ste` 始终开启。

```python
import torch
from msmodelslim.ir.qal.dtype_ops import (
    int8_calc_qparam_per_channel_sym, int8_quantize_per_channel_sym,
    int8_dequantize_per_channel_sym, int8_quantize_dynamic_per_channel_sym,
)

scale = int8_calc_qparam_per_channel_sym(x.detach(), axis=-1).detach()
q, scale = int8_quantize_per_channel_sym(x, scale, axis=-1, dtype=torch.float32)
y = int8_dequantize_per_channel_sym(q, scale, axis=-1)

q, scale = int8_quantize_per_channel_sym(
    x, scale, axis=-1, dtype=torch.float32, ste_clamp=False
)

q, scale = int8_quantize_dynamic_per_channel_sym(x, axis=-1, dtype=torch.float32)
y = int8_dequantize_per_channel_sym(q, scale, axis=-1)
```

**输出**：float32 网格可对损失 `backward`；低精度 `q` 只适合保存或推理。

## 6. 验收条件

- 同一条路径上的 `calc_qparam`、`quantize`、`dequantize` 使用同一粒度与同一 `fmt`。
- 默认路径返回的 `q` 与输入同形状；传入低精度 `dtype` 后形状仍不变。
- `dequantize` 输出为 float32，形状与输入一致。
- `per_group` 在不能整除时失败；MX 在不能整除时由内部 pad 处理。

## 7. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 量化与反量化 | scale、zero_point、静态动态与粒度 | [量化与反量化](term_quantization.md) |
| INT8 | 8位有符号整数仿射 | [INT8](term_int8.md) |
| INT4 | 4位有符号整数仿射 | [INT4](term_int4.md) |
| FP8 | OCP OFP8 逐元素浮点 | [FP8](term_fp8.md) |
| MXFP8 | OCP MX 8bit 块级格式 | [MXFP8](term_mxfp8.md) |
| MXFP4 | OCP MX 4bit 块级格式 | [MXFP4](term_mxfp4.md) |
| 量化模式 | 按对象选用上述格式 | [量化模式](../quantization_mode/README.md) |
