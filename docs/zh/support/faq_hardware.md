---
title: FAQ - 硬件兼容
description: 硬件兼容 FAQ；昇腾硬件选型、驱动固件与算子编译类问题请查阅下方指引，并覆盖 A3 SoC、310P/300I 算子与 BF16 限制、aclnn/kernel 自检。
keywords: [硬件, 兼容, Ascend, 昇腾, 300I, 300T, A2, A3, 310P, BF16, 算子编译, aclnn, kernel, 950]
---

# 硬件兼容 FAQ

> 本节用于沉淀昇腾硬件选型、驱动与固件配套、算子编译与硬件算子不支持等主题的常见问题。
>
> 950PR 等新平台的适配状态以《[msModelSlim 安装指南](../install_guide/install_guide.md)》与模型支持矩阵为准。

硬件不在支持范围内、算子编译失败或硬件算子不支持等问题，请参见：

- 《[传统量化 V0 使用指南](../user_guide/traditional_quantization_v0/README.md)》——旧代硬件上的量化链路与算子适配说明。

模型部署与量化产物使用请参见《[主流模型量化部署](../user_guide/process_mainstream_model_deployment.md)》。

新增条目请按 `1.`、`2.` 顺序在文末追加，并保持“问题现象 / 问题原因 / 解决方法 / 关联文档”的四段结构（段内小节编号依次为 `x.1`~`x.4`）。

## 1. 为什么 A3 上报 Unsupported soc version: Ascend910_9382？

### 1.1 问题现象

加载模型或执行量化时出现类似以下报错：

```text
Unsupported soc version: Ascend910_9382
```

### 1.2 问题原因

当前 CANN 或 torch_npu 版本过旧，不支持 A3（Ascend910_9382）SoC；常见于沿用旧镜像或在宿主环境中混装依赖。

### 1.3 解决方法

1. 按《安装指南》的版本配套升级 CANN 与 torch_npu。
2. 优先使用与 A3 配套的官方镜像，避免宿主环境混装。
3. 升级后执行 `source set_env` 并重启进程。

### 1.4 关联文档

硬件与软件版本配套要求请参见《[msModelSlim 安装指南](../install_guide/install_guide.md)》。

## 2. 为什么 310P/300I 量化时报 DT_BFLOAT16 not implemented 或算子不支持？

### 2.1 问题现象

在 310P/300I 上执行量化时出现类似以下报错：

```text
aclnnAmin ... DT_BFLOAT16 not implemented
InplaceIndexAdd ... not supported
```

### 2.2 问题原因

310P/300I DUO 不支持 BF16 或部分算子；msModelSlim 量化主流程面向 910 系列，310 系列只保证有限的推理能力，量化精度不保证。

### 2.3 解决方法

1. 在 A2/A3 上完成量化，310 侧只做推理。
2. 必须在 310P 上运行时，改用硬件支持的精度（如 fp16），并重新评测精度。
3. 确认镜像与硬件型号匹配，避免使用面向 910 的算子包。

### 2.4 关联文档

- 硬件与版本要求请参见《[msModelSlim 安装指南](../install_guide/install_guide.md)》。
- 精度评估与调优请参见《[精度调优方法](../user_guide/process_quantization_precision_tuning.md)》。

## 3. 为什么出现 aclnnCast failed、Parse dynamic kernel config fail 或 aicore timeout？

### 3.1 问题现象

执行量化或推理时出现类似以下报错：

```text
aclnnCast failed
Parse dynamic kernel config fail
aicore timeout, retCode=0x25
```

### 3.2 问题原因

CANN kernel 包、镜像与硬件版本不匹配，或驱动/算子执行超时；也可能是卡被占用、batch/seq 过大导致。

### 3.3 解决方法

1. 使用官方发布镜像，避免宿主环境混装。
2. 执行 `source set_env`，确认 CANN 与 torch_npu 版本配套。
3. 检查卡状态与占用，减小 batch/seq 后重试。
4. 仍失败时收集完整 traceback、`npu-smi info` 与 CANN 版本，按《[FAQ - 运行与资源](./faq_runtime.md)》继续排查。

### 3.4 关联文档

- 环境安装与版本配套请参见《[msModelSlim 安装指南](../install_guide/install_guide.md)》。
- 运行与资源问题请参见《[FAQ - 运行与资源](./faq_runtime.md)》。
