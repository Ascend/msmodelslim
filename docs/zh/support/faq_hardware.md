---
title: FAQ - 硬件兼容
description: 硬件兼容类问题的原因与解决；不覆盖安装、运行、精度、部署问题。
keywords: [硬件, 兼容, Ascend, 300I, 300T, DT_BFLOAT16, float16, InplaceIndexAdd, JIT]
---

# 硬件兼容 FAQ

> 本节覆盖 msModelSlim 与 Ascend 硬件兼容性相关的常见问题。新增条目按 1.x 顺序追加，并保持“问题现象 / 问题原因 / 解决方法 / 关联文档”的结构。

## 1.1 为什么量化权重时出现报错“PTA call acl api failed... The param dtype not implemented for DT_BFLOAT16”？

### 问题现象

量化权重时报错：

```text
PTA call acl api failed. *** The param dtype not implemented for DT_BFLOAT16, should be in dtype support list [***]
```

### 问题原因

部分Ascend硬件（例如Atlas 300I/300T系列）只支持float16精度推理，如果模型权重采用bfloat16（DT_BFLOAT16）精度量化，可能导致量化失败。

### 解决方法

修改模型权重路径下`config.json`中的`torch_dtype`为`float16`后进行量化。

### 关联文档

模型部署与量化产物使用请参见《[主流模型部署流程](../user_guide/process_mainstream_model_deployment.md)》。

## 1.2 为什么在300I/300T系列硬件上量化权重时会报错“RuntimeError: The Inner error is reported as above... InplaceIndexAdd”？

### 问题现象

在300I/300T系列硬件上量化权重时报错：

```text
RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is InplaceIndexAdd.
```

### 问题原因

在300I/300T系列硬件上进行传统量化（V0）时，JIT（Just-In-Time，即时编译）编译模式与该系列硬件存在兼容性问题，导致`InplaceIndexAdd`算子编译失败，从而引发运行时错误。

### 解决方法

在传统量化（V0）模型量化脚本（`msmodelslim/example`路径下）中添加`torch_npu.npu.set_compile_mode(jit_compile=False)`来禁用JIT编译模式。

**示例代码：**

```python
import torch_npu

# 在量化脚本开头添加以下代码
torch_npu.npu.set_compile_mode(jit_compile=False)

# 然后执行量化操作
# ... 后续的量化代码
```

### 关联文档

传统量化（V0）的使用说明请参见《[传统量化 V0 使用指南](../user_guide/traditional_quantization_v0/README.md)》。
