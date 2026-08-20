---
title: FAQ - 告警与提示
description: 告警与提示类问题的处理说明；不覆盖安装、运行、硬件、精度、部署问题。
keywords: [告警, 提示, DeprecationWarning, swigvarlink, SWIG, sentencepiece]
---

# 告警与提示 FAQ

> 本节覆盖 msModelSlim 运行过程中告警与提示类常见问题。新增条目按 1.x 顺序追加，并保持“问题现象 / 问题原因 / 解决方法”的结构。

## 1.1 量化结束告警“sys:1: DeprecationWarning: builtin type swigvarlink has no module attribute”？

### 问题现象

量化结束时出现告警：

```text
sys:1: DeprecationWarning: builtin type swigvarlink has no module attribute
```

### 问题原因

旧版SWIG（Simplified Wrapper and Interface Generator，简化包装器与接口生成器）生成的第三方库与Python 3.10+不兼容。

### 解决方法

此告警不影响量化结果，升级触发该告警的第三方库即可消除（如sentencepiece）。
