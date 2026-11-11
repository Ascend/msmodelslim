---
title: FAQ - 精度与调优
description: 精度与调优 FAQ；覆盖量化精度异常定位、自动调优、敏感层分析等问题。
keywords: [精度, 调优, 敏感层, 自动调优, 回退, badcase, 精度异常, VLM, 多模态, lm_head, visual, resampler]
---

# 精度与调优 FAQ

> 本节用于沉淀量化精度异常定位、自动调优、敏感层分析等主题的常见问题。

量化精度不达标、badcase 增多、无法对齐浮点输出等问题，请参见：

- 《[精度调优方法](../user_guide/process_quantization_precision_tuning.md)》
- 《[量化推理精度异常定位](../user_guide/process_quantization_accuracy_anomaly_locating.md)》

新增条目请按 `1.`、`2.` 顺序在文末追加，并保持“问题现象 / 问题原因 / 解决方法 / 关联文档”的四段结构（段内小节编号依次为 `x.1`~`x.4`）。

## 1. 为什么 VLM 量化后精度明显下降，如何排查敏感层？

### 1.1 问题现象

多模态模型（VLM）量化后输出乱码、重复或评测指标明显下降，怀疑视觉塔、resampler 或 lm_head 被误量化。

### 1.2 问题原因

VLM 的视觉编码器、resampler、lm_head 等模块对量化更敏感；是否量化取决于量化配置中的 process 匹配范围，复用纯文本模型配方时容易把这些模块一起量化。

### 1.3 解决方法

1. 查看量化 YAML 的 process，确认 `lm_head`、visual/vision tower、resampler 是否在量化匹配范围内。
2. 同口径对比浮点与量化输出，先排除 tokenizer、processor、部署格式等“假精度问题”。
3. 对敏感层设置排除或单独保护后重新量化；具体字段以《[量化配置 API](../api_reference/config/quant/modelslim_v1.md)》和模型 example 为准。
4. 每次只改一个变量并记录基线；仍不达标时转自动调优或敏感层分析。

### 1.4 关联文档

- 精度调优路径请参见《[精度调优方法](../user_guide/process_quantization_precision_tuning.md)》。
- 异常定位流程请参见《[量化推理精度异常定位](../user_guide/process_quantization_accuracy_anomaly_locating.md)》。
- 具体模型的官方配方与基线请参见对应模型 `example/<family>/README.md`。
