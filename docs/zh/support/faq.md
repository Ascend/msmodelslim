---
title: FAQ 总览（索引）
description: msModelSlim 常见问题统一索引，按安装/运行/硬件/精度/部署/告警六类导航至对应 FAQ 子文档；本页不含具体问答。
keywords: [FAQ, 常见问题, 故障排查, 索引, Killed, pydantic, DT_BFLOAT16]
---

# FAQ

> 本文档为 msModelSlim 常见问题（FAQ）的统一索引。FAQ 按主题拆分为多篇子文档，便于 AI 检索与快速定位；具体问答内容请在对应分类文档中查阅。

## 分类索引

| 分类 | 覆盖问题 | 文档 |
|------|---------|------|
| 1. 安装与依赖 | pydantic 版本冲突、accelerate 依赖安装失败 | [faq_installation.md](./faq_installation.md) |
| 2. 运行与资源 | 进程 Killed、显存/内存不足（OOM） | [faq_runtime.md](./faq_runtime.md) |
| 3. 硬件兼容 | DT_BFLOAT16 不支持、InplaceIndexAdd 编译失败 | [faq_hardware.md](./faq_hardware.md) |
| 4. 精度与调优 | 预留分类（精度调优、异常定位指引） | [faq_precision.md](./faq_precision.md) |
| 5. 部署与推理 | 预留分类（MindIE / vLLM-Ascend 部署指引） | [faq_deployment.md](./faq_deployment.md) |
| 6. 告警与提示 | DeprecationWarning 等告警处理 | [faq_warning.md](./faq_warning.md) |

## 检索指引

- **按主题定位**：从上方分类索引进入对应子文档。
- **按关键词搜索**：直接使用关键词（如 `Killed`、`pydantic`、`DT_BFLOAT16`）在各子文档或全文检索中定位。
- **AI 检索**：各子文档头部均含 AI 可读元数据（`title` / `description` / `keywords`），AI 可通过元数据快速判断文档相关性，减少全文读取带来的 token 开销。

## 修订记录

| 日期 | 版本 | 变更说明 |
| --- | --- | --- |
| 2026-08-17 | v2.1 | 精简各文档头部元数据：由 type/category/scope/related 收拢为 title/description/keywords 三字段，scope（排除域）并入 description，降低文档头冗余，保持 AI 检索与 token 优化能力。 |
| 2026-08-17 | v2.0 | 重构：FAQ 拆分为多篇分类子文档（安装/运行/硬件/精度/部署/告警），本文档改为统一索引；各子文档新增 AI 可读元数据头。 |
| 2026-08-17 | v1.0 | 按《msModelSlim资料规范》重构：新增分类编号结构、统一条目三段式、补齐缩写全称与关联文档链接。 |
