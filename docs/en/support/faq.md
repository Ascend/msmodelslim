---
title: FAQ Overview (Index)
description: A unified index of msModelSlim frequently asked questions (FAQ), navigating to the corresponding FAQ sub-document by six categories (installation, runtime, hardware, accuracy, deployment, and warnings); this page contains no specific questions or answers.
keywords: [FAQ, frequently asked questions, troubleshooting, index, Killed, pydantic, DeprecationWarning]
---

# FAQ

> This document is the unified index of msModelSlim frequently asked questions (FAQ). The FAQ is split into multiple sub-documents by topic to facilitate AI retrieval and quick access; see the corresponding category document for specific questions and answers.

## 1. Category Index

| Category | Covered Problems | Document |
|------|---------|------|
| 1. Installation and Dependencies | pydantic version conflict, accelerate dependency installation failure | [faq_installation.md](./faq_installation.md) |
| 2. Runtime and Resources | Process Killed, insufficient NPU memory/system memory (OOM) | [faq_runtime.md](./faq_runtime.md) |
| 3. Hardware Compatibility | Reserved category (guidance on the hardware environment and operator compatibility) | [faq_hardware.md](./faq_hardware.md) |
| 4. Accuracy and Tuning | Reserved category (guidance on accuracy tuning and anomaly locating) | [faq_precision.md](./faq_precision.md) |
| 5. Deployment and Inference | Reserved category (guidance on MindIE / vLLM-Ascend deployment) | [faq_deployment.md](./faq_deployment.md) |
| 6. Warnings and Notices | Handling of warnings such as DeprecationWarning | [faq_warning.md](./faq_warning.md) |

## 2. Search Guide

- **Locate by topic**: open the corresponding sub-document from the category index above.
- **Search by keyword**: use keywords directly (such as `Killed`, `pydantic`, and `DT_BFLOAT16`) to locate the content in each sub-document or in a full-text search.
- **AI retrieval**: the header of each sub-document contains AI-readable metadata (`title` / `description` / `keywords`), allowing AI to quickly assess document relevance from the metadata and reduce the token overhead of reading the full text.

## 3. Revision History

| Date | Version | Change Description |
| --- | --- | --- |
| 2026-08-17 | v1.0 | First release: the FAQ system was restructured in accordance with the *msModelSlim Documentation Standards*, split into six categories of sub-documents (installation, runtime, hardware, accuracy, deployment, and warnings), and this document was established as the unified index; entries were standardized into three sections, with the full forms of abbreviations and links to related documents added; the header of each sub-document uses the three readable metadata fields `title`/`description`/`keywords`. |
