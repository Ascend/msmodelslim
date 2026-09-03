---
name: installation
description: |
  msModelSlim 安装与环境校验。当用户询问如何安装 msmodelslim、验证 import/CLI、或端到端调优进入环境准备阶段需要安装检查时使用。
  安装步骤以仓库安装指南为准，本 skill 只提供检查清单与链接，不复写安装正文。
metadata:
  version: 0.1.0
  domain: quant
  framework: msmodelslim
  skill_class: workflow
---

# 安装（installation）

## 概述

确认 msModelSlim 可在当前环境安装并被调用。详细安装步骤见《[msModelSlim 安装指南](../../docs/zh/install_guide/install_guide.md)》，**不要在本 skill 中复写安装章节**。

## 适用

- 用户要安装 / 重装 / 源码安装 msmodelslim
- 调优编排（`tuning`）进入环境准备，需要核对安装与 import

## 不做

- 不写适配器、不跑量化/评测/调优策略
- 不复制安装指南长文

## 检查清单

1. 阅读并按《[安装指南](../../docs/zh/install_guide/install_guide.md)》完成在线 / 离线 / 源码安装之一。
2. NPU 场景确认 Ascend/CANN 与 TorchNPU 按指南就绪。
3. 验证：

```bash
python -c "import msmodelslim; print('ok')"
msmodelslim quant --help
```

> **源码目录内安装的注意（实测 F0-2）**：若在仓库根目录内执行过 `bash install.sh` 源码安装，验证 CLI **请到仓库根目录之外**的目录执行——源码树内 `import` 会命中 `msmodelslim/msmodelslim/`（无 `config` 子目录），导致 `SecurityError: msmodelslim/config doesn't exist` 假失败；site-packages 安装版才带 config。同时 CLI 首启较慢（约 20~60s，import torch_npu 等重依赖），勿误判卡死。

1. 若刚用 `bash install.sh` 注册过新 adapter，在同一 Python 环境确认 `import` 无报错即可（无需额外重启工具服务）。

向用户回显：安装方式、Python 环境、`import`/`CLI` 是否通过。获得确认后再进入后续适配或调优。

## 使用方式

- **直接使用**：用户询问「如何安装 / 环境是否就绪」时，按本文检查清单独立完成安装与校验。
- **被 `tuning` 编排**：端到端调优在「环境准备」阶段引用本 Skill（见 `tuning/references/prepare_environment.md`）；同时作为 `quantization` / `evaluation/evaluate` 的绑定依赖（frontmatter `bind`），为其提供环境校验能力。

## 经验条目（Experiences，追加制）

> 追加规范见 `skills/README.md`「经验条目」。连续编号 `[E-序号]`；正文保留权威展开，本表只做索引 + 元数据登记；来源：实测编号（F0-x / Dx / Tx）| 用户反馈 | 代码实证；验证状态：已回归 | 待验证 | 已上流 docs。

| 条目 | 主题 | 适用条件 / 触发信号 | 结论要点（一句话） | 正文位置 | 来源 | 验证状态 |
|------|------|------|------|------|------|------|
| E-001 | 仓库根内验证会假失败 | 源码安装后验证 CLI | 在仓库根目录内 `import msmodelslim` 命中源码树 `msmodelslim/msmodelslim/`（无 `config` 子目录）报 `SecurityError: msmodelslim/config doesn't exist`；验证请到**仓库根目录外**执行（命中 site-packages 完整安装） | 检查清单 3 | 实测 F0-2 | 已回归 |
| E-002 | CLI 首启慢勿误判 | 首次执行 CLI | 首启约 20~60s（import torch_npu 等重依赖），勿把启动慢误判为卡死 | 检查清单 3 | 实测 F0-1 | 已回归 |
