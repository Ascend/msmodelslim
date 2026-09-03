---
name: adaptation
description: |
  msModelSlim 模型适配总入口。适配准备由 4个子任务组成，每个子任务自带门禁，主 Agent 在收尾统一验收：
  ① analyze（分析，门禁：next_step 判定）；
  ② calibration（写 adapter，门禁：verify 四步；含条件前置 dequant 反量化）；
  ③ ep（多卡 MoE EP 并行改造，门禁：[EP_CHECK]+[EP_ACT_GATE]，多卡才触发）；
  ④ verify（主 Agent 适配验收，门禁：四步全 PASS = NPU 前向推理判定基准；兼作 calibration 内部门禁）。
  触发：新模型接入、写 adapter、适配器验证、FP8 反量化接入、逐层加载、MoE EP 并行适配等。
metadata:
  version: 0.2.0
  domain: quant
  framework: msmodelslim
  skill_class: workflow
---

# 模型适配（adaptation）

## 是什么

「模型适配」= 让目标模型能在 msModelSlim 量化流程中被正确加载、遍历与量化导出。按适配目标分为两类：

| 适配类型 | 目录 | 目标 |
|---|---|---|
| **校准适配**（Calibration Adaptation） | [calibration/](calibration/SKILL.md) | 把模型接入**校准量化**（PTQ / W8A8 / W4A16）流程：写 Model Adapter，使 `msmodelslim quant` 可加载、校准、量化、导出 |
| **并行适配**（Parallel Adaptation） | [ep/](ep/SKILL.md) | 把 MoE 模型改造成**多卡 EP** 并行状态，保证后续量化全程跑在 EP 上 |

> **命名澄清**：`calibration/` 是「校准适配」的缩写，指为**校准量化流程**做适配（写 Model Adapter），**不是**「跑校准集计算 scale 的那一步」。后者在量化执行时由 `msmodelslim quant` 内部完成，不在此目录。同理，`ep/` 指「EP 并行适配」，两者都是模型适配（`adaptation`）的子类。

## 使用方式

- **直接使用**：用户要求「接入新模型 / 写适配器 / 适配器验证 / FP8 反量化 / 逐层加载 / MoE EP 并行适配」时，按下方决策树与路由速查执行，由本 Skill 独立完成适配（含各子任务门禁），不依赖 `tuning`。
- **被 `tuning` 编排**：端到端调优在「模型准备」阶段委派本 Skill 的可委派子 Skill（`analyze` / `calibration` / `dequant` / `ep` / `verify`），委派时机与 input/output 契约见 `tuning/references/prepare_model.md`。本 Skill 是**领域入口**，自身不作为 subagent 被委派。

## 适配顺序（决策树）

适配准备由 4个适配子任务构成；每个子任务完成时自带**门禁**，通过后才进入下一步；主 Agent 在适配准备收尾统一验收（触发 verify 门禁），**不跳步**：

```text
适配准备
├─ ① analyze（分析）             门禁: next_step 判定
├─ ② calibration（写适配器）      门禁: verify 四步（内部门禁，随 verification_steps 回传）
│     └─ (条件) dequant（反量化）  门禁: adapter_updated=true → 回到 ②
├─ ③ ep（EP 并行改造，多卡才触发） 门禁: [EP_CHECK] + [EP_ACT_GATE]
└─ ④ verify（主 Agent 适配验收）   门禁: 四步全 PASS（NPU 前向推理判定基准）
```

**门禁说明**：

- **① analyze**：判断适配可行性（实现来源解析），输出风险结论与 `next_step`（`model-adapt` / `dequant` / `blocked` / `need_user_input`）。`blocked` / `need_user_input` 即停，回到本子 skill 修复或向用户索要材料，不得跳过继续。
- **② calibration**：写 Model Adapter（`handle_dataset` / `init_model` / `generate_model_visit` / `generate_model_forward` / `enable_kv_cache` + `config.ini` 注册 + `bash install.sh`）。完成时**自带 verify 四步作为内部门禁**，随 `verification_steps` 回传。
- **②' dequant（条件前置）**：仅当 ① 判定为原生量化模型（`next_step: dequant`）时，先执行反量化（FP8 per-block / per-channel → 写 `convert_*_to_bf16.py` 接入 adapter）；门禁 `adapter_updated=true`，完成后**回到 ②** 继续写 Model Adapter；普通 FP 模型从 ① 直接到 ②。
- **③ ep（条件触发）**：仅当调优需多卡（≥2卡）且命中 EP 路由时走；MoE 检查 → EP 就绪检查/改造，门禁 `[EP_CHECK]` + `[EP_ACT_GATE]`；非 MoE / 单卡跳过。
- **④ verify（主 Agent 适配验收）**：适配准备收尾的统一验收，也是「NPU 前向推理」判定基准——若 ② 的内部门禁缺失或未全过，主 Agent 独立下发 ④，四步全 PASS（Step2 全回退量化 + Step3 权重一致性/可加载保存）才视为模型可在目标设备正常前向推理。
- **可选高阶 layer_wise**：逐层量化（CPU 内存不足或用户明确要求时）依赖 ② 完成后启用，不占主链路。

> **verify 双重角色**：既是 ② calibration 的内部门禁（随 `verification_steps` 回传），又是 ④ 主 Agent 的适配验收门禁（独立下发，不得另造命令代替）。两层共用同一套四步验证。

## 路由速查

| 用户诉求 | 入口 |
|---|---|
| 新模型接入 / 写 adapter / FP8 反量化 / 逐层加载 | `calibration/`（必要时先 `analyze/`） |
| 适配器验证 / 主 Agent 适配验收（四步验证） | `verify/` |
| MoE 多卡 EP 并行 / EP 就绪检查 | `ep/` |
| 调优多卡 MoE：先校准再 EP | 先 `calibration/` 完成校准适配（含四步验证），再 `ep/` |
