---
name: evaluate
description: 执行模型测评。通过 scripts/run_evaluation.py 依据 Evaluation YAML 对量化模型进行评测。
license: Apache-2.0
metadata:
  version: 0.3.1
  domain: quantization
  framework: msmodelslim
  protocol: script
  skill_class: tool
  aliases:

    - evaluator
    - evaluation-run

  trigger_intents:

    - 执行测评
    - 运行 run_evaluation
    - 评测模型

  keywords:

    - run_evaluation
    - evaluate
    - aisbench
    - service_oriented

  # subagent 绑定声明：id = 委派标识（subagent_type）；bind = 委派时须绑定加载的 skill 根目录（含 references/、scripts/）
  # evaluate 依赖 installation：错误处理/环境校验（vLLM 服务、msmodelslim 安装检查）
  subagent:
    id: "evaluation/evaluate"
    bind:

      - "evaluation/evaluate"
      - "installation"

---

# 评测执行（evaluate）

## 概述

**解决什么**：依据 Evaluation YAML 配置，通过 `scripts/run_evaluation.py` 对量化模型进行评测。

**不解决什么**：

- 不生成/修改 Evaluation YAML → 见 `evaluation/evaluation-cfg` Agent
- 不执行量化 → 见 `quantization` Agent
- 不做策略决策 → 见 `tuning` Skill

**执行主体**：`scripts/run_evaluation.py`

---

## 协作关系

```text
tuning (workflow)
        │
        ▼ 调用
evaluation/evaluate (tool)
        │
        ▼ Script
  run_evaluation.py
        │
        ▼ 输出
  评测结果 (精度分数)
```

---

## 执行步骤

```text
┌─────────────────┐
│ 输入检查        │
│ - config_path   │
│ (Evaluation YAML)│
└────────┬────────┘
         ▼
┌─────────────────┐
│ 服务启动检查    │
│ - 检查 vLLM     │
│   是否就绪      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ execute:        │
│ run_evaluation  │
│ .py             │
│ (启动推理服务   │
│  + 执行评测)    │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 结果处理        │
│ - 解析精度分数   │
│ - 检查是否达标   │
│ - 错误上报       │
└─────────────────┘
```

---

## 输入

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `quant_model_path` | string | ✅ | 量化后模型路径 |
| `evaluate_id` | string | ✅ | 本轮评测 ID |
| `evaluate_config_path` | string | ✅ | Evaluation YAML 路径（编排层常称 `config_path`） |
| `save_path` | string | ✅ | 评测工作目录 |
| `device` | string | ❌ | 设备类型，默认 `npu` |
| `device_indices` | list[int] | ❌ | 设备索引列表，如 `[0,1]` |

---

## 脚本调用

```bash
python skills/evaluation/evaluate/scripts/run_evaluation.py \
  --quant-model-path /path/to/quantized \
  --evaluate-id eval-round-1 \
  --evaluate-config-path /path/to/evaluate.yaml \
  --save-path /path/to/workdir \
  --device npu \
  --device-indices 0,1
```

> **执行位置注意（实测 F0-2）**：本脚本会 `import msmodelslim`（含框架初始化）。务必在**仓库外**工作目录、以脚本**绝对路径**调用，命中 site-packages 完整安装；在仓库根内执行会命中源码树残缺包（缺 `config/`）报 `SecurityError: msmodelslim/config doesn't exist`。脚本不依赖 cwd，可在任意仓库外目录运行。

### 错误处理

| 错误类型 | 处理 |
|----------|------|
| msmodelslim 未安装 | 按 installation skill 安装（该 skill 已绑定给本 subagent） |
| 推理服务启动失败 | 检查端口占用、设备可用性 |
| 评测超时 | 检查 `aisbench.timeout` 配置 |
| 精度不达标 | 正常返回结果，由 orchestrator 决策 |

---

## 输出结果

### 成功

```json
{
  "ok": true,
  "evaluate_result": {
    "accuracies": [
      { "dataset": "gsm8k", "accuracy": "83.5" },
      { "dataset": "aime25", "accuracy": "52.0" }
    ],
    "expectations": [
      { "dataset": "gsm8k", "target": "83.0", "tolerance": "0" },
      { "dataset": "aime25", "target": "50.0", "tolerance": "0" }
    ],
    "is_satisfied": true
  }
}
```

> 注：`accuracies` / `expectations` 中的数值为 JSON 模式序列化后的十进制字符串，与 `EvaluateResult` 定义一致；下游 `EvaluateResult.model_validate()` 读取时恢复为 `Decimal`。不得用 Python 模式 `model_dump()` 直接输出（`Decimal` 无法被标准 `json.dumps()` 序列化）。

### 失败

```json
{
  "ok": false,
  "error": "推理服务启动失败",
  "error_code": "INFERENCE_ERROR"
}
```

## 执行流程

### 1. 服务启动

调用 `scripts/run_evaluation.py`（`execute`）

### 2. 结果解析

`run_evaluation.py` 输出 `{ok, evaluate_result}`，`evaluate_result` 结构与 msmodelslim `EvaluateResult` 一致：

| 字段 | 说明 |
|------|------|
| `accuracies[]` | 各数据集实测精度，`{ dataset, accuracy }` |
| `expectations[]` | 各数据集目标（来自 Evaluation YAML `demand.expectations`），`{ dataset, target, tolerance }` |
| `is_satisfied` | 所有数据集是否都达标（`accuracy >= target - tolerance`） |

`run_evaluation.py` 必须在调用公共 `emit_result()` 前使用 Pydantic JSON 模式序列化 `EvaluateResult`。禁止直接返回 `model_dump()` 的 Python 模式结果，因为其中的 `Decimal`（如 `accuracy`、`target`、`tolerance`）无法由标准 `json.dumps()` 序列化。JSON 模式会将十进制值无损转换为字符串；下游通过 `EvaluateResult.model_validate()` 读取时会恢复为 `Decimal`。

### 0 分处置（EVALUATION_INVALID / VALIDATION_ERROR）

`run_evaluation.py` 会在拉起服务前做长度配置预检（`VALIDATION_ERROR`），并在返回 `accuracy=0.0` 时诊断根因（`EVALUATION_INVALID` + `evidence`）。执行者收到这两类错误**不得直接向编排层上报**，按以下顺序自愈，修复多次仍失败才上报：

1. **确认根因**：读 `error.evidence`（`vllm_server.log` / aisbench summary 路径）。`rejected_by_vllm=true` 或 summary 显示请求全失败 → 服务/配置问题；summary 可解析 accuracy 且 =0.0 → 真实 0 分（如实回传，由编排层判不达标）。
2. **配置自洽性修复（唯一例外）**：确认是 `max_out_len ≥ max-model-len` 等**配置自洽性**问题时，允许修正评测配置后重测一次——这是「禁止派生或修改评测配置」的唯一例外，且**必须**：只做最小修正（如调小 `max_out_len`），回显修改内容与原因，并在审计中记录。不得改动评测口径（数据集/目标/设备）。
3. **重测仍为 0 / 无法确认根因** → 才向编排层上报 `EVALUATION_INVALID`，附 `evidence` 与已尝试的修复。

---

## 执行示例

### 标准调用

```bash
python skills/evaluation/evaluate/scripts/run_evaluation.py \
  --quant-model-path /workspace/output/round_1/quantized \
  --evaluate-id round-1 \
  --evaluate-config-path /workspace/output/evaluate.yaml \
  --save-path /workspace/output \
  --device npu \
  --device-indices 0,1
```

> 执行位置同「脚本调用」注：仓库外 cwd + 脚本绝对路径，规避源码树遮蔽（F0-2）。

### 结果返回给 orchestrator

```text
评测完成:
- gsm8k: 83.5% (目标: 83.0%) ✅
- aime25: 52.0% (目标: 50.0%) ✅
- 总体: 达标
- 耗时: 1800.5s
```

---

## 约束

- **Script-only**：禁止用裸 CLI 替代 `run_evaluation.py`
- **路径格式**：必须是 JSON 字符串
- **单轮单次**：每次调用只执行一次完整评测
- **禁止派生或修改评测配置**：只能按编排层给定的 `evaluate_config_path` 执行评测，**不得**自行派生、修改或生成新的 Evaluation YAML。子集不达标时 fast-fail 跳过后序数据集属正常行为，按原样回传结果，由编排层决策后续流程。评测配置仅由编排层 `evaluation-generator` 生成。（**唯一例外**：`0 分处置`中确认属配置自洽性问题时，允许最小修正后重测一次。）
- **服务生命周期**：由脚本内部评测服务管理。如果你需要测多个数据集，请你在测完所有数据集后再关闭服务化，**避免**重复多次拉起。服务化测评运行时长可能较长，超过 timeout 3600s，**务必避免**在测评的中途关闭服务化和测评，你应该等待测评完成，必要时可以通过看日志（如vllm_server.log）最新的消息时间来确认测评任务是否还活跃。

---

## 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `VALIDATION_ERROR` | 长度配置自洽性失败（`max_out_len ≥ max-model-len`） | 按「0 分处置」最小修正 `max_out_len` 后重测 |
| `EVALUATION_INVALID` | `accuracy=0.0` 且未能确认为真实 0 分（见 `error.evidence`） | 按「0 分处置」自查根因并自愈；仍失败向编排层上报证据 |
| `port already in use` | 端口被占用 | 更换端口或等待释放 |
| `HCCL init failed` | NPU 通信失败 | 检查 `device_indices` 和设备状态 |
| `evaluate.yaml not found` | 配置文件不存在 | 检查 `config_path` |
| `out of memory` | 设备内存不足 | 换设备 |
| `Object of type Decimal is not JSON serializable` | 使用了 Python 模式 `model_dump()` | 改用 `model_dump(mode="json")`，且不得重新执行已完成的评测 |

若错误不在上述常见错误中或者多次解决后依然未解决，依据[错误上报](references/error_handling.md)，按照错误上报格式返回至`evaluation/evaluate` Agent

---

## 检查清单

- [ ] `config_path` 指向的 Evaluation YAML 格式正确
- [ ] `device` 与 `device_indices` 匹配
- [ ] YAML 中的 `ASCEND_RT_VISIBLE_DEVICES` 与 `device_indices` 一致
- [ ] `device_indices` 长度与 `tensor-parallel-size` 对齐
- [ ] 目标端口未被占用
- [ ] NPU/GPU 设备可用
- [ ] msmodelslim 已安装
- [ ] 成功结果可被标准 `json.dumps()` 序列化，并可由 `EvaluateResult.model_validate()` 重新读取

## 经验条目（Experiences，追加制）

> 追加规范见 `skills/README.md`「经验条目」。连续编号 `[E-序号]`；正文保留权威展开，本表只做索引 + 元数据登记；来源：实测编号（F0-x / Dx / Tx）| 用户反馈 | 代码实证；验证状态：已回归 | 待验证 | 已上流 docs。

| 条目 | 主题 | 适用条件 / 触发信号 | 结论要点（一句话） | 正文位置 | 来源 | 验证状态 |
|------|------|------|------|------|------|------|
| E-001 | 脚本执行位置 | `run_evaluation.py` 调用前 | 脚本 `import msmodelslim`（含框架初始化），在仓库根内跑命中源码树残缺包（缺 `config/`）报 `SecurityError`；在**仓库外** cwd + 脚本绝对路径调用 | 脚本调用 | 实测 F0-2 | 已回归 |
| E-002 | 长度配置自洽预检 | 拉起服务前 | 脚本先做长度预检（`VALIDATION_ERROR`），`max_out_len ≥ max-model-len` 会被拦截在服务拉起前，不产生无效 0 分 | 0 分处置 | 实测 F0-5 | 已回归 |
| E-003 | Decimal JSON 序列化 | 回传评测结果时 | 必须 `model_dump(mode="json")`（Decimal→字符串），禁止 Python 模式 `model_dump()` 直出（`Decimal` 无法被标准 `json.dumps` 序列化）；下游 `model_validate` 自动恢复 Decimal | 输出结果 / 结果解析 | 代码实证 | 已回归 |
| E-004 | 0 分先自愈后上报 | 收到 `EVALUATION_INVALID` / `VALIDATION_ERROR` | 按「确认根因 → 配置自洽最小修正重测一次（唯一例外，需回显+审计）→ 仍 0/无法确认才上报」顺序处理，不得直接上抛 | 0 分处置 | 用户反馈 | 已回归 |
| E-005 | 服务生命周期 | 多数据集评测 / 长评测 | 测完所有数据集再关闭服务，避免重复拉起；服务化单次可能超 3600s，中途勿关，可用 `vllm_server.log` 最新消息时间确认活跃 | 约束-服务生命周期 | 用户反馈 | 待验证 |
