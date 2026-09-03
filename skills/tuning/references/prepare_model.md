# 模型准备

## 阶段说明

**模型准备阶段**是端到端自动量化与调优流程编排的第 3 阶段。在本阶段，你需要确保目标模型已被 msModelSlim 支持并完成适配，使后续量化配置调优阶段可以正常调用。

> 适配实现细节（Model Adapter 编写、`config.ini` 注册、接口约定等）以 docs 为准：《[LLM 大模型接入指南](../../../docs/zh/knowledge_base/model/integrating_models.md)》、《[LLM 量化集成指南](../../../docs/zh/knowledge_base/ptq/llm/integration_guide_large_language_model_quantization.md)》；本文只定义编排层的委派契约与判定标准。

## 执行依赖项

### 子代理

模型分析与适配工作由**专用子代理**承载；主代理 **不要**在本会话中代替 subagent 完成分析或适配。适配域的子任务划分：每个适配任务 = 独立子任务 + 自有门禁；主 Agent 在阶段收尾统一验收（触发 verify 门禁）。

| 子代理 | 功能用途 | 触发条件 | 门禁 |
|--------|----------|----------|------|
| `adaptation/calibration/analyze` | 适配前分析：实现来源解析、结构/MoE/逐层加载等风险评估 | 必选（模型未注册） | `next_step` 判定 |
| `adaptation/calibration` | 分析通过后：适配模板、注册、`config.ini` 与四步验证 | `next_step: "model-adapt"` | verify 四步（内部门禁） |
| `adaptation/calibration/dequant` | 原生量化权重反量化（FP8 per-block/per-channel）并接入 adapter | `next_step: "dequant"` | `adapter_updated=true` |
| `adaptation/ep` | MoE 多卡 EP 并行适配（分片构造、权重按 rank 加载、量化映射本地化） | 用户设备卡数 ≥ 2（路由决策命中） | `[EP_CHECK]` + `[EP_ACT_GATE]` |
| `adaptation/calibration/verify` | 适配功能性验收：四步验证（测试模型/全回退量化/权重一致性/规则校验） | 主 Agent 验收下发（或 calibration 内自验收） | 四步全 PASS |

> **verify 双重角色**：`adaptation/calibration/verify` 既是 calibration 子任务的内部门禁（适配完成时自带四步验证并随 `verification_steps` 回传），又是主 Agent 的适配验收门禁（模型准备收尾判定「NPU 前向推理」的基准）。两层共用同一套四步验证，见下文「步骤 4 最终验证」。

## 执行流程

### 1. 检查模型是否已支持

查询用户提供的 `model_type` 是否已在 `msmodelslim/config/config.ini` 的 `[ModelAdapter]` 中注册。注意 `model_type` 不是模型权重路径中 `config.json` 里的 `model_type`，一般形如 `Qwen3-32B`、`DeepSeek-V3`。如果已注册且适配器存在，则跳过本文档的后续 subagent 委派。

### 2. 委派模型分析

若模型未注册，委派 `adaptation/calibration/analyze` subagent。委派描述**必须**包含 SUBAGENT_IO 块，字段见下文。

### 3. 委派模型适配

仅当分析回传 `next_step: "model-adapt"` 时，委派 `adaptation/calibration` subagent。`next_step: "dequant"` 时**先委派 `adaptation/calibration/dequant`**（契约见下文），完成反量化适配后**回到本步骤重新委派 `adaptation/calibration`** 继续写 Model Adapter；`blocked` / `need_user_input` 时停止并向用户说明（细节见 `summary` 与 `report_path`）。`description` **必须**包含 SUBAGENT_IO 块，字段见下文。

### 3.5 委派 EP 并行适配（多卡条件触发）

仅当 [routing.md](./routing.md) 路由决策命中「多卡 EP」时（用户设备卡数 ≥ 2 且未声明不用多卡），在步骤 3 模型适配**之后**委派 `adaptation/ep` subagent（契约见下文）。本步骤归入**模型适配过程**：ep 与 calibration 同属 adaptation 层，将模型改造为可 EP 并行运行状态；但它作为独立子任务，拥有独立的门禁 `[EP_CHECK]`（结构分片验证）+ `[EP_ACT_GATE]`（单卡 vs 多卡激活余弦相似度）。

- 回传 `requires_ep=true`（MoE，EP 已适配）→ 后续量化/评测全程开启 EP 并行（每轮多卡、日志须含 `[EP_CHECK]`，中途不得退回单卡 / DP）；
- 回传 `requires_ep=false`（非 MoE）→ 普通多卡 / 单卡流程，不涉及专家分片。
- 路由决策本身（何时触发）见 `routing.md`；本文档只定义委派契约与回传判定。

### 4. 最终验证（主 Agent 适配验收）

确认以下条件均已满足后，方可进入下一阶段（量化配置调优）：

- [ ] 模型适配已完成，适配器已注册
- [ ] 模型权重文件完整可加载
- [ ] 模型可在目标设备（NPU）上正常执行前向推理
- [ ] （多卡场景）EP 适配已通过 `[EP_CHECK]` + `[EP_ACT_GATE]` 门禁

「NPU 前向推理」判定手段：以 `adaptation/calibration/verify` 四步验证为基准——Step2 全回退量化 + Step3 权重一致性/可加载保存验证通过，即视为「可在目标设备正常执行前向推理」；若四步验证未执行，须先委派 `adaptation/calibration/verify` 完成，不得另造命令代替。

**verify 的双重角色**（本步骤为第二层「主 Agent 验收门禁」）：

1. **calibration 内部门禁**：步骤 3 中 `adaptation/calibration` 适配完成时自带四步验证，回传 `verification_steps`（四步全 `passed: true` 即通过）；
2. **主 Agent 验收门禁**：本步骤收尾时，若 `verification_steps` 缺失或未全通过，主 Agent **必须**下发独立委派 `adaptation/calibration/verify`（契约见下文）完成四步验证，不得另造命令代替。

若上述任何步骤失败，须向用户明确报告原因并停止流程。

## 注意事项

- 禁止在本会话中代替 subagent 完成分析或适配代码编写
- 分析阶段判定阻塞时，不得强行进入适配或调优
- 适配完成后，按 `config.ini` 注册格式确认 `model_type` 已正确添加

## 拉起 subagent 的格式（SUBAGENT_IO v1）

协议总则见 [subagent_io_protocol.md](./subagent_io_protocol.md)。本文档面向**主 Agent**：定义委派 `input` 与回传 `output` 业务字段；`commands` 见协议。完整 output 示例见各 subagent prompt。

调用子 agent 委派工具时，委派描述**必须**包含一个 ` ```subagent-io v1 ` JSON 块。

### Agent: adaptation/calibration/analyze

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_type` | string | ✓ | msModelSlim 注册名，如 `Qwen3-8B` |
| `model_path` | string | ✓ | 模型权重目录 |
| `trust_remote_code` | bool | | 默认 `true` |
| `save_path` | string | | 工作目录；分析报告写入 `{save_path}/model_analysis_report.md` |

回传 `output` 必填：`next_step`，`implementation_source`，`summary`，`report_path`；有 shell 执行时填 `commands`

委派模板：

````markdown
```subagent-io v1

{
  "protocol": "subagent.subagent_io",
  "subagent_type": "adaptation/calibration/analyze",
  "input": {
    "model_type": "Qwen3-8B",
    "model_path": "/data/models/Qwen3-8B/",
    "trust_remote_code": true,
    "save_path": "/path/to/workdir/"
  }
}

```text
````

### Agent: adaptation/calibration

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_type` | string | ✓ | msModelSlim 注册名 |
| `model_path` | string | ✓ | 模型权重目录 |
| `trust_remote_code` | bool | | 默认 `true` |
| `analysis_report_path` | string | ✓ | 步骤 2 产出的分析报告路径 |
| `save_path` | string | | 适配工作目录 |

回传 `output` 必填：`adapter_registered`，`verification_steps`（四步全 `passed: true` 即通过），`artifact_paths`（可选），`commands`（须含 `install` 与 `verification_step1`～`verification_step4`）

委派模板：

````markdown
```subagent-io v1

{
  "protocol": "subagent.subagent_io",
  "subagent_type": "adaptation/calibration",
  "input": {
    "model_type": "Qwen3-8B",
    "model_path": "/data/models/Qwen3-8B/",
    "trust_remote_code": true,
    "analysis_report_path": "/path/to/workdir/model_analysis_report.md",
    "save_path": "/path/to/workdir/"
  }
}

```text
````

### Agent: adaptation/calibration/dequant

仅当 analyze 回传 `next_step: "dequant"` 时委派。完成 FP8 反量化（per-block / per-channel）并写入 `convert_*_to_bf16.py` 接入 adapter。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_type` | string | ✓ | msModelSlim 注册名 |
| `model_path` | string | ✓ | 模型权重目录 |
| `trust_remote_code` | bool | | 默认 `true` |
| `analysis_report_path` | string | ✓ | 分析报告路径（含量化判定依据） |
| `save_path` | string | | 工作目录 |

回传 `output` 必填：`dequant_script_path`（`convert_*_to_bf16.py` 路径），`adapter_updated`（bool，是否已接入 adapter），`commands`（须含 `dequant_convert`）。`adapter_updated=false` 时主 Agent 不得继续委派 `adaptation/calibration`，须先向用户索要反量化脚本或修复。

委派模板：

````markdown
```subagent-io v1

{
  "protocol": "subagent.subagent_io",
  "subagent_type": "adaptation/calibration/dequant",
  "input": {
    "model_type": "DeepSeek-V3",
    "model_path": "/data/models/DeepSeek-V3/",
    "trust_remote_code": true,
    "analysis_report_path": "/path/to/workdir/model_analysis_report.md",
    "save_path": "/path/to/workdir/"
  }
}

```text
````

**完成后流转**：dequant 回传成功（`adapter_updated=true`）后，回到「步骤 3 委派模型适配」重新委派 `adaptation/calibration` 继续完成 Model Adapter；dequant 失败或缺少反量化脚本时，按「注意事项」向用户报告并停止，不得跳过。

### Agent: adaptation/ep

仅当 [routing.md](./routing.md) 路由决策命中「多卡 EP」时委派（步骤 3.5）。完成 MoE 检查 + EP 就绪检查与适配 + `[EP_CHECK]` / `[EP_ACT_GATE]` 验证。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_type` | string | ✓ | msModelSlim 注册名，如 `DeepSeek-V3` |
| `model_path` | string | ✓ | 模型权重目录 |
| `device_list` | int[] | ✓ | 物理设备索引列表，如 `[0,1,2,3]` |
| `device_count` | int | ✓ | 卡数，等于 `device_list` 长度 |
| `quant_type` | string | ✓ | 量化方案：`w8a8` / `w4a8`（用于判断 mapping 本地化项） |
| `save_path` | string | ✓ | 工作目录 |
| `trust_remote_code` | bool | | 默认 `true` |

回传 `output` 必填：`EP_ADAPT_RESULT`（`PASS` / `FAIL`），`requires_ep`（bool），`moE` 判定结论；数值门禁失败时回传 `first_diverged_layer`（可选）。`commands` 须含 `ep_check` 与 `ep_act_gate`（未执行时 `skipped: true`）。

委派模板：

````markdown
```subagent-io v1

{
  "protocol": "subagent.subagent_io",
  "subagent_type": "adaptation/ep",
  "input": {
    "model_type": "DeepSeek-V3",
    "model_path": "/data/models/DeepSeek-V3/",
    "device_list": [0, 1, 2, 3],
    "device_count": 4,
    "quant_type": "w8a8",
    "save_path": "/path/to/workdir/",
    "trust_remote_code": true
  }
}

```text
````

回传示例：

````markdown
```subagent-io v1

{
  "protocol": "subagent.subagent_io",
  "subagent_type": "adaptation/ep",
  "status": "ok",
  "output": {
    "EP_ADAPT_RESULT": "PASS",
    "requires_ep": true,
    "first_diverged_layer": null
  }
}

```text
````

### Agent: adaptation/calibration/verify

用于模型适配完成后的功能性验收（四步验证）。作为「主 Agent 验收门禁」独立下发时委派（步骤 4）；作为 calibration 内部门禁时由 `adaptation/calibration` 自带并回传 `verification_steps`，两层结构一致。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_type` | string | ✓ | msModelSlim 注册名 |
| `model_path` | string | ✓ | 模型权重目录 |
| `trust_remote_code` | bool | | 默认 `true` |
| `analysis_report_path` | string | | 分析报告路径（有则传入，用于 Step4 规则校验对照） |
| `save_path` | string | | 验证工作目录 |

回传 `output` 必填：`passed`（bool，四步全过为 `true`），`verification_steps`（`[{step, name, passed}]`，Step1 测试模型 / Step2 全回退量化 / Step3 权重一致性·可加载保存 / Step4 实际量化规则校验）；失败时回传 `failed_step` 与 `fix_suggestion`。`commands` 须含 `verification_step1`～`verification_step4`（对应 `scripts/step1_generate_test_model.py` 等）。

委派模板：

````markdown
```subagent-io v1

{
  "protocol": "subagent.subagent_io",
  "subagent_type": "adaptation/calibration/verify",
  "input": {
    "model_type": "Qwen3-8B",
    "model_path": "/data/models/Qwen3-8B/",
    "trust_remote_code": true,
    "save_path": "/path/to/workdir/"
  }
}

```text
````
