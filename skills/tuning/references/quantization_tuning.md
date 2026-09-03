# 量化配置调优（阶段）

## 阶段说明

**量化配置调优阶段**是**端到端自动量化与调优流程**编排的第4阶段。这个阶段将循环迭代生成多组量化配置，进行测评，并从中选择最优策略用作结果。

## 执行依赖项

### Agent

| 委派角色 | 功能用途 |
| --- | --- |
| `evaluation/evaluation-cfg` | 生成测评配置 |
| `strategy/practice-cfg` | 生成量化配置 |
| `quantization` | 依据量化配置进行量化 |
| `evaluation/evaluate` | 对量化后的模型进行测评 |

### Scripts（编排层）

| Skill | 脚本 | 功能用途 |
| --- | --- | --- |
| `tuning` | `scripts/history_clear.py` | 任务初始化时清空调优历史（**循环外执行一次**，位于循环开始之前） |
| `tuning` | `scripts/history_append.py` | 记录一条调优历史 |
| `tuning` | `scripts/accuracy_append.py` | 将 practice + evaluate 评测结果写入精度缓存 |
| `tuning` | `scripts/accuracy_lookup.py` | 查询精度缓存，避免重复计算 |

### CLI / 脚本（子 Skill，由 subagent 调用）

| Skill | 命令 / 脚本 | 功能用途 |
| --- | --- | --- |
| `sensitive-layer-analysis` | `msmodelslim analyze layer ...` | 敏感层分析（一次；由 `strategy/practice-cfg` 步骤②再委派，或主 Agent 直接委派） |
| `strategy/practice-cfg` | `scripts/validate_practice_yaml.py` | Practice YAML 校验 |
| `quantization` | `msmodelslim quant --config ...` | 执行量化 |
| `evaluation/evaluate` | `scripts/run_evaluation.py` | 执行评测 |

## 详细步骤

完整的调优循环流程图如下。在执行该流程前和流程中，须遵守后续小节列出的约束规则。

```plaintext
             ┌───────────────────┐
             │ evaluation/       │
             │ evaluation-cfg    │
             │ (生成测评配置)     │
             └─────────┬─────────┘
                       ▼
            ┌──────────────────────────┐
            │ expert-rules             │
            │ (standing_high_with_     │
            │  experience；二分前)      │
            └───────────┬──────────────┘
                        ▼
            ┌──────────────────────────┐
            │ Script: history_clear    │
            │ (任务初始化历史，循环外)  │
            └───────────┬──────────────┘
                        ▼
             (>>> 循环开始 <<<) ◄────────────┐
                     ▼                      │
            ┌────────────────────┐          │
            │ 输出:              │          │
            │ "第X次调优循环"     │          │
            └────────┬───────────┘          │
                     ▼                      │
            ┌─────────────────┐             │
            │ strategy/       │             │
            │ practice-cfg    │             │
            │ (生成量化配置，   │             │
            │ 应用结构化回退)   │             │
            └────────┬────────┘             │
                     ▼                      │
            ┌─────────────────┐             │
            │ Script:         │             │
            │ accuracy_lookup │             │
            │ (查询精度缓存)   │             │
            └────────┬────────┘             │
               ┌─ 缓存命中? ─┐               │
               │            │               │
         Yes   │        No  ▼               │
(跳过量化/评估) │    ┌───────────────┐       │
               │    │ quantization  │       │
               │    │ (执行量化)     │       │
               │    └────────┬──────┘       │
               │             ▼              │
               │    ┌───────────────┐       │
               │    │ evaluation/   │       │
               │    │ evaluate      │       │
               │    │ (模型测评)     │       │
               │    └────────┬──────┘       │
               │             ▼              │
               │    ┌─────────────────┐     │
               │    │ Script:         │     │
               │    │ accuracy_append │     │
               │    │ (写入精度缓存)   │     │
               │    └────────┬────────┘     │
               └─────┬───────┘              │
                     ▼                      │
            ┌────────────────┐              │
            │ Script:        │              │
            │ history_append │              │
            │ (记录调优历史)  │              │
            └────────┬───────┘              │
                     ▼                      │
            ┌─────────────────┐             │
            │ 检查退出条件:    │             │
            │ 1. 当前策略收敛? │             │
            │ 2. 达到最大次数? │             │
            └────────┬────────┘             │
         ┌───── 满足退出条件? ──────┐        │
         │                         │        │
      No ▼                     Yes ▼        │
(继续循环)                      [ 任务结束 ] │
         └──────────────────────────────────┘
```

## 调优约束规则

以下规则对上方流程图中的各个环节施加额外约束，执行调优时必须一并遵守。

### FP baseline 获取（循环前前置）

**分两种情况**：

1. **用户提供了绝对精度目标**（如「gsm8k 精度 ≥ 83%」）：直接将该数值作为 target，**不需要**执行浮点基线测评。
2. **用户未提供绝对目标**（如「精度损失 ≤ 2%」「尽量保持精度」等相对/模糊描述）：**必须**先对浮点模型执行评测获取 baseline，禁止猜测或使用默认值：
   1. 生成浮点模型的评测配置（不含 quantization 相关参数）
   2. 对浮点模型执行评测，记录 baseline 精度
   3. 将 baseline 值写入 evaluate_config.yaml 的 target 字段（target = FP baseline - tolerance, 其中 tolerance 为用户容差）
   4. 然后才进入循环

注意：target 不要随意计算。参照以下示例进行计算：
（1）如果用户描述"不低于浮点精度超过 1%"，则 target = FP baseline - 1%。
（2）如果描述"某数据集相比浮点模型多错一道题"，则需要通过计算该数据集一道题目对应的精度数值，然后 target = FP baseline - 一题对应的数值。

**FP 基线不落 accuracy 缓存**：FP 基线评测结果由主 Agent 记录（建议写入 `{save_path}/baseline.yaml`），**不得**通过 `accuracy_append.py` 写入精度缓存。原因：精度缓存 key 为 `md5(evaluation_config) + md5(quant_config)`，其中 `evaluation_config` 含 `demand.target`——FP 基线评测按约定用 `target=100` 占位（见 `evaluation/evaluation-cfg`），测完把真实 target 写入后续 evaluate_config 后 key 必然变化，缓存无法命中；且 FP 评测没有对应的 Practice（`accuracy_append.py` 入参要求 practice_path）。量化轮次的精度才写入缓存。

### standing_high 摸高二分搜索

采用 standing_high 策略时，调优循环不是"达标即停"，而是二分搜索最少回退层数：

1. **上界**：上界为全部敏感层回退，此时量化结果**近似**等价于浮点权重，因而可以直接复用浮点基线精度，而不需要进行量化和测评。

   > **风险说明**：「等价于浮点」仅在无离群值抑制、且无非敏感层被量化时严格成立。若 Practice 仍带 `iter_smooth`/`flex_smooth_quant` 等离群值抑制、或仍有非敏感层被量化（如 MoE experts、attention 投影），上界配置的实际精度可能略低于 FP 基线。此时建议对最终上界配置**抽查量化评测一次**（可选项，计入迭代轮次）；若实测不达标，以最近一个已实测达标的配置作为最终结果。

2. **Round 1**：0层回退（下界）
3. **后续轮**：取上下界中位数，保持 gate_proj/up_proj 配对完整性（见下方约束）
4. **终止条件**：上界与下界不可再二分（如差距 ≤ 配对粒度）
5. **最终结果**：上界（最少回退且达标的配置）为最优

退出条件（判断优先级）：

- 二分收敛（上界与下界不可再分）→ 输出上界为最优
- 达到最大迭代次数 → 从历史达标配置中输出回退层数最少（即量化层数最多）的配置
- **"某轮达标"不是退出条件**，达标只标记上界

### 二分搜索约束（exclude 列表截断规则）

生成 Practice YAML 的 exclude 列表时，必须遵守以下规则：

- gate_proj 和 up_proj 在 vllm 中融合为 gate_up_proj，**必须同退同量化**
- 截断 exclude 列表时，不能在 gate_proj/up_proj 配对中间截断
- 若中位数截断点落在配对中间，向上取整到配对末尾

### 结构化回退经验（二分前接入）

采用 `standing_high_with_experience` 策略时，**进入二分搜索前**，主 Agent 应先委派 `strategy/standing_high_with_experience/expert-rules` 取得「结构化回退意见」，作为 practice-generator 生成/修改 Practice YAML 时 `exclude` / 高精度档位 / 提级项的**初始化候选**，再跑二分。

- **接入时机**：FP baseline 获取与压缩数据集来源确认之后、`(>>> 循环开始 <<<)` 之前执行一次；拿到回退意见后随每轮 `prev_result` 一起传给 practice-generator，由后者在生成 Practice 时落地（因此该意见**不参与、也不替代** standing_high 的二分搜索逻辑本身）。
- **委派 `input` 参考**：模型结构类型、`quant_type`（`w8a8`/`w4a8`/`w4a4`）、是否 MoE / EP、routed/shared experts 与 gate/router 命名、当前 `include`/`exclude`、浮点基线与敏感层分析、可引用 `lab_practice` YAML 路径。
- **回退意见用途**：优先回退候选（如 `mlp.down_proj`、`o_proj`、MoE `gate`/`router`、`shared_experts`、MLA 低秩投影等）与置信度/证据等级，供 practice-generator 作为初值；最终是否回退、回退哪些层，仍以敏感层分析 + 本轮精度结果 + 二分收敛为准。
- **只回答问题、不执行**：`strategy/standing_high_with_experience/expert-rules` 仅输出「哪些层需要回退」，不改 YAML、不量化、不做 EP 检查 / 服务化 / 评测；这些动作仍由 practice-generator / quantizer / evaluator / EP 适配承接。
- **策略为 `standing_high`（不含 experience）时**：不强制委派该 skill，可按需作为参考，不改变二分流程。

#### 委派契约（Agent: expert-rules）

调用委派工具时，委派描述**必须**包含 ` ```subagent-io v1 ` JSON 块。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_type` | string | ✓ | msModelSlim 注册名 |
| `structure_family` | string | ✓ | 结构模型类（MoE / MLA / 混合 attention / Dense FFN / VLM 等，可多属） |
| `quant_type` | string | ✓ | `w8a8` / `w4a8` / `w4a4`（含 int / mxfp 系） |
| `is_moe` | bool | ✓ | 是否 MoE |
| `requires_ep` | bool | | EP 适配结果（多卡任务且已适配时为 `true`） |
| `current_include` | string[] | | 当前 Practice 的 `include` |
| `current_exclude` | string[] | | 当前 Practice 的 `exclude`（含 `protected_exclude`） |
| `fp_baseline` | object \| null | | 浮点基线精度（数据集 → 精度值） |
| `analysis_result_path` | string \| null | | 敏感层分析结果 `analysis_result.yaml` 路径 |
| `lab_practice_path` | string \| null | | 可引用的既有 Practice YAML 路径 |

回传 `output` 必填：`experience_hints`，结构如下：

```json
{
  "priority_fallback_candidates": [
    { "module": "*.mlp.down_proj", "reason": "FFN 输出投影，量化误差易累积", "evidence_level": "high" }
  ],
  "default_quantized": ["*gate*", "*router*"],
  "evidence_level": "high",
  "suggested_order": "离群值抑制 → 校准集 → 回退"
}
```

`experience_hints` 仅作为 practice-generator 生成/修改 Practice 时的 `exclude`/高精度档位/提级项**初值**；`priority_fallback_candidates[].evidence_level` 取值 `high` / `medium` / `low`。

委派模板：

````markdown
```subagent-io v1

{
  "protocol": "subagent.subagent_io",
  "subagent_type": "strategy/standing_high_with_experience/expert-rules",
  "input": {
    "model_type": "Qwen3-32B",
    "structure_family": "MoE",
    "quant_type": "w8a8",
    "is_moe": true,
    "requires_ep": false,
    "current_include": ["*"],
    "current_exclude": ["*.gate"],
    "fp_baseline": { "gsm8k": 83.5 },
    "analysis_result_path": "/path/to/workdir/analysis_result.yaml",
    "lab_practice_path": null
  }
}

```text
````

### 压缩数据集的使用（默认）

**默认的量化调优过程均使用压缩数据集**进行快速迭代，只有在用户确认不使用压缩数据集时，才退回全集测试。

进入调优循环前，主 Agent **必须**向用户确认压缩数据集的来源，三选一：

1. **用户自备已压缩数据集**：用户直接提供精简数据集路径，调优迭代直接使用。
2. **委派 `evaluation/dataset-compression-herding` skill 生成**：用户未自备但同意生成时，委派该 skill，用 RBF Kernel Herding 算法从全集生成 coreset 子集。
   - 当前仅支持 `aime2025` 与 `gpqa` 两个数据集；
   - 压缩耗时约 30 分钟（CPU 实现），执行前须向用户说明成本并获确认；
   - 生成「全集 + 子集」两份测试脚本，子集携带 `indices.json` 可追溯复现；
   - 压缩集结果仅用于调优迭代的快速反馈，最终精度仍以全量数据集验收为准。
3. **用户两者都不愿意**：既不提供已压缩数据集、也不生成时，退回**全集测试**——直接使用全量数据集进行调优迭代，并向用户说明反馈周期会变长。

> **数据集支持面**：herding 压缩（第 2 种来源）当前仅支持 `aime2025` 与 `gpqa`。若调优数据集不在其中（如 `gsm8k`、`mmlu`、`bfcl-simple`），且用户不自备已压缩子集，则**直接走全集测试**进行调优迭代，并向用户回显：
> 「数据集 {dataset} 暂不支持 coreset 压缩（当前仅支持 aime2025/gpqa），且未提供已压缩子集，将使用全集进行调优迭代，反馈周期会变长；如需加速可自备该数据集的压缩子集。」
> 此情形下不存在「子集出口标准」，仅以全集出口标准（用户目标或全集 FP 基线）为达标线。

#### 两个出口标准（子集 + 全集）

使用压缩数据集时，主流程存在**两个独立的出口标准**，都必须先于调优循环确定：

| 出口标准 | 数据集 | 作用 | 达标判定 |
|---------|--------|------|----------|
| **子集出口标准** | coreset 子集 | 子集调优循环的「达标」线 | 子集精度 ≥ 子集 FP 基线（或用户给出的子集绝对目标） |
| **全集出口标准** | 全集 | 最终验收的「达标」线 | 全集精度 ≥ 全集 FP 基线（或用户给出的全集绝对目标） |

两个出口标准的获取方式（进入调优循环前**必须**完成）：

1. **先询问用户**：是否分别提供「子集出口标准」与「全集出口标准」（可给绝对精度值，或只给一方、另一方由基线测得）。
2. **用户不提供某一方标准时**：在当前环境**跑浮点模型**测得对应数据集上的 FP 基线精度，作为该方出口标准。**此时必须向用户提示**：浮点基线评测会**额外占用卡数资源**（与量化/评测卡数可能不同），请用户确认可用卡后再执行；每个出口标准（子集 / 全集）对应的 FP 基线**各测一次**并由主 Agent 记录（写入 `{save_path}/baseline.yaml`），供该标准后续复用。

**注意**：子集与全集是两个不同的评测配置（`config_name` 不同），各自的 FP 基线**不能互相替代**——全集不达标时不能拿「子集基线」当作全集出口，反之亦然。

#### 主流程：子集先行 → 全集兜底

采用「**子集调优 → 全集验证 → 不通过改全集调优**」的闭环，**不再采用**「子集达标但全集不达标时按固定步长逐步收紧子集出口标准」的容忍性做法（该做法无法保证与全集一致）：

```text
1. 用子集调优（子集出口标准 = 子集 FP 基线 / 用户给定子集目标）
      ↓ 直到子集验证通过（子集精度达标）
2. 全集验证：用全集测当前最优量化权重
      ├─ 达标 → 调优完成，输出结果
      └─ 不达标 → 进入第 3步（不逐步收紧，不猜测）
3. 直接用全集调优（全集出口标准 = 全集 FP 基线 / 用户给定全集目标）
      ↓ 在全集上跑二分搜索 / 摸高，直到全集达标
```

关键约束：

- **子集只负责快速迭代**：子集验收通过只代表「可以进入全集验证」，不代表最终达标。
- **全集不达标 → 直接切全集调优**：切全集后以「全集出口标准」为准重跑摸高二分搜索，让最终权重与全集必然对齐；**不再**通过反复收紧子集标准来「逼近」全集。
- **切全集前先回显确认**：第 3 步切换前须向用户回显：当前全集验证结果、全集出口标准、以及「改在全集上进行调优」的动作，获得用户认可后执行。
- **浮点基线记录**：子集 FP 基线与全集 FP 基线**各测一次**，由主 Agent 记录（写入 `{save_path}/baseline.yaml`），再写入对应 evaluate_config 的 `demand.expectations[].target`；**不写入 accuracy 精度缓存**（key 含 `demand.target` 且 FP 评测无 practice，详见「FP baseline 获取」）。

> 前置约束：进入调优循环前，须先确定两个出口标准——用户给出绝对目标则以其为准；否则跑浮点基线测得（见上文「FP baseline 获取」，注意此处需要**子集与全集两份**基线，或用户分别给出的两份绝对目标）。

#### 评测配置管理（子集 / 全集切换）

评测配置（`evaluate.yaml`）采用**单份文件 + config_name 切换**的方式管理，由 `evaluation/evaluation-cfg` 生成/修改，**不得**由 quantization / evaluate 子 agent 派生：

- **迭代期**：`datasets.<key>.config_name` 使用 coreset config_name（herding 场景 B 回传的 `<原 config_name>_coreset`）；`demand.expectations[].target` 为子集出口标准。
- **全集验证 / 全集调优**：切换 `config_name` 为全集 config_name，**同时**更新 `demand.expectations[].target` 为全集出口标准；两处必须同步修改，切换后由 evaluation-cfg 重新校验（字段一致性、`tensor-parallel-size` 与 `device_indices` 对齐等）。
- **切换时机**：子集调优达标、准备全集验证前执行一次；全集验证不达标转全集调优前再执行一次（此时向用户回显全集验证结果、全集出口标准与「改在全集上调优」的动作，获认可后执行）。
- **accuracy 缓存自动隔离**：缓存 key 为 `md5(evaluation_config)+md5(quant_config)`，`evaluation_config` 含 `config_name`，全集与 coreset 自然形成不同缓存键，无需额外处理。

### 服务化推理脚本（可选加速）

进入调优循环前，主 Agent **必须询问用户**是否提供了服务化推理脚本，用于加速评测阶段。

#### 适用场景

评测阶段每轮需启动 vLLM 推理服务加载量化模型，模型加载耗时 1~3 分钟。若提供常驻服务脚本，可在后续轮次跳过服务启停，直接 reload 模型执行评测。

#### 用户提供方式

| 方式 | 说明 | 用户操作 |
|------|------|---------|
| **预启动** | 调优开始前用户已启动服务 | 告知 agent 服务地址（如 `localhost:8000`）和 reload 接口 |
| **代启动** | 调优过程中由 agent 启动常驻服务 | 提供脚本路径，agent 首轮启动，后续轮次 reload |

#### 脚本接口约定

若用户提供脚本，建议遵循以下接口（agent 按此调用）：

| 操作 | 命令 | 说明 |
|------|------|------|
| 启动服务 | `python service.py start --model-path <path> --port <port>` | 首轮启动，加载模型 |
| 热加载 | `python service.py reload --model-path <path>` | 后续轮次，替换模型权重 |
| 执行评测 | `python service.py evaluate --config <path>` | 调用 AISBench 评测 |
| 关闭服务 | `python service.py stop` | 调优结束后关闭 |

脚本输出须为 stdout JSON，含 `success` 字段，错误时含 `error` 字段（与编排层脚本约定一致）。

#### 接入编排（替换默认 evaluator）

用户确认提供服务化脚本后，主 Agent 按以下方式接入，**不再委派 `evaluation/evaluate` 子 agent** 执行评测：

| 环节 | 原默认流程 | 服务化脚本接入后 |
|------|-----------|----------------|
| 首次评测（round 1） | evaluator 启动服务 → 评测 → 关服务 | 执行 `start --model-path <量化权重路径>`（代启动）或复用预启动服务（用户已给地址），再 `evaluate --config <path>` |
| 后续轮次 | 每轮重新启停服务 | 只执行 `reload --model-path <新量化权重路径>` + `evaluate --config <path>`，不重启服务 |
| 调优结束 | — | 执行 `stop` 关闭服务（代启动时）；预启动的服务由用户自行管理，agent 不关 |

接入判定：

- **有服务化脚本**：评测改用 `start/reload/evaluate/stop` 命令驱动，跳过 `evaluation/evaluate` 委派；
- **无服务化脚本**：保持默认流程，委派 `evaluation/evaluate`。

#### 无服务化脚本时的默认行为

用户未提供或不确定时，走默认流程：每轮由 `evaluation/evaluate` 子 agent 启动 vLLM 服务、执行评测、关闭服务。

## 调优结束与 Practice 入库

退出条件满足（二分收敛 / 达到最大迭代次数 / 全集验收达标）后：

1. 确认最终达标配置：`{save_path}/practice_round_{N}.yaml` 与对应 `round_{N}/quantized` 权重。
2. 向用户回显最终结果（格式见 [output_format.md](./output_format.md)），获得认可后视为收敛。
3. 调用 `tuning/scripts/finalize_practice_repo.py` 将最终 Practice 写入实践库：

```bash
python skills/tuning/scripts/finalize_practice_repo.py \
  --model-type "${MODEL_TYPE}" \
  --model-path "${MODEL_PATH}" \
  --final-practice-path "${save_path}/practice_round_{N}.yaml" \
  [--trust-remote-code]
```

> **执行位置注意（实测 F0-2）**：本脚本 `import msmodelslim`（PluginModelFactory 等）。请在**仓库外**工作目录以脚本**绝对路径**调用；仓库根内执行会命中源码树残缺包（缺 `config/`）报 `SecurityError`。

- 脚本内部校验 `model_adapter` 是否实现 `ModelInfoInterface` 且 practice 仓库支持保存；不支持时返回 `ok: false` + `error_code=PRACTICE_SAVE_SKIPPED`（显式失败，不静默视为成功）。此时执行**替代归档**：保留 `practice_round_{N}.yaml`，在回显与审计中记录未入库原因与可复现配置路径；不阻塞交付。

## 拉起 subagent 的格式（SUBAGENT_IO v1）

协议总则见 [subagent_io_protocol.md](./subagent_io_protocol.md)。本文档面向**主 Agent**，重点定义委派 `input`；回传 `output` 一行简述主 Agent 需读取的业务字段。

调用子 agent 委派工具时，委派描述**必须**包含一个 ` ```subagent-io v1 ` JSON 块；`input` 按下表填写。收到回传后从 `output` 读取下表字段驱动流程，完整 output 示例见各 subagent prompt。每次收到回传后，按 [subagent_io_protocol.md](./subagent_io_protocol.md)「执行者关键节点记录与审计汇总」将 `input`/`commands`/`deliverables`/`output` 聚合写入 `{save_path}/audit/audit_log.md`，供人类一次性复盘（执行者漏报命令时如实标注，不得虚构）。

编排脚本（`accuracy_lookup`、`history_clear`、`history_append`、`accuracy_append`）由主 Agent 直接执行，**不得**通过 subagent 委派。accuracy 系脚本（`accuracy_lookup`/`accuracy_append`）已内置插件冷启动超时重试（3次退避，见 `tuning-loop-lib/scripts/script_utils.py` 的 `with_plugin_timeout_retry`）；重试后仍报 `Code 303` 才按错误上报，首次超时不判失败。

### Agent: strategy/practice-cfg

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_type` | string | ✓ | 模型类型名 |
| `model_path` | string | ✓ | 模型路径 |
| `save_path` | string | ✓ | 工作目录，Practice YAML 写入此目录 |
| `device` | string | ✓ | 如 `npu:2,3` |
| `strategy` | string | ✓ | 搜索算法：`standing_high` 或 `standing_high_with_experience`； |
| `calib_dataset` | string | ✓ | 已确定并经用户确认的校准数据集 |
| `max_iterations` | int | ✓ | 最大迭代轮次 |
| `round` | int | ✓ | 当前调优轮次 |
| `prev_result` | object\|null | | 上轮评测结果，首轮 `null` |
| `anchor_practice` | string\|null | | 已知最优且达标的 Practice 路径 |
| `experience_hints` | object\|null | | 结构化回退意见（由 `strategy/standing_high_with_experience/expert-rules` 回传），供生成 Practice 时作为 `exclude`/高精度档位初值；`standing_high` 策略时为 `null` |

回传 `output` 必填：`practice_path`，`validation: { ok, valid, errors }`，`commands`（须含 `sensitive_layer_analysis` 与 `validate_practice_yaml`；跳过敏感层分析时前者 `skipped: true`）。其中 `sensitive_layer_analysis` 由 practice-cfg 步骤②**再委派** `sensitive-layer-analysis` subagent 完成（嵌套委派，见 `subagent_io_protocol.md`），命令细节按 [sensitive-layer-analysis SKILL](../../sensitive-layer-analysis/SKILL.md)；实践上主 Agent 无需感知该再委派，只读取 practice-cfg 回传即可。

委派模板：

````markdown
```subagent-io v1

{
  "protocol": "subagent.subagent_io",
  "subagent_type": "strategy/practice-cfg",
  "input": {
    "model_type": "",
    "model_path": "",
    "save_path": "",
    "device": "",
    "strategy": "standing_high_with_experience",
    "calib_dataset": "",
    "max_iterations": 10,
    "round": 1,
    "prev_result": null,
    "anchor_practice": null,
    "experience_hints": null
  }
}

```text
````

### Agent: evaluation/evaluation-cfg

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_name` | string | ✓ | 量化后的模型标识符 |
| `save_path` | string | ✓ | 工作目录 |
| `datasets` | object[] | ✓ | 评测数据集列表，每项见下表 |
| `service_host` | string | | 默认 `localhost` |
| `service_port` | int | | 默认 `8000` |
| `device_type` | string | | 默认 `ascend` |
| `device_indices` | int[] | ✓ | 用户选择的物理设备索引，如 `[7]`；用于生成 vLLM 的 `ASCEND_RT_VISIBLE_DEVICES` |
| `allowed_local_media_path` | string\|null | | VLM 路径任务的显式覆盖目录；`null` 时由配置生成 Skill 尝试从数据集 README 推导 |

`datasets[]` 每项：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | ✓ | 数据集名称（如 `gpqa`） |
| `config_name` | string\|null | | 用户明确指定或复用已确认配置时填写；否则为 `null`，由配置生成 Skill 查询 AISBench README 后选择 |
| `target` | number | ✓ | 目标精度（百分比） |
| `tolerance` | number | | 容差，默认 `0` |

回传 `output` 必填：`evaluate_config_path`；若执行了 YAML 校验，建议填 `commands`（`name: validate_yaml`）

委派模板：

````markdown
```subagent-io v1

{
  "protocol": "subagent.subagent_io",
  "subagent_type": "evaluation/evaluation-cfg",
  "input": {
    "model_name": "",
    "save_path": "",
    "datasets": [
      {
        "name": "gpqa",
        "config_name": null,
        "target": 79.0,
        "tolerance": 1.0
      }
    ],
    "service_host": "localhost",
    "service_port": 8000,
    "device_type": "ascend",
    "device_indices": [0],
    "allowed_local_media_path": null
  }
}

```text
````

### Agent: quantization

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `config_path` | string | ✓ | Practice YAML 路径 |
| `model_path` | string | ✓ | 原始模型路径 |
| `save_path` | string | ✓ | 量化产物目录，如 `.../round_N/quantized` |
| `model_type` | string | ✓ | msModelSlim 注册的模型适配器名称，用于 `msmodelslim quant --model_type` |
| `device` | string | ✓ | 如 `npu:2,3` |
| `trust_remote_code` | bool | | 默认 `true` |
| `round` | int | | 建议填写当前轮次 |

回传 `output` 必填：`success`，`quantized_path`，`exit_code`，`commands`（含 `name: quantize` 的量化命令）

委派模板：

````markdown
```subagent-io v1

{
  "protocol": "subagent.subagent_io",
  "subagent_type": "quantization",
  "input": {
    "config_path": "",
    "model_path": "",
    "save_path": "",
    "model_type": "",
    "device": "",
    "trust_remote_code": true,
    "round": 1
  }
}

```text
````

### Agent: evaluation/evaluate

> **前置判定**：若用户已提供服务化推理脚本（见上文「服务化推理脚本」），评测**不委派本 subagent**，改由主 Agent 直接 `execute` 服务化脚本的 `reload` + `evaluate` 命令。仅当无服务化脚本时才按本小节委派。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `evaluate_config_path` | string | ✓ | Evaluation YAML 路径（即编排层常用 `config_path`） |
| `quant_model_path` | string | ✓ | 量化模型路径 |
| `save_path` | string | ✓ | 工作目录 |
| `device` | string | ✓ | 如 `npu` |
| `device_indices` | int[] | ✓ | 如 `[0, 1]` |
| `evaluate_id` | string | | 评测标识 |
| `round` | int | | 建议填写当前轮次 |

回传 `output` 必填：`evaluate_result`（结构见下方，与 msmodelslim `EvaluateResult` 一致），`commands`（须含 `inference_service` 与 `evaluation`）

**`evaluate_result` 结构**（`run_evaluation.py` 实际输出 `{ok, evaluate_result}` 中的 `evaluate_result` 字段）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `accuracies` | array | 各数据集实测精度，每项 `{ dataset: string, accuracy: Decimal }` |
| `expectations` | array | 各数据集目标，每项 `{ dataset, target: Decimal, tolerance: Decimal }`；与 Evaluation YAML `demand.expectations` 一一对应 |
| `is_satisfied` | bool | 是否全部满足（`accuracy >= target - tolerance`） |

> 主 Agent **原样**将 `output.evaluate_result` 作为 `--evaluate-result` 传给 `tuning/scripts/accuracy_append.py` 与 `tuning/scripts/history_append.py`，**不得**自行改写字段。`expectations[].tolerance` 由 `evaluation/evaluation-cfg` 生成的 YAML `demand.expectations[].tolerance` 保证存在（默认 `0`）。

委派模板：

````markdown
```subagent-io v1

{
  "protocol": "subagent.subagent_io",
  "subagent_type": "evaluation/evaluate",
  "input": {
    "evaluate_config_path": "",
    "quant_model_path": "",
    "save_path": "",
    "device": "npu",
    "device_indices": [0, 1],
    "evaluate_id": "",
    "round": 1
  }
}

```text
````
