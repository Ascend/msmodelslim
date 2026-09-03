---
name: quantization
description: 模型量化执行与导引。调优闭环中依据 Practice YAML 调用 msmodelslim quant 执行量化；直接使用时按「是否提供 YAML / 支持矩阵是否收录」导引到权重量化或一键量化。
license: Apache-2.0
metadata:
  version: 0.3.1
  domain: quantization
  framework: msmodelslim
  protocol: cli
  skill_class: tool
  aliases:

    - quantizer
    - quantization-run

  trigger_intents:

    - 执行量化
    - 运行 quantization_run
    - 量化模型

  keywords:

    - msmodelslim quant
    - quantize
    - practice.yaml

  # subagent 绑定声明：id = 委派标识（subagent_type）；bind = 委派时须绑定加载的 skill 根目录（含 references/、scripts/）
  # quantization 依赖 installation：环境/CLI 安装校验（msmodelslim quant --help）
  subagent:
    id: "quantization"
    bind:

      - "quantization"
      - "installation"

---

# 量化执行（quantization）

## 职责

在量化调优闭环的「量化配置调优」阶段，依据已生成的 Practice YAML 调用 `msmodelslim quant` 执行量化，回传产物路径与成败。

**不解决什么**：

- 不生成/修改 Practice YAML → `strategy/practice-cfg`
- 不执行评测 → `evaluation/evaluate`
- 不做策略决策 → `tuning`

**权威参考**（CLI 参数、命令格式、退出码等以 docs 为准，不在此重复）：

- `msmodelslim quant` 参数与命令 → 《[msmodelslim quant CLI](../../docs/zh/api_reference/cli/msmodelslim_quant.md)》
- 量化执行完整流程（预检、下载、适配、配置、量化、校验交付件）→ 《[权重量化使用指南](../../docs/zh/user_guide/usage_weight_quantization.md)》
- 一键量化（支持矩阵内已收录模型）→ 《[一键量化使用指南](../../docs/zh/user_guide/usage_one_click_quantization.md)》，仅导引不展开
- 支持矩阵 → 《[大模型支持矩阵](../../docs/zh/knowledge_base/model/README.md)》

## 使用导引（直接使用）

用户要求「量化模型」时，按「是否提供量化配置 YAML」导引，不自行展开 docs 已有流程：

| 用户输入 | 导引去向 |
|----------|----------|
| 已提供量化配置 YAML | 按配置执行量化，流程见《[权重量化使用指南](../../docs/zh/user_guide/usage_weight_quantization.md)》 |
| 未提供 YAML | 查《[大模型支持矩阵](../../docs/zh/knowledge_base/model/README.md)》：

  - 模型 + 量化模式**已收录 / 已验证** → 走《[一键量化使用指南](../../docs/zh/user_guide/usage_one_click_quantization.md)》（指定 `quant_type` 自动匹配最佳实践），**仅导引，不展开**；
  - 模型**未收录 / 未验证** → 需先适配与编写配置，走《[权重量化使用指南](../../docs/zh/user_guide/usage_weight_quantization.md)》 |

## 委派契约（SUBAGENT_IO v1）

编排层委派本 subagent 时，`input` 按下表填写；`commands` 与信封格式见 `tuning/references/subagent_io_protocol.md`。

| input 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `config_path` | string | ✓ | Practice YAML 路径（映射为 CLI `--config`） |
| `model_path` | string | ✓ | 原始模型路径 |
| `save_path` | string | ✓ | 产物路径，`{workdir}/round_{N}/quantized`（N 为本次量化序号） |
| `model_type` | string | ✓ | 模型类型名 |
| `device` | string | ✓ | 设备类型，取 `npu` / `cpu`（单值）；多卡经 `device_id` 传列表 |
| `trust_remote_code` | bool | | 默认 `false` |

执行 `msmodelslim quant`（`--config` 传 Practice，与 `--quant_type` 互斥；`--trust_remote_code` 默认 `false`）。

**回传 `output`**：`success`（bool）、`artifact_path`（量化产物目录）、`stderr`（失败时摘要）、`commands`。

## 执行步骤

1. **输入检查**：`config_path` / `model_path` / `save_path` / `model_type` / `device` 齐备且路径存在可读。
2. **执行**：调用 `msmodelslim quant`，参数按委派契约传入。
3. **结果处理**：
   - **成功判定**：exit code 为 0，且 `save_path` 下生成量化权重产物；
   - 失败 → 报命令名 + stderr 关键摘要，立即中止，不兜底续跑，等待编排层决策。

## 多卡写法（实证，实测 D2）

`msmodelslim quant --help` 权威形态：`--device {npu,cpu}` 为**单值枚举**，多卡通过 **`--device_id [<ID> ...]`** 传多个物理卡 ID（如 `--device npu --device_id 0 1 2 3`）。**`--device npu:0,1,2,3` 的形式在 `--help` 中未声明**，仅见于部分旧文档示例——执行前以 `msmodelslim quant --help` 实际输出为准，多卡一律用 `--device <type> --device_id <ids...>`。

## 错误处理

| 错误类型 | 处理 |
|----------|------|
| `msmodelslim` 未安装 | 按 installation skill 安装后重试（已绑定） |
| 路径不存在 / 配置解析失败 | 检查路径与 YAML 后重试或中止 |
| 量化失败 | 报 stderr 摘要，停止并等待编排层决策 |
| 超时 | 按 Agent execution_timeout 处理，不上层续跑 |

## 约束与磁盘管理

- **错误即停**：命令失败后立即中止，不兜底续跑；**单轮单次**：每次调用只执行一次量化。
- **产物命名**：`{workdir}/round_{N}/quantized`，N 为本次量化序号，便于编排层定位各次产物。
- **磁盘**：量化产物写入 `save_path`；磁盘空间由编排层管理（最多保留 2份完整权重），本 skill 不主动清理历史产物。

## 检查清单（执行前）

- [ ] `config_path` 指向的 Practice YAML 已通过校验
- [ ] `device` 格式正确：`--device {npu,cpu}` 单值 + 多卡 `--device_id <ids...>`（勿用 `--device npu:0,1,2,3` 未声明形态）
- [ ] `save_path` 为 `{workdir}/round_{N}/quantized` 形式且磁盘空间充足
- [ ] `msmodelslim quant --help` 可正常执行

## 被 `tuning` 编排

调优编排委派本 Skill，按上述委派契约执行，不参与配置生成、评测或策略决策。

## 经验条目（Experiences，追加制）

> 追加规范见 `skills/README.md`「经验条目」。连续编号 `[E-序号]`；正文保留权威展开，本表只做索引 + 元数据登记；来源：实测编号（F0-x / Dx / Tx）| 用户反馈 | 代码实证；验证状态：已回归 | 待验证 | 已上流 docs。

| 条目 | 主题 | 适用条件 / 触发信号 | 结论要点（一句话） | 正文位置 | 来源 | 验证状态 |
|------|------|------|------|------|------|------|
| E-001 | 多卡 CLI 写法 | 使用 ≥2 张卡量化 | `--device {npu,cpu}` 是单值枚举，多卡走 `--device_id [<ID> ...]`（如 `--device npu --device_id 0 1 2 3`）；`--device npu:0,1,2,3` 未在 `--help` 声明，仅见旧文档，执行前以 `--help` 为准 | 多卡写法 | 实测 D2 | 已回归 |
| E-002 | 单轮单次 + 错误即停 | 任意一次量化调用 | 每次调用只执行一次量化；失败报 stderr 摘要即停，不兜底续跑、不换未文档化命令规避，交由编排层决策 | 约束与磁盘管理 | 用户反馈 | 已回归 |
