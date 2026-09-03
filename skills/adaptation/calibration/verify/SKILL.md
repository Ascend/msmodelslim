---
name: verify
description: 为 msModelSlim 适配器执行功能性验证。语义为减层结构覆盖（非全量）：Step1 按结构类型生成覆盖完备的最小前缀测试模型，再跑全回退量化、权重一致性、W8A8 描述校验四步门禁。
metadata:
  # subagent 绑定声明：id = 委派标识（subagent_type）；bind = 委派时须绑定加载的 skill 根目录（含 references/、scripts/）
  # 双重角色：① calibration 子任务的内部门禁（适配完成时自带四步验证）；② 主 Agent 的适配验收门禁（NPU 前向推理判定基准）
  subagent:
    id: "adaptation/calibration/verify"
    bind:

      - "adaptation/calibration/verify"

---

# 适配器功能性验证（verify）

用于在基础适配器开发完成后，自动帮助用户进行功能性验证。

## 验证语义（硬约定）

- **减层结构覆盖验证，不是全量验证。**
- 大模型由若干**结构类型**（如 `attn:linear_attention` / `attn:full_attention` × `ffn:dense` / `ffn:moe`）堆叠而成；验证只需每种结构至少出现一次。
- Step1 默认取「覆盖完备的**最小前缀层数**」生成随机权重测试模型；**禁止**盲目前 N 层（会漏 MoE、full_attn 等）。
- **不要求**加载或量化原始全层权重；全量精度/性能属于后续量化调优，不在本 skill 范围。

## 触发条件

- `adaptation/calibration` 已完成适配器开发与注册安装。
- 用户希望确认适配器是否可用，或要求执行标准验证流程。

## 执行要求

- 必须按顺序执行四步验证，不可跳步。
- 每一步失败都要立即停止并返回失败原因与下一步修复建议。
- 仅当四步全部通过时，返回“功能性验证通过”。
- 修改代码后要执行 `bash install.sh` 重新安装msModelslim
- 若验证过程中出现模型实现文件（如权重目录内 `modeling_*.py`）报错，必须先判断是否为 `transformers` 版本不契合导致。
- 对疑似版本不契合问题，必须先告知用户并确认版本需求（目标版本或可接受版本区间）；未确认前不得切换版本。
- 仅在用户确认后，才可执行 `transformers` 版本切换与重试验证。

## 四步验证流程

1. Step1：按结构覆盖计划生成减层随机权重测试模型（见下方硬门禁）。
2. Step2：执行全回退量化，验证流程与注册生效。
3. Step3：验证 Step2 与 Step1 的权重严格一致，且产物可完整加载/保存。
4. Step4：执行实际量化（W8A8 静态/动态）并校验描述文件规则。

## 硬门禁：Step1 结构覆盖

- 执行前可用 `--plan-only` 打印覆盖计划；正式生成须落盘 `structure_cover_plan.json`。
- 覆盖完备：`incomplete=false`，且日志中每种结构标签均有「首次覆盖 layer[i]」。
- 若人为传入 `--num-layers` 导致漏盖 → **FAIL**（除非显式 `--allow-incomplete-cover`，仅调试）。
- 常见漏盖：MoE 从 layer≥K 才开始、hybrid attn 的 full 层按 interval 出现——前缀必须延伸到首次出现该类结构的层。

## Buffer 权重说明（Step3 常见问题）

- 若 Step3 出现“全回退权重缺失/键不一致”，需优先检查缺失项是否来自模型 `buffer`。
- `msmodelslim` 通常不会保存 `buffer` 类型权重，因此可能导致全回退产物缺少对应键。
- 适配器需要主动将这类关键 `buffer` 转为 `nn.Parameter`，以确保量化导出和一致性校验可覆盖该权重。

## transformers 版本兼容处理（验证期）

- 触发条件：验证阶段出现模型实现文件导入/模型forward错误，且报错指向 `transformers` API 变更、缺失符号或签名不匹配。
- 必做沟通：向用户说明“当前报错疑似版本兼容问题”、给出关键报错摘要、请求确认目标版本策略（指定版本或版本区间）。
- 搜索策略：获用户确认后，使用二分法在确认范围内搜索可用 `transformers` 版本（每次切换版本后需重装并重跑触发失败的验证步骤）。
- 收敛标准：找到“可成功加载并通过对应验证步骤”的版本后停止搜索，并将最终版本写入msModelslim的config.ini。
- 失败处理：若二分搜索后仍无可用版本，返回阻塞结论并要求用户提供官方建议版本或模型实现修订方案。

## 自动化脚本

- `scripts/step1_generate_test_model.py`
- `scripts/step2_run_quantization.py`
- `scripts/step3_verify_weights.py`
- `scripts/step4_verify_quant_description.py`

## 参考资料

- [适配器验证指南](references/verification_guide.md)

## 输出格式要求

- 给出每一步的执行结果（PASS/FAIL）。
- 若失败，标注失败步骤、错误要点、建议修复方向。
- 最后给出总结结论：通过 / 未通过。

## 经验条目（Experiences，追加制）

> 追加规范见 `skills/README.md`「经验条目」。连续编号 `[E-序号]`；正文保留权威展开，本表只做索引 + 元数据登记；来源：实测编号（F0-x / Dx / Tx）| 用户反馈 | 代码实证；验证状态：已回归 | 待验证 | 已上流 docs。

| 条目 | 主题 | 适用条件 / 触发信号 | 结论要点（一句话） | 正文位置 | 来源 | 验证状态 |
|------|------|------|------|------|------|------|
| E-001 | buffer 缺失优先自查 | Step3 权重缺失/键不一致 | 缺失项多来自模型 `buffer`（msmodelslim 通常不保存 buffer）；适配器须将关键 buffer 主动转为 `nn.Parameter` 使其可被导出与校验覆盖 | Buffer 权重说明 | 代码实证 | 已回归 |
| E-002 | tie 权重等价克隆豁免 | Step3 严格键集合比对 | tie 权重模型（如 Qwen3）量化产物会克隆 `lm_head.weight = embed_tokens.weight`，属等价克隆而非真不一致；比对前先判 shape+数值容差内相等再豁免 | Step3（脚本 `step3_verify_weights.py`） | 实测 G6 | 已回归 |
| E-003 | transformers 版本兼容二分 | 验证期模型实现报错疑似版本不契合 | 先向用户说明并确认版本策略（目标版本或区间），确认后才二分搜索可用版本；收敛后把最终版本写入 config.ini；无可用版本返回阻塞并要官方建议 | transformers 版本兼容处理 | 用户反馈 | 待验证 |
| E-004 | 修改代码后重装 | 任何 adapter 代码变更后 | 必须执行 `bash install.sh` 重新安装 msmodelslim 后再重试验证（脚本路径为 console 命令，非 `python -m msmodelslim`） | 执行要求 | 实测 G5 | 已回归 |
| E-005 | 减层结构覆盖非全量 | Step1 生成测试模型 | 验证=结构覆盖完备的最小前缀减层，不是全量、也不是盲目前 N 层；MoE/hybrid 等晚出现结构必须被前缀覆盖 | 验证语义 / 硬门禁：Step1 结构覆盖 | 用户反馈 | 待验证 |
| E-006 | 禁止盲目前 N 层 | `--num-layers` 过小 | 若上限导致 `incomplete`，step1 须 FAIL；扩大前缀或去掉上限，勿用 `--allow-incomplete-cover` 混过门禁 | 硬门禁：Step1 结构覆盖 | 用户反馈 | 待验证 |
| E-007 | remote tied_weights list | Step1 `save_pretrained` AttributeError keys | 旧式 remote modeling 的 `_tied_weights_keys` 可能是 list；测试模型可清空为 `{}` 再保存 | Step1 脚本 | 实测 | 已回归 |
