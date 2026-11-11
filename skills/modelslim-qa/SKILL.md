---
name: modelslim-qa
description: |
  仅限用户显式主动调用（如 /modelslim-qa），不得由模型自动触发或由其他 skill 流程加载。
  用于回答 msModelSlim 的客户咨询类问题：先按用户意图判定分类（安装环境、能力边界/支持判定、
  操作指引、故障排查、部署推理、精度性能、概念释疑），再加载对应分类 reference，
  输出结论、依据、操作、适用条件与风险、下一步；资料缺失时声明“资料暂未覆盖”并给出谨慎结论。
  需要执行量化/调优时转交对应 skill。
license: Apache-2.0
---

# msModelSlim 问答（modelslim-qa）

## 调用约束（必须遵守）

- 本 skill **仅供用户显式主动调用**，不得由模型在执行其他 skill 流程时自动加载、自动路由或作为旁路 context 引入。
- 安装、适配、量化、调优、评测等流程中遇到相似问题，仍按原流程处理，**不读取本 skill 的 reference**，避免影响原有流程。
- 本 skill **只回答客户咨询问题**，不承担执行类任务；需要执行时按 §6 转交对应 skill。

## 概述

客户咨询问答场景的**分类器与路由器**。职责只有三件事：

1. 从用户问题中抽取实体，判定问题属于哪个**分类**；
2. 只加载该分类对应的 reference，按其中“必查证据 → 结论形态 → 回答要点”作答；
3. 资料未覆盖时声明“资料暂未覆盖”并给出谨慎结论；需要执行动作时转交对应 skill。

```text
问题 → 分类 → 分类 reference → 权威证据 → 结论/操作/风险/下一步
```

**不解决什么**：

- 不执行量化 → `quantization`
- 不安装环境 → `installation`
- 不写模型适配器 → `adaptation`
- 不做精度调优策略/回退搜索 → `strategy`、`sensitive-layer-analysis`、`tuning`
- 不实现评测 → `evaluation`
- 不替代 `docs/`：docs 是权威，本 skill 只做分类、索引、判定和增量经验

---

## 1. 分类与路由

分类**按用户意图收敛**，不按主题单列：权重转换按“是否支持 / 怎么做”分别归入 C2 / C3；安全问题不设独立分类，作为跨分类红线由 §8 统一约束。

先定**主分类**，再读对应 reference。一次只读一个主 reference；只有主 reference 明确要求时才读第二个。

| 分类 | 判定信号 | 参考文件 | 典型结论 |
|---|---|---|---|
| **C1 安装环境** | 装不上、依赖冲突、CANN/torch_npu/transformers/Python 版本、镜像、SoC/算子、310P/950 平台 | [C1-installation-environment.md](references/C1-installation-environment.md) | 版本/镜像/自检命令 |
| **C2 能力边界 / 支持判定** | 模型/模式/算法是否支持；C8/KV Cache；FP8/FP4→BF16、INT4 反量化、AWQ 迁移是否支持 | [C2-feature-support.md](references/C2-feature-support.md) | 官方一键/官方实践/仅适配器/未验证/不支持 |
| **C3 操作指引** | 怎么量化、一键命令、YAML/CLI 参数、新模型怎么接、V0→V1、权重格式转换怎么做 | [C3-quantization-execution.md](references/C3-quantization-execution.md) | 命令或流程 |
| **C4 故障排查** | traceback、ImportError、OOM、config 不存在、算子错误、state_dict 不匹配 | [C4-error-troubleshooting.md](references/C4-error-troubleshooting.md) | 根因 + 修复 |
| **C5 部署推理** | vLLM-Ascend/MindIE 拉不起、服务初始化失败、量化格式与引擎不匹配、W8A8S 部署 | [C5-deployment.md](references/C5-deployment.md) | 格式/引擎/版本匹配结论 |
| **C6 精度性能** | 掉点、乱码、重复/截断、余弦相似度、BFCL/GPQA、无性能收益、吞吐下降 | [C6-accuracy-performance.md](references/C6-accuracy-performance.md) | 调优路径 / 引擎适配建议 |
| **C7 概念释疑** | 概念、量化模式区别、per-token/per-channel、算法选择、校准数据、量化格式、体积为何不变 | [C7-quantization-basics.md](references/C7-quantization-basics.md) | 解释 + 权威文档 |

> 各分类对应的 FAQ 文档（`docs/zh/support/faq_*.md`）见 §11「FAQ 文档」；检索时先查分类 reference，再按需查对应 FAQ。

### 分类判定优先级

1. **有完整报错/traceback** → C4；报错里若明确是版本/镜像问题，C4 的 reference 会转到 C1。脚本报错同样先 C4，是否属 V0 由 C4 内部转 C3。
2. **问“是否支持/有没有”** → C2；即使提到报错，只要核心是确认能力，先 C2。权重转换“是否支持”也归 C2。
3. **问“怎么做/命令/YAML/转换步骤”** → C3；若模型本身未收录，C3 会先指向 C2 的判定结果。
4. **量化已完成，问题在拉起/推理** → C5。
5. **能跑但效果差/没收益** → C6；无报错，不要放 C4。
6. **问概念/模式区别/为什么（含“体积为何没变小”）** → C7。
7. **涉及不可信代码/漏洞/凭据** → 不进入分类路由，按 §8 安全边界与红线处理；需要排障时回 C4。

> 禁止把“多个分类的内容”都塞进回答。先用主分类给结论；需要第二个分类时，在回答中明确“该部分转交/参见”。

---

## 2. 流程

```text
① 抽取实体：模型名、quant_type、硬件（A2/A3/310P/950）、CANN/torch/transformers、
   推理引擎、报错文本、用户最终目标（确认支持 / 执行 / 排错 / 部署 / 调优）
② 判定主分类 C1–C7；低置信度时只问一个澄清问题
③ 打开对应 reference，按“必查证据”顺序检索；不跨 reference 拼结论
④ 组织回答：结论 → 依据 → 操作 → 适用条件与风险 → 下一步
⑤ 资料未命中：声明“资料暂未覆盖”，给谨慎结论；反复出现时补 FAQ
⑥ 需要执行/调优：转交对应 skill
```

### 澄清原则

只在“不同答案会导致不同操作”时提问，且一次只问一个：

| 场景 | 必问 | 原因 |
|---|---|---|
| 模型名模糊 | 具体模型名/参数量/是否含 MTP | 适配器与实践不同 |
| 量化模式模糊 | W4A8C8 还是 W4A4C8；是否要 KV Cache | C8 有两条实现路径 |
| 硬件不明 | A2/A3/310P/950 + CANN 版本 | 支持矩阵/算子差异大 |
| 报错不完整 | 完整命令 + traceback 最后50行 | 仅错误码无法定位 |
| 版本不明 | transformers/torch/torch_npu/Python | 多数 ImportError 是版本问题 |
| 部署引擎不明 | vLLM-Ascend 还是 MindIE + 版本 | 量化格式支持不同 |

---

## 3. 输入 / 输出

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `question` | string | ✓ | 用户原始问题 |
| `category` | enum | ✓ | C1–C7 |
| `model_name` | string | | 从问题中抽取 |
| `quant_type` | string | | W8A8/W4A8/W4A4/C8/W8A8S/MXFP8 等 |
| `hardware` | string | | A2/A3/310P/950 等 |
| `engine` | string | | vLLM-Ascend/MindIE + 版本 |
| `error_text` | string | | 报错原文或 traceback |
| `evidence` | list[path] | ✓ | 命中的权威资料/代码位置 |
| `answer` | markdown | ✓ | 五段式回答 |
| `fallback` | string | | 资料未覆盖时的谨慎结论 |

---

## 4. 回答规范

固定五段：

```text
【结论】支持 / 不支持 / 条件支持 / 资料暂未覆盖（一句话）
【依据】支持矩阵条目、lab_practice、docs、代码位置
【操作】最小命令或 YAML 片段
【适用条件/风险】硬件、CANN/torch/transformers、推理引擎、是否官方验证
【下一步】需要执行→转交 skill；信息不足→需要哪些日志
```

**必须做**：

- 先分类再回答，依据来自主分类 reference 的“必查证据”；
- 至少给1个真实存在的权威路径；
- 区分“已注册 / 有一键 / 有实践 / 已验证 / 可部署”；
- 环境类问题给版本自检命令；
- 未验证路径明确标注；
- 需要执行时给出明确的 skill 转交建议。

**不得做**：

- 不得把 V0 脚本量化作为新方案推荐；
- 不得建议在 msmodelslim 源码根目录执行量化；
- 不得只凭 `config.ini` 有 adapter 就宣称官方支持；
- 不得编造不存在的 YAML、CLI 参数、模型支持；
- 不得在未确认推理引擎时承诺量化权重可部署；
- 不得复现安全漏洞细节。

---

## 5. 资料未覆盖与 FAQ 补齐

本 skill **不维护缺口台账**。检索未命中时按以下方式处理：

1. 已按主分类 reference 的“必查证据”完整检索，`docs/`、FAQ、`example/` 仍无可执行结论 → 在回答中声明“资料暂未覆盖”，给基于代码/配置的谨慎结论，并明确标注“未经官方验证”；
2. 若同一问题反复出现（≥2次）或客户确认是通用问题 → 用 `docs-management` skill 在 `docs/zh/support/faq_*.md` 补 FAQ；
3. FAQ 保持四段格式：问题现象 / 问题原因 / 解决方法 / 关联文档；
4. 补完后同步分类 reference 的证据索引。

> 支持矩阵未收录、仅适配器、无 `lab_practice`、通用安装/报错自检等，按 C2/C3/C4 的既有协议直接回答，不视为“资料未覆盖”。

---

## 6. 与其他 Skill 协同

| 后续动作 | 转交 |
|---|---|
| 安装环境修复 | [installation](../installation/SKILL.md) |
| 新模型适配 | [adaptation](../adaptation/SKILL.md) |
| 执行量化 | [quantization](../quantization/SKILL.md) |
| 精度调优/回退 | [strategy](../strategy/SKILL.md)、[sensitive-layer-analysis](../sensitive-layer-analysis/SKILL.md)、[tuning](../tuning/SKILL.md) |
| 评测 | [evaluation](../evaluation/SKILL.md) |
| 补 FAQ/文档 | [docs-management](../docs-management/SKILL.md) |

---

## 7. 硬门禁

回答前必须满足：

- 未读取主分类 reference 的“必查证据”，不得给出结论；
- 支持类问题未查支持矩阵、`config.ini`、`lab_practice` 前，不得回答“支持/不支持”；
- 环境类问题未给出至少一条版本自检命令，不得结束回答；
- 资料未命中未声明“资料暂未覆盖”并给出谨慎结论，不得结束回答；
- 回答中引用的 YAML/文档路径必须真实存在。

---

## 8. 红线

贯穿全流程的禁止行为：

- 不得把 `msmodelslim/pytorch/llm_ptq` 等 V0 脚本作为新方案推荐；
- 不得建议用户在 msmodelslim 源码根目录执行量化命令；
- 不得把“有 adapter”等同于“官方验证”；
- 不得在未确认推理引擎支持时承诺量化格式可部署；
- 不得编造支持矩阵条目、最佳实践 YAML 或 CLI 参数。

### 安全边界（跨分类）

- **不可信权重**：不执行来源不明模型目录中的 Python 代码；不得为了“先跑通”直接开启 `trust_remote_code`。
- **必须开启时**：先确认模型官方资料明确要求、来源可信，并在隔离环境（容器/最小权限/无网络）中执行。
- **漏洞披露**：只描述影响与缓解，不复现利用细节；按 `docs/zh/legal/SECURITY.md` 的渠道私密上报，不在公开 issue 贴 POC。
- **信息收集脱敏**：要求客户提供命令、traceback、版本与配置前，先提示删除令牌、Cookie、URL 凭据、私网 IP、内部主机名和用户路径，只保留最小必要日志。

---

## 9. 检查清单

- [ ] 已抽取模型名、quant_type、硬件、引擎、版本、报错等实体
- [ ] 已判定主分类 C1–C7，且只读了对应 reference
- [ ] 已按该 reference 的“必查证据”顺序检索
- [ ] 环境类问题已给出或执行版本自检命令
- [ ] 回答含结论、依据、操作、风险、下一步
- [ ] 未验证路径已标注
- [ ] 资料未覆盖时已声明并给出谨慎结论
- [ ] 需要执行时已给出 skill 转交

## 10. 通过/失败标准

- **通过** = 分类正确、结论有依据、操作可执行、风险已说明、资料未覆盖已声明。
- **失败** = 分类错误导致答非所问、无依据断言、推荐 V0、建议源码目录运行、把未验证说成已验证。

---

## 11. 参考资料

**分类 reference（按用户意图只读一个）**：

- [C1 安装环境](references/C1-installation-environment.md)
- [C2 能力边界/支持判定](references/C2-feature-support.md)
- [C3 操作指引](references/C3-quantization-execution.md)
- [C4 故障排查](references/C4-error-troubleshooting.md)
- [C5 部署推理](references/C5-deployment.md)
- [C6 精度性能](references/C6-accuracy-performance.md)
- [C7 概念释疑](references/C7-quantization-basics.md)

**FAQ 文档（`docs/zh/support/`，按分类索引）**：

| 对应分类 | FAQ 文档 | 覆盖问题 |
|---|---|---|
| 总览 | [faq.md](../../docs/zh/support/faq.md) | FAQ 统一索引与检索指引 |
| C1 安装环境 | [faq_installation.md](../../docs/zh/support/faq_installation.md) | 安装与依赖、pydantic / accelerate 冲突 |
| C1 安装环境 / C4 故障排查 | [faq_runtime.md](../../docs/zh/support/faq_runtime.md) | 进程 Killed、显存/内存不足（OOM） |
| C1 安装环境 | [faq_hardware.md](../../docs/zh/support/faq_hardware.md) | A3 SoC、310P/300I 算子与 BF16 限制、aclnn/kernel 自检 |
| C1 安装环境 / C4 故障排查 | [faq_warning.md](../../docs/zh/support/faq_warning.md) | DeprecationWarning 等依赖告警 |
| C6 精度性能 | [faq_precision.md](../../docs/zh/support/faq_precision.md) | VLM 敏感层排查、精度调优与异常定位 |
| C3 操作指引 / C5 部署推理 | [faq_deployment.md](../../docs/zh/support/faq_deployment.md) | FP8/INT4→BF16、BF16→MXFP8 权重转换、引擎匹配 |

> C2、C7 暂无专属 FAQ 文档，按对应分类 reference 的“必查证据”检索；新增 FAQ 时同步更新本表与 `docs/zh/support/faq.md` 索引。

---

## 12. 维护约定

1. **新增分类**：同步更新 §1 分类表、§3 `category` 枚举、§11 参考资料列表，再新建 `references/<category>.md`；不要往现有分类里硬塞。
2. **新增经验**：写入所属分类 reference 的对应小节；模型/版本个案结论不写入 skill，归支持矩阵或 FAQ。
3. **新增证据**：写入所属分类 reference 的“必查证据”；跨分类的通用入口只保留一份，其他分类引用路径，不复制正文。
4. **删除/合并分类**：同步更新 §1、§3、§11 与 FAQ 索引的分类列。
5. **docs 更新后**：把分类 reference 中“临时结论”改为“以 docs 为准”，并在经验规则表更新状态。
