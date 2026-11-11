# C2 能力边界 / 支持判定

> 只回答“支持 / 不支持 / 条件支持”，以及支持到哪一档；包括权重格式/精度转换是否支持。
> 命令怎么执行、权重转换怎么做见 [C3 操作指引](C3-quantization-execution.md)；报错见 [C4](C4-error-troubleshooting.md)；概念解释见 [C7](C7-quantization-basics.md)。

## 1. 适用问题

- 某模型是否支持量化、支持哪些量化模式；
- W8A8 / W4A8 / W4A4 / W8A8C8 / W4A4C8 / C8 / W8A8S / MXFP8 是否支持；
- 某算法（GPTQ / AWQ / QuaRot / LAOS / SmoothQuant / AutoRound）是否支持；
- C8 / KV Cache 是否支持、走哪条实现路径；
- 未收录模型能否复用同族适配器；
- 某模型某模式是否“官方验证 / 仅适配器 / 仅示例 / 不支持”；
- FP8/FP4→BF16、INT4 反量化、原生 AWQ 迁移等权重转换是否支持。

## 2. 不适用（转交）

| 问题 | 转到 |
|---|---|
| 怎么给命令、YAML 怎么写、新模型怎么接 | [C3 操作指引](C3-quantization-execution.md) |
| 权重格式/精度转换怎么做、转换脚本与字段 rename | [C3 操作指引](C3-quantization-execution.md) |
| 报错根因 | [C4 故障排查](C4-error-troubleshooting.md) |
| 量化后能否在 vLLM/MindIE 拉起 | [C5 部署推理](C5-deployment.md) |
| 量化模式/概念区别、体积为何不变 | [C7 概念释疑](C7-quantization-basics.md) |

## 3. 必查证据（按顺序）

| 顺序 | 资料 | 用途 |
|---|---|---|
| 1 | `docs/zh/knowledge_base/model/README.md` | 模型 + quant_type 是否登记“一键量化” |
| 2 | `config/config.ini` 的 `[ModelAdapter]` | `model_type` 是否注册、对应哪个 adapter |
| 3 | `lab_practice/**/*.yaml` | `verified_model_types` / `verified_tags` / `label` 是否有匹配实践 |
| 4 | `example/<family>/README.md` | 模型版本、量化模式、依赖、注意事项 |
| 5 | `msmodelslim/model/<adapter>/` | 是否实现该模式所需接口（如 KV/FA3 接口） |
| 6 | `docs/zh/knowledge_base/quantization_algorithms/README.md` 与各 `usage_*.md` | 算法支持与最小配置 |
| 7 | `docs/zh/knowledge_base/ptq/convert/usage_weight_conversion.md` | 权重格式/精度转换的支持范围与示例 |

> 结论按**最保守**的一档输出；证据冲突时说明差异。

## 4. 结论枚举

| 枚举 | 判定条件 | 回答要求 |
|---|---|---|
| `OFFICIAL_ONE_CLICK` | 矩阵标记“一键量化”，且有匹配的已验证实践 | 给一键命令；注明硬件/引擎 tag |
| `OFFICIAL_PRACTICE` | `lab_practice` 的 `verified_model_types`/`verified_tags` 命中，但未登记一键 | 给 `--config` 或 example 命令 |
| `ADAPTER_ONLY` | `config.ini` 注册了模型，但无匹配实践 | 标明“可尝试、未验证”；给同族 YAML 起点 |
| `LISTED_BUT_NO_PRACTICE` | 矩阵列出模型，但目标模式无实践 | 标明未验证/需自行配置 |
| `UNVERIFIED_TRY` | 仅在 example/issue/代码中可见，无矩阵/实践 | 明确“未官方验证” |
| `UNSUPPORTED` | 无 adapter、无矩阵、无实践 | 明示暂不支持 + 新模型流程 |

**冲突处理**：

- 矩阵有一键量化，但 `lab_practice` 无对应 `quant_type` → 先按矩阵，再提示以实际匹配结果为准；
- `config.ini` 有 adapter、无实践 → `ADAPTER_ONLY`；
- 支持矩阵未列出、example 有命令 → `UNVERIFIED_TRY`，说明来源是 example 而非官方矩阵。

### 权重格式 / 精度转换的判定

1. 先确认源格式/精度：看模型 `config.json`、量化产物 `quant_model_description.json`、`example/<family>/README.md`；
2. 查 `usage_weight_conversion.md` 是否有该模型/格式的转换示例与字段说明；
3. 无示例但有通用脚本 → `UNVERIFIED_TRY`；文档明确不支持 → `UNSUPPORTED`。

稳定判定口径（不在 skill 维护逐模型清单，具体以文档为准）：

- BF16/FP16 → 低比特属于**量化**，不是“转换”，走 C3 量化流程；
- FP8 → BF16 需按转换指南处理 `.scale` / `.weight_scale_inv` 等字段；
- FP4 → FP16/BF16 多数场景不支持，必须先查文档再回答；
- INT4/AWQ → BF16 需先反量化为 BF16，再走量化；**原生 AWQ 不能直接作为量化输入**；
- 源权重本来就是低比特时，转换/量化后体积不变属预期，解释归 C7。

## 5. quant_type 与 label 匹配（协议，不维护映射表）

`quant_type` 与 YAML `label` 的对应关系以 `docs/zh/api_reference/config/quant/modelslim_v1.md` 和 `lab_practice` 实际 YAML 为准，本 skill 不维护静态映射表。

匹配时按以下顺序：

1. 读目标 YAML 的 `label`（`w_bit` / `a_bit` / `is_sparse` / `kv_cache` / `fa_quant`）；
2. 与目标 `quant_type` 的语义逐项比对，字段不相等即不匹配；
3. `label` 缺失时，按 `quant_type` 匹配**不会命中**（匹配逻辑要求 label 字段相等），需用 `--config` 显式指定或补全 `label`；不要声称存在自动推断机制；
4. 无法判断时，以 `example` 命令或 `lab_practice` 已验证 YAML 为准。

## 6. KV Cache 8bit（C8）判定

1. 确认模型是否有 KV Cache 实践（`label.kv_cache=True` 或 `quant_type` 含 c8）；
2. 确认实现路径：
   - `dynamic_cache` → KVCache Quant；
   - `fa3_quant` → FA3 Quant；
3. 确认适配器是否实现对应接口：
   - `dynamic_cache` 需要 KV 量化/平滑接口；
   - `fa3_quant` 需要 `FA3QuantAdapterInterface`；
4. 确认推理引擎是否支持该格式；
5. 四条都满足才回答“支持”，否则“条件支持 / 需适配”。

> C8 指 KV Cache 8bit，不等同于普通 W8A8。回答必须写清：**哪条路径 + 哪个 YAML + 哪个部署引擎**。

## 7. 算法能力判定（协议，不维护算法清单）

算法列表与 `quant_type` 映射会随版本变化，本 skill 不维护逐算法表。判定顺序：

1. 打开 `docs/zh/knowledge_base/quantization_algorithms/README.md` 与对应 `usage_*.md`，确认算法能力与配置字段；
2. 查 `lab_practice` 是否有该模型的已验证实践；算法文档存在不等于该模型有实践；
3. 按 §4 的结论档位输出，并给出对应算法文档的真实路径。

> 不要凭记忆罗列算法支持；不要用静态表替代文档检索。


## 8. 参考文件索引

| 主题 | 路径 |
|---|---|
| 大模型支持矩阵 | `docs/zh/knowledge_base/model/README.md` |
| 模型注册表 | `config/config.ini` |
| 最佳实践 | `lab_practice/**/*.yaml` |
| 模型示例 | `example/<family>/README.md` |
| 新模型流程 | `docs/zh/user_guide/process_new_model_quantization_tuning.md` |
| 算法总览 | `docs/zh/knowledge_base/quantization_algorithms/README.md` |
| 量化模式 | `docs/zh/knowledge_base/quantization_mode/` |
| KV Cache | `docs/zh/knowledge_base/quantization_algorithms/kvcache_quant/usage_kvcache_quant.md` |
| FA3 | `docs/zh/knowledge_base/quantization_algorithms/fa3_quant/usage_fa3_quant.md` |
| 权重转换 | `docs/zh/knowledge_base/ptq/convert/usage_weight_conversion.md` |

## 9. 回答骨架

**官方一键**：

```markdown
【结论】支持，且已登记一键量化。
【依据】支持矩阵 + `lab_practice/...yaml`。
【操作】`msmodelslim quant ... --quant_type <type>`
【适用条件/风险】transformers 版本；部署引擎 tag；硬件。
【下一步】执行量化 → quantization。
```

**仅适配器**：

```markdown
【结论】仓库注册了该模型适配器，但没有已验证实践，属于“可尝试、未官方验证”。
【依据】`config.ini` 有 model_type；`lab_practice` 无匹配 verified_model_types。
【操作】从同族/ default YAML 起步，或按新模型流程调优。
【适用条件/风险】精度/部署兼容性不保证。
【下一步】适配 → adaptation；调优 → strategy/tuning。
```

**未支持**：

```markdown
【结论】当前仓库未收录该模型/模式。
【依据】支持矩阵、`config.ini`、`lab_practice` 均未命中。
【操作】参考《新模型量化调优流程》适配与量化。
【适用条件/风险】未验证，不要硬套其他模型 YAML。
【下一步】适配 → adaptation。
```

## 10. 常见错误

| 错误 | 纠正 |
|---|---|
| 只看 `config.ini` 有 adapter 就说支持 | 补查矩阵 + `lab_practice` + example，按最保守档位 |
| 把 C8 当普通 W8A8 | 按 KV Cache / FA3 两条路径判定 |
| 算法文档存在就说模型支持 | 算法能力 ≠ 模型有实践 |
| 矩阵有一键但目标模式无 practice | 标明冲突，按矩阵 + 实际匹配结果回答 |
| 未收录模型直接套同族 YAML | 明确“未验证”，先看结构是否一致 |
| 声称 `label` 缺失时可自动推断 | 按 `quant_type` 不会命中；用 `--config` 或补全 `label` |
| 用固定算法清单回答支持性 | 以算法总览与 `usage_*.md` 为准 |
| 把权重转换说成量化（或反之） | BF16→低比特是量化；格式/精度互转才是转换 |
