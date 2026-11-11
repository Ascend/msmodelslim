# C7 概念释疑

> 只解释“概念、模式、算法、格式、校准数据”等基础知识。
> 具体模型是否支持 → [C2](C2-feature-support.md)；怎么执行/转换 → [C3](C3-quantization-execution.md)。

## 1. 适用问题

- W8A8 / W4A8 / W4A4 / W8A8C8 / C8 / W8A8S / MXFP8 有什么区别；
- per-token / per-channel / per-group / per-tensor 是什么；
- 量化算法怎么选（GPTQ / AWQ / QuaRot / LAOS / SmoothQuant / AutoRound）；
- 校准集怎么选、多少条、长度如何覆盖；
- AscendV1 / compressed_tensors 等量化格式是什么；
- 为什么量化后权重体积没变小；
- 某模式的参数量/显存估算。

## 2. 不适用（转交）

| 问题 | 转到 |
|---|---|
| 某模型是否支持某模式 | [C2 能力边界/支持判定](C2-feature-support.md) |
| 命令/YAML 怎么写、权重怎么转换 | [C3 操作指引](C3-quantization-execution.md) |
| 量化报错 | [C4 故障排查](C4-error-troubleshooting.md) |
| 部署格式与引擎 | [C5 部署推理](C5-deployment.md) |
| 掉了多少点、怎么调优 | [C6 精度性能](C6-accuracy-performance.md) |

## 3. 必查证据

| 主题 | 路径 |
|---|---|
| 量化模式总览 | `docs/zh/knowledge_base/quantization_mode/README.md` |
| 线性层量化 | `docs/zh/knowledge_base/quantization_mode/linear_layer_quantization/README.md` |
| KV Cache 量化 | `docs/zh/knowledge_base/quantization_mode/kv_cache_quantization/README.md` |
| FA 量化 | `docs/zh/knowledge_base/quantization_mode/fa_quantization/README.md` |
| 算法总览 | `docs/zh/knowledge_base/quantization_algorithms/README.md` |
| 量化格式 | `docs/zh/knowledge_base/quantization_format/README.md` |
| AscendV1 | `docs/zh/knowledge_base/quantization_format/ascendv1/ascendv1_usage.md` |
| 一键量化 | `docs/zh/user_guide/usage_one_click_quantization.md` |

## 4. 核心概念

### 4.1 量化模式

| 模式 | 含义 | 关键点 |
|---|---|---|
| W8A8 | 权重 8bit、激活 8bit | 最常见；不是所有模型/引擎都支持 |
| W4A8 | 权重 4bit、激活 8bit | 压缩更高，精度更敏感 |
| W4A4 | 权重/激活均 4bit | 需要更强算法（如 LAOS/QuaRot） |
| W8A8C8 | W8A8 + KV Cache 8bit | C8 是 KV Cache，不是普通 W8A8 |
| W4A4C8 | W4A4 + KV Cache 8bit | 混合精度，配置更复杂 |
| C8 | KV Cache 8bit | 两条路径：KVCache Quant / FA3 Quant |
| W8A8S | 稀疏 + W8A8 | 需要引擎支持稀疏格式 |
| MXFP8 | 微缩放 FP8 | 受硬件/模型/引擎限制 |

### 4.2 粒度

| 粒度 | 含义 | 影响 |
|---|---|---|
| per-tensor | 整个张量一组 scale | 最粗，压缩高，精度风险大 |
| per-channel | 每个通道一组 scale | 权重常用 |
| per-token | 每个 token 一组 scale | 激活常用，动态范围适配好 |
| per-group | 每 N 个元素一组 scale | GPTQ/AWQ 常见，group size 影响精度 |

### 4.3 算法分类

按原理可分为训练后量化、旋转/变换、平滑/离群值抑制、混合低比特等几类。具体算法有哪些、支持哪些 `quant_type`、如何配置，以《[量化算法总览](../../../docs/zh/knowledge_base/quantization_algorithms/README.md)》与各 `usage_*.md` 为准，本 skill 不维护算法清单。

**算法支持 ≠ 模型有实践**：模型是否有已验证配方，回到 C2 判定。

### 4.4 校准数据

- 与业务同分布优先；不要只用通用语料；
- 覆盖实际输入长度（含长文本/多模态样本）；
- 去除重复、空样本、异常样本；
- 校准时 OOM 可截断/减少样本，或换大显存机器做最后校准；
- 具体条数/格式以对应算法文档与 `lab_practice` 为准。

### 4.5 为什么权重体积没变小

> “体积为何不变”在本 skill 中**只由本节解释**，C2/C3/C6 不再重复。

1. 原始权重可能本来就是低比特（FP8/INT8/AWQ），再量化不会继续减半；
2. 只量化了部分模块（如只量化 linear，不量化 embedding/lm_head）；
3. 保存格式包含额外 scale/描述文件，体积不是简单按 bit 数折算；
4. 先看原始 `config.json` 与量化产物 `quant_model_description.json` 再判断。


## 5. 回答骨架

```markdown
【结论】<概念/模式> 的含义是……；与 <相近概念> 的区别是……。
【依据】<量化模式/算法/格式文档路径>
【操作】<如需配置，给最小片段；纯概念题可省略>
【适用条件/风险】<硬件/引擎/精度影响；不要把它当成某模型已支持>
【下一步】确认模型支持 → feature-support；执行量化 → quantization-execution。
```

## 6. 常见错误

| 错误 | 纠正 |
|---|---|
| 把 C8 解释成普通 W8A8 | C8 是 KV Cache 8bit |
| 用概念解释代替模型支持判断 | 概念题答概念；是否支持回 C2 |
| 说“W4A4 一定比 W8A8 快” | 取决于硬件/引擎/模型，不能一概而论 |
| 说“量化后体积一定减半” | 看原始精度、量化范围、保存格式 |
| 推荐某算法但无模型实践 | 明确“算法能力存在，模型未验证” |
