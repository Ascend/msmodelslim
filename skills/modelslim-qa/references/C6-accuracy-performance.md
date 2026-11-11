# C6 精度性能

> 只处理“能跑，但结果不对或没有收益”。
> 有报错 → [C4](C4-error-troubleshooting.md)；部署拉不起 → [C5](C5-deployment.md)。

## 1. 适用问题

- 量化后精度掉点、输出乱码、重复、截断；
- 余弦相似度 / BFCL / GPQA / 业务数据集指标下降；
- 量化后没有性能收益、吞吐下降、时延变差；
- 310/310P 精度不保证的说明；
- 需要判断“是量化问题，还是评测/部署问题”。

## 2. 不适用（转交）

| 问题 | 转到 |
|---|---|
| 有 traceback / 报错 | [C4 故障排查](C4-error-troubleshooting.md) |
| 模型/模式是否支持 | [C2 能力边界/支持判定](C2-feature-support.md) |
| 服务拉不起、格式不匹配 | [C5 部署推理](C5-deployment.md) |
| 权重体积没有变小 | [C7 概念释疑](C7-quantization-basics.md) |
| 需要自动搜索/敏感层分析 | `tuning`、`sensitive-layer-analysis` skill |

## 3. 必查证据（按顺序）

| 顺序 | 资料 | 用途 |
|---|---|---|
| 1 | `docs/zh/user_guide/process_quantization_precision_tuning.md` | 精度调优总路径 |
| 2 | `docs/zh/user_guide/process_quantization_accuracy_anomaly_locating.md` | 异常定位流程 |
| 3 | `docs/zh/support/faq_precision.md` | 精度 FAQ（如为预留页，按前两项） |
| 4 | `docs/zh/user_guide/usage_auto_precision_tuning.md` | 自动调优 |
| 5 | `example/<family>/README.md` | 官方基线、评测口径 |
| 6 | 推理引擎文档 | 量化内核是否启用、batch/seq 配置 |

## 4. 精度问题处理顺序

```text
① 先确认问题真实：同口径复测
   - 相同 prompt / 模板 / 截断 / 超时 / batch
   - 浮点 vs 量化只改一个变量
② 排除部署问题：量化格式、引擎版本、算子回退、是否真的加载了量化权重
③ 再谈量化调优：算法、粒度、校准集、离群值抑制、敏感层回退
④ 每次只改一个变量，记录基线
```

### 调优路径

1. 调整离群值抑制：IterSmooth / Flex Smooth / QuaRot；
2. 调整量化算法/粒度：GPTQ / AWQ / per-group / per-channel；
3. 优化校准集：业务同分布、长度覆盖、去重复；
4. 敏感层回退：lm_head / visual / 首尾层 / MoE gate；
5. 仍不达标 → `strategy` / `tuning` / `sensitive-layer-analysis`。

### 先排除“假精度问题”

| 现象 | 先检查 |
|---|---|
| 全是 `????` / 乱码 | tokenizer、模板、部署格式、引擎是否支持该量化格式 |
| 余弦相似度接近 0 | Embedding 模型是否被当生成模型评测；池化方式 |
| 输出重复/截断 | 生成参数、max_tokens、stop token、模板 |
| 只有个别数据集掉点 | 评测口径、few-shot、后处理 |
| 310/310P 掉点 | 量化主流程面向 910；310 精度不保证 |

## 5. 性能问题处理顺序

```text
① 确认对比基线：同硬件、同引擎、同 batch/seq 的 BF16
② 确认引擎真的启用量化内核（如 `--quantization ascend`）
③ 确认模型规模与量化收益预期（小模型可能无收益）
④ 确认瓶颈：prefill / decode / 通信 / 显存
⑤ 仍无收益 → 引擎侧适配/特性开发，不是继续调量化参数
```

| 现象 | 判断 |
|---|---|
| W8A8 无加速 | 引擎未启用量化内核 / 小模型 / batch 太小 |
| 吞吐下降 | 量化反量化开销、引擎未适配、通信瓶颈 |
| 时延变差 | 首 token、KV Cache、算子回退 |
| 小模型无收益 | 计算量小，量化收益被开销抵消，属预期 |


## 6. 回答骨架

```markdown
【结论】先判断这是量化问题、评测问题还是部署问题；当前属于 <类别>。
【依据】<精度调优/异常定位文档；模型 example 基线>
【操作】
1. 同口径复测（浮点 vs 量化，只改一个变量）
2. 排除部署/引擎问题
3. 按 <算法/校准集/敏感层回退> 调优
【适用条件/风险】每次只改一个变量；回退会增加显存/降低压缩；310 精度不保证。
【下一步】需要自动搜索 → tuning；需要敏感层分析 → sensitive-layer-analysis。
```

## 7. 常见错误

| 错误 | 纠正 |
|---|---|
| 一上来就调量化参数 | 先确认评测口径与部署格式 |
| 把引擎不支持格式当成精度问题 | 先按 C5 确认格式/引擎 |
| 同时改算法、校准集、回退 | 无法归因；每次只改一个变量 |
| 承诺 310P 精度 | 310 精度不保证 |
| 无性能收益继续调量化 | 先确认引擎是否启用量化内核 |
