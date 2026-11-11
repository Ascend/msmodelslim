# C5 部署推理

> 只处理“量化权重已经产出，如何被推理引擎正确加载”。
> 量化怎么做见 [C3 操作指引](C3-quantization-execution.md)；拉起失败的报错定位见 [C4](C4-error-troubleshooting.md)。

## 1. 适用问题

- 量化权重在 vLLM-Ascend / MindIE 拉不起、初始化失败；
- 量化格式与推理引擎不匹配（AscendV1 / compressed_tensors / W8A8S / C8）；
- 选择哪个推理引擎、哪个版本；
- 量化时没加 `--tags`，部署时如何确认引擎支持；
- W8A8S 在 vLLM-Ascend 不支持；
- 多机/PD 分离场景下的量化格式选择。

## 2. 不适用（转交）

| 问题 | 转到 |
|---|---|
| 怎么量化、命令/YAML | [C3 操作指引](C3-quantization-execution.md) |
| 模型/模式是否支持 | [C2 能力边界/支持判定](C2-feature-support.md) |
| 拉起失败的具体 traceback | [C4 故障排查](C4-error-troubleshooting.md)（定位后回本文） |
| 量化后掉点 | [C6 精度性能](C6-accuracy-performance.md) |
| 引擎安装/驱动问题 | [C1 安装环境](C1-installation-environment.md) |

## 3. 必查证据（按顺序）

| 顺序 | 资料 | 用途 |
|---|---|---|
| 1 | `docs/zh/user_guide/process_mainstream_model_deployment.md` | 主流模型量化部署流程 |
| 2 | `docs/zh/knowledge_base/quantization_format/README.md` | 格式总览与引擎对应关系 |
| 3 | `docs/zh/knowledge_base/quantization_format/ascendv1/ascendv1_usage.md` | AscendV1 产物结构、C8/FA 字段 |
| 4 | 推理引擎官方文档 / release notes | **引擎版本**与量化格式的兼容矩阵 |
| 5 | `docs/zh/knowledge_base/model/README.md` | 矩阵中的部署备注（如“仅 vLLM Ascend 支持”） |
| 6 | `example/<family>/README.md` | 该模型实际验证过的引擎与命令 |
| 7 | `docs/zh/support/faq_deployment.md` | 部署类 FAQ（如为预留页，按前几项） |
| 8 | 量化产物 | `quant_model_description.json` / `config.json` / `--tags` |

## 4. 四维匹配（必须先确认）

```text
量化格式  ↔  推理引擎及版本  ↔  硬件平台  ↔  模型/量化模式
```

四条都匹配才回答“可部署”；缺一条只能回答“条件支持/需确认”。

| 维度 | 要确认什么 |
|---|---|
| 量化格式 | AscendV1 / compressed_tensors / 其他；W8A8 / W8A8C8 / W8A8S / MXFP8 |
| 推理引擎 | vLLM-Ascend / MindIE；**具体版本号** |
| 硬件 | A2 / A3 / 310P / 950；多机/PD 分离 |
| 模型/模式 | 模型家族、是否 MoE/MLA/VLM、是否 C8、是否 MTP |

## 5. 推理引擎版本选择（重点）

**这是 C5 最容易出错的地方。** 不要只回答“用 vLLM”或“升级 MindIE”，必须给出可验证的版本判断：

1. 以**该模型 example README 验证过的引擎版本**为第一依据；
2. 以**引擎官方 release notes** 的格式支持矩阵为第二依据；
3. 以支持矩阵中的部署备注为第三依据（例如“仅 vLLM Ascend 支持”）；
4. 客户现场版本低于验证版本时，优先建议**升级到验证版本**，而不是改量化格式；
5. 客户现场版本高于验证版本时，要求确认该版本的格式兼容性，不能默认兼容；
6. 无法确认时，明确回答“需引擎侧确认”，不要承诺可部署。

```text
推荐顺序：
模型 example 验证版本 > 引擎 release notes > 支持矩阵备注 > 社区经验
```

### 版本确认清单

- [ ] 量化时用的 `--tags` / quant_type / 格式
- [ ] 引擎名称与完整版本（如 vLLM-Ascend x.y.z）
- [ ] 模型 family 与是否 MoE/MLA/VLM/MTP
- [ ] 硬件（A2/A3/310P/950）与驱动/CANN
- [ ] example README 中该模型验证过的引擎版本
- [ ] 是否存在“仅某引擎支持”的矩阵备注

## 6. 常见格式与引擎结论

| 场景 | 结论要点 |
|---|---|
| W8A8 + vLLM-Ascend | 通常可用；确认引擎版本与 `--quantization ascend` |
| W8A8S | vLLM-Ascend 不支持稀疏格式；改普通 W8A8 或联系 vllm-ascend 适配 |
| W8A8C8 / C8 | 先确认走 KVCache Quant 还是 FA3 Quant；再确认引擎支持该 KV/FA 格式 |
| AscendV1 | 确认产物描述文件、引擎是否读取 `quant_model_description.json` |
| compressed_tensors | 确认引擎版本是否支持该格式 |
| 量化时未加 `--tags` | 部署前用 example/矩阵确认引擎支持，不要假设 |


## 7. 回答骨架

```markdown
【结论】<格式/引擎/版本> 是否匹配；能否部署。
【依据】<example README 验证版本 / 引擎 release notes / 支持矩阵备注 / 产物描述>
【操作】
1. 确认引擎版本：<版本>
2. 确认量化格式：<AscendV1/W8A8/C8/...>
3. 启动参数：<--quantization ascend / tags / 其他>
【适用条件/风险】<引擎版本、硬件、是否官方验证；不确认时明确“需引擎侧确认”>
【下一步】需要引擎适配 → 提 issue/联系引擎团队；需要重新量化 → quantization。
```

## 8. 常见错误

| 错误 | 纠正 |
|---|---|
| 只说“用 vLLM”不给版本 | 必须给出验证过的引擎版本 |
| 未确认引擎就承诺可部署 | 明确“需确认”，不要承诺 |
| 把 W8A8S 当普通 W8A8 部署 | 先确认引擎是否支持稀疏格式 |
| 量化时未加 tags，部署时假设支持 | 回查 example/矩阵，必要时重新量化 |
| 格式不匹配就建议改引擎源码 | 优先换格式/升级官方版本；提 issue |
