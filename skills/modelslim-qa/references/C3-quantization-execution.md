# C3 操作指引

> 处理“怎么做”：量化命令、YAML、`model_type`、新模型接入、V0→V1、权重格式/精度转换。
> 是否支持见 [C2 能力边界/支持判定](C2-feature-support.md)；跑起来的报错见 [C4 故障排查](C4-error-troubleshooting.md)；概念见 [C7 概念释疑](C7-quantization-basics.md)。

## 1. 适用问题

- 一键量化命令怎么写；
- `model_type` / `quant_type` / `--config` / `--device` / `--device_id` 怎么选；
- 新模型怎么接入、怎么复用同族 YAML；
- V0 脚本（`pytorch.llm_ptq`、`quant_*.py`）怎么迁移到 V1；
- 量化产物保存到哪里、执行目录有什么要求；
- FP8/FP4→BF16、INT4/AWQ 反量化、字段 rename 等权重转换怎么做；
- `trust_remote_code` 什么时候可以开启、如何安全开启。

## 2. 不适用（转交）

| 问题 | 转到 |
|---|---|
| 模型/模式/转换是否支持 | [C2 能力边界/支持判定](C2-feature-support.md) |
| 命令报错、traceback | [C4 故障排查](C4-error-troubleshooting.md) |
| 量化权重如何部署 | [C5 部署推理](C5-deployment.md) |
| 量化后掉点/调优 | [C6 精度性能](C6-accuracy-performance.md) |
| 量化模式/体积概念 | [C7 概念释疑](C7-quantization-basics.md) |
| 新模型适配器实现 | `adaptation` skill |

## 3. 必查证据（按顺序）

| 顺序 | 资料 | 用途 |
|---|---|---|
| 1 | `docs/zh/user_guide/usage_one_click_quantization.md` | 支持矩阵内模型的一键量化 |
| 2 | `docs/zh/user_guide/usage_weight_quantization.md` | 自定义 YAML / 权重量化流程 |
| 3 | `docs/zh/api_reference/cli/msmodelslim_quant.md` | CLI 参数语义 |
| 4 | `docs/zh/api_reference/config/quant/modelslim_v1.md` | YAML 字段与校验 |
| 5 | `lab_practice/**/*.yaml` + `example/<family>/README.md` | 该模型的最小可执行命令 |
| 6 | `config/config.ini` `[ModelAdapter]` / `[ModelAdapterDependencies]` | `model_type` 与版本 |
| 7 | `docs/zh/knowledge_base/ptq/convert/usage_weight_conversion.md` | 权重转换的命令、脚本与字段 rename |
| 8 | `docs/zh/support/faq_deployment.md` | 权重转换与部署的常见问题（FP8/INT4→BF16、BF16→MXFP8） |

## 4. 执行流程

```text
① 先确认支持档位（C2）：OFFICIAL_ONE_CLICK / OFFICIAL_PRACTICE / ADAPTER_ONLY
② 找最小可执行样本：example 命令 > lab_practice YAML > 一键量化文档
③ 抽参数：model_path / save_path / model_type / quant_type / device / device_id / config
④ 确认执行目录：安装后的非源码目录
⑤ 确认版本：transformers / torch_npu / CANN
⑥ 给出命令或 YAML 片段 + 风险
```

## 5. 参数要点

### `--device` 与 `--device_id`

- `--device` 只接受 `npu` / `cpu`；
- 多卡索引用 `--device_id 0 1 2 3`，不要写 `--device npu:0,1`；
- 旧写法可能被 CLI 兼容转换，但帮助信息里仍可能看到旧格式；
- **能否真正用多卡取决于 YAML 的 `spec.runner`**：
  - `model_wise` runner 会忽略 `device_indices`；
  - `LayerWiseRunner` 绑定第一张卡；
  - `DPLayerWiseRunner` 才会使用全部卡；
- `ASCEND_RT_VISIBLE_DEVICES` 会把可见卡重编号，`--device_id` 是可见设备内的逻辑索引。

### `model_type`

- 必须是 `config.ini` 注册的模型名或 `model_type`；
- 未注册的名称会回退到 `default` adapter，不会自动命中同族适配器；
- 借用同族适配器时要明确说明“未验证”（回到 C2 的档位结论）。

### `--config` / YAML

- 优先复用 `lab_practice` 的官方 YAML；
- `--config` 使用**绝对路径**；在非源码目录执行时，不要把仓库内相对路径直接传给 `--config`；
- 自定义 YAML 至少包含 `apiversion`、`metadata.config_id`、`process`、`spec`；
- 不要手工改官方 YAML 的 `verified_model_types` 来“制造支持”。

### `--trust_remote_code`

- **默认 `false`**；不要为了“先跑通”直接开启；
- 仅当模型官方资料明确要求、且权重来源可信时才可开启；
- 开启时在隔离环境（容器/最小权限/无网络）中执行，并向用户说明代码执行风险；
- 详见 SKILL.md §8「安全边界」。

### 执行目录

- 安装后在**非源码根目录**执行；
- 在源码根目录执行会出现 `config/config.ini` 相对路径错误（见 C1/C4）。

## 6. V0 → V1

| 场景 | 处理 |
|---|---|
| 用户跑 `msmodelslim/pytorch/llm_ptq` / `quant_*.py` | 说明 V0 脚本已逐步淘汰；引导最新仓库 + `bash install.sh` + V1 CLI |
| 旧接口报错 | 换当前 V1 命令；到对应模型 example 找命令 |
| MindSpore PTQ / 自动回退 | 属于 V0 功能；统一引导 V1，参考 `docs/zh/user_guide/README.md` |

## 7. 新模型接入

1. 先按 C2 判定是否已收录；未收录走《新模型量化调优流程》：
   `docs/zh/user_guide/process_new_model_quantization_tuning.md`
2. 检查 `config.ini` 是否需要新增 `model_type` 与 adapter（实现工作转 `adaptation`）；
3. 选择同结构模型的 YAML 作为起点，逐项核对：
   - `model.layers.*` 排除项；
   - MTP / 视觉塔 / lm_head 是否量化；
   - `quant_type` 与 `label` 匹配。
4. 明确标注“未验证”，不要声称官方支持。

## 8. 权重格式 / 精度转换

> 是否支持先按 C2 判定；本节只给操作步骤。命令、脚本和字段名以 `usage_weight_conversion.md` 与该模型 README 为准，不凭记忆编造。

```text
① 确认源格式/精度：config.json、quant_model_description.json、example README
② 查转换指南是否有该模型/格式示例；无示例时明确“未验证”
③ 按文档执行转换脚本/命令，或给出字段 rename 清单
④ 校验产物字段与权重分片
⑤ 转换完成后：需要量化 → 回 C3 量化流程；需要部署 → C5
```

常见路径：

- **FP8 → BF16**：按转换指南处理 `.scale` / `.weight_scale_inv` 等字段，产物落到本地目录；
- **FP4 → FP16/BF16**：多数场景不支持，先查文档；不支持时明确告知；
- **INT4 / 原生 AWQ → BF16**：先反量化为 BF16，再走 msModelSlim 量化；原生 AWQ 不能直接作为量化输入；
- **BF16 → 低比特**：属于量化，不是转换，走 §4 量化流程；
- **体积不变**：若源权重本来就是低比特，属预期，解释见 C7。


## 9. 常见错误

| 错误 | 纠正 |
|---|---|
| 推荐 V0 脚本 | 违反红线；只给 V1 |
| 让用户在源码根目录执行 | 违反红线；先 install 再换目录 |
| 只给 `--quant_type` 不给 YAML/命令 | 客户无法落地；至少给最小命令 |
| 多卡只写 `--device npu:0,1` | 用 `--device npu --device_id 0 1`，并说明 runner 限制 |
| 把未注册模型名直接传给 `--model_type` | 会回退 default；说明需同族名称或适配 |
| 自定义 YAML 声称官方验证 | 明确“未验证” |
| 默认开启 `--trust_remote_code true` | 默认 false；仅官方要求且来源可信时开启 |
| 把量化说成转换（或反之） | BF16→低比特是量化；格式/精度互转才是转换 |
| 编造转换脚本/字段名 | 以 `usage_weight_conversion.md` 和模型 README 为准 |

## 10. 回答骨架

````markdown
【结论】<模型/模式> 的量化方式：<一键 / 官方实践 / 未验证尝试>。
【依据】<example 命令 / lab_practice YAML / 一键量化文档>
【操作】
```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu --device_id 0 1 2 3 \
  --model_type <model_type> \
  --quant_type <quant_type> \
  --trust_remote_code false
```
【适用条件/风险】transformers 版本；runner 是否支持多卡；是否官方验证；是否可信来源。
【下一步】执行量化 → quantization；新模型适配 → adaptation；报错 → 提供完整 traceback。
````

> `--trust_remote_code` 仅在模型官方资料明确要求且来源可信时改为 `true`，并说明风险。
