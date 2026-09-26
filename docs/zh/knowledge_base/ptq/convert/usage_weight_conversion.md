# 权重转换使用指南

## 1. 适用范围

本指南面向需要对**已有量化/浮点权重**进行格式或精度变换的用户（如 FP8 block 反量化到 BF16、BF16 离线量化为 W8A8 MXFP8、INT4 packed 反量化到 BF16 等）。**重点是给出可上手的推荐配置、可复制的执行命令，并说明每个配置项的含义以及何时需要调整。**

与常规一键量化的区别：

| 对比项 | 常规一键量化 | 权重转换（modelslim_convert） |
|--------|--------------|-------------------------------|
| 是否需要校准集 | 是（激活值统计等） | **否** |
| 是否需要 `model_type` | 必选 | **不需要** |
| 是否需要 `quant_type` | 使用 `--quant_type` 匹配最佳实践时需要；使用 `--config` 时不需要 | **不需要**（须通过 `--config` 指定转换配置） |
| 典型场景 | 浮点模型 → W8A8 等 | FP8 → BF16、BF16 → MXFP8、FP8 → MXFP8 等 |

命令行书写规范见 [步骤 3](#步骤-3执行转换命令)，参数总表见《[msmodelslim quant 命令行](../../../api_reference/cli/msmodelslim_quant.md)》，字段说明见《[modelslim_convert 配置说明](../../../api_reference/config/quant/modelslim_convert.md)》。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 源权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被读取，含 `model.safetensors.index.json` 或单文件权重 |
| 输入 | 转换配置 YAML | `--config` 指定 | `apiversion: modelslim_convert`，含 `spec.linears`、`spec.save` 等 | 配置校验通过 |
| 交付件 | 转换后权重目录 | `--save_path` 指定路径 | 目标 IR 对应格式（`ascend_v1` / `huggingface`） | 含完整权重文件，可被目标推理框架加载 |

`--save_path` 不得指向源模型目录或其子目录，以免覆盖源权重。

## 3. 流程总览

入门时建议先复用一组**同族示例配置**建立基线，再按层名、目标 IR 与保存格式调整。

```mermaid
flowchart LR
    A[确定源权重格式与目标 IR] --> B[编写/复用转换配置]
    B --> C[执行转换命令]
    C --> D[验证转换结果]
    D --> E[部署到目标推理框架]
```

## 4. 操作步骤

### 步骤 1：确认源与目标

**操作**：先确定三件事——源权重当前是什么格式（FP8 block / BF16 / INT4 packed）、目标输出是什么 IR（`W8A8_MXFP8` / `FLOAT`）、目标部署框架是什么（决定 `save` 格式）。权重转换是**纯离线、data-free** 操作，不需要校准集、不需要 `model_type`、不需要 `quant_type`。

当前支持的转换：

| 源格式 | 目标 | 说明 | 有损/无损 |
|-------|---------|------|-----------|
| `FP8_BLOCK` | `FLOAT` | FP8 block 权重反量化为 BF16 | 无损 |
| `INT4_PACKED` | `FLOAT` | INT4 packed（分组）权重反量化为 BF16 | 无损 |
| `FLOAT` | `W8A8_MXFP8` | BF16/FP16 浮点权重离线 MXFP8 量化 | 有损 |
| `FP8_BLOCK` | `W8A8_MXFP8` | 先反量化再量化为 MXFP8 | 有损（后半段量化） |

工具会根据 checkpoint 中的权重自动识别源格式。

### 步骤 2：建立推荐配置

**操作**：下面给出三个最常见转换场景的推荐配置。如果目标模型已有同族转换示例，优先复用。

**场景 A：FP8 block → BF16（HF 无损反量化）**

```yaml
apiversion: modelslim_convert
spec:
  linears:
    - match:
        - "model.layers.*.self_attn.q_proj"
        - "model.layers.*.self_attn.k_proj"
        - "model.layers.*.self_attn.v_proj"
        - "model.layers.*.self_attn.o_proj"
        - "model.layers.*.mlp.gate_proj"
        - "model.layers.*.mlp.up_proj"
        - "model.layers.*.mlp.down_proj"
      target: FLOAT
      route: auto
  save:
    - type: huggingface
      part_file_size: 4
```

**场景 B：BF16 → W8A8_MXFP8（昇腾部署）**

```yaml
apiversion: modelslim_convert
spec:
  linears:
    - match:
        - "model.layers.*.self_attn.q_proj"
        - "model.layers.*.self_attn.k_proj"
        - "model.layers.*.self_attn.v_proj"
        - "model.layers.*.self_attn.o_proj"
        - "model.layers.*.mlp.gate_proj"
        - "model.layers.*.mlp.up_proj"
        - "model.layers.*.mlp.down_proj"
      target: W8A8_MXFP8
      route: auto
  save:
    - type: ascend_v1
      part_file_size: 4
  parallel:
    cpu_workers: 8
```

**场景 C：FP8 block → W8A8_MXFP8**

```yaml
apiversion: modelslim_convert
spec:
  linears:
    - match:
        - "model.layers.*.self_attn.q_proj"
        - "model.layers.*.self_attn.k_proj"
        - "model.layers.*.self_attn.v_proj"
        - "model.layers.*.self_attn.o_proj"
        - "model.layers.*.mlp.gate_proj"
        - "model.layers.*.mlp.up_proj"
        - "model.layers.*.mlp.down_proj"
      target: W8A8_MXFP8
      # 显式路径：先反量化再量化。不写或写 auto 时工具选同一条最短路径。
      route: [FP8_BLOCK, FLOAT, W8A8_MXFP8]
  save:
    - type: ascend_v1
      part_file_size: 4
```

完整示例配置：

| 转换场景 | 配置 |
|---------|------|
| BF16 → W8A8_MXFP8 | [qwen3_8b_bf16_to_mxfp8.yaml](./qwen3_8b_bf16_to_mxfp8.yaml) |
| FP8 block → W8A8_MXFP8 | [qwen3_8b_fp8_to_mxfp8.yaml](./qwen3_8b_fp8_to_mxfp8.yaml) |
| FP8 block → BF16（HF 格式） | [qwen3_8b_fp8_to_bf16.yaml](./qwen3_8b_fp8_to_bf16.yaml) |
| INT4 packed → BF16 | [kimi_k2_5_int4_per_group_to_bf16.yaml](./kimi_k2_5_int4_per_group_to_bf16.yaml) |

<a id="步骤-3执行转换命令"></a>

### 步骤 3：执行转换命令

**操作**：入口与一键量化相同，都是 `msmodelslim quant`，用 `--config` 指定上一步的 YAML。

最小可运行命令：

```bash
msmodelslim quant \
  --model_path "${MODEL_PATH}" \
  --save_path "${SAVE_PATH}" \
  --config "${CONFIG_PATH}"
```

`${MODEL_PATH}` 为源权重目录，`${SAVE_PATH}` 为转换输出目录，`${CONFIG_PATH}` 为上一份 YAML（示例文件或自定义文件均可）。若模型目录含自定义代码且加载时需要执行，再补充 `--trust_remote_code true`。

NPU 多卡加速（推荐大模型 / MoE）：

```bash
msmodelslim quant \
  --model_path "${MODEL_PATH}" \
  --save_path "${SAVE_PATH}" \
  --config "${CONFIG_PATH}" \
  --device npu \
  --device_id 0 1 2 3
```

`--device_id` 指定使用哪些 NPU。CPU 运行时再设 `parallel.cpu_workers`。

### 步骤 4：选择并调整参数

**操作**：先确认转换了哪些层、目标格式和保存格式是否匹配；需要加速时再调并行。

`W8A8_MXFP8` 配 `ascend_v1`，`FLOAT` 配 `huggingface`。

| 配置项 | 含义 | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `linears.match` | 待转换的线性层模块路径模式，支持 `*` 通配符。仅匹配到的 Linear 权重参与 IR 转换；**未匹配的权重（如 Norm、Embedding、Head 等）会原样拷贝**到输出目录。 | 列出目标模型中所有 Linear 投影层：`q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/`down_proj`。 | 匹配范围过大会尝试转换非 Linear 层（可能失败），过小则某些 Linear 权重保持原格式。建议先查看 checkpoint 的 `model.safetensors.index.json` 确认 key 名，再按层名前缀与投影名写通配符。 |
| `linears.target` | 转换目标 IR 类型，决定输出权重的数值格式。当前支持：`FLOAT`（BF16 浮点）、`W8A8_MXFP8`（昇腾 MXFP8）。 | FP8 反量化 → `FLOAT`；昇腾部署 → `W8A8_MXFP8`。 | `target` 必须与 `save.type` 匹配：`W8A8_MXFP8` 必须用 `ascend_v1` 落盘，`FLOAT` 用 `huggingface`。配置校验会检查此项。 |
| `linears.route` | 源到目标的 IR 转换路径。`auto` 按权重推断源 IR 并走最短路径；也可写成 IR 列表，首元素为源 IR、末元素为 `target`。 | 默认 `auto`。需要固定路径时写列表，例如只反量化 `[FP8_BLOCK, FLOAT]`，或 FP8 转到 MXFP8 `[FP8_BLOCK, FLOAT, W8A8_MXFP8]`。 | 列表中相邻 IR 必须有已注册的转换边。见场景 C。 |
| `save.type` | 保存格式：`ascend_v1`（昇腾，对应 MXFP8）、`huggingface`/`hf`/`compressed_tensors`（HF 生态，对应 BF16 浮点）。 | `ascend_v1`（目标 IR 为 `W8A8_MXFP8`）；`huggingface`（目标 IR 为 `FLOAT`）。 | 格式与目标 IR 必须严格匹配，配错会导致权重无法被目标框架加载。 |
| `save.part_file_size` | 权重分片文件大小，单位 GB；`0` 表示不分片。 | 默认 `4`（4GB 分片）。 | 小模型可设 `0` 不分片，大模型保持 `4` 便于管理超大 checkpoint。 |
| `parallel.cpu_workers` | CPU 转换的并行 worker 数：`1` 单进程；`>1` 多进程并行（突破 GIL），默认 `8`。NPU 转换固定每卡一个子进程、组内串行（进程数=卡数），此字段不参与。 | 默认 `8` 适合多数场景；超大模型或 MoE 可适当增大。 | 仅 `--device cpu` 时生效。`cpu_workers` 越大，CPU 并行度越高，但受内存带宽和磁盘 I/O 限制；NPU 多卡并行由 `--device_id` 控制，无需设置。 |
| `preprocess` | 权重图预处理，在构建虚拟模块树之前对 checkpoint key 做结构性变换。支持 `rename`（重命名 key）和 `convert`（chunk 拆分 fused 权重 / merge 合并权重）。 | 一般不需要；MoE 模型 fused gate_up_proj 拆分时需配置。 | 仅当 checkpoint 中有 fused 权重（如 `gate_up_proj`）需要拆分为独立投影时才需要。参考 `convert/` 目录下的 MoE 示例。 |
| `defaults.src_format` | 源权重格式；`auto` 由模型适配器/权重目录自动推断。 | 默认 `auto`。 | 一般不需要改。工具会从 checkpoint 中权重 dtype 自动推断。 |
| `defaults.dst_format` | 目标保存格式；与 `save.type` 同义，`save` 为空时回退到此值。代码默认值是 `ascendv1`（无下划线），对应 `save.type` 的 `ascend_v1`。 | 默认 `ascendv1`。 | 一般不需要改，直接通过 `save.type` 控制。 |
| `defaults.dst_ir` | 目标 IR 类型；不设置时由目标格式决定。 | 默认 `null`。 | 一般不需要改，直接通过 `linears.target` 控制。 |

> 设备由命令行 `--device`/`--device_id` 决定（YAML 不配置设备，语义与量化命令一致）：`--device npu`（默认）走 NPU 转换，未传 `--device_id` 时默认卡 0，多卡传 `--device_id 0 1 2 3`；`--device cpu` 强制 CPU 转换并忽略 `--device_id`。指定 NPU 但环境无可用卡（或卡号非法）时任务直接报错退出，不会静默回落 CPU。

### 参数组合与选择顺序

1. **先确定源格式与目标 IR**：`FP8_BLOCK → FLOAT`（无损反量化）还是 `FLOAT → W8A8_MXFP8`（有损量化）或 `FP8_BLOCK → W8A8_MXFP8`（自动路由）。
2. **再固定 `linears.match` 匹配范围**：参考同模型族示例 YAML 的层名模式，通配符写法保持一致。
3. **最后调整并行度和分片大小**：`parallel.cpu_workers` 影响转换速度，`part_file_size` 影响输出文件管理。

### 步骤 5：根据结果收敛参数方案

**操作**：转换完成后检查输出目录是否包含预期的权重文件，并使用目标推理框架加载进行冒烟测试。

- **无损转换**（FP8 / INT4 → BF16）：无需再调转换参数。
- **有损转换**（目标为 `W8A8_MXFP8`）：精度不够时应改走带校准集的常规量化。
- **MoE 拆分失败**：核对 `config.json` 里的 `num_experts`（也可能在 `text_config` / `language_config` 下）。若权重已经是分开的 `gate_proj` / `up_proj`，不要再配 `preprocess` 拆分，直接写 `match`。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 权重转换 | 离线、data-free 的权重格式/精度变换 | [权重转换词条](./term_weight_conversion.md) |
| IR | 中间表示，抽象表示张量格式 | [权重转换词条 - 核心原理](./term_weight_conversion.md#3-核心原理) |

## 6. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim quant` | 命令行入口、参数与权重转换示例 | [msmodelslim quant 命令行](../../../api_reference/cli/msmodelslim_quant.md) |
| modelslim_convert 配置说明 | `linears` / `save` / `parallel` 等字段说明 | [modelslim_convert 配置说明](../../../api_reference/config/quant/modelslim_convert.md) |
| 格式支持矩阵 | 量化格式与存储格式说明 | [格式支持矩阵](../../quantization_format/README.md) |
| AscendV1 量化结果 | AscendV1 落盘结果说明 | [一键量化生成结果](../../quantization_format/ascendv1/ascendv1_usage.md) |
