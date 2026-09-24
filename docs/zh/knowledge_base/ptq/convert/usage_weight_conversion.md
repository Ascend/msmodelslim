# 权重转换使用指南

## 1. 适用范围

本指南面向需要对**已有量化/浮点权重**进行格式或精度变换的用户（如 FP8 block 反量化到 BF16、BF16 离线量化为 W8A8 MXFP8、INT4 packed 反量化到 BF16 等）。**重点是给出可上手的推荐配置、可复制的执行命令，并说明每个配置项的含义以及何时需要调整。**

与常规一键量化的区别：

| 对比项 | 常规一键量化 | 权重转换（modelslim_convert） |
|--------|--------------|-------------------------------|
| 是否需要校准集 | 是（激活值统计等） | **否** |
| 是否需要 `model_type` | 必选 | **不需要** |
| 是否需要 `quant_type` | 方式 1 需要 | **不需要**（须通过 `--config` 指定转换配置） |
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
    workers: 8
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
      route: auto
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

`--device_id` 指定使用哪些 NPU。CPU 运行时再设 `parallel.workers`。

### 步骤 4：选择并调整参数

**操作**：先确认转换了哪些层、目标格式和保存格式是否匹配；需要加速时再调并行。

`W8A8_MXFP8` 配 `ascend_v1`，`FLOAT` 配 `huggingface`。

| 配置项 | 含义 | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `linears.match` | 待转换的线性层路径，支持 `*` 通配符。未匹配的权重（Norm、Embedding、Head 等）原样保留。 | 列出目标模型中的 Linear 投影层，如 `q_proj` / `k_proj` / `v_proj` / `o_proj` / `gate_proj` / `up_proj` / `down_proj`。 | 过宽会把 Norm 等也算进去，过窄则部分 Linear 保持原格式。层名以 `model.safetensors.index.json` 为准。 |
| `linears.target` | 转换目标。常用 `FLOAT`、`W8A8_MXFP8`。 | FP8 / INT4 反量化 → `FLOAT`；昇腾 MXFP8 → `W8A8_MXFP8`。 | `W8A8_MXFP8` 必须配 `ascend_v1`。 |
| `save.type` | 保存格式：`ascend_v1`（昇腾）、`huggingface`（HF）。 | `ascend_v1`（MXFP8）；`huggingface`（BF16）。 | 与 `target` 对齐即可。 |
| `save.part_file_size` | 分片大小，单位 GB；`0` 表示不分片。 | 默认 `4`。 | 小模型可设 `0`。 |
| `parallel.workers` | CPU 并行 worker 数。NPU 多卡不使用此项。 | 小模型（8B dense）用 `4~8`。 | 越大越快，受内存和磁盘限制。 |
| `parallel.max_group_size` | 单个依赖组的最大任务数，超过则拆成多个子组分散到不同进程/卡并行。 | 默认 `null`（不拆分）；MoE 模型可设为约「每层专家任务数 / 卡数」（如 8 卡设为 96 左右）。 | 仅在 MoE 等含大量专家任务的场景下设置，用于缓解大组单进程/单卡计算导致的拖尾与空闲。 |

### 参数组合与选择顺序

1. **先确定源格式与目标**：反量化到 `FLOAT`，还是转到 `W8A8_MXFP8`。
2. **再写 `linears.match` 和 `save.type`**：层名对齐 checkpoint；MXFP8 用 `ascend_v1`。
3. **最后选设备与并行**：NPU 加 `--device npu --device_id ...`；CPU 调 `parallel.workers`；MoE 大模型按需配置 `parallel.max_group_size`。

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
