# 权重转换使用指南

## 1. 适用范围

本流程适用于对**已有量化/浮点权重**进行格式或精度变换的场景，无需加载模型代码、无需校准数据集。

**适用角色**：算法工程师、模型部署工程师

**典型场景**：

- FP8 block 权重反量化为 BF16（HuggingFace 格式导出）
- BF16 浮点权重离线量化为 W8A8 MXFP8（昇腾 NPU 部署）
- FP8 block 权重直接转换为 W8A8 MXFP8（自动路由串联）
- INT4 分组量化权重反量化为 BF16

**与常规一键量化的区别**：权重转换不需要校准数据、不需要 `model_type`、不需要 `quant_type`，仅通过 `--config_path` 指定转换配置。

| 对比项 | 常规一键量化 | 权重转换（modelslim_convert） |
|--------|--------------|-------------------------------|
| 是否需要校准集 | 是（激活值统计等） | **否** |
| 是否需要 `model_type` | 必选 | **可选**（YAML 中 `apiversion: modelslim_convert` 时可省略） |
| 是否需要 `quant_type` | 方式 1 需要 | **不需要**（须通过 `config_path` 指定转换配置） |
| 典型场景 | 浮点模型 → W8A8 等 | FP8 → BF16、BF16 → MXFP8、FP8 → MXFP8 等 |

当前已注册的 IR 转换边包括：

| 源 IR | 目标 IR | 说明 | 有损/无损 |
|-------|---------|------|-----------|
| `FP8_BLOCK` | `FLOAT` | FP8 block 权重反量化为 BF16 | 无损 |
| `FLOAT` | `W8A8_MXFP8` | BF16/FP16 浮点权重离线 MXFP8 量化 | 有损 |

配置 `route: auto` 时，工具会根据 checkpoint 中张量 dtype 与 `weight_scale_inv` 等字段**自动推断源 IR**，并在 IR 图上选择最短转换路径。例如 FP8 block 权重转 MXFP8 时，实际路径为 `FP8_BLOCK → FLOAT → W8A8_MXFP8`。

**落盘格式约束**：

- 目标 IR 为 **`W8A8_MXFP8`** 时，须使用 **`ascend_v1`** 保存（昇腾 NPU 部署路径）。
- 目标 IR 为 **`FLOAT`**（如 FP8 反量化到 BF16）时，可使用 **`huggingface`** / **`compressed_tensors`** 保存，供 HuggingFace 生态推理。

## 2. 流程关系与前置条件

**上级流程**：已有量化/浮点权重，需要调整格式或精度以适配不同推理框架。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[安装指南](../../../install_guide/install_guide.md)》）
- 已获取源权重目录（HuggingFace 格式 checkpoint，含 `config.json` 及 `*.safetensors`）
- 已编写或选用转换 YAML 配置（参考 `convert/` 目录下的示例配置）

**后续操作**：转换后权重部署到目标推理框架，或进入精度验证环节。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | msModelSlim 工具 | 已安装环境 | 可执行 `msmodelslim` 命令 | `msmodelslim --help` 正常输出 |
| 输入 | 源权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被读取，含 `model.safetensors.index.json` 或单文件权重 |
| 输入 | 转换配置 YAML | `--config_path` 指定 | `apiversion: modelslim_convert`，含 `spec.linears`、`spec.save` 等 | 配置校验通过 |
| 交付件 | 转换后权重目录 | `--save_path` 指定路径 | 目标 IR 对应格式（ascend_v1 / huggingface） | 含完整权重文件，可被目标推理框架加载 |
| 交付件 | 精度评估结果 | 用户记录 | 转换前后精度对比报告 | 精度在可接受范围内 |

## 4. 流程总览

```mermaid
flowchart LR
  A[准备源权重与转换配置] --> B[执行转换命令]
  B --> C[验证转换结果]
  C --> D[部署到目标推理框架]
```

## 5. 操作步骤

### 步骤 1：安装 msModelSlim 工具

**目标**：完成 msModelSlim 工具的安装与验证。

**操作**：

```bash
# 安装完成后验证工具可用
msmodelslim --help
```

> 详细的安装方式（在线安装、离线安装、源码安装）请参阅《[msModelSlim 安装指南](../../../install_guide/install_guide.md)》。

**输出**：msModelSlim 工具安装成功，可正常执行 `msmodelslim` 命令。

### 步骤 2：准备源权重与转换配置

**目标**：确认源权重目录和转换 YAML 配置。

**操作**：

1. 确认源权重为 HuggingFace 风格 checkpoint（含 `config.json` 及 `*.safetensors` 或 `model.safetensors.index.json` 分片索引）。
2. 编写或选用转换 YAML 配置，明确 `spec.linears`（需转换的线性层匹配规则）、`spec.save`（落盘格式）与 `spec.parallel`（并行度）。字段含义详见本文档第6节「转换配置协议详解」。
3. 确认 `save.type` 与目标 IR 匹配（`ascend_v1` 对应 MXFP8，`huggingface` 对应 BF16）。

文档目录下提供了 Qwen3-8B 的参考配置，可直接复用或按需修改：

- [qwen3_8b_bf16_to_mxfp8.yaml](./qwen3_8b_bf16_to_mxfp8.yaml)：BF16 → W8A8_MXFP8
- [qwen3_8b_fp8_to_mxfp8.yaml](./qwen3_8b_fp8_to_mxfp8.yaml)：FP8 block → W8A8_MXFP8
- [qwen3_8b_fp8_to_bf16.yaml](./qwen3_8b_fp8_to_bf16.yaml)：FP8 block → BF16（HF 格式）
- [kimi_k2_5_int4_per_group_to_bf16.yaml](./kimi_k2_5_int4_per_group_to_bf16.yaml)：INT4 packed → BF16

**输出**：源权重路径（`${MODEL_PATH}`）和配置路径（`${CONFIG_PATH}`）。

### 步骤 3：执行权重转换

**目标**：对源权重执行格式/精度变换，生成转换后权重。

**操作**：

权重转换复用一键量化 CLI 入口，通过 **`--config_path`** 指定 `modelslim_convert` 配置：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config_path ${CONFIG_PATH}
```

**参数说明**：

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| `model_path` | 必选 | 源权重目录路径。 |
| `save_path` | 必选 | 转换后权重保存路径。 |
| `config_path` | 必选 | YAML 转换配置路径，`apiversion` 须为 `modelslim_convert`。 |
| `-h, --help` | 可选 | 命令行帮助信息。 |

**注意事项**：

- 权重转换**不需要** `--model_type`、`--quant_type`、`--trust_remote_code` 等参数。
- 未在 `linears.match` 中匹配的权重（如 `embed_tokens`、`lm_head`、LayerNorm 等）会**原样保留**并写入输出目录。
- 可通过环境变量 `MSMODELSLIM_LOG_LEVEL` 设置日志级别（可选值：`INFO`（默认）、`DEBUG`）。

**使用示例**：

示例 1：FP8 block 权重反量化为 BF16（HF 格式）

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config_path ./qwen3_8b_fp8_to_bf16.yaml
```

其中 `${MODEL_PATH}` 为含 FP8 block 权重（`.weight` + `.weight_scale_inv`）的源目录，配置文件见 [qwen3_8b_fp8_to_bf16.yaml](./qwen3_8b_fp8_to_bf16.yaml)。

示例 2：BF16 权重离线量化为 W8A8_MXFP8（AscendV1 格式）

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config_path ./qwen3_8b_bf16_to_mxfp8.yaml
```

配置文件见 [qwen3_8b_bf16_to_mxfp8.yaml](./qwen3_8b_bf16_to_mxfp8.yaml)。

示例 3：FP8 block 权重直接转换为 W8A8_MXFP8

在已有 FP8 block 权重、希望部署到昇腾 MXFP8 推理路径时，可一步完成反量化与 MXFP8 量化（`route: auto` 自动串联 `FP8_BLOCK → FLOAT → W8A8_MXFP8`）：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --config_path ./qwen3_8b_fp8_to_mxfp8.yaml
```

配置文件见 [qwen3_8b_fp8_to_mxfp8.yaml](./qwen3_8b_fp8_to_mxfp8.yaml)。

**输出**：转换后权重目录 `${SAVE_PATH}`。

### 步骤 4：验证转换结果

| save.type | 典型目标 IR | 输出特征 |
|-----------|-------------|----------|
| `ascend_v1` | `W8A8_MXFP8` | 生成 `quant_model_description.json`、`quant_model_weights*.safetensors` 等 AscendV1 量化权重，详见《[一键量化生成结果](../../quantization_format/ascendv1/term_ascendv1.md)》 |
| `huggingface` / `compressed_tensors` | `FLOAT` | 生成 HF 风格 `config.json`、`model*.safetensors` 等，权重为 BF16 浮点 |

**操作**：

1. 检查输出目录是否包含预期的权重文件。输出目录取决于 YAML 中 `save.type`：

   | save.type | 典型目标 IR | 输出特征 |
   |-----------|-------------|----------|
   | `ascend_v1` | `W8A8_MXFP8` | 生成 `quant_model_description.json`、`quant_model_weights*.safetensors` 等 AscendV1 量化权重，详见《[一键量化生成结果](../../quantization_format/ascendv1/ascendv1.md)》 |
   | `huggingface` / `compressed_tensors` | `FLOAT` | 生成 HF 风格 `config.json`、`model*.safetensors` 等，权重为 BF16 浮点 |

2. 确认目标 IR 与保存格式匹配（MXFP8 → ascend_v1，FLOAT → huggingface）。
3. 使用目标推理框架加载转换后权重进行冒烟测试。

> 无论哪种落盘格式，未纳入 `linears.match` 的非线性层权重均会从源 checkpoint 拷贝至输出目录。

**输出**：转换结果验证通过。

### 步骤 5：精度测评

**目标**：对转换后权重进行精度验证，确认转换精度满足要求。

**操作**：

方式一（使用 curl 请求推理服务进行抽样验证）：

```bash
curl -X POST ${INFERENCE_ENDPOINT} \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请用一句话解释什么是量子计算。",
    "max_tokens": 128
  }'
```

通过对比转换前后模型对同一组测试 prompt 的生成结果，评估质量损失。

方式二（使用标准评测工具进行系统评估）：

推荐使用 [AISbench](https://github.com/AISBench/benchmark) 评测工具，在转换前后模型上运行相同的评测任务，对比精度指标。对于无损转换（如 FP8 反量化到 BF16），结果应与原始权重一致；对于有损转换（如 BF16 到 MXFP8），需评估精度损失是否在可接受范围内。

> 详细精度调优方法请参阅《[量化精度调优指南](../../../user_guide/process_quantization_precision_tuning.md)》。

**输出**：转换精度评估报告，确认转换后精度在可接受范围内。

## 6. 转换配置协议详解

### 6.1 配置协议概述

#### 6.1.1 基本结构

权重转换配置采用 YAML 描述，顶层固定为：

```yaml
apiversion: modelslim_convert   # 协议版本，固定值
spec:                           # 转换任务具体配置
  preprocess: [ ]               # 可选：权重图结构预处理
  linears: [ ]                  # 必选：线性层匹配与目标 IR
  save: [ ]                     # 必选：落盘格式
  parallel: { }                 # 可选：并行与设备
```

与 `modelslim_v1` 量化配置不同，`modelslim_convert` **不包含** `runner`、`process`、`dataset` 等校准相关字段；转换流水线为固定顺序：读 catalog → 预处理 → 建虚拟模块树 → IR 路由 → 转换 → 落盘。

#### 6.1.2 协议版本说明

| 参数 | 可选/必选 | 说明 |
|------|-----------|------|
| apiversion | 必选 | 固定为 `"modelslim_convert"`，用于选择 convert 量化服务后端。 |
| spec | 必选 | 转换规则、落盘与并行参数。 |

### 6.2 linears - 线性层转换规则

**作用**：声明哪些模块参与 IR 转换，以及目标精度/格式。

**特点**：

- **列表结构**：可配置多组 `match`，每组共享相同的 `target` 与 `route`。
- **通配符匹配**：`match` 支持 `*` 通配（如 `model.layers.*.self_attn.q_proj`）。
- **自动路由**：`route: auto` 时由工具推断源 IR 并选择最短转换链；也可显式指定 IR 序列。

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| match | 字符串列表 | 待转换模块路径模式，支持 `*` 通配。 |
| target | IRKind | 目标 IR。当前支持：`FLOAT`（BF16 浮点）、`W8A8_MXFP8`。 |
| route | `"auto"` 或 IRKind 列表 | 源到目标的 IR 路径约束。默认 `"auto"`。 |

**配置示例**（Qwen3-8B dense，7×36个 linear 转 MXFP8）：

```yaml
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
```

**源 IR 推断规则**（`route: auto` 时）：

- checkpoint 中存在 `weight_scale_inv` 或 weight dtype 为 float8 → 源 IR 为 `FP8_BLOCK`
- 仅存在 BF16/FP16 的 `.weight` → 源 IR 为 `FLOAT`

### 6.3 save - 落盘格式

**作用**：指定转换结果的保存格式与分片策略。

**配置示例**：

```yaml
# AscendV1（MXFP8 部署）
spec:
  save:
    - type: ascend_v1
      part_file_size: 4

# HuggingFace / compressed_tensors（BF16 浮点导出）
spec:
  save:
    - type: huggingface
      part_file_size: 4
```

**字段说明**：

| 字段 | 作用 | 说明 |
|------|------|------|
| type | 保存器类型 | `ascend_v1` / `ascendv1` / `ascendv1_saver` → AscendV1；`huggingface` / `hf` / `compressed_tensors` → HF 生态格式。 |
| part_file_size | 分片大小 | 单位 GB；`0` 表示不分片，类型为 int。 |

**格式与目标 IR 对应关系**：

| target IR | 推荐 save.type | 说明 |
|-----------|----------------|------|
| `W8A8_MXFP8` | `ascend_v1` | **必选**；MXFP8 权重仅面向昇腾 NPU 部署。 |
| `FLOAT` | `huggingface` | FP8 反量化等 HF 侧推理场景。 |

若 `target` 为 `W8A8_MXFP8` 而 `save.type` 为 `huggingface`，配置校验将报错。

### 6.4 parallel - 并行配置

**作用**：控制 IR 任务执行的并行度与设备策略。

**配置示例**：

```yaml
spec:
  parallel:
    workers: 8              # 进程/worker 数量
```

**字段说明**：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| workers | 1 | `1`：单进程；`>1`：多进程并行。 |

### 6.5 preprocess - 权重图预处理（可选）

**作用**：在构建虚拟模块树之前，对 checkpoint key 做结构性变换（如 fused gate/up 拆分）。

**支持类型**：

| type | 说明 |
|------|------|
| rename | 按 pattern 重命名 checkpoint key。 |
| convert | 对一组 source key 执行 chunk（拆分 fused 权重）或 merge（合并 gate/up）等操作；须配合 `source`、`target` 与 `ops` 使用。 |

**convert 字段说明**（`type: convert` 时）：

| 字段 | 可选/必选 | 说明 |
|------|-----------|------|
| source | 必选 | 待变换的源模块路径模式列表，支持 `*` 通配。 |
| target | 必选 | 变换后的目标模块路径模式列表，条目数须与 `ops` 中拆分结果数量一致。 |
| ops | 必选 | 结构变换算子列表，按顺序对匹配的源权重执行操作，详见下文。 |

**ops 算子说明**：

`ops` 为列表结构，每个元素描述一步结构变换。预处理阶段仅改写 checkpoint 中的 key 与元数据，**不物化大张量**；实际切片/合并在后续虚拟模块加载时完成。

| 算子 type | 说明 | 典型场景 |
|-----------|------|----------|
| `chunk` | 沿指定维度将 fused 权重拆分为多路逻辑子权重 | `gate_up_proj` → `gate_proj` + `up_proj` |
| `merge` | 将多路逻辑子权重合并为 fused 权重 | 与 `chunk` 互为逆操作 |

各算子支持的字段如下：

| 字段 | 可选/必选 | 适用算子 | 默认值 | 说明 |
|------|-----------|----------|--------|------|
| type | 必选 | 全部 | — | 算子类型，取 `chunk` 或 `merge`。 |
| dim | 可选 | `chunk` / `merge` | `chunk` 为 `1`，`merge` 为 `0` | 拆分或合并所沿的张量维度。 |
| projections | 可选 | `chunk` | `["gate_proj", "up_proj"]` | 拆分后各子权重的逻辑投影名，须与 `target` 列表条目一一对应。 |

**注意事项**：

1. MoE 模型拆分 `*.mlp.experts.gate_up_proj` 时，工具会从模型 `config.json` 读取 `num_experts`；若缺失将报错。
2. 可配置多条 `ops`，按 YAML 列表顺序依次执行。
3. `chunk` 在内部映射为 `split_fused_gate_up`，`merge` 映射为 `merge_gate_up`。

**rename 示例**：

```yaml
spec:
  preprocess:
    - type: rename
      patterns:
        - from: "model.layers.0.mlp.gate_up_proj.weight"
          to: "model.layers.0.mlp.gate_proj.weight"
```

**convert（chunk）示例**：

将 MoE 模型中 fused 的 `gate_up_proj` 拆分为 per-expert 的 `gate_proj` 与 `up_proj`：

```yaml
spec:
  preprocess:
    - type: convert
      source:
        - "model.layers.*.mlp.experts.gate_up_proj"
      target:
        - "model.layers.*.mlp.experts.*.gate_proj.weight"
        - "model.layers.*.mlp.experts.*.up_proj.weight"
      ops:
        - type: chunk
          dim: 1
          projections: ["gate_proj", "up_proj"]
```

**convert（merge）示例**：

将独立的 `gate_proj` / `up_proj` 合并回 fused `gate_up_proj`（与 chunk 互为逆操作）：

```yaml
spec:
  preprocess:
    - type: convert
      source:
        - "model.layers.*.mlp.experts.*.gate_proj.weight"
        - "model.layers.*.mlp.experts.*.up_proj.weight"
      target:
        - "model.layers.*.mlp.experts.gate_up_proj"
      ops:
        - type: merge
          dim: 0
```

### 6.6 完整配置示例

以下示例与 [qwen3_8b_fp8_to_mxfp8.yaml](./qwen3_8b_fp8_to_mxfp8.yaml) 等价，展示 FP8 → MXFP8 的完整 spec：

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

## 7. 验收条件

- 转换后权重目录包含预期的权重文件。
- 目标 IR 与保存格式匹配（MXFP8 → ascend_v1，FLOAT → huggingface）。
- 未匹配的线性层权重从源 checkpoint 拷贝至输出目录。
- 转换后权重可被目标推理框架成功加载。
- 精度评估结果在可接受范围内。

- 一键量化总体流程与常规量化配置：《[一键量化完整指南](usage.md)》
- AscendV1 量化权重文件说明：《[一键量化生成结果](../../quantization_format/ascendv1/term_ascendv1.md)》
- 格式支持矩阵：《[格式支持矩阵](../../quantization_format/README.md)》

- **配置校验失败**：检查 YAML 中 `apiversion` 是否为 `modelslim_convert`，`spec` 字段是否完整。
- **转换失败**：检查 `linears.match` 中的模块路径是否与 checkpoint 中的 key 匹配。
- **保存格式不匹配**：确认 `save.type` 与 `target` IR 的对应关系（MXFP8 必须用 `ascend_v1`）。
- **MoE 拆分失败**：确认模型 `config.json` 中存在 `num_experts` 字段。
- 详细排查可参考本文档第9节「常见问题」。

## 9. 常见问题

### Q1: 权重转换与常规一键量化如何选择？

- 已有 **FP8 / BF16 等 checkpoint**，仅需改精度或落盘格式、**不需要重新校准** → 使用权重转换（本文档）。
- 从 **原始浮点模型** 出发，需要校准集统计激活并做 W8A8 等完整量化 → 使用《[训练后量化（PTQ）知识库](../README.md)》中的对应模型量化流程。

### Q2: 为什么 MXFP8 必须用 ascend_v1 落盘？

W8A8_MXFP8 权重面向昇腾 NPU 推理栈设计，元数据与 packing 方式与 HF `compressed_tensors` 不兼容。BF16 浮点导出（如 FP8 反量化）才应使用 `huggingface` 保存。

### Q3: 如何确定 linears.match 应写哪些层？

1. 查看源 checkpoint 的 `model.safetensors.index.json` 或单文件 key 列表。
2. 仅对需要变更 IR 的 **Linear 权重** 配置 match；Norm、Embedding、Head 等通常排除在外，会自动 passthrough。
3. 可参考同模型族文档目录下的示例 YAML，按层名前缀与投影名调整通配符。

### Q4: workers 设多少合适？

- 小模型（如 8B dense）：`workers: 4~8` 通常即可充分利用 CPU。
- 超大模型或 MoE：可适当增大 `workers`。
- 校准集、NPU 与此流程无关；convert 为纯权重离线计算。

### Q5: route 何时需要显式指定？

多数场景使用 `route: auto` 即可。仅在需要**强制中间 IR**（例如调试单步 `FP8_BLOCK → FLOAT`）时，可写显式路径：

```yaml
route:
  - FP8_BLOCK
  - FLOAT
  - W8A8_MXFP8
```

显式 route 的每一步须在工具已注册的 IR 图上有对应转换边，否则会报错。

## 10. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 权重转换 | 离线、data-free 的权重格式/精度变换 | [权重转换词条](./term_weight_conversion.md) |
| IR | 中间表示，抽象表示张量格式 | [权重转换词条 - 原理](./term_weight_conversion.md#3-原理) |
| 自动路由 | route: auto 自动推断源 IR 并选择最短路径 | [权重转换使用指南 - 配置协议](#62-linears---线性层转换规则) |
| PTQ | 训练后量化 | [PTQ 总览](../README.md) |

## 11. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| 权重转换配置协议 | modelslim_convert YAML 配置详解 | [权重转换使用指南 - 配置协议](#6-转换配置协议详解) |
| 格式支持矩阵 | 量化格式与存储格式说明 | [格式支持矩阵](../../quantization_format/README.md) |
| AscendV1 量化结果 | 一键量化生成结果说明 | [一键量化生成结果](../../quantization_format/ascendv1/ascendv1.md) |
