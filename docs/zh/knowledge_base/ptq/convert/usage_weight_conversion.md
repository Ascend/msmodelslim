# 权重转换使用指南

## 1. 适用范围

本指南面向需要对**已有量化/浮点权重**进行格式或精度变换的用户（如 FP8 block 反量化到 BF16、BF16 离线量化为 W8A8 MXFP8 等）。**重点不是展开完整配置协议，而是给出可上手的推荐配置，并说明每个配置项的含义（原理）以及何时需要调整、怎么选。**

与常规一键量化的区别：

| 对比项 | 常规一键量化 | 权重转换（modelslim_convert） |
|--------|--------------|-------------------------------|
| 是否需要校准集 | 是（激活值统计等） | **否** |
| 是否需要 `model_type` | 必选 | **可选** |
| 是否需要 `quant_type` | 方式 1 需要 | **不需要**（须通过 `--config` 指定转换配置） |
| 典型场景 | 浮点模型 → W8A8 等 | FP8 → BF16、BF16 → MXFP8、FP8 → MXFP8 等 |

命令行怎么写不在此展开，完整执行命令见[《一键量化完整指南》](../../../user_guide/usage_one_click_quantization.md)。

## 2. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 源权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被读取，含 `model.safetensors.index.json` 或单文件权重 |
| 输入 | 转换配置 YAML | `--config` 指定 | `apiversion: modelslim_convert`，含 `spec.linears`、`spec.save` 等 | 配置校验通过 |
| 交付件 | 转换后权重目录 | `--save_path` 指定路径 | 目标 IR 对应格式（ascend_v1 / huggingface） | 含完整权重文件，可被目标推理框架加载 |

## 3. 流程总览

入门时建议先用一组**推荐配置**建立基线，再根据实际转换场景调整参数。

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

当前已注册的 IR 转换边：

| 源 IR | 目标 IR | 说明 | 有损/无损 |
|-------|---------|------|-----------|
| `FP8_BLOCK` | `FLOAT` | FP8 block 权重反量化为 BF16 | 无损 |
| `FLOAT` | `W8A8_MXFP8` | BF16/FP16 浮点权重离线 MXFP8 量化 | 有损 |

配置 `route: auto` 时，工具会根据 checkpoint 中张量 dtype 与 `weight_scale_inv` 等字段**自动推断源 IR**，并在 IR 图上选择最短转换路径。例如 FP8 block 权重转 MXFP8 时，实际路径为 `FP8_BLOCK → FLOAT → W8A8_MXFP8`。

### 步骤 2：建立推荐配置

**操作**：下面给出三个最常见转换场景的推荐配置。如果目标模型已有同族转换示例（在 `convert/` 目录下），优先复用。

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

**场景 C：FP8 block → W8A8_MXFP8（自动路由串联）**

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

> 以上三个场景的完整示例配置可在 `convert/` 目录下找到，可直接复用或按需修改。

**参考配置示例**（文档目录下提供的参考配置）：

| 转换场景 | 配置路径 |
|---------|---------|
| BF16 → W8A8_MXFP8 | `convert/qwen3_8b_bf16_to_mxfp8.yaml` |
| FP8 block → W8A8_MXFP8 | `convert/qwen3_8b_fp8_to_mxfp8.yaml` |
| FP8 block → BF16（HF 格式） | `convert/qwen3_8b_fp8_to_bf16.yaml` |
| INT4 packed → BF16 | `convert/kimi_k2_5_int4_per_group_to_bf16.yaml` |

### 步骤 3：选择并调整参数

**操作**：参数选择按两个层次理解：先确认 `linears.match` 匹配了哪些层、`target` 目标 IR 是否合法；其次调整 `save` 格式、`parallel` 并行度等执行参数。以下列出所有关键配置项。

| 配置项 | 含义（原理） | 推荐配置 | 选择与调整建议 |
| --- | --- | --- | --- |
| `linears.match` | 待转换的线性层模块路径模式，支持 `*` 通配符。仅匹配到的 Linear 权重参与 IR 转换；**未匹配的权重（如 Norm、Embedding、Head 等）会原样拷贝**到输出目录。 | 列出目标模型中所有 Linear 投影层：`q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/`down_proj`。 | 匹配范围过大会尝试转换非 Linear 层（可能失败），过小则某些 Linear 权重保持原格式。建议先查看 checkpoint 的 `model.safetensors.index.json` 确认 key 名，再按层名前缀与投影名写通配符。 |
| `linears.target` | 转换目标 IR 类型，决定输出权重的数值格式。当前支持：`FLOAT`（BF16 浮点）、`W8A8_MXFP8`（昇腾 MXFP8）。 | FP8 反量化 → `FLOAT`；昇腾部署 → `W8A8_MXFP8`。 | `target` 必须与 `save.type` 匹配：`W8A8_MXFP8` 必须用 `ascend_v1` 落盘，`FLOAT` 用 `huggingface`。配置校验会检查此项。 |
| `linears.route` | 源到目标的 IR 转换路径。`auto` 让工具自动推断源 IR 并选择最短路径；显式指定 IR 列表可强制中间步骤（如 `[FP8_BLOCK, FLOAT, W8A8_MXFP8]`）。 | 默认 `auto`，满足绝大多数场景。 | 仅在调试或需要强制中间 IR 时显式指定。显式 route 的每一步都须有已注册的转换边。 |
| `save.type` | 保存格式：`ascend_v1`（昇腾，对应 MXFP8）、`huggingface`/`hf`/`compressed_tensors`（HF 生态，对应 BF16 浮点）。 | `ascend_v1`（目标 IR 为 `W8A8_MXFP8`）；`huggingface`（目标 IR 为 `FLOAT`）。 | 格式与目标 IR 必须严格匹配，配错会导致权重无法被目标框架加载。 |
| `save.part_file_size` | 权重分片文件大小，单位 GB；`0` 表示不分片。 | 默认 `4`（4GB 分片）。 | 小模型可设 `0` 不分片，大模型保持 `4` 便于管理超大 checkpoint。 |
| `parallel.workers` | 并行 worker 数：`1` 表示单进程组内线程（可配 NPU）；`>1` 表示组间多进程 + 组内线程（CPU，突破 GIL）。 | 小模型（8B dense）用 `4~8`；超大模型或 MoE 可适当增大。 | 此为纯权重离线计算，与校准集/NPU 无关。`workers` 越大，CPU 并行度越高，但受内存带宽和磁盘 I/O 限制。 |
| `parallel.worker_device` | worker 运行设备：`cpu` 或 `npu`。 | 默认 `cpu`。 | 纯 CPU 计算即可满足多数场景；仅当权重转换涉及 NPU 算子时才设为 `npu`。 |
| `preprocess` | 权重图预处理，在构建虚拟模块树之前对 checkpoint key 做结构性变换。支持 `rename`（重命名 key）和 `convert`（chunk 拆分 fused 权重 / merge 合并权重）。 | 一般不需要；MoE 模型 fused gate_up_proj 拆分时需配置。 | 仅当 checkpoint 中有 fused 权重（如 `gate_up_proj`）需要拆分为独立投影时才需要。参考 `convert/` 目录下的 MoE 示例。 |
| `defaults.src_format` | 源权重格式；`auto` 由模型适配器/权重目录自动推断。 | 默认 `auto`。 | 一般不需要改。工具会从 checkpoint 中权重 dtype 自动推断。 |
| `defaults.dst_format` | 目标保存格式；与 `save.type` 同义，`save` 为空时回退到此值。 | 默认 `ascendv1`。 | 一般不需要改，直接通过 `save.type` 控制。 |
| `defaults.dst_ir` | 目标 IR 类型；不设置时由目标格式决定。 | 默认 `null`。 | 一般不需要改，直接通过 `linears.target` 控制。 |

### 参数组合与选择顺序

1. **先确定源格式与目标 IR**：`FP8_BLOCK → FLOAT`（无损反量化）还是 `FLOAT → W8A8_MXFP8`（有损量化）或 `FP8_BLOCK → W8A8_MXFP8`（自动路由）。
2. **再固定 `linears.match` 匹配范围**：参考同模型族示例 YAML 的层名模式，通配符写法保持一致。
3. **最后调整并行度和分片大小**：`parallel.workers` 影响转换速度，`part_file_size` 影响输出文件管理。

### 步骤 4：根据结果收敛参数方案

**操作**：转换完成后检查输出目录是否包含预期的权重文件，并使用目标推理框架加载进行冒烟测试。

- **无损转换**（如 FP8 → BF16）：结果应与原始权重一致，无需精度调优。
- **有损转换**（如 BF16 → MXFP8）：若精度不达标，需回到常规量化流程（使用校准集），而非继续调整转换参数。
- **MoE 拆分失败**：确认模型 `config.json` 中存在 `num_experts` 字段。

## 5. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| 权重转换 | 离线、data-free 的权重格式/精度变换 | [权重转换词条](./term_weight_conversion.md) |
| IR | 中间表示，抽象表示张量格式 | [权重转换词条 - 核心原理](./term_weight_conversion.md#3-核心原理) |
| PTQ | 训练后量化 | [PTQ 词条](../term_ptq.md) |

## 6. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim quant` | 一键量化 CLI 入口与完整命令说明 | [一键量化完整指南](../../../user_guide/usage_one_click_quantization.md) |
| modelslim_convert 配置说明 | `preprocess`/`linears`/`save`/`parallel`/`defaults` 等配置的字段类型、默认值与完整约束 | [modelslim_convert 配置说明](../../../api_reference/config/task/modelslim_convert.md) |
| 格式支持矩阵 | 量化格式与存储格式说明 | [格式支持矩阵](../../quantization_format/README.md) |
| AscendV1 量化结果 | 一键量化生成结果说明 | [一键量化生成结果](../../quantization_format/ascendv1/ascendv1_usage.md) |

> **高阶功能**：如需深度探索 IR 路由、自定义预处理算子、多级串联转换等高级配置，可查阅 [api_reference/config/task 高阶配置文档](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/api_reference/config/task)。入门阶段无需使用这些能力。
