# RA Compress 使用指南

## 1. 适用范围

本流程适用于在 msModelSlim 中使用 RA Compress（`ra_compress`）长序列压缩算法。RA Compress 作为 `msmodelslim analyze attn_head` 的 metrics 指标，用于 KV 注意力头粒度的筛选，输出归纳头（induction heads）与回声头（echo heads）的入选索引，供后续 KV cache 压缩流程使用。

适用角色：需要进行算法构建以及模型部署的开发者

适用场景：

- 需要为长序列推理生成 KV cache 压缩配置（`head.pt`）的场景。
- 需要了解模型中哪些 KV head 承担跨段信息传递（归纳 / 复制头）的场景。
- 使用重复段合成校准数据进行注意力头能力分析。

不适用场景：

- 校准集 tokenize 后总长度不足 10000 tokens（段边界错位，得分被置空）。
- 目标 `model_type` 的适配器未实现 `RaCompressAnalysisInterface` 且 Q/K 投影层命名与默认模式不匹配。

## 2. 流程关系与前置条件

**上级流程**：模型适配与验证通过后，执行长序列 KV cache 压缩前的注意力头筛选阶段。

**前置条件**：

- 已安装兼容版本的 msModelSlim 工具（详见《[msModelSlim 工具安装指南](../../../install_guide/install_guide.md)》）。
- 已确认校准集 tokenize 后总长度 ≥ 2500 × 4 = 10000 tokens（默认使用 `calib_dummy.jsonl` 即可满足）。
- 若目标模型 Q/K 投影层命名非 `q_proj` / `k_proj` / `qkv_proj`，需在适配器中实现 `RaCompressAnalysisInterface`。

**后续操作**：将产出的 `head.pt` 送入 RA Compress KV cache 压缩量化流程，生成压缩后的模型。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型权重目录 | 模型下载或本地路径 | HuggingFace 格式，含 `config.json` 及 `*.safetensors` | 可被目标 Transformers 版本加载 |
| 输入 | 校准数据集 | 工具默认 `calib_dummy.jsonl` 或用户指定 | JSONL 格式，tokenize 后长度 ≥ 10000 | 段边界对齐，得分非空 |
| 交付件 | `head.pt` | `--save_path` 指定目录 | dict 序列化 `.pt` 文件，含 `prefix_matching` 与 `copying` 字段 | 文件存在且可被 torch.load 读取 |
| 交付件 | 入选 head 列表 | 命令行输出 | 每层 KV head 索引列表 | 可读且包含入选 head |

## 4. 流程总览

```mermaid
flowchart LR
    A[准备模型与 calib_dummy.jsonl] --> B[执行 analyze attn_head 命令]
    B --> C[Q@K^T 段间偏移注意力聚合]
    C --> D[按 ratio 选 induction/echo heads]
    D --> E[输出 head.pt 与列表]
```

## 5. 操作步骤

### 步骤 1：确认适配器与校准数据

**目标**：确认目标 `model_type` 能正确采集 Q/K 投影层输出，且校准数据长度满足段对齐要求。

**操作**：

1. 检查 Q/K 投影层命名：默认匹配 `q_proj`、`k_proj`、`qkv_proj`，若命名不一致，需在模型适配器中实现 `RaCompressAnalysisInterface.get_ra_compress_proj_patterns()`，返回 `{"q": "...", "k": "...", "qkv": "..."}`。
2. 使用默认校准集 `calib_dummy.jsonl`（推荐）；若使用自定义校准集，需保证 tokenize 后总长度 ≥ 2500 × 4 = 10000 tokens，且段边界严格对齐。

接口约定详见《[RA Compress 词条](./term_ra_compress.md)》。

**输出**：适配器命名模式与校准数据确认。

### 步骤 2：执行注意力头筛选命令

**目标**：使用 `ra_compress` 指标完成 KV head 粒度筛选，生成 `head.pt`。

**操作**：

```bash
msmodelslim analyze attn_head \
    --model_type Qwen2.5-7B-Instruct \
    --model_path ${model_path} \
    --metrics ra_compress \
    --device npu \
    --trust_remote_code True \
    --save_path ./head_result
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `attn_head` | KV 注意力头粒度分析（`ra_compress` 为默认 metrics） |
| `--metrics` | 指定分析算法，取值为 `ra_compress` 时使用本算法 |
| `--save_path` | `head.pt` 与结果文件保存目录 |
| `--calib_dataset` | 校准数据集路径（默认 `calib_dummy.jsonl`，通常无需修改） |

完整参数见《[Attention Head 筛选分析使用指南](../../../user_guide/usage_sensitive_attn_head_analysis.md)》。

**输出**：命令行打印入选的 induction heads / echo heads 列表；`--save_path` 目录下生成 `head.pt`。

### 步骤 3：解读结果并用于压缩

**目标**：将 `head.pt` 送入 RA Compress KV cache 压缩量化流程。

**操作**：

1. 确认 `head.pt` 结构为 `{"prefix_matching": {layer_idx: [kv_head_idx, ...]}, "copying": {layer_idx: [kv_head_idx, ...]}}`。
2. 在后续量化流程中，将该文件路径配置到 RA Compress 压缩相关参数中。
3. 执行量化并验证长序列推理精度与显存占用。

**输出**：RA Compress 压缩配置接入完成。

## 6. 验收条件

- 分析命令执行成功，`head.pt` 已生成在 `--save_path` 目录下。
- 命令行输出中 induction heads / echo heads 列表非空（若为空，请检查校准数据长度）。
- `head.pt` 可被后续 RA Compress 压缩量化流程正确读取。

## 7. 异常处置

- **得分或 head 列表为空**：校准集 tokenize 后总长度不足 10000 tokens。换用默认 `calib_dummy.jsonl`，或扩充自定义校准集至 ≥ 2500 × 4 tokens。
- **未命中 Q/K 投影层**：目标模型 Q/K 命名与默认模式不匹配。在适配器中实现 `RaCompressAnalysisInterface.get_ra_compress_proj_patterns()`，返回正确的模块名模式。
- **GQA 分组相关报错**：确认模型 `config.json` 中 `num_attention_heads` 与 `num_key_value_heads` 字段正确，工具会自动按组取 max。

## 8. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| RA Compress | 基于重复段结构的 KV head 筛选算法，用于长序列 KV cache 压缩 | [RA Compress 词条](./term_ra_compress.md) |
| attn_head 范围分析 | KV 注意力头粒度的筛选分析 | [Attention Head 筛选分析使用指南](../../../user_guide/usage_sensitive_attn_head_analysis.md) |
| Induction Head | 归纳头，在段边界处体现 prefix matching 行为的关键 KV head | [RA Compress 词条](./term_ra_compress.md) |
| Echo Head | 回声头，在段边界处体现 copying matching 行为的关键 KV head | [RA Compress 词条](./term_ra_compress.md) |
| RaCompressAnalysisInterface | 模型适配器可选实现的 Q/K 投影层命名接口 | [RA Compress 词条](./term_ra_compress.md) |

## 9. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| `msmodelslim analyze attn_head` | KV 注意力头粒度筛选命令，`--metrics ra_compress` 启用本算法 | [RA Compress 词条](./term_ra_compress.md) |
| `RaCompressAnalysisInterface` | 模型适配器可选实现的 Q/K 投影层命名接口 | [RA Compress 词条](./term_ra_compress.md) |
