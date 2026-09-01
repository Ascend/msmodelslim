# compressed-tensors 使用指南

本指南说明如何在 msModelSlim 中选用 **compressed-tensors** 量化格式，按 **确认模式支持 → 配置 save → 执行量化并核对产物** 完成落盘，并将产物部署到 vLLM 等支持 HuggingFace `quantization_config` 的推理框架。格式字段与 Preset 见《[compressed-tensors](term_compressed_tensors.md)》；一键量化命令总览见《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md)》。

## 1. 适用范围

本流程适用于：需将 msModelSlim 量化结果导出为 **compressed-tensors** 格式，并在 **vLLM** 等支持 HuggingFace `config.json` → `quantization_config` 的推理框架中加载的用户。

不适用：昇腾 MindIE / vLLM Ascend 默认 AscendV1 路径（见《[AscendV1 使用指南](../ascendv1/ascendv1_usage.md)》）；多模态 MindIE-SD（见《[MindIE-SD 使用指南](../mindie_sd/mindie_sd_usage.md)》）。

## 2. 流程关系与前置条件

**上级流程**：《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md)》

**前置条件**：

- 已安装兼容版本的 msModelSlim，且目标模型在《[大模型支持矩阵](../../model/README.md)》中可量化
- 量化方案落在当前 compressed-tensors 已支持的 QIR preset（目前为 W8A8 Static / W8A8 Dynamic，见《[compressed-tensors](term_compressed_tensors.md#mode-support)》）
- 目标推理框架可识别 `quant_method: "compressed-tensors"`

**后续操作**：用 vLLM 等加载 `${SAVE_PATH}`；需要 W4A8 等未实现 handler 的模式时，先确认格式词条限制或改用其他格式；精度不达标则进入调优流程。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型目录 | 本地或 ModelScope/HF | 可被目标 Transformers 版本加载 | `from_pretrained` 冒烟通过 |
| 输入 | 量化 YAML / 最佳实践 | 最佳实践库或自定义 `config` | 含 `spec.save` 且 `type` 为 `compressed_tensors` | 字段通过配置协议校验 |
| 交付件 | compressed-tensors 权重目录 | `${SAVE_PATH}` | 含注入 `quantization_config` 的 `config.json` 与 `model*.safetensors` | 见《[compressed-tensors](term_compressed_tensors.md#export-artifacts)》导出产物 |

## 4. 流程总览

```mermaid
flowchart LR
  adapt[确认模式支持] --> config[配置 save] --> run[执行量化并核对产物]
```

## 5. 操作步骤

### 步骤 1：确认目标推理框架支持所选量化模式

**目标**：对照支持表，确认目标推理框架可按 compressed-tensors / HF `quantization_config` 加载所选 Preset（当前为 W8A8 Static / Dynamic）。

**操作**：

1. 确认目标框架读取 `config.json` 中 `quantization_config.quant_method == "compressed-tensors"`（或显式 `quantization="compressed-tensors"`），而非 Ascend 私有加载路径。
2. 对照《[compressed-tensors](term_compressed_tensors.md#mode-support)》量化模式支持情况，确认当前仅 W8A8 Static / Dynamic 有导出 handler；不支持分布式导出与 KV Cache scheme。
3. 规划推理启动参数：避免误用 `--quantization ascend`。

**输出**：明确的目标框架、Preset（W8A8 Static / Dynamic）与加载参数说明。

**通过条件**：框架文档声明支持 compressed-tensors / HF `quantization_config`；所选 Preset 落在支持表内。

### 步骤 2：配置 compressed-tensors 保存器

**目标**：在一键量化 YAML 中启用 `compressed_tensors`。

**操作**：在 `spec.save` 中配置：

```yaml
spec:
  save:
    - type: "compressed_tensors"
      part_file_size: 4
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `type` | string | `"compressed_tensors"` | 保存器类型标识，固定值 |
| `part_file_size` | int | `4` | 权重分片大小（GB）；`>0` 时分片，`0` 表示不分片 |

量化过程**无需**安装 `compressed-tensors` Python 包；导出时由 msModelSlim 内置 schema 写入元数据。完整协议见《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md#5252-compressed_tensors)》中 `compressed_tensors` 字段说明。

**输出**：可用的 YAML 配置文件路径。

**通过条件**：`type` 为 `compressed_tensors`；所选量化模式在《[compressed-tensors](term_compressed_tensors.md#mode-support)》Preset 表内。

### 步骤 3：执行量化并核对产物

**目标**：生成可被 vLLM 等加载的权重并完成加载与推理验证。

**操作**：

1. 准备最小可执行 YAML（保存为 `${CONFIG_PATH}`）。以下以 W8A8 静态量化 + `compressed_tensors` 为例；`process` 可按所选 Preset 替换（动态量化将 `act.scope` 改为 `"per_token"` 并调整 `symmetric`），`save` 须保留 `compressed_tensors`：

   ```yaml
   apiversion: modelslim_v1
   spec:
     process:
       - type: "linear_quant"
         qconfig:
           act:
             scope: "per_tensor"
             dtype: "int8"
             symmetric: False
             method: "minmax"
           weight:
             scope: "per_channel"
             dtype: "int8"
             symmetric: True
             method: "minmax"
         include: ["*"]
     save:
       - type: "compressed_tensors"
         part_file_size: 4
   ```

2. 按《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md)》执行量化：

   ```bash
   msmodelslim quant \
     --model_path ${MODEL_PATH} \
     --save_path ${SAVE_PATH} \
     --device npu \
     --model_type ${MODEL_TYPE} \
     --config ${CONFIG_PATH} \
     --trust_remote_code true
   ```

3. 核对 `${SAVE_PATH}` 中至少存在：
   - `config.json` 含 `quantization_config`，且 `quant_method` 为 `"compressed-tensors"`
   - `model.safetensors` 或分片权重 + `model.safetensors.index.json`
   - 自源模型复制的 HF 辅助文件齐全  
   目录树与字段细则见《[compressed-tensors](term_compressed_tensors.md#export-artifacts)》导出产物及各量化模式交付件格式。
4. 使用目标推理框架加载该目录，完成至少 1 条 generate 或 API 请求，确认量化权重可正常加载且推理返回正常。该步骤为部署前的快速验证，不要求完整精度评测。

**输出**：可部署的 `${SAVE_PATH}` 目录与验证日志。

**通过条件**：无权重 shape / dtype mismatch；`quantization_config` 字段合法；至少 1 条请求返回正常。

## 6. 验收条件

- 仅有 safetensors、缺少合法 `quantization_config` 时不得宣称 compressed-tensors 交付完成。
- 使用未实现 handler 的量化模式导致导出不完整时，不得进入精度验收。
- Preset 与张量键须与《[compressed-tensors](term_compressed_tensors.md)》说明一致。

## 7. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| compressed-tensors | HF / vLLM 生态量化格式 | 《[compressed-tensors](term_compressed_tensors.md)》 |
| 量化格式 | 工具与推理框架的落盘协议 | 《[量化格式](../README.md)》 |
