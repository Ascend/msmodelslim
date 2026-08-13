# MindIE-SD 使用指南

本指南说明如何在 msModelSlim 中选用 **MindIE-SD** 量化格式，按 **确认模式支持 → 配置 → 执行** 完成落盘，并将产物部署到 MindIE 多模态生成路径。格式字段与枚举见《[MindIE-SD](term_mindie_sd.md)》；一键量化与多模态协议见《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md)》。

## 1. 适用范围

本流程适用于：对 **多模态生成** 模型执行 msModelSlim 一键量化，并需将结果保存为 **MindIE-SD** 格式（`mindie_format_saver`）、供 MindIE 加载的用户。

不适用：通用 LLM 的 AscendV1 导出（《[AscendV1 使用指南](../ascendv1/ascendv1_usage.md)》）；HF / vLLM compressed-tensors 导出（《[compressed-tensors 使用指南](../compressed_tensors/compressed_tensors_usage.md)》）。

## 2. 流程关系与前置条件

**上级流程**：《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md)》；模型侧参见《[多模态生成模型接入](../../model/integrating_multimodal_generation_model.md)》。

**前置条件**：

- 已安装兼容版本的 msModelSlim，目标 `model_type` 已在支持矩阵或最佳实践中验证
- 确认部署目标为 MindIE 多模态生成路径

**后续操作**：按 MindIE 文档加载 `${SAVE_PATH}`；精度或画质不达标则按多模态调优实践迭代。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 多模态生成浮点模型 | 本地路径 | 适配器可加载 | 适配器 init 成功 |
| 输入 | YAML（`apiversion: multimodal_sd_modelslim_v1`）或官方 `quant_type` | 最佳实践 / lab_practice | `save` 含 `mindie_format_saver` | 配置校验通过 |
| 交付件 | MindIE-SD 量化目录 | `${SAVE_PATH}` | 含描述 JSON 与 `quant_model_weight*.safetensors` | 见《[MindIE-SD](term_mindie_sd.md#export-artifacts)》导出产物 |

## 4. 流程总览

```mermaid
flowchart LR
  adapt[确认模式支持] --> config[配置save] --> run[执行量化并核对产物]
```

各阶段对应下文步骤 1～3：**确认**目标推理框架是否支持所选量化模式、**配置** `mindie_format_saver`、**执行**一键量化并核对交付件。

> 说明：MindIE-SD 无 AscendV1 侧的 `AscendV1SaveInterface` 钩子；新模型接入见《[多模态生成模型接入](../../model/integrating_multimodal_generation_model.md)》，本流程不单独设「适配器适配 save」步骤。

## 5. 操作步骤

### 步骤 1：确认目标推理框架支持所选量化模式

**目标**：对照支持表，确认 MindIE 多模态生成路径可加载所选量化模式及 MindIE-SD 约定。

**操作**：

1. 确认业务为多模态生成（如 Wan2.2 T2V / I2V / TI2V），且 MindIE 版本支持对应量化权重；走 MindIE-SD，而非 AscendV1 / compressed-tensors。
2. 对照《[MindIE-SD](term_mindie_sd.md#mode-support)》「量化模式支持情况」与 example / lab_practice 推荐 YAML，确认所选模式在 MindIE-SD 支持表内；不支持的模式改走《[AscendV1](../ascendv1/term_ascendv1.md)》。
3. 明确 `multimodal_sd_config.inference_config`（如 `task`、`size`、`frame_num`）须与 `--model_type` 场景一致。
4. 明确是否启用 dump（`dump_config.enable_dump`）及 `dump_data_dir`（见《[MindIE-SD](term_mindie_sd.md#optional-dump)》可选导出）。

**输出**：目标模型场景、MindIE 版本与量化模式决策记录。

**通过条件**：存在对应最佳实践或接入文档声明支持 `mindie_format_saver`；所选模式落在 MindIE-SD 支持表内。

### 步骤 2：配置 MindIE-SD 保存器

**目标**：在量化配置中启用 `mindie_format_saver`。

**操作**：在 `spec.save` 中配置：

```yaml
spec:
  save:
    - type: "mindie_format_saver"
      part_file_size: 0
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `type` | string | `"mindie_format_saver"` | 保存器类型标识，固定值 |
| `part_file_size` | int | `4`（代码默认；多模态示例常写 `0`） | 权重分片大小（GB）；`0` 表示不分片 |
| `ext` | object | `{}` | 可选扩展配置；常规导出可省略 |

也可直接使用官方 `quant_type` 最佳实践（多模态最佳实践通常已含 `mindie_format_saver`）。完整协议见《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md#5341-mindie_format_saver)》及 `#53-multimodal_sd_modelslim_v1-配置详解`。

**输出**：可用 YAML 配置路径，或确认采用官方 `quant_type` 一键路径。

**通过条件**：`type` 为 `mindie_format_saver`；`apiversion` 为 `multimodal_sd_modelslim_v1`（或等价官方入口）。

### 步骤 3：执行量化并核对产物

**目标**：生成 MindIE 可加载目录并完成冒烟核对。

**操作**：

1. 准备最小可执行 YAML（保存为 `${CONFIG_PATH}`）。以下以 Wan2.2 T2V 场景的线性层 MXFP8 + `mindie_format_saver` 为例；`process` / `inference_config` 可按场景替换，`save` 须保留 `mindie_format_saver`，`apiversion` 须为 `multimodal_sd_modelslim_v1`：

   ```yaml
   apiversion: multimodal_sd_modelslim_v1
   spec:
     process:
       - type: "linear_quant"
         qconfig:
           act:
             scope: "per_block"
             dtype: "mxfp8"
             symmetric: True
             method: "minmax"
           weight:
             scope: "per_block"
             dtype: "mxfp8"
             symmetric: True
             method: "mse_round"
         include: ["*"]
     dataset: wan2_2_t2v
     save:
       - type: "mindie_format_saver"
         part_file_size: 0
     multimodal_sd_config:
       dump_config:
         enable_dump: False
         capture_mode: "args"
         dump_data_dir: ""
       inference_config:
         size: "1280*720"
         frame_num: 81
         sample_steps: 40
         convert_model_dtype: True
         task: "t2v-A14B"
   ```

2. 按《[多模态生成模型接入](../../model/integrating_multimodal_generation_model.md)》或《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md)》执行量化：

   ```bash
   msmodelslim quant \
     --model_path ${MODEL_PATH} \
     --save_path ${SAVE_PATH} \
     --device npu \
     --model_type Wan2.2-T2V-A14B \
     --config_path ${CONFIG_PATH} \
     --trust_remote_code True
   ```

3. 核对 `${SAVE_PATH}` 中至少存在：
   - `quant_model_description.json`（或带量化类型后缀的变体）
   - `quant_model_weight.safetensors`（或分片 + index / 带量化类型后缀的变体）
   - 自源模型复制的 `.json` / `.py` 配置与代码文件  
   目录树与字段细则见《[MindIE-SD](term_mindie_sd.md#export-artifacts)》导出产物及「各量化模式交付件格式」。
4. 按 MindIE 部署文档加载该目录，完成一次生成冒烟（分辨率 / 帧数等与 `inference_config` 一致）。

**输出**：可部署的 `${SAVE_PATH}` 与冒烟结果。

**通过条件**：描述文件与权重齐全；MindIE 可加载；生成流程无权重 shape / dtype 加载错误。

## 6. 验收条件

- 误用 AscendV1 / compressed-tensors 保存器产出的权重，不得按 MindIE-SD 路径验收。
- 缺少描述 JSON 或 `quant_model_weight*.safetensors` 时不得进入推理评测。
- `inference_config.task` 等与 `model_type` 不一致时，须先修正再评测。
- 量化模式字段须与《[MindIE-SD](term_mindie_sd.md#mode-support)》支持表一致。

## 7. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| MindIE-SD | 多模态生成 MindIE 落盘格式 | 《[MindIE-SD](term_mindie_sd.md)》 |
| 量化格式 | 工具与推理框架的落盘协议 | 《[量化格式](../README.md)》 |

## 8. 接口文档列表

| 接口或能力 | 简述 | 链接 |
| --- | --- | --- |
| 一键量化 | 命令与配置协议 | 《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md)》 |
| mindie_format_saver | save 字段 | 《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md#5341-mindie_format_saver)》 |
| multimodal_sd 配置 | dump / inference_config | 《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md#53-multimodal_sd_modelslim_v1-配置详解)》 |
| 多模态生成接入 | 模型与示例 | 《[多模态生成模型接入](../../model/integrating_multimodal_generation_model.md)》 |
| 格式接入 | 新格式开发对照 | 《[量化格式接入指南](../iformat_integration_guide.md)》 |
