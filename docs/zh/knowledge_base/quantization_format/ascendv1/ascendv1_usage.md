# AscendV1 使用指南

本指南说明如何在 msModelSlim 中选用 **AscendV1** 量化格式，按 **确认模式支持 →（可选）适配器适配 save → 配置 → 执行** 完成落盘，并将产物部署到 vLLM Ascend、SGLang、MindIE。格式字段与枚举见《[AscendV1](term_ascendv1.md)》；一键量化命令总览见《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md)》。

## 1. 适用范围

本流程适用于：已选定 **AscendV1** 作为落盘格式，需在 msModelSlim 中完成配置、执行一键量化，并将产物部署到 **vLLM Ascend**、**SGLang** 或 **MindIE** 等昇腾推理栈的用户。

不适用：仅导出 HuggingFace / vLLM 通用 `compressed-tensors` 权重（请改用《[compressed-tensors 使用指南](../compressed_tensors/compressed_tensors_usage.md)》）；多模态生成 MindIE-SD 场景（请改用《[MindIE-SD 使用指南](../mindie_sd/mindie_sd_usage.md)》）。

## 2. 流程关系与前置条件

**上级流程**：《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md)》

**前置条件**：

- 已安装兼容版本的 msModelSlim，且目标模型在《[大模型支持矩阵](../../model/README.md)》中可量化
- 已确认目标推理框架为 vLLM Ascend、SGLang 或 MindIE

**后续操作**：按推理框架文档加载 `save_path` 产物；精度不达标则进入调优流程。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型目录 | 本地或 ModelScope/HF | 可被目标 Transformers 版本加载 | `from_pretrained` 冒烟通过 |
| 输入 | 量化 YAML / 最佳实践 | 最佳实践库或自定义 `config` | 含 `spec.save` 且 `type` 为 `ascendv1_saver` | 字段通过配置协议校验 |
| 交付件 | AscendV1 量化权重目录 | `${SAVE_PATH}` | 含 `quant_model_description.json` 与权重 safetensors | 见《[AscendV1](term_ascendv1.md#export-artifacts)》导出产物 |

## 4. 流程总览

```mermaid
flowchart LR
  adapt[确认模式支持] --> adapter["适配器适配save（可选）"] --> config[配置save] --> run[执行量化并核对产物]
```

## 5. 操作步骤

### 步骤 1：确认目标推理框架支持所选量化模式

**目标**：对照支持表，确认目标推理框架（vLLM Ascend、SGLang 或 MindIE）可加载所选量化模式及 AscendV1 约定。

**操作**：

1. 对照《[AscendV1](term_ascendv1.md#engine-support)》推理引擎支持情况与《[AscendV1](term_ascendv1.md#mode-support)》量化模式支持情况，以及目标框架版本说明，确认所选量化模式可被该栈加载。
2. 确认推理侧通过 `quant_model_description.json` 识别各张量量化类型，并从 `quant_model_weights*.safetensors`（或分片 + index）加载参数。
3. 若启用 QuaRot 且需导出旋转矩阵，确认推理框架可消费 `optional/quarot.safetensors`（见《[AscendV1](term_ascendv1.md#optional-quarot)》可选导出：QuaRot 相关文件）。

**输出**：明确的目标框架、量化模式与是否导出 optional 的决策记录。

**通过条件**：已选定 vLLM Ascend、SGLang 或 MindIE 作为部署目标；已确认采用 AscendV1 约定（`quant_model_description.json` + 权重 safetensors）加载；若启用 QuaRot 可选导出，推理侧已确认可加载 `optional/`。

### 步骤 2（可选）：模型适配器实现 AscendV1SaveInterface

**目标**：仅当落盘需要模型侧预处理 / 后处理时，在模型适配器中实现 `AscendV1SaveInterface`；多数模型可跳过本步骤，直接进入步骤 3。

**何时需要**：默认 `ascendv1_saver` 已能正确遍历并导出时不必实现。仅当出现以下情况之一时才适配：

- 保存前需改写模块前缀或替换待导出模块（如结构映射、专家子模块重定向）
- 导出完成后需对产物目录做后处理（如改写 `config.json`、补充辅助文件）

**操作**：

1. 让模型适配器继承 [`AscendV1SaveInterface`](../../../../../msmodelslim/core/quant_service/modelslim_v1/save/interface.py)，按需实现：
   - `ascendv1_save_module_preprocess(prefix, module, model)`：保存模块前返回新的 `(prefix, module)`
   - `ascendv1_save_postprocess(model, save_directory)`：全部导出件写完后的目录后处理
2. 保存器仅在 `isinstance(adapter, AscendV1SaveInterface)` 时调用上述钩子；未实现则走默认落盘路径。
3. 参考已实现该接口的适配器源码核对行为是否与目标推理栈一致，例如：
   - [`msmodelslim/model/deepseek_v3/`](../../../../../msmodelslim/model/deepseek_v3/)
   - [`msmodelslim/model/qwen3_next/`](../../../../../msmodelslim/model/qwen3_next/)
   - [`msmodelslim/model/minimax_m2/`](../../../../../msmodelslim/model/minimax_m2/)

**输出**：已实现钩子的适配器，或“无需实现、跳过本步骤”的结论。

**通过条件**：不需要特殊钩子则可跳过；若实现了接口，则预处理 / 后处理与目标 AscendV1 产物约定一致。

### 步骤 3：配置 AscendV1 保存器

**目标**：在一键量化 YAML 中启用 `ascendv1_saver`。

**操作**：在 `spec.save` 中配置：

```yaml
spec:
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `type` | string | `"ascendv1_saver"` | 保存器类型标识，固定值 |
| `part_file_size` | int | `4` | 权重分片大小（GB）；`0` 表示不分片 |
| `ext` | object | `{}` | 可选扩展配置；当前 `AscendV1Saver` 不读取此字段，常规导出可省略 |

也可直接使用官方 `quant_type` 最佳实践（多数 LLM 最佳实践默认已含 `ascendv1_saver`）。完整协议见《[一键量化使用指南](../../../user_guide/usage_quick_quantization.md#5251-ascendv1_saver)》中 `ascendv1_saver` 字段说明。

**输出**：可用的 YAML 配置文件路径，或确认采用官方 `quant_type` 一键路径。

**通过条件**：`type` 为 `ascendv1_saver`；所选量化模式落在《[AscendV1](term_ascendv1.md#mode-support)》支持表内。

### 步骤 4：执行量化并核对产物

**目标**：生成 AscendV1 权重并完成加载与推理验证。

**操作**：

1. 准备最小可执行 YAML（保存为 `${CONFIG_PATH}`）。以下以全局 W8A8 动态量化 + `ascendv1_saver` 为例；`process` 可按所选量化模式替换，`save` 须保留 `ascendv1_saver`：

   ```yaml
   apiversion: modelslim_v1
   spec:
     process:
       - type: "linear_quant"
         qconfig:
           act:
             scope: "per_token"
             dtype: "int8"
             symmetric: True
             method: "minmax"
           weight:
             scope: "per_channel"
             dtype: "int8"
             symmetric: True
             method: "minmax"
         include: ["*"]
         exclude: ["*down_proj*"]
     save:
       - type: "ascendv1_saver"
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
   - `quant_model_description.json`（含 `model_quant_type` 与各张量类型）
   - `quant_model_weights.safetensors` 或分片权重 + index
   - 自源模型复制的 `config.json` / tokenizer 等辅助文件  
   目录树与字段细则见《[AscendV1](term_ascendv1.md#export-artifacts)》导出产物及各量化模式交付件格式。
4. 使用目标推理框架加载该目录，完成至少 1 条 generate 或 API 请求，确认量化权重可正常加载且推理返回正常。该步骤为部署前的快速验证，不要求完整精度评测。

**输出**：可部署的 `${SAVE_PATH}` 目录与验证日志。

**通过条件**：描述文件与权重齐全；推理加载正常；至少 1 条请求返回正常。

## 6. 验收条件

- 未确认目标推理栈支持 AscendV1 加载约定前，不得宣称可部署。
- 缺少 `quant_model_description.json` 或权重文件时不得进入推理评测。
- 量化模式字段须与《[AscendV1](term_ascendv1.md)》枚举一致。

## 7. 术语

| 术语 | 简述 | 链接 |
| --- | --- | --- |
| AscendV1 | 昇腾侧量化落盘格式 | 《[AscendV1](term_ascendv1.md)》 |
| 量化格式 | 工具与推理框架的落盘协议 | 《[量化格式](../README.md)》 |
