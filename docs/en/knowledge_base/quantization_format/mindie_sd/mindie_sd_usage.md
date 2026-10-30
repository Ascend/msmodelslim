# MindIE SD Usage Guide

This guide explains how to select the **MindIE SD** quantization format in msModelSlim, complete the write-to-disk step in the order **confirm mode support -> configure save -> run quantization and verify the artifacts**, and deploy the artifacts to the MindIE multimodal generation path. For format fields and enumerations see *[MindIE SD](term_mindie_sd.md)*; for model integration and running quantization with `--config` see the *[Weight Quantization Usage Guide](../../../user_guide/usage_weight_quantization.md)*; for the configuration protocol and best-practice entry point see the *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*.

## 1. Scope

This workflow applies to users who run msModelSlim one-click quantization on a **multimodal generation** model and need to save the result in the **MindIE SD** format (`mindie_format_saver`) for MindIE to load.

Not applicable: AscendV1 export for general LLMs (the *[AscendV1 Usage Guide](../ascendv1/ascendv1_usage.md)*); HF / vLLM compressed-tensors export (the *[compressed-tensors Usage Guide](../compressed_tensors/compressed_tensors_usage.md)*).

## 2. Process Relationships and Prerequisites

**Parent process**: *[Weight Quantization Usage Guide](../../../user_guide/usage_weight_quantization.md)*; for model-side integration details see the *[Multimodal Generation Model Integration Guide](../../ptq/dit/integration_guide_diffusion_transformer_quantization.md)*; for the configuration protocol and best-practice entry point see the *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*.

**Prerequisites**:

- A compatible version of msModelSlim is installed, and the target `model_type` has been verified in the support matrix or a best practice
- The deployment target has been confirmed as the MindIE multimodal generation path

**Follow-up operations**: load `${SAVE_PATH}` according to the MindIE documentation; if the accuracy or image quality is insufficient, iterate using the multimodal tuning practices.

## 3. Inputs and Deliverables

| Type | Name | Source or Location | Format or Constraint | Acceptance Method |
| --- | --- | --- | --- | --- |
| Input | Multimodal generation floating-point model | Local path | Can be loaded by the adapter | The adapter initializes successfully |
| Input | YAML (`apiversion: multimodal_sd_modelslim_v1`) or the official `quant_type` | Best practice / lab_practice / `${CONFIG_PATH}` | `save` contains `mindie_format_saver` | Configuration validation passes |
| Deliverable | MindIE SD quantization directory | `${SAVE_PATH}` | Contains the description JSON and `quant_model_weight*.safetensors` | See the *[MindIE SD](term_mindie_sd.md#export-artifacts)* export artifacts |

## 4. Process Overview

```mermaid
flowchart LR
  adapt[Confirm mode support] --> config[Configure save] --> run[Run quantization and verify artifacts]
```

## 5. Procedure

### Step 1: Confirm the Target Inference Framework Supports the Chosen Quantization Mode

**Objective**: Referring to the support tables, confirm that the MindIE multimodal generation path can load the chosen quantization mode and the MindIE SD conventions.

**Action**:

1. Confirm that the business scenario is multimodal generation (such as Wan2.2 T2V / I2V / TI2V) and that the MindIE version supports the corresponding quantized weights; the MindIE SD path (`mindie_format_saver`) must be used. Do not switch multimodal generation scenarios to AscendV1 / compressed-tensors.
2. Referring to *[MindIE SD](term_mindie_sd.md#mode-support)* Quantization Mode Support and the recommended YAML in example / lab_practice, confirm that the chosen mode is within the MindIE SD support table.
3. Clarify that `multimodal_sd_config.inference_config` (such as `task`, `size`, and `frame_num`) must be consistent with the `--model_type` scenario.
4. Clarify whether dump is enabled (`dump_config.enable_dump`) and what the `dump_data_dir` is (see *[MindIE SD](term_mindie_sd.md#optional-dump)* Optional Export).

**Output**: A decision record of the target model scenario, the MindIE version, and the quantization mode.

**Pass conditions**: A corresponding best practice or integration document states support for `mindie_format_saver`; the chosen mode falls within the MindIE SD support table.

### Step 2: Configure the MindIE SD Saver

**Objective**: Enable `mindie_format_saver` in the quantization configuration.

**Action**: Configure the following in `spec.save`:

```yaml
spec:
  save:
    - type: "mindie_format_saver"
      part_file_size: 0
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | string | `"mindie_format_saver"` | Saver type identifier; a fixed value |
| `part_file_size` | int | `4` (the code default; multimodal examples often write `0`) | Weight shard size (GB); `0` means no sharding |
| `ext` | object | `{}` | Optional extension configuration; it can be omitted for regular exports |

Alternatively, use the official `quant_type` best practice directly (multimodal best practices usually already include `mindie_format_saver`). For the complete protocol see the *[mindie_format_saver Configuration Description](../../../api_reference/config/format/mindie_format_saver.md)* and the *[multimodal_sd_modelslim_v1 Configuration Description](../../../api_reference/config/quant/multimodal_sd_modelslim_v1.md)*.

**Output**: A usable YAML configuration path, or confirmation that the official `quant_type` one-click path is used.

**Pass conditions**: `type` is `mindie_format_saver`; `apiversion` is `multimodal_sd_modelslim_v1` (the official `quant_type` best-practice entry point can also be used directly, without hand-writing a complete YAML).

### Step 3: Run Quantization and Verify the Artifacts

**Objective**: Generate a directory loadable by MindIE and complete the loading and generation validation.

**Action**:

1. Prepare a minimal runnable YAML (save it as `${CONFIG_PATH}`). The following takes the linear layer MXFP8 plus `mindie_format_saver` for a Wan2.2 T2V scenario as an example; `process` / `inference_config` can be replaced according to the scenario, `save` must keep `mindie_format_saver`, and `apiversion` must be `multimodal_sd_modelslim_v1`.

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

2. Confirm that the target model has completed multimodal integration, then run quantization with the `${CONFIG_PATH}` written in point 1 of this step. For the complete procedure of integration and explicit configuration see the *[Weight Quantization Usage Guide](../../../user_guide/usage_weight_quantization.md)*:
   - **Not yet integrated**: first complete the adapter development and registration as described in [Step 2: Complete Model Adaptation](../../../user_guide/usage_weight_quantization.md#step-2-complete-model-adaptation) of that guide, then proceed to quantization.
   - **Already integrated**: after confirming that `--model_type` can hit the adapter, run quantization with `--config ${CONFIG_PATH}` as described in [Step 4: Run the One-Click Quantization Command](../../../user_guide/usage_weight_quantization.md#step-4-run-the-one-click-quantization-command) of that guide:

   ```bash
   msmodelslim quant \
     --model_path ${MODEL_PATH} \
     --save_path ${SAVE_PATH} \
     --device npu \
     --model_type ${MODEL_TYPE} \
     --config ${CONFIG_PATH} \
     --trust_remote_code true
   ```

3. Verify that the following exist under `${SAVE_PATH}`:
   - `quant_model_description.json` (or a variant with a quantization type suffix)
   - `quant_model_weight.safetensors` (or sharded weights plus an index, or a variant with a quantization type suffix)
   - The `.json` / `.py` configuration and code files copied from the source model
   For the directory tree and field details see *[MindIE SD](term_mindie_sd.md#export-artifacts)* Export Artifacts and Deliverable Formats of Each Quantization Mode.
4. Load this directory according to the MindIE deployment documentation and complete one generation validation (the resolution and frame count must match `inference_config`). This step is a quick pre-deployment validation and does not require a full image-quality evaluation.

**Output**: A deployable `${SAVE_PATH}` and the validation result.

**Pass conditions**: The description file and weights are complete; MindIE can load them; the generation flow reports no weight shape / dtype loading error.

## 6. Acceptance Conditions

- Weights produced by mistakenly using the AscendV1 / compressed-tensors saver must not be accepted through the MindIE SD path.
- Do not proceed to inference evaluation when the description JSON or `quant_model_weight*.safetensors` is missing.
- When `inference_config.task` and so on are inconsistent with `model_type`, correct them before evaluation.
- The quantization mode field must be consistent with the support table in *[MindIE SD](term_mindie_sd.md#mode-support)*.

## 7. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| MindIE SD | Multimodal generation MindIE write-to-disk format | [MindIE SD](term_mindie_sd.md) |
| Quantization Format | The write-to-disk protocol between the quantization tool and the inference framework | *[Quantization Format](../README.md)* |
| Weight Quantization | The business path of integrating the model, writing the configuration, and running quantization with --config | *[Weight Quantization Usage Guide](../../../user_guide/usage_weight_quantization.md)* |
