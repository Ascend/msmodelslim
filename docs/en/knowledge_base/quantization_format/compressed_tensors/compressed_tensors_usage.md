# compressed-tensors Usage Guide

This guide explains how to select the **compressed-tensors** quantization format in msModelSlim, complete the write-to-disk step in the order **confirm mode support -> configure save -> run quantization and verify the artifacts**, and deploy the artifacts to inference frameworks such as vLLM that support the HuggingFace `quantization_config`. For format fields and Presets see *[compressed-tensors](term_compressed_tensors.md)*; for an overview of the one-click quantization commands see the *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*.

## 1. Scope

This workflow applies to users who need to export msModelSlim quantization results in the **compressed-tensors** format and load them in inference frameworks such as **vLLM** that support the HuggingFace `config.json` -> `quantization_config` path.

Not applicable: the default AscendV1 path for Ascend MindIE / vLLM Ascend (see the *[AscendV1 Usage Guide](../ascendv1/ascendv1_usage.md)*); multimodal MindIE SD (see the *[MindIE SD Usage Guide](../mindie_sd/mindie_sd_usage.md)*).

## 2. Process Relationships and Prerequisites

**Parent process**: *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*

**Prerequisites**:

- A compatible version of msModelSlim is installed, and the target model is quantizable according to the *[Foundation Model Support Matrix](../../model/README.md)*
- The quantization scheme falls within the QIR presets currently supported by compressed-tensors (currently W8A8 Static / W8A8 Dynamic; see *[compressed-tensors](term_compressed_tensors.md#mode-support)*)
- The target inference framework can recognize `quant_method: "compressed-tensors"`

**Follow-up operations**: load `${SAVE_PATH}` with vLLM and so on; when a mode whose handler is not implemented (such as W4A8) is needed, first check the limitations in the format entry or switch to another format; if the accuracy is insufficient, enter the tuning workflow.

## 3. Inputs and Deliverables

| Type | Name | Source or Location | Format or Constraint | Acceptance Method |
| --- | --- | --- | --- | --- |
| Input | Floating-point model directory | Local or ModelScope/HF | Can be loaded by the target Transformers version | `from_pretrained` smoke test passes |
| Input | Quantization YAML / best practice | The best-practice library or a custom `config` | Contains `spec.save` with `type` set to `compressed_tensors` | Fields pass configuration protocol validation |
| Deliverable | compressed-tensors weight directory | `${SAVE_PATH}` | Contains `config.json` with an injected `quantization_config` and `model*.safetensors` | See the *[compressed-tensors](term_compressed_tensors.md#export-artifacts)* export artifacts |

## 4. Process Overview

```mermaid
flowchart LR
  adapt[Confirm mode support] --> config[Configure save] --> run[Run quantization and verify artifacts]
```

## 5. Procedure

### Step 1: Confirm the Target Inference Framework Supports the Chosen Quantization Mode

**Objective**: Referring to the support table, confirm that the target inference framework can load the chosen Preset (currently W8A8 Static / Dynamic) through compressed-tensors / the HF `quantization_config`.

**Action**:

1. Confirm that the target framework reads `quantization_config.quant_method == "compressed-tensors"` from `config.json` (or an explicit `quantization="compressed-tensors"`) rather than an Ascend private loading path.
2. Referring to *[compressed-tensors](term_compressed_tensors.md#mode-support)* Quantization Mode Support, confirm that only W8A8 Static / Dynamic currently have an export handler, and that distributed export and KV Cache schemes are not supported.
3. Plan the inference startup parameters: avoid mistakenly using `--quantization ascend`.

**Output**: A description of the confirmed target framework, the Preset (W8A8 Static / Dynamic), and the loading parameters.

**Pass conditions**: The framework documentation states support for compressed-tensors / the HF `quantization_config`; the chosen Preset falls within the support table.

### Step 2: Configure the compressed-tensors Saver

**Objective**: Enable `compressed_tensors` in the one-click quantization YAML.

**Action**: Configure the following in `spec.save`:

```yaml
spec:
  save:
    - type: "compressed_tensors"
      part_file_size: 4
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | string | `"compressed_tensors"` | Saver type identifier; a fixed value |
| `part_file_size` | int | `4` | Weight shard size (GB); values `>0` enable sharding, while `0` means no sharding |

The quantization process does **not** require installing the `compressed-tensors` Python package; during export, msModelSlim writes the metadata using its built-in schema. For the complete protocol see the `compressed_tensors` field description in the *[compressed_tensors Configuration Description](../../../api_reference/config/format/compressed_tensors.md)*.

**Output**: A usable YAML configuration file path.

**Pass conditions**: `type` is `compressed_tensors`; the chosen quantization mode is within the Preset table of *[compressed-tensors](term_compressed_tensors.md#mode-support)*.

### Step 3: Run Quantization and Verify the Artifacts

**Objective**: Generate weights that can be loaded by vLLM and so on, and complete the loading and inference validation.

**Action**:

1. Prepare a minimal runnable YAML (save it as `${CONFIG_PATH}`). The following takes W8A8 static quantization plus `compressed_tensors` as an example; `process` can be replaced according to the chosen Preset (for dynamic quantization, change `act.scope` to `"per_token"` and adjust `symmetric`), while `save` must keep `compressed_tensors`:

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

2. Run quantization as described in the *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)*:

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
   - `config.json` contains `quantization_config`, and `quant_method` is `"compressed-tensors"`
   - `model.safetensors`, or sharded weights plus `model.safetensors.index.json`
   - The HF auxiliary files copied from the source model are complete
   For the directory tree and field details see *[compressed-tensors](term_compressed_tensors.md#export-artifacts)* Export Artifacts and Deliverable Formats of Each Quantization Mode.
4. Load this directory with the target inference framework and complete at least one generate or API request to confirm that the quantized weights load correctly and inference returns normally. This step is a quick pre-deployment validation and does not require a full accuracy evaluation.

**Output**: A deployable `${SAVE_PATH}` directory and a validation log.

**Pass conditions**: No weight shape / dtype mismatch; the `quantization_config` fields are valid; at least one request returns normally.

## 6. Acceptance Conditions

- Do not claim that compressed-tensors delivery is complete when only safetensors exist and a valid `quantization_config` is missing.
- When an unimplemented handler for a quantization mode causes an incomplete export, do not proceed to accuracy acceptance.
- The Preset and tensor keys must be consistent with the description in *[compressed-tensors](term_compressed_tensors.md)*.

## 7. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| compressed-tensors | The HF / vLLM ecosystem quantization format | *[compressed-tensors](term_compressed_tensors.md)* |
| Quantization Format | The write-to-disk protocol between the quantization tool and the inference framework | *[Quantization Format](../README.md)* |
