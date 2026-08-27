# LAOS: W4A4 Quantization Scheme

## Overview

- **Background**: In low-bit quantization scenarios (such as W4A4), model accuracy loss is particularly significant. The core challenge is that extreme outliers in weights and activations significantly distort the quantization range. This leads to a sharp decline in numerical representation precision, which cannot be resolved by traditional methods.
- **Core idea**: This algorithm combines rotation matrix optimization with low-bit quantization (implemented through rounding offset parameter training). First, this scheme uses [Adapt Rotation](../outlier_suppression_algorithms/adapt_rotation.md) to perform rotation matrix optimization on the Qwen3 dense model in stages to effectively suppress outliers. Then, it uses [AutoRound](autoround.md) to perform low-bit quantization and optimize rounding offset parameters, thereby improving accuracy and stability in W4A4 scenarios.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Application Requirements

- **Low-bit quantization**: This algorithm is suitable for 4-bit quantization in extremely low-bit quantization scenarios.
- **High precision**: This algorithm maintains high model accuracy even under low-bit conditions.
- **Compute resources**: Additional optimization is required, and the computing cost is higher than that of simple quantization methods.
- **Usage limitations**:

  - Sufficient calibration data or training iterations are required to optimize parameters. Because iterative optimization is involved, the quantization duration is longer than that of other methods.
  - Currently, this scheme is mainly intended for low-bit quantization of Qwen3 dense series models (such as Qwen3-8B, Qwen3-14B, and Qwen3-32B). Generalization to other model series is not guaranteed.

## Function Description

**Note: The algorithm implementation includes a training process that requires a minimum amount of NPU memory. Only devices with NPU memory of 64 GB or more are supported.**

### YAML Configuration Example

The following example shows a YAML configuration when the algorithm is used as a processor:

```yaml
apiversion: modelslim_v1

metadata:
  config_id: qwen3-32b-dense-w4a4
  score: 90
  verified_model_types:
    - Qwen3-32B
  label:
    w_bit: 4
    a_bit: 4
    is_sparse: False
    kv_cache: False

default_w8a8_dynamic: &default_w8a8_dynamic
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: true
    method: "autoround"
    ext:
      scale_dtype: "bfloat16"
  act:
    scope: "per_token"
    dtype: "int8"
    symmetric: true
    method: "minmax"
    ext:
      scale_dtype: "bfloat16"


default_w4a4_dynamic: &default_w4a4_dynamic
  weight:
    scope: "per_channel"
    dtype: "int4"
    symmetric: true
    method: "autoround"
    ext:
      scale_dtype: "bfloat16"
  act:
    scope: "per_token"
    dtype: "int4"
    symmetric: true
    method: "minmax"
    ext:
      scale_dtype: "bfloat16"


spec:
  prior:
    - process:
      - type: "adapt_rotation"
        stage: 1
        steps: 20
        layer_type:
          - "up_proj"

  process:
    - type: "adapt_rotation"
      stage: 2
      online: false
      block_size: -1
      max_tp_size: 1

    - type: "autoround_quant"
      iters: 400
      enable_round_tuning: true
      strategies:
        - qconfig: *default_w8a8_dynamic
          include:
            - "*self_attn*"
            - "*.down_proj"
            - "model.layers.{1,2,3,4,5,6,7,8,30,31,32,43,44,45,46,52,60,61,62,63}.mlp.up_proj"
            - "model.layers.{1,2,3,4,5,6,7,8,30,31,32,43,44,45,46,52,60,61,62,63}.mlp.gate_proj"

        - qconfig: *default_w4a4_dynamic
          include:
            - "*.up_proj"
            - "*.gate_proj"
          exclude:
            - "model.layers.{1,2,3,4,5,6,7,8,30,31,32,43,44,45,46,52,60,61,62,63}.mlp.up_proj"
            - "model.layers.{1,2,3,4,5,6,7,8,30,31,32,43,44,45,46,52,60,61,62,63}.mlp.gate_proj"

  save:
    - type: "ascendv1_saver"
      part_file_size: 4

  dataset: laos_calib.jsonl

```

### YAML Configuration Fields

Configuration fields are derived from the Adapt Rotation and AutoRound processors. For details, see [Adapt Rotation YAML Configuration Fields](../outlier_suppression_algorithms/adapt_rotation.md#yaml-configuration-fields) and [AutoRound-YAML-Configuration-Fields](autoround.md#yaml-configuration-fields).

## Model Adaptation

### Adaptation Procedure

- **Prerequisites**

  - Ensure that all returned module references are actual module objects in the model.
  - Module paths must be identical to the paths returned by `model.named_modules()`.

- **Procedure**

  1. Define the quantization strategy in the configuration file. Different quantization policies are supported for different layers.
  2. Configure `adapt_rotation` (`stage1` + `stage2`) in the configuration file to complete the rotation matrix optimization.
  3. Use `autoround_quant` in the configuration file to specify the `AutoRound` processor and configure related parameters and policy matching rules.
  4. To use a custom calibration dataset, refer to `msmodelslim/lab_calib` to add the dataset and specify the dataset name in the configuration file.
