# LAOS：w4a4量化方案说明

## 简介

- **背景**：在低比特量化（如W4A4）场景下，模型精度损失尤为显著，其核心难点在于权重和激活值中的极端离群值会显著扭曲量化区间，导致数值表示精度急剧下降，传统方法难以解决。
- **核心思想**：核心思想是“旋转矩阵优化 + 基于舍入偏移参数训练的低比特量化”。先通过 [Adapt Rotation](../outlier_suppression_algorithms/adapt_rotation.md) 在Qwen3稠密模型上进行旋转矩阵优化（分阶段执行），实现有效的离群值抑制，再通过 [AutoRound](autoround.md) 进行低比特量化与舍入偏移参数优化，从而提升 W4A4 场景下的精度与稳定性。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

## 适用要求

- **低比特量化**：适合极低比特量化场景中的4比特量化。
- **高精度需求**：在低比特条件下仍能保持较高的模型精度。
- **计算资源**：需要额外的优化过程，计算成本高于简单量化方法。
- **使用限制**：

  - 需要足够的校准数据或训练迭代次数来优化参数，由于涉及到迭代优化，量化时长相对其他方法相对较久。
  - 当前该方案主要面向Qwen3稠密系列模型（如Qwen3-8B/14B/32B）的低比特量化场景，不保证可泛化到其他系列模型。

## 功能介绍

### 昇腾AI处理器支持情况

| 产品系列 | 支持 |
|---------|------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | ✓ |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | ✓ |
| Atlas 推理系列产品 | ✗ |

**注：算法实现包含训练过程，对NPU显存有一定的要求，仅支持NPU显存>=64G的设备。**

### YAML配置示例

作为Processor使用，YAML配置示例如下：

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

### YAML配置字段详解

配置字段来自 Adapt Rotation 与 AutoRound 处理器，详见 [Adapt Rotation YAML配置字段详解](../outlier_suppression_algorithms/adapt_rotation.md#yaml-配置字段详解)、[AutoRound YAML配置字段详解](autoround.md#yaml配置字段详解)。

## 模型适配

### 适配步骤

- **前置要求**：

  - 确保所有返回的模块引用都是实际模型中的模块对象。
  - 模块路径必须与model.named_modules()返回的路径完全一致。

- **步骤**：

  1. 在配置文件中定义量化策略，支持针对不同的层使用不同的量化策略。
  2. 在配置文件中配置 `adapt_rotation`（stage1 + stage2）完成旋转矩阵优化。
  3. 在配置文件中使用 `autoround_quant` 指定 AutoRound 处理器，并配置相关参数与策略匹配规则。
  4. 如需使用自定义校准集，可参考 `msmodelslim/lab_calib`添加数据集，并在配置文件中指定数据集名称。
