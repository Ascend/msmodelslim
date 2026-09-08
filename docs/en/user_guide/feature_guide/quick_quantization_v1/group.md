# Group Processor

## 1. Overview

The Group Processor is a core component in msModelSlim for implementing **refined quantization strategies**. It allows users to encapsulate multiple processors (such as linear quantization and smooth processing) into a logical group and apply differentiated quantization configurations to different layers of a model.

Through the Group Processor, you can easily implement mixed quantization. For example, you can use static quantization for some layers in the same model to maximize performance, and use dynamic quantization for other layers to ensure accuracy.

## 2. Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## 3. Principle and Implementation

### 3.1 Principle

When processing ultra-large-scale models, such as LLMs, the activation distribution characteristics of different modules, such as Attention and MLP, are often completely different. A single global quantization configuration cannot achieve the optimal balance between accuracy and performance across all layers.

The core principles of the Group Processor are **divide-and-conquer and process sharing**.

1. **Logical grouping**: Model layers are divided into different logical sets through `include` and `exclude` pattern matching.
2. **Local optimality**: Each set is assigned the processor type and quantization parameters that are most suitable for its distribution characteristics.
3. **Process sharing and acceleration**: This is the core advantage of using the Group Processor. In the quantization calibration phase, each independent processor, such as `linear_quant`, usually needs to execute a complete inference (forward) process on the calibration dataset.
   - **Without Group container**: If there are M independent `linear_quant` processors and the calibration dataset size is N, then M × N inference operations are required in total.
   - **With Group container**: After multiple processors are mounted to the Group container, they share the inference process, requiring only N inference operations in total.
   Therefore, using the Group container significantly improves the quantization speed. While mixed quantization can be implemented through multiple independent processors without using the Group container, the efficiency is lower. In addition, the results may have subtle differences due to different execution timing, though this usually does not affect the final accuracy.
4. **Resource optimization**: In layer-wise quantization mode, the Group Processor ensures that multiple processing operations for the same layer are completed in a single loading, significantly reducing I/O overhead.

### 3.2 Implementation

The Group Processor is implemented in [msmodelslim/processor/container/group.py](../../../../../msmodelslim/processor/container/group.py). It functions as a container to sequentially schedule the sub-processors in its `configs` list.

## 4. Function

### 4.1 Application Scenario

- **Mixed quantization strategy**: For example, W8A8 static quantization is used for the Attention layer, and W8A8 dynamic quantization is used for the MLP layer.
- **Sensitive layer protection**: Higher bit-widths or specific smooth processing algorithms are used for layers that are extremely sensitive to accuracy, such as `down_proj` or `gate`.
- **Multi-algorithm combination**: Smooth processing and linear quantization are applied to the same group of layers in sequence.

### 4.2 YAML Configuration Example

The following example demonstrates how to use the Group Processor to implement **W8A8 static and dynamic mixed quantization**:

```yaml
# 1. Define the static quantization template (anchor).
default_w8a8_static: &w8a8_static
  act:
    scope: "per_tensor"        # Static quantization
    dtype: "int8"
    symmetric: false
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: true
    method: "minmax"

# 2. Define the dynamic quantization template (anchor).
default_w8a8_dynamic: &w8a8_dynamic
  act:
    scope: "per_token"         # Dynamic quantization
    dtype: "int8"
    symmetric: true
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: true
    method: "minmax"

spec:
  process:
    - type: "group"            # Use the Group Processor.
      configs:
        - type: "linear_quant" # Sub-processor 1: targeted at the Attention layer.
          qconfig: *w8a8_static
          include: ["*self_attn*"]

        - type: "linear_quant" # Sub-processor 2: targeted at the MLP layer.
          qconfig: *w8a8_dynamic
          include: ["*mlp*"]
          exclude: ["*gate*"]  # Exclude the gating layer for finer control.
```

### 4.3 YAML Configuration Fields

| Field| Purpose| Type| Description|
|--------|------|------|------|
| type | Specifies the processor type identifier.| `string` | The fixed value is `"group"`.|
| configs | Specifies the sub-processor configuration list.| `list[object]` | Contains configurations of multiple sub-processors. The fields of each sub-processor, such as `type`, `qconfig`, and `include`, are identical to those used when the sub-processor is operated independently.|

## 5. FAQs

What Is the Execution Order in the Group Processor?

The Group Processor strictly processes the matched layers in the order specified in the `configs` list.

What Happens If a Layer Is Matched by Multiple Sub-processors?

If a layer meets the `include` conditions of multiple sub-processors, it will be processed by these sub-processors in sequence. **Note**: Do not repeatedly apply quantization operations to the same layer, as this may cause accuracy exceptions or errors. You are advised to use the `exclude` field to achieve precise exclusion.

Why Should the Group Container Be Preferred Over Multiple Independent Processors?

Although mixed quantization can be implemented by configuring multiple independent processors in the `process` list, the Group container greatly improves efficiency by **sharing the inference process (forward)**.

If the Group container is not used, each independent processor will call the inference process on the calibration dataset N times. When they are mounted to the Group container, they only invoke the inference process N times in total. This significantly reduces the quantization time when dealing with large-scale calibration datasets or complex foundation models. In addition, using the Group container may result in subtle differences in the final output compared to using independent processors, but this usually does not cause accuracy degradation issues.
