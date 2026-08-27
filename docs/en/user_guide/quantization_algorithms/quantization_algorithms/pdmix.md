# PDMIX: Phase-wise Mixed Activation Quantization Algorithm

## Overview

- **Problem**: Traditional W8A8 static quantization uses fixed activation parameters. This often leads to significant quantization error in long-context or distribution drift scenarios. Controlling accuracy loss typically requires falling back a large number of layers, which sacrifices performance gains.
- **Objective**: Combine dynamic and static W8A8 strategies across different phases. This enables accuracy loss control by falling back only a few layers, while achieving performance gains near those of static W8A8 quantization during output.
    - Prefilling phase: performs dynamic W8A8 quantization (`per-token`) to minimize quantization information loss in the input context, thereby mitigating accuracy drops.
    - Decoding phase: performs static W8A8 quantization (`per_tensor`) to maximize quantization performance gains during output generation, thereby improving inference efficiency.

> **Note**: Weight quantization methods must remain consistent across phases. Otherwise, two sets of quantized weights must be stored. Therefore, PDMIX is categorized as an activation quantization algorithm, used in combination with W8 `per_channel` weight quantization.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Principle and Implementation

### Principle

- The weight quantization process remains unchanged, while activation quantization adopts a phase-adaptive mixed approach.
  - Prefilling phase: performs dynamic quantization (`per-token`), where quantization parameters are calculated online at token-level granularity to minimize quantization error.
  - Decoding phase: performs static quantization (`per_tensor`), where activation quantization parameters are calculated offline to reduce online parameter calculation, thereby reducing inference latency and increasing throughput.

### Implementation

The specific implementation components are located as follows:

- Quantization calibration: `ActPDMixMinmax` in [`msmodelslim/core/quantizer/impl/minmax.py`](../../../../../msmodelslim/core/quantizer/impl/minmax.py)
- Quantization mode IR: `W8A8PDMixFakeQuantLinear` in [`msmodelslim/ir/w8a8_pdmix.py`](../../../../../msmodelslim/ir/w8a8_pdmix.py)
- Related constants: `int8_pd_mix_asym` defined in [`msmodelslim/ir/const.py`](../../../../../msmodelslim/ir/const.py)

## Application Requirements

- Currently, only MindIE inference and deployment are supported.
- When static quantization results in high accuracy loss and requires falling back a large number of layers, you are advised to replace the algorithm with PDMIX quantization.
- This algorithm is an activation quantization method for linear layers. Any linear layer implemented using `torch.nn.Linear` meets the algorithm requirements.

## Function Description

### YAML Configuration Example

The following example shows a YAML configuration when the algorithm is used as a processor:

```yaml
spec:
  process:
    - type: "linear_quant"     # Specifies the linear quantization processor.
      qconfig:
        act: # Activation quantization configuration
          scope: "pd_mix"      # prefilling: per-token; decoding: per_tensor
          dtype: "int8"        # Currently, only INT8 is supported.
          symmetric: false     # PDMIX quantization is fundamentally asymmetric.
          method: "minmax"     # Currently, only the MinMax algorithm is supported.
        weight: # Weight quantization configuration
          scope: "per_channel" # Currently, only per_channel weight quantization is supported.
          dtype: "int8"        # Currently, only INT8 weight quantization is supported.
          symmetric: true      # Only symmetric weight quantization is supported.
          method: "minmax"     # Specifies the weight quantization algorithm.
```

Currently, only MindIE supports the W8A8 PDMIX quantization mode. Therefore, `qconfig.weight.method` is the only adjustable parameter, and other configuration combinations do not have corresponding implementations.

### YAML Configuration Fields

#### `qconfig.weight` - Weight Quantization Configuration

| Parameter      | Purpose    | Value                           | Description                                           | Default Value                        |
|-----------|--------|--------------------------------|-----------------------------------------------|-----------------------------|
| scope     | Specifies the quantization scope.  | `"per_channel"` | `per_channel`: uses independent parameters for each channel. Only `per_channel` weight quantization is supported.| `"per_channel"` |
| dtype     | Specifies the quantization data type.| `"int8"` | `int8`: enables 8-bit integer quantization. Only INT8 weight quantization is supported.| `"int8"` |
| symmetric | Specifies whether to enable symmetric quantization.| `true` | `true`: enables symmetric quantization, with a zero point of `0`. Only symmetric weight quantization is supported.| `true` |
| method    | Specifies the quantization algorithm.  | `"minmax"` | `minmax`: enables `minmax` quantization.| `"minmax"` |
