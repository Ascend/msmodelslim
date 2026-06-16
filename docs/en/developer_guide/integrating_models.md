# LLM Model Integration Guide

## Overview

This document is for developers who need to integrate custom models into msModelSlim.
msModelSlim recognizes that quantization mechanisms and algorithms each have their own applicable scope and limitations, and that new model architectures keep emerging, so no quantization method works in every case.
To simplify quantization for custom models as much as possible, msModelSlim extracts the model conditions required by quantization mechanisms and algorithms and expresses them as interfaces.
A model adapter describes the model. It combines interface implementations and enables quantization for a custom model by implementing the interfaces required by different mechanisms and algorithms.

## Concepts

### Interfaces

* Interfaces are defined in quantization mechanisms and algorithms. They describe what each component expects from the model. See the corresponding component documentation and code for interface definitions and usage.
* You only need to implement an interface when you use the corresponding component.
* Interface summary: [`msmodelslim/model/interface_hub.py`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/interface_hub.py)

### Model Adapters

* A model adapter combines component interface implementations and describes model characteristics and behavior that serve those specific interface implementations.
* A set of structurally similar models can reuse one model adapter, and one model adapter can register multiple models.
* The common `model_type` parameter in the `msmodelslim` command corresponds to the model name registered by the model adapter. It is used to match and create the model adapter.

## Model Integration

The following uses the [Qwen3-32B](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/qwen3/model_adapter.py) W8A8 dynamic quantization scenario, referred to as the "scenario example", as the model integration example.

### Creating a Model Adapter `.py` File

You are advised to place it in [`msmodelslim/model/`](https://gitcode.com/Ascend/msmodelslim/tree/master/msmodelslim/model) and name it `qwen3.py`, for example.

### Identifying the Components Involved in the Quantization Process and Define the Adapter Class by Combining Component Interfaces

The model adapter class must inherit from [`BaseModelAdapter`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/base.py).

Based on experience, W8A8 dynamic quantization usually causes only a small accuracy loss, so it does not need an outlier suppression algorithm and rarely requires fallback. Therefore, in the scenario example, we only need to support quantization scheduling. We do not need outlier quantization, sensitive-layer analysis, or other extra functions. If you need to integrate other algorithms, refer to [`Algorithm Overview`](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/)

```python
from msmodelslim.model.interface_hub import ModelSlimPipelineInterfaceV1
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.utils.logging import logger_setter


@logger_setter()
class Qwen3ModelAdapter(TransformersModel,  # Inherits BaseModelAdapter behavior and simplifies interface implementation with common Transformer model traits and behavior.
                        ModelSlimPipelineInterfaceV1,  # Required. Supports quantization scheduling.
                        ):
    pass
```

### Implementing Component Interface Methods

Implement the methods required by the interfaces. The methods describe model characteristics and behavior. If you use an IDE, you can use IDE features to generate the interface methods quickly and then fill in the implementation.

```python
from typing import List, Any, Generator
from torch import nn
from msmodelslim.core.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.interface_hub import ModelSlimPipelineInterfaceV1
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func, \
    transformers_generated_forward_func
from msmodelslim.utils.logging import logger_setter


@logger_setter()
class Qwen3ModelAdapter(TransformersModel,
                        ModelSlimPipelineInterfaceV1
                        ):
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:  # Describe how to convert the calibration set into batched inputs.
        return self._get_tokenized_data(dataset, device)  # TransformersModel provides a default implementation based on Transformer model behavior.

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:  # Describe how to initialize the model.
        return self._load_model(device)  # TransformersModel provides a default implementation based on Transformer model behavior.

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:  # Describe how to segment the model. It must match the model structure.
        # msmodelslim/model/common/layer_wise_forward.py provides a default implementation that segments the model based on the DecoderLayer class.
        yield from generated_decoder_layer_visit_func(model)

    def generate_model_forward(self, model: nn.Module, inputs: Any,
                               ) -> Generator[ProcessRequest, Any, None]:  # Describe how to segment the model forward process. It must match the model forward process.
        # msmodelslim/model/common/layer_wise_forward.py provides a default implementation that segments the model forward process based on the DecoderLayer class.
        yield from transformers_generated_forward_func(model,
                                                       inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:  # Describe whether to disable KVCache to reduce memory usage.
        return self._enable_kv_cache(model, need_kv_cache) # TransformersModel provides a default implementation based on Transformer model behavior.
```

### Registering the Model

Register the model name in the configuration file [`config.ini`](https://gitcode.com/Ascend/msmodelslim/blob/master/config/config.ini) so that models in the same series can reuse one model adapter.

```ini
# Register the Qwen3-32B model in the qwen3 series in ModelAdapter. qwen3 maps to Qwen3ModelAdapter below.
[ModelAdapter]
default = default
deepseek_v3 = DeepSeek-V3, DeepSeek-V3-0324, DeepSeek-R1, DeepSeek-R1-0528, DeepSeek-V3.1
deepseek_v3_2 = DeepSeek-V3.2-Exp
qwen2_5 = Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct, Qwen2.5-72B-Instruct, Qwen2.5-Coder-7B-Instruct
qwen3 = Qwen3-8B, Qwen3-14B, Qwen3-32B  # Add here.
qwen3_moe = Qwen3-30B, Qwen3-235B
qwq = Qwen-QwQ-32B, QwQ-32B
wan2_1 = Wan2_1, Wan2.1
qwen3_next = Qwen3-Next-80B-A3B-Instruct
wan2_2 = Wan2_2, Wan2.2

# If you add a model adapter, you need to add it to ModelAdapterEntryPoints. Note that the keys in ModelAdapter and ModelAdapterEntryPoints must stay consistent, or the configuration will not take effect.
[ModelAdapterEntryPoints]
default = msmodelslim.model.default.model_adapter:DefaultModelAdapter
deepseek_v3 = msmodelslim.model.deepseek_v3.model_adapter:DeepSeekV3ModelAdapter
deepseek_v3_2 = msmodelslim.model.deepseek_v3_2.model_adapter:DeepSeekV32ModelAdapter
qwen2_5 = msmodelslim.model.qwen2_5.model_adapter:Qwen25ModelAdapter
qwen3 = msmodelslim.model.qwen3.model_adapter:Qwen3ModelAdapter
qwen3_moe = msmodelslim.model.qwen3_moe.model_adapter:Qwen3MoeModelAdapter
qwq = msmodelslim.model.qwq.model_adapter:QwqModelAdapter
wan2_1 = msmodelslim.model.wan2_1.model_adapter:Wan2Point1Adapter
qwen3_next = msmodelslim.model.qwen3_next.model_adapter:Qwen3NextModelAdapter
wan2_2 = msmodelslim.model.wan2_2.model_adapter:Wan2Point2Adapter
```

## Automatic Tuning and Sensitivity Analysis

When using [Automatic Tuning Usage Guide](../feature_guide/auto_precision_tuning/usage.md) and the strategy must **automatically build rollback candidates** (`standing_high` always; `standing_high_with_experience` delegates to Standing High), the model adapter must implement **`ModelSlimPipelineInterfaceV1`**. Import it as follows:

```python
from msmodelslim.model.interface_hub import ModelSlimPipelineInterfaceV1
```

This is the same protocol as the CLI **`msmodelslim analyze`** command and `PipelineAnalysisService`. Implement `init_model`, `handle_dataset`, `generate_model_visit`, `generate_model_forward`, and related pipeline methods. Tuning strategies invoke them via `PipelineAnalysisService`; the strategy **does not** call `load_model` upfront.

| Tuning strategy | Sensitivity analysis | Extra interface |
|-----------------|----------------------|-----------------|
| `standing_high` | Always automatic | None |
| `standing_high_with_experience` | Delegated to Standing High | **`StandingHighWithExperienceInterface`** (`load_model`, outlier suppression probe); inherit **`ModelSlimPipelineInterfaceV1`** separately |

See [Automatic Tuning Configuration Protocol](../feature_guide/auto_precision_tuning/configuration_protocols.md) and each strategy algorithm document.

## Quantization of Self-Owned Models

After you finish writing and registering the model adapter, you can use quick quantization to quantize the model.

### Creating a W8A8 Dynamic Quantization YAML Configuration File

```yaml
apiversion: modelslim_v1
spec:
  process:
    - type: "linear_quant" # Linear layer quantization
      qconfig:
        act: # Activation quantization
          scope: "per_token" # Dynamic quantization
          dtype: "int8" # 8-bit integer quantization
          symmetric: True # Symmetric quantization
          method: "minmax" # Use the minmax algorithm.
        weight: # Weight quantization
          scope: "per_channel" # per_channel quantization
          dtype: "int8" # 8-bit integer quantization
          symmetric: True # Symmetric quantization
          method: "minmax" # Use the minmax algorithm.
      include: [ "*" ] # Global W8A8 dynamic quantization
      exclude: [ "down_proj" ] # Fall back to the down_proj layer.

  save:
    - type: "ascendv1_saver"
      part_file_size: 4 # The maximum size of each safetensors file is 4 GB.
```

### Quantizing Self-Owned Models

You can use the following command to quantize your own model. Note that when `trust_remote_code` is `True`, code files in the floating-point model weights may be executed. Ensure that the floating-point model comes from a safe and reliable source. `${MODEL_PATH}` is the path to the original floating-point weights, `${SAVE_PATH}` is the user-defined path for saving the quantized weights, `${MODEL_TYPE}` is the registered model name, and `${CONFIG_PATH}` is the path to the YAML config file.

```bash
msmodelslim quant --model_path ${MODEL_PATH} \
                  --save_path ${SAVE_PATH} \
                  --device npu \
                  --model_type ${MODEL_TYPE} \
                  --config_path ${CONFIG_PATH} \
                  --trust_remote_code False
```

- For details about the usage and parameters, see [Quick Quantization Guide](../feature_guide/quick_quantization_v1/usage.md).
