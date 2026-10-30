# VLM Model Integration Quantization Workflow Guide

## 1. Scope

This guide is intended for developers who need to integrate a **new multimodal vision-language model** (VLM, such as a new Qwen-VL or InternVL structure) into the msModelSlim quantization framework. msModelSlim uses an interface-decoupled design, so developers only need to write a model adapter (Model Adapter) that implements the corresponding interfaces to connect their own VLM model to the multimodal quantization pipeline.

Applicable scenarios:

- The target multimodal understanding model is not in the official support matrix, so a new `model_type` must be added and the quantization chain connected;
- An existing `model_type` needs to be extended to support new quantization features (such as outlier smoothing, rotation quantization, or MoE mixed quantization).

To check whether a model is already in the support list see the *[Foundation Model Support Matrix](../../model/README.md)*; for standard multimodal understanding quantization operations see the *[VLM Quantization Usage Guide](usage_vision_transformer_quantization.md)*; for pure-text LLM integration see the *[LLM Integration Guide](../llm/integration_guide_large_language_model_quantization.md)*; for diffusion generation model integration see the *[DiT Integration Guide](../dit/integration_guide_diffusion_transformer_quantization.md)*.

## 2. Inputs and Deliverables

| Type | Name | Source or Location | Format or Constraint | Acceptance Method |
| --- | --- | --- | --- | --- |
| Input | Floating-point model weight directory | Local path or download repository | HuggingFace format, containing `config.json` and sharded weights | The target Transformers version can load it normally |
| Input | Multimodal calibration dataset | Local path | Calibration image directory plus text, or `index.jsonl`; 50 recommended | Can be successfully parsed and processed by the adapter's `handle_dataset` |
| Deliverable | Model adapter code | `msmodelslim/model/{model_name}/` | Python source containing `loader.py`, `model_adapter.py`, and so on | Code review passes and the import raises no error |
| Deliverable | Model registration configuration | `config/config.ini` | `[ModelAdapter]` and `[ModelAdapterEntryPoints]` configuration items | Keys stay consistent and can be resolved through `--model_type` |
| Deliverable | Quantization configuration file | `lab_practice/{model_name}/` | A YAML file (complying with the `multimodal_vlm_modelslim_v1` specification) | Passes quantization configuration schema validation |
| Deliverable | Quantized weight directory | Path specified by `--save_path` | Contains `quant_model_description.json` and shards | Export is complete and the inference smoke test passes |

## 3. Procedure

### Step 1: Create the Model Adapter Project Directory

**Objective**: Establish the directory structure of a new multimodal model adapter package under the framework model library.

**Action**:

Create a dedicated subdirectory named after the model series under `msmodelslim/model/` (such as `qwen3_vl` and `internvl3`):

```text
msmodelslim/model/{model_name}/
├── __init__.py          # Module initialization (can be empty)
├── loader.py            # Adapter lazy loader
├── model_adapter.py     # Main adapter class implementing the component interfaces
└── moe_utils.py         # (Optional) MoE 3D fused weight conversion utility
```

> A regular VLM usually needs only the two files `loader.py` and `model_adapter.py`. For an MoE architecture, `moe_utils.py` can be added; if it contains custom structures such as MTP, auxiliary modules can be added at the same level.

**Output**: The adapter directory and source files are initialized.

### Step 2: Implement the Loader (Lazy Loading)

**Objective**: Register the class path of the adapter class with the framework plugin factory, avoiding the pre-loading of non-essential modules at startup.

**Action**:

In `loader.py`, inherit from [BaseModelAdapterLoader](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/plugin_factory/base_loader.py), and specify the import path of the adapter class through `ADAPTER_CLASS_PATH`:

```python
# msmodelslim/model/{model_name}/loader.py
from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class {ModelName}AdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.{model_name}.model_adapter:{ModelName}ModelAdapter"
```

**Output**: A `loader.py` that the framework factory can invoke through dynamic reflection.

### Step 3: Define the Adapter Class and Inherit the Component Interfaces

**Objective**: Define the model adapter main class, inheriting the VLM base adapter class and the component interfaces required by the quantization algorithms.

**Action**:

1. **Base Model Class Selection**:
   - Inheriting from [VLMBaseModelAdapter](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/common/vlm_base.py) is recommended (it already provides generic multimodal implementations such as `_load_config`, `_collect_inputs_to_device`, and sharded weight reading).
   - If the model structure is highly customized and the base class cannot be reused, inherit from [BaseModelAdapter](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/base.py) and implement the config loading and data collection logic yourself.

2. **Core Quantization Interface**:
   - **[`ModelSlimPipelineInterfaceV1`](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/runner/pipeline_interface.py) (must be implemented)**: the core model pipeline interface, which the scheduler uses to drive multimodal quantization.

3. **Optional Algorithm Extension Interfaces** (all defined in [Interface Hub](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/interface_hub.py)):
   - **`IterSmoothInterface` / `FlexSmoothQuantInterface`**: if the scheme needs outlier smoothing suppression, inherit the corresponding interface and configure the operator mapping on the language Decoder.
   - **`QuaRotInterface`**: if the scheme needs a rotation matrix to smooth outliers, inherit this interface (it applies to the language part only).

```python
from pathlib import Path

from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.model.interface_hub import ModelSlimPipelineInterfaceV1
from msmodelslim.utils.logging import logger_setter


@logger_setter("msmodelslim.model.{model_name}")
class {ModelName}ModelAdapter(
    VLMBaseModelAdapter,             # Inherits the VLM base model adapter class
    ModelSlimPipelineInterfaceV1,    # Required: core interface for quantization pipeline scheduling
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._processor = None
        self._tokenizer = None
        super().__init__(model_type, model_path, trust_remote_code)
```

**Output**: The adapter class skeleton is defined.

### Step 4: Implement the Pipeline Core Interface Methods

**Objective**: Implement the methods declared in [ModelSlimPipelineInterfaceV1](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/runner/pipeline_interface.py) so that the scheduler (Runner) can drive multimodal data processing, loading, and layer-wise quantization as a whole.

**Action**:

The adapter must implement the following core interface methods; the signatures and responsibilities are as follows:

```python
def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
    # Load the Processor to build multimodal inputs and move them to the device via self._collect_inputs_to_device

def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
    # Instantiate the model: load the vision encoder in full, and load only the first layer of the language model to control memory

def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
    # Topological traversal generator: first yield the "model.visual" vision encoder, then traverse the language model's Decoder layers

def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
    # Instantiate and load the language Decoder Layer on demand; for an MoE architecture the 3D weight conversion is performed at this stage

def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
    # Three-stage calibration forward pass: vision encoder forward -> visual feature fusion and placeholder replacement -> language Decoder layer-by-layer forward

def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
    # Controls whether the calibration forward pass enables KVCache; PTQ calibration disables it explicitly to reduce memory overhead
```

**Implementation Notes**:

- **Full vision + layer-wise language**: for a VLM it is recommended to load the complete vision encoder in `init_model`, but to initialize only the first layer of the language model (temporarily setting `num_hidden_layers=1`); the remaining layers are loaded and released on demand through `generate_decoder_layer` during the forward pass, avoiding out-of-memory during multimodal quantization of large models.
- **Feature fusion alignment**: the visual feature extraction and fusion in `generate_model_forward` (such as replacing the `<image>` placeholder with `masked_scatter`, or the DeepStack cross-layer injection) must be exactly consistent with the original model's official forward logic.
- **MoE weight conversion**: if the model stores MoE experts as 3D fused weights, they must be converted to standard Linear layers when the Decoder is loaded, so that the quantization framework can process them layer by layer.

**Output**: Adapter code that implements every interface.

### Step 5: Register the Model in the Global Configuration File

**Objective**: Register the model name and loader entry point in `config/config.ini` so that the CLI can route correctly through `--model_type`.

**Action**:

Add configuration items in `config/config.ini` (note: the key names in the three sections must stay strictly consistent):

```ini
[ModelAdapter]
# Map the model names commonly used by users to a unified key
{model_alias_key} = {ModelName1}, {ModelName2}

[ModelAdapterEntryPoints]
# Specify the corresponding loader class path
{model_alias_key} = msmodelslim.model.{model_name}.loader:{ModelName}AdapterLoader

[ModelAdapterDependencies]
# (Optional) Declare model-specific third-party library version dependency constraints
{model_alias_key} = {"transformers": "==4.48.2"}
```

**Output**: Configuration registration is complete; the model can be invoked through `--model_type {ModelName}`.

### Step 6: Write the Quantization YAML Plan and Run Validation

**Objective**: Create a quantization plan YAML adapted to this VLM model, and run an end-to-end quantization test through the CLI.

**Action**:

1. **Write the YAML quantization plan** (using standard W8A8 static quantization as an example; the visual projection layers must be excluded):

   ```yaml
   # lab_practice/{model_name}/{model_name}_w8a8.yaml
   apiversion: multimodal_vlm_modelslim_v1
   spec:
     runner: auto
     process:
       - type: linear_quant
         qconfig:
           act:
             scope: per_tensor
             dtype: int8
             symmetric: false
             method: minmax
           weight:
             scope: per_channel
             dtype: int8
             symmetric: true
             method: minmax
         include: ["*"]
         exclude:
           - "*merger*"                 # Exclude the visual feature projection layer from quantization
           - "*linear_fc2"
           - "*deepstack_merger_list*"
           - "*mlp.gate"                # Exclude the MoE router from quantization
     save:
       - type: ascendv1_saver
         part_file_size: 4
     dataset: calib_images              # Calibration image directory or index.jsonl
     default_text: "Describe this image in detail."
   ```

2. **Run the CLI quantization validation**:

   ```bash
   msmodelslim quant \
     --model_path <float_model_dir> \
     --save_path <quantized_weight_output_dir> \
     --model_type <registered model adapter name> \
     --config <quantization configuration file path> \
     --device npu \
     --device_id 0
   ```

**Output**: The quantization command succeeds, and the output directory contains `quant_model_description.json` and sharded weights `*.safetensors`.

## 4. Case List

| Case | Brief Description | Link |
| --- | --- | --- |
| Kimi-K3 new model integration and quantization | End-to-end practice for integrating a new multimodal understanding model structure and W4A8 quantization | [Kimi-K3 New Model W4A8 Quantization Case](../../../best_practices/kimi_k3_adaptation_case.md) |

## 5. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| PTQ | Post-Training Quantization | [Post-Training Quantization Entry](../term_ptq.md) |
| VLM Quantization | Post-training quantization for multimodal vision-language models | [Multimodal Understanding Model (VLM) Quantization Entry](./term_vision_transformer_quantization.md) |

## 6. Interface Documentation List

| API or Capability | Brief Description | Link |
| --- | --- | --- |
| Interface Hub | A centralized definition and summary of the interfaces that quantization mechanisms and algorithm components require from models | [Interface Hub](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/interface_hub.py) |
| VLM Quantization Usage Guide | The complete standard multimodal understanding model quantization workflow, configuration protocol, and out-of-the-box examples | [VLM Quantization Usage Guide](usage_vision_transformer_quantization.md) |
| multimodal_vlm_modelslim_v1 Configuration Description | Field descriptions for multimodal understanding configuration such as `process`/`save`/`dataset` | [multimodal_vlm_modelslim_v1 Configuration Description](../../../api_reference/config/quant/multimodal_vlm_modelslim_v1.md) |
| One-Click Quantization Usage Guide | One-click quantization CLI entry point and complete command parameter reference | *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)* |
| Quantization Algorithm Overview | Detailed description of outlier smoothing suppression and the various quantization algorithms | [Quantization Algorithm Overview](../../quantization_algorithms/README.md) |
