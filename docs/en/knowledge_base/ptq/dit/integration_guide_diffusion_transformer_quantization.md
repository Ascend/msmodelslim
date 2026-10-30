# DiT Model Integration Quantization Workflow Guide

## 1. Scope

This guide is intended for developers who need to integrate a **new multimodal generation model** (DiT, Diffusion Transformer, such as the new structures of FLUX.1, HunyuanVideo, Wan2.1, and Wan2.2) into the msModelSlim quantization framework. msModelSlim uses an interface-decoupled design, so developers only need to write a model adapter (Model Adapter) that implements the multimodal generation pipeline interface to connect the floating-point inference dump calibration data with the layer-wise quantization scheduling chain.

Applicable scenarios:

- The target multimodal generation model is not in the official support matrix, so a new `model_type` must be added and the quantization chain connected;
- An existing `model_type` needs to be extended to support new quantization modes (such as extending from W8A8 MXFP8 to W4A4 MXFP4, or differentiated quantization for dual experts).

To check whether a model is already in the support list see the *[Foundation Model Support Matrix](../../model/README.md)*; for standard DiT quantization operations see the *[DiT Quantization Usage Guide](usage_diffusion_transformer_quantization.md)*; for pure-text LLM integration see the *[LLM Integration Guide](../llm/integration_guide_large_language_model_quantization.md)*; for multimodal understanding model integration see the *[VLM Integration Guide](../vlm/integration_guide_vision_transformer_quantization.md)*.

## 2. Inputs and Deliverables

| Type | Name | Source or Location | Format or Constraint | Acceptance Method |
| --- | --- | --- | --- | --- |
| Input | Floating-point DiT weight directory | Local path or downloaded repository | Contains model weight files and configuration files | Can be loaded by the official inference pipeline |
| Input | Official inference repository | Local path, added to `PYTHONPATH` | Can run floating-point denoising inference | The official inference example runs normally |
| Input | Calibration dataset and task configuration | `dataset` and `inference_config` in the YAML | Specifies the task type, the inference resolution, and the step count; 50 samples recommended | The adapter parses successfully and completes the floating-point inference dump |
| Deliverable | Model adapter code | `msmodelslim/model/{model_name}/` | Python source containing `loader.py`, `model_adapter.py`, and so on | Code review passes and the import raises no error |
| Deliverable | Model registration configuration | `config/config.ini` | Sections such as `[ModelAdapter]` and `[ModelAdapterEntryPoints]` | Keys stay consistent and can be matched through `--model_type` |
| Deliverable | Quantization configuration file | `lab_practice/{model_name}/` | A YAML file (complying with the `multimodal_sd_modelslim_v1` specification) | Passes quantization configuration schema validation |
| Deliverable | Quantized weight directory | Path specified by `--save_path` | MindIE SD format; multi-expert models contain one subdirectory per expert | Export is complete and the inference smoke test passes |

## 3. Procedure

### Step 1: Create the Model Adapter Project Directory

**Objective**: Establish the directory structure of a new DiT model adapter package under the framework model library.

**Action**:

Create a dedicated subdirectory named after the model series under `msmodelslim/model/` (such as `wan2_2` and `flux`):

```text
msmodelslim/model/{model_name}/
├── __init__.py              # Module initialization (can be empty)
├── loader.py                # Adapter lazy loader
├── model_adapter.py         # Main adapter class
├── base_model_adapter.py    # (Optional) Base adapter for multi-expert models
├── expert_sub_adapter.py    # (Optional) Expert sub-adapter (required for multi-expert models)
└── {task}/                  # (Optional) Submodules split by generation task (such as t2v/i2v)
```

> A single-network DiT usually needs only `loader.py` and `model_adapter.py`; a multi-expert model (such as the Wan2.2 dual-expert model) additionally needs `base_model_adapter.py` and `expert_sub_adapter.py`.

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

**Objective**: Define the model adapter main class, inheriting the base adapter class and the multimodal generation quantization interfaces.

**Action**:

1. **Base adapter class**: DiT relies on the official inference pipeline for loading, so it must inherit from [BaseModelAdapter](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/base.py) (it does not inherit `TransformersModel`).
2. **Core pipeline interface**: it must inherit from [MultimodalPipelineInterface](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/quant_service/multimodal_sd_v1/pipeline_interface.py) (a refactored path).
3. **Optional algorithm extension interfaces**: inherit `OnlineQuaRotInterface`, `FA3QuantAdapterInterface`, `IterSmoothInterface`, and so on as required by the quantization algorithms (all defined in [Interface Hub](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/interface_hub.py)).

```python
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.model.interface_hub import ModelInfoInterface
from msmodelslim.core.quant_service.multimodal_sd_v1.pipeline_interface import (
    MultimodalPipelineInterface,
)
from msmodelslim.utils.logging import logger_setter


@logger_setter("msmodelslim.model.{model_name}")
class {ModelName}ModelAdapter(
    BaseModelAdapter,                    # Base adapter class
    ModelInfoInterface,                  # Model information interface
    MultimodalPipelineInterface,         # Required: core interface of the multimodal generation pipeline
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self.pipeline = None
        self.transformer = None
        self.model_args = None
```

**Output**: The adapter class skeleton is defined.

### Step 4: Implement the Pipeline Core Interface Methods

**Objective**: Implement the methods declared in [MultimodalPipelineInterface](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/quant_service/multimodal_sd_v1/pipeline_interface.py) to support inference parameter validation, runtime configuration, floating-point inference calibration dump, and layer-wise scheduling.

**Action**:

The adapter must implement the following core interface methods; the signatures and responsibilities are as follows:

```python
def get_inference_config_class(self) -> Type[BaseModel]:
    # Return the inference parameter Pydantic configuration class (defining fields such as size/frame_num/sample_steps and their validation rules)

def configure_runtime(self, inference_config: Any) -> None:
    # Set the validated inference parameters on the official inference pipeline's model_args

def inference_dump_calib_data(self, dataset: Any = None, inference_config: Any = None):
    # When enable_dump is set, run the complete floating-point denoising inference, capture the activations of each Timestep, and write them to disk as calibration data

def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[VlmCalibSample]:
    # Validate and normalize the calibration dataset samples

def init_model(self, device: DeviceType = DeviceType.NPU) -> Dict[str, nn.Module]:
    # Load the inference pipeline and return the dictionary of models to be quantized (a single network returns {'': transformer}; a dual-expert model returns the sub-model of each expert)

def prepare_calib_data(self, models, dump_config, save_path, dataset, inference_config) -> Dict[str, Any]:
    # Calibration data preparation: when enable_dump is set, run floating-point inference to save the activations as a pth cache; otherwise read the existing cache directly

def quantization_context(self) -> AbstractContextManager:
    # Provide the autocast, no_grad, and device context environment for quantization

def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
    # Locate the Attention Block by keyword (such as attentionblock) and traverse it block by block

def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
    # Layer-by-layer forward generator: after intercepting the first Block input through a pre-hook, yield block by block to drive the forward pass

def get_expert_adapter(self, expert_name: str) -> PipelineInterface:
    # A multi-expert model returns the corresponding expert sub-adapter; a single-expert model returns self

def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
    # DiT is a dense forward network with no KV Cache, so it is usually implemented as a no-op to satisfy the interface contract
```

**Implementation Notes**:

- **Floating-point inference dump calibration**: DiT quantization calibration relies on running the complete floating-point denoising inference pipeline to capture activation features; `inference_dump_calib_data` must fix the random seed to guarantee reproducibility.
- **Layer-wise scheduling (LayerWise)**: DiT models always use `runner: layer_wise` in `multimodal_sd_modelslim_v1`, intercepting the Attention Block input through a hook and quantizing block by block.
- **Explicit inheritance for multi-expert models**: the expert sub-adapter of a multi-expert model (such as the Wan2.2 dual-expert model) must **explicitly inherit** the extension interfaces implemented by the main adapter (such as `OnlineQuaRotInterface` and `IterSmoothInterface`); it cannot rely solely on the main adapter's inheritance and `__getattr__` proxying.

**Output**: Adapter code that implements the entire interface contract.

### Step 5: Register the Model in the Global Configuration File

**Objective**: Register the model name and loader entry point in `config/config.ini` so that the CLI can route correctly through `--model_type`.

**Action**:

Add configuration items in `config/config.ini` (splitting into independent `model_type` values per scenario is recommended):

```ini
[ModelAdapter]
# Model name mapping (aliases supported)
{model_name} = {ModelName1}, {ModelName2}
{model_name}_t2v = {ModelName}-T2V-{Size}
{model_name}_i2v = {ModelName}-I2V-{Size}

[ModelAdapterEntryPoints]
# Specify the loader class path
{model_name} = msmodelslim.model.{model_name}.loader:{ModelName}AdapterLoader
{model_name}_t2v = msmodelslim.model.{model_name}.t2v.loader:{ModelName}T2VAdapterLoader
{model_name}_i2v = msmodelslim.model.{model_name}.i2v.loader:{ModelName}I2VAdapterLoader
```

**Output**: Configuration registration is complete; the model can be invoked through `--model_type {ModelName}`.

### Step 6: Write the Quantization YAML Plan and Run Validation

**Objective**: Create a quantization plan YAML adapted to this DiT model, and run an end-to-end quantization test through the CLI.

**Action**:

1. **Write the YAML quantization plan** (using standard W8A8 MXFP8 static quantization as an example):

   ```yaml
   # lab_practice/{model_name}/{model_name}_w8a8_mxfp8.yaml
   apiversion: multimodal_sd_modelslim_v1
   spec:
     runner: layer_wise                  # DiT uses layer_wise (fixed)
     process:
       - type: linear_quant
         qconfig:
           act:
             scope: per_block
             dtype: mxfp8
             symmetric: true
             method: minmax
           weight:
             scope: per_block
             dtype: mxfp8
             symmetric: true
             method: mse_round
         include: ["*"]
     save:
       - type: mindie_format_saver       # Deployment format for Ascend MindIE SD
     dataset: {model_name}_{task}
     multimodal_sd_config:
       dump_config:
         enable_dump: true               # Enable floating-point dump for the first quantization run
         capture_mode: "args"
         dump_data_dir: ""
       inference_config:
         size: "1280*720"
         frame_num: 81
         sample_steps: 40
         task: "{task}"
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

**Output**: The quantization command runs successfully, and the output directory contains MindIE SD weight files (multi-expert models contain an independent subdirectory per expert).

## 4. Case List

| Case | Brief Description | Link |
| --- | --- | --- |
| Wan2.2 New Model Quantization Case | End-to-end low-bit W4A4F4 quantization practice for a dual-expert DiT text-to-video model | [Wan2.2 New Model W4A4F4 Quantization Case](../../../best_practices/wan2_2_w4a4f4_new_dit_quantization_case.md) |

## 5. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| PTQ | Post-Training Quantization | [Post-Training Quantization Entry](../term_ptq.md) |
| DiT Quantization | Post-training quantization for Diffusion Transformer multimodal generation models | [DiT Quantization Entry](./term_diffusion_transformer_quantization.md) |

## 6. Interface Documentation List

| API or Capability | Brief Description | Link |
| --- | --- | --- |
| Interface Hub | A centralized definition and summary of the interfaces that quantization mechanisms and algorithm components require from models | [Interface Hub](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/interface_hub.py) |
| Pipeline Interface | Pipeline interface definition of the multimodal generation quantization service | [pipeline_interface.py](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/core/quant_service/multimodal_sd_v1/pipeline_interface.py) |
| DiT Quantization Usage Guide | The complete standard DiT model quantization workflow, configuration protocol, and out-of-the-box examples | [DiT Quantization Usage Guide](usage_diffusion_transformer_quantization.md) |
| multimodal_sd_modelslim_v1 Configuration Description | Field descriptions for multimodal generation configuration such as `process`/`save`/`multimodal_sd_config` | [multimodal_sd_modelslim_v1 Configuration Description](../../../api_reference/config/quant/multimodal_sd_modelslim_v1.md) |
| mindie_format_saver Configuration Description | MindIE SD save format and multi-expert write-to-disk description | [mindie_format_saver Configuration Description](../../../api_reference/config/format/mindie_format_saver.md) |
| One-Click Quantization Usage Guide | One-click quantization CLI entry point and complete command parameter reference | *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)* |
| Quantization Algorithm Overview | Detailed description of micro-scaling floating point, online rotation, and the various quantization algorithms | [Quantization Algorithm Overview](../../quantization_algorithms/README.md) |
