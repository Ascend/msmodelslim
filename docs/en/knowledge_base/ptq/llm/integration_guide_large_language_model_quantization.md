# LLM Model Integration Quantization Workflow Guide

## 1. Scope

This guide is intended for developers who need to integrate a **new Decoder-only large language model** (such as a LLaMA variant, a Qwen variant, or a new DeepSeek structure) into the msModelSlim quantization framework. msModelSlim uses an interface-decoupled design that extracts the model capabilities required by the quantization mechanism and algorithms into generic interfaces. Developers only need to write a model adapter (Model Adapter) that implements the corresponding interfaces to connect their own model to the quantization pipeline.

Applicable scenarios:

- The target LLM is not in the official support matrix, so a new `model_type` must be added and the quantization chain connected;
- An existing `model_type` needs to be extended to support new quantization features (such as outlier smoothing, rotation quantization, or sensitive layer tuning).

To check whether a model is already in the support list see the *[Large Model Support Matrix](../../model/README.md)*; for standard model quantization operations see the *[LLM Quantization Usage Guide](usage_large_language_model_quantization.md)*; for multimodal model integration see the *[VLM Integration Guide](../vlm/integration_guide_vision_transformer_quantization.md)* or the *[DiT Integration Guide](../dit/integration_guide_diffusion_transformer_quantization.md)*.

## 2. Inputs and Deliverables

| Type | Name | Source or Location | Format or Constraint | Acceptance Method |
| --- | --- | --- | --- | --- |
| Input | Floating-point model weight directory | Local path or download repository | HuggingFace format, containing `config.json` and sharded weights | The target Transformers version can load it normally |
| Input | Calibration dataset | Local path or `lab_calib/` | JSONL or JSON text prompts, 50 recommended | Can be successfully encoded into tensors by the adapter's `handle_dataset` |
| Deliverable | Model adapter code | `msmodelslim/model/{model_name}/` | Python source containing `loader.py` and `model_adapter.py` | Code review passes and the import raises no error |
| Deliverable | Model registration configuration | `config/config.ini` | `[ModelAdapter]` and `[ModelAdapterEntryPoints]` configuration items | Keys stay consistent and can be resolved through `--model_type` |
| Deliverable | Quantization configuration file | `lab_practice/{model_name}/` | A YAML file (complying with the `modelslim_v1` specification) | Passes quantization configuration schema validation |
| Deliverable | Quantized weight directory | Path specified by `--save_path` | Contains `quant_model_description.json` and shards | Export is complete and the inference smoke test passes |

## 3. Procedure

### Step 1: Create the Model Adapter Project Directory

**Objective**: Establish the directory structure of a new model adapter package under the framework model library.

**Action**:

Create a dedicated subdirectory named after the model series under `msmodelslim/model/` (such as `qwen3` and `deepseek_v4`):

```text
msmodelslim/model/{model_name}/
├── __init__.py          # Module initialization (can be empty)
├── loader.py            # Adapter lazy loader
└── model_adapter.py     # Main adapter class implementing the component interfaces
```

> For a standard open-source model built on HuggingFace Transformers, usually only the two files `loader.py` and `model_adapter.py` need to be created. If the model contains special structures (such as an FP8 weight dequantization script or custom MTP layers), auxiliary modules can be added at the same level.

**Output**: The adapter directory and source files are initialized.

### Step 2: Implement the Loader (Lazy Loading)

**Objective**: Register the class path of the adapter class with the framework plugin factory, avoiding the pre-loading of non-essential modules at startup.

**Action**:

In `loader.py`, inherit from [BaseModelAdapterLoader](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/plugin_factory/base_loader.py), and specify the import path of the adapter class through `ADAPTER_CLASS_PATH`:

```python
# msmodelslim/model/{model_name}/loader.py
from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class {ModelName}AdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.{model_name}.model_adapter:{ModelName}ModelAdapter"
```

**Output**: A `loader.py` that the framework factory can invoke through dynamic reflection.

### Step 3: Define the Adapter Class and Inherit the Component Interfaces

**Objective**: Define the model adapter main class, combining the base adapter class with the component interfaces required by the quantization algorithms.

**Action**:

1. **Base Model Class Selection**:
   - If the model is implemented on HuggingFace Transformers, inheriting from [TransformersModel](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/common/transformers.py) is recommended (it already provides generic implementations of tokenization batching, base model loading, and KVCache control).
   - If the model structure is highly customized and cannot be loaded through Transformers, inherit directly from [BaseModelAdapter](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/base.py).

2. **Core Quantization Interface**:
   - **[`ModelSlimPipelineInterfaceV1`](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/core/runner/pipeline_interface.py) (must be implemented)**: the core model pipeline interface (equivalent to `PipelineInterface`), which the scheduler uses to drive quantization.

3. **Optional Algorithm Extension Interfaces** (all defined in [Interface Hub](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/interface_hub.py)):
   - **`IterSmoothInterface` / `FlexSmoothQuantInterface`**: if the quantization scheme needs outlier smoothing suppression (such as IterSmooth / FlexSmoothQuant), inherit this interface and declare the smoothable Norm-Linear and Up-Down subgraph operator pairs in the network.
   - **`QuaRotInterface`**: if the scheme needs a rotation matrix to smooth outliers, inherit this interface.
   - **`StandingHighWithExperienceInterface`**: inherit this when using the "expert-experience ceiling-probing strategy" in automatic precision tuning.

```python
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.model.interface_hub import ModelSlimPipelineInterfaceV1
from msmodelslim.utils.logging import logger_setter


@logger_setter("msmodelslim.model.{model_name}")
class {ModelName}ModelAdapter(
    TransformersModel,               # Inherits the base model adapter class
    ModelSlimPipelineInterfaceV1,    # Required: core interface for quantization pipeline scheduling
):
    pass
```

**Output**: The adapter class skeleton is defined.

### Step 4: Implement the Pipeline Core Interface Methods

**Objective**: Implement the methods declared in [ModelSlimPipelineInterfaceV1](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/core/runner/pipeline_interface.py) so that the scheduler (Runner) can drive the model's data preprocessing, loading, and layer-wise quantization as a whole.

**Action**:

The adapter must implement the following 5 core interface methods; the signatures are as follows (a standard model that inherits `TransformersModel` can directly reuse the base class defaults):

```python
def handle_dataset(self, dataset, device=DeviceType.NPU) -> List[Any]
    # Tokenize, truncate, and convert the raw text prompt list into tensors, returning [ {"input_ids": tensor, ...} ]
    # A standard model directly reuses self._get_tokenized_data(dataset, device)

def init_model(self, device=DeviceType.NPU) -> nn.Module
    # Instantiate the model and load the weights, returning the model instance
    # A standard model directly calls self._load_model(device)

def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]
    # Network structure traversal generator (data-free quantization); yields ProcessRequest layer by layer in topological order
    # A standard Decoder model directly calls generated_decoder_layer_visit_func(model)

def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]
    # Calibration forward generator; forwards a single sample layer by layer to capture the activation distribution, yielding and passing it down at each layer
    # A standard model directly calls transformers_generated_forward_func(model, inputs)

def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None
    # Controls whether the calibration forward pass enables KVCache
    # A standard model directly reuses self._enable_kv_cache(model, need_kv_cache)
```

**Implementation Notes**:

- **Layer-by-layer loading to control memory**: for models with a large parameter count, load only the first Decoder Layer in `init_model` (temporarily setting `num_hidden_layers=1`) and load the remaining layers on demand during the quantization forward pass; if the source model uses FP8 weights, perform dequantization at this stage.
- **Strictly identical layer order**: the layer visit order of `generate_model_visit` and `generate_model_forward` must be exactly the same, otherwise the layer-wise scheduling will be misaligned.
- **Disable KVCache**: PTQ calibration only needs the Prefill features, so explicitly disabling it in `enable_kv_cache` greatly reduces memory overhead.

**Output**: Adapter code that implements the entire interface contract.

### Step 5: Register the Model in the Global Configuration File

**Objective**: Register the model name and loader entry point in `config/config.ini` so that the CLI can route correctly through `--model_type`.

**Action**:

Add configuration items in `config/config.ini` (note: the key names in the three sections must stay strictly consistent):

```ini
[ModelAdapter]
# Map the model names commonly used by users (an alias list is supported) to a unified key
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

**Objective**: Create a quantization plan YAML adapted to this model, and run an end-to-end quantization test through the CLI.

**Action**:

1. **Write the YAML quantization plan** (using standard W8A8 dynamic quantization as an example):

   ```yaml
   # lab_practice/{model_name}/{model_name}_w8a8_dynamic.yaml
   apiversion: modelslim_v1
   spec:
     runner: auto                    # layer_wise for single-card, dp_layer_wise for multi-card
     process:
       - type: linear_quant
         qconfig:
           act:
             dtype: int8
             scope: per_token
             symmetric: true
             method: minmax
           weight:
             dtype: int8
             scope: per_channel
             symmetric: true
             method: minmax
         include: ["*"]
         exclude: ["*gate*"]          # Exclude sensitive layers such as the router
     save:
       - type: ascendv1_saver        # Recommended format for Ascend inference deployment
     dataset: mix_calib.jsonl        # General calibration set
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
| DeepSeek-V4-Pro new model integration and quantization | End-to-end integration practice for W4A8 mixed quantization of a complex MoE+MLA architecture large model | [DeepSeek-V4-Pro New Model W4A8 Quantization Case](../../../best_practices/deepseek_v4_pro_w4a8_new_llm_quantization_case.md) |
| Qwen3-32B accuracy tuning in practice | A case study of W8A8 dynamic quantization and automatic sensitive-layer tuning for a typical open-source large model | [Qwen3-32B Model W8A8 Quantization Tuning Case](../../../best_practices/qwen3_32b_w8a8_precision_tuning_case.md) |

## 5. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| PTQ | Post-Training Quantization | [Post-Training Quantization Entry](../term_ptq.md) |
| LLM Quantization | Post-training quantization for Decoder-only large language models | [LLM Quantization Entry](./term_large_language_model_quantization.md) |

## 6. Interface Documentation List

| API or Capability | Brief Description | Link |
| --- | --- | --- |
| Interface Hub | A centralized definition and summary of the interfaces that quantization mechanisms and algorithm components require from models | [Interface Hub](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/msmodelslim/model/interface_hub.py) |
| LLM Quantization Usage Guide | The complete standard LLM quantization workflow, configuration protocol, and out-of-the-box examples | [LLM Quantization Usage Guide](usage_large_language_model_quantization.md) |
| modelslim_v1 Configuration Description | Field descriptions of task-level configuration such as `runner`/`process`/`save`/`dataset` | [modelslim_v1 Configuration Description](../../../api_reference/config/quant/modelslim_v1.md) |
| One-Click Quantization Usage Guide | One-click quantization CLI entry point and complete command parameter reference | *[One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md)* |
| Auto Tuning Usage Instructions | Sensitive layer analysis, rollback candidates, and tuning strategy descriptions | [Auto Tuning Usage Instructions](../../../user_guide/usage_auto_precision_tuning.md) |
| Quantization Algorithm Overview | Detailed description of outlier smoothing suppression and the various quantization algorithms | [Quantization Algorithm Overview](../../quantization_algorithms/README.md) |
