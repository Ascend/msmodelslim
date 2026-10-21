# QuaRot Rotation Quantization Algorithm Usage Guide

## 1. Scope

This guide is intended for users who need to use the [QuaRot Rotation Quantization Algorithm](./term_quarot.md). As an outlier-suppression algorithm, QuaRot usually serves as a preprocessing step before quantization, smoothing the activation distribution through orthogonal rotation and improving low-bit quantization accuracy.

Applicable scenarios:

- The accuracy of low-bit quantization such as W8/W4 falls short, and you need orthogonal rotation to break up activation outlier channels;
- You need to make the activation and weight distributions more uniform without changing the model's forward output.

To check whether the model is on the officially pre-validated list, see the [Foundation Model Support Matrix](../../model/README.md); for a quick introduction to basic CLI usage, see the [One-Click Quantization Usage Guide](../../../user_guide/usage_one_click_quantization.md).

## 2. Inputs and Deliverables

| Type | Name | Source or Storage Location | Format or Constraints | Acceptance Criteria |
| --- | --- | --- | --- | --- |
| Input | Floating-point model weight directory | Model download or a local path | HuggingFace format containing `config.json` and `*.safetensors` shards | Loadable by the target Transformers version |
| Input | Model adapter | User adaptation code, invoked via `--model_type` | Implements `PipelineInterface` and the QuaRot-related interfaces | Properly driven by the runner and processor |
| Input | Calibration dataset | Built-in `lab_calib/` or a user-defined path | Text prompts in JSONL or JSON format; 50 samples recommended | Successfully encoded into forward tensors by the adapter's `handle_dataset` |
| Input | Quantization configuration file | Local YAML file | Complies with the `modelslim_v1` protocol specification | Passes schema validation |
| Deliverable | Quantized weight directory | Path specified by `--save_path` | Contains `quant_model_description.json` and `*.safetensors` shards | Export is complete and the inference smoke test passes |

## 3. Process Overview

The overall workflow of QuaRot is as follows:

```mermaid
flowchart LR
    A[Adapt the model rotation interfaces] --> B[Determine the<br/>offline/online rotation mode]
    B --> C[Choose the rotation block size]
    C --> D[Apply the orthogonal rotation]
    D --> E[Connect to low-bit quantization]
    E --> F[Validate accuracy<br/>and deployment info]
```

Key details of each stage are as follows:

- **Adapt the model rotation interfaces**: the adapter must implement `QuaRotInterface` (offline rotation), `OnlineQuaRotInterface` (online rotation), and other interfaces, providing network structure information such as rotation mappings, Norm fusion, and mean fusion. This is the precondition for all subsequent rotation steps.
- **Determine the offline/online rotation mode**: prefer offline rotation (`online: false`), where the rotation matrix is fused into the weights during quantization and inference has no extra overhead; enable online rotation only when the deployment chain explicitly supports runtime rotation, and evaluate inference latency and operator compatibility at the same time. By default the rotation matrix is generated automatically as a Walsh-Hadamard orthogonal matrix by `QuaRotInterface.get_rotate_command`, and is moved automatically to the current device when an NPU is available; a custom matrix can also be provided through `RotationConfig` of `OnlineQuaRotInterface`.
- **Choose the rotation block size**: `block_size: -1` means rotation over the full hidden_dim, which best breaks up outliers across all dimensions; specifying a power of 2 (such as `32`) enables blocked rotation to fit specific parallelism/hardware constraints.
- **Apply the orthogonal rotation**: before rotating, fuse LayerNorm via `get_ln_fuse_map` and apply mean fusion to the necessary Linear layers via `get_bake_names` to guarantee mathematical equivalence before and after rotation, then apply the orthogonal rotation layer by layer to break up outlier channels.
- **Connect to low-bit quantization**: after rotation the tensor distribution is more uniform, so connecting to low-bit quantization such as `linear_quant` improves accuracy; keep the subsequent quantization configuration unchanged and adjust only the rotation parameters to attribute the gain.
- **Validate accuracy and deployment info**: when `export_extra_info: true`, the saver exports metadata such as the global rotation matrix for the downstream inference chain to rebuild the rotation; do not disable it casually. The end-to-end accuracy and deployment-chain compatibility verification are authoritative.

## 4. Procedure

### Step 1: Adapt the Model Pipeline Interface

**Goal**: The quantization tool cannot operate directly on models of arbitrary structure; it requires each model to be wrapped in an "adapter" first that translates the model's various operations into standard methods the framework can call uniformly. This step confirms and completes that adapter layer. Model adaptation must be completed first before the QuaRot algorithm can run.

**Action**: The adapter must implement the basic pipeline interface `PipelineInterface` (located at `msmodelslim.core.runner.pipeline_interface`, inheriting from the basic attribute interface `IModel`). This interface defines the 5 standard methods the framework needs to drive a model: load data (`handle_dataset`), load the model (`init_model`), layer-wise traversal (`generate_model_visit`), layer-wise forward pass (`generate_model_forward`), and control the KV Cache (`enable_kv_cache`). The quantization runner and processor drive the model only through these standard methods, independently of the model's internal structure.

For a standard open-source LLM implemented with HuggingFace Transformers, it is recommended to build the adapter quickly through compositional inheritance:

```python
from msmodelslim.model.interface_hub import (
    IModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1 as PipelineInterface,
)

class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface):
    pass
```

`TransformersModel` provides the generic LLM loading and tokenizer logic, and the framework interfaces provide standardized driving capability; together they form a complete adapter. The adapter is then referenced in the quantization command via `--model_type MyModelAdapter`.

To develop new model adaptation code, see the [LLM Model Quantization Integration Guide](../../ptq/llm/integration_guide_large_language_model_quantization.md).

**Output**: A usable model adapter registered in the framework (invoked via `--model_type`).

### Step 2: Adapt the Model Rotation Interfaces

**Goal**: Complete the QuaRot rotation-related model adaptation in the adapter, providing the network structure information (rotation mappings, Norm fusion, mean fusion, etc.) that the rotation processor needs. The QuaRot algorithm cannot run before adaptation is complete.

**Action**: The rotation-related adaptation code in the model adapter must implement the following interfaces, all of which are aggregated and provided by `msmodelslim.model.interface_hub`:

1. **`QuaRotInterface`** (located at `msmodelslim.processor.quarot.offline_quarot.quarot_interface`): the offline rotation core interface.

   ```python
   class QuaRotInterface:
       QuaRotMode = QuaRotMode
       RotatePair = RotatePair

       @staticmethod
       def get_rotate_command(mode: QuaRotMode,
                  size: int,
                  block_size: int = -1,
                  rot_step: int = 1,
                  eye_step: tuple = (-1,)):
           ...

       @abstractmethod
       def get_ln_fuse_map(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
           ...

       @abstractmethod
       def get_bake_names(self) -> Tuple[List[str], List[str]]:
           ...

       @abstractmethod
       def get_rotate_map(self, block_size: int) -> Tuple[List[RotatePair], List[RotatePair]]:
           ...
   ```

   - **`get_rotate_command`**: calls `create_rot` to generate the orthogonal rotation matrix (Walsh-Hadamard, etc.), and moves it automatically to the current NPU device when an NPU is available.
   - **`get_ln_fuse_map`**: returns the fusion map between LayerNorm layers and Linear layers (the two dictionaries `pre_run_fused_ln` and `fused_map`), used for Norm fusion before rotation.
   - **`get_bake_names`**: returns the list of Linear layer names that need mean fusion (`pre_run_bake_names` and `bake_names`); usually needed when the model uses `nn.LayerNorm`.
   - **`get_rotate_map`**: returns the rotation map, including `pre_run_pairs` for the pre-run stage (usually embedding-layer rotation) and `rotate_pairs` for the preprocess stage; each entry is a list of `RotatePair` carrying left/right rotation matrices.
2. **`OnlineQuaRotInterface`** (located at `msmodelslim.processor.quarot.online_quarot.online_quarot_interface`): the simplified online rotation interface.

   ```python
   class OnlineQuaRotInterface:
       RotationConfig = RotationConfig
       QuaRotMode = QuaRotMode

       @abstractmethod
       def get_online_rotation_configs(self, model: Optional[nn.Module] = None) -> Dict[str, RotationConfig]:
           ...
   ```

   Returns a mapping from module names (such as `model.layers.0.self_attn.q_rot`) to `RotationConfig`; `rotation_type` supports `"input"`, `"output"`, `"replace"`, and `"offline"`, and you can provide a custom rotation matrix or specify `rotation_size` for automatic generation.
3. **`LAOSOnlineRotationInterface`** (located at `msmodelslim.processor.quarot.offline_quarot.quarot_interface`): the LAOS online rotation interface, used to describe the attention head dimension and the rotation pairs of the O/V and up/down projections of each decoder layer.

   ```python
   class LAOSOnlineRotationInterface:
       @abstractmethod
       def get_head_dim(self):
           ...

       @abstractmethod
       def get_num_attention_heads(self):
           ...

       @abstractmethod
       def get_layer_wise_ov_pair(self, decoder_module):
           ...

       @abstractmethod
       def get_layer_wise_up_down_pair(self, decoder_module):
           ...
   ```

4. **`AdaptRotationInterface`** (located at `msmodelslim.processor.adapt_rotation`): the adaptation interface for the two-stage AdaptRotation rotation (stage1/stage2); it only needs to be implemented when orchestrating the `adapt_rotation` processor.

   All the interfaces above are exported together from `msmodelslim.model.interface_hub`; during adaptation development, simply import them from that module.

**Recommended adapter implementation**:

1. **Choose interfaces by rotation scenario**: the adapter does not need to implement all interfaces; choose according to the actual rotation mode:
   - **Offline rotation (`online: false`, recommended starting point)**: implement `QuaRotInterface`, focusing on providing `get_rotate_map` (left/right rotation maps of each decoder layer), `get_ln_fuse_map` (LayerNorm-to-Linear fusion map), and `get_bake_names` (mean fusion layers).
   - **Online rotation (`online: true`)**: implement `OnlineQuaRotInterface`, returning each module's `RotationConfig` (`rotation_type` is `"input"`/`"output"`/`"replace"`/`"offline"`). If `down_proj_online_layers` or LAOS features are used, `LAOSOnlineRotationInterface` must also be implemented.
   - **When orchestrating the `adapt_rotation` processor**: additionally implement `AdaptRotationInterface`.

2. **Recommended compositional inheritance**: on top of the Step 1 adapter, inherit the QuaRot-related interfaces compositionally:

   ```python
   from msmodelslim.model.interface_hub import (
       # ---- Rotation algorithm adaptation interfaces (focus of this step) ----
       QuaRotInterface,          # Algorithm adaptation: provides the LayerNorm fusion map (get_ln_fuse_map) and rotation pairs (get_rotate_map)
       OnlineQuaRotInterface,    # Algorithm adaptation: provides online rotation capability
   )

   class MyModelAdapter(TransformersModel, ModelInfoInterface, PipelineInterface,
                        QuaRotInterface, OnlineQuaRotInterface):
       pass
   ```

3. **Refer to existing adaptation implementations**: the built-in model adapters in the repository can be referenced directly, such as `msmodelslim/model/glm_5/quarot.py` and `msmodelslim/model/deepseek_v3/quarot.py` (QuaRot interface implementations), and `msmodelslim/model/glm_5/model_adapter.py` (adapter composition approach).

4. **Develop new model adaptation code**: see the [LLM Model Quantization Integration Guide](../../ptq/llm/integration_guide_large_language_model_quantization.md).

**Output**: A usable model adapter registered in the framework (invoked via `--model_type`).

### Step 3: Select and Orchestrate the QuaRot Quantization Algorithm

**Goal**: Determine QuaRot's rotation mode and key parameters, and connect to the subsequent low-bit quantization algorithm chain as needed.

**Action**:

1. **QuaRot algorithm introduction**:
   QuaRot introduces an orthogonal Walsh-Hadamard rotation matrix into the inter-layer computation, breaking up the outlier channels in the activation distribution without changing the model's forward output so that the activation and weight distributions become more uniform; it usually serves as a preprocessing step before quantization.

2. **Recommended configuration**:

   ```yaml
   spec:
     process:
       - type: "quarot"                     # Fixed to `quarot`; specifies the processor type.
         online: False                      # Whether to enable online rotation; default False.
         block_size: -1                     # Block size when the rotation matrix uses a block-diagonal matrix; the value is -1 or a power of 2. If it is greater than 0 it must be a power of 2; -1 means no block-diagonal processing
         max_tp_size: 4                     # Maximum tensor-parallel size, default 4; takes effect only when online rotation is enabled, and must be greater than 0 and a power of 2
         down_proj_online_layers: [ ]       # Specifies which layers' down_proj use online rotation; empty by default
         export_extra_info: True            # Whether to export the global rotation matrix: with ascend_v1_saver it creates an optional directory under the quantized weight path and appends fields to quant_model_description.json; default True
   ```

3. **Parameter selection and tuning guide**:

   Change only one variable per round and keep the same calibration set and evaluation set.

   | Configuration Item | Meaning (Rationale) | Recommended Configuration | Selection and Tuning Advice |
   | --- | --- | --- | --- |
   | `type` | Processor identifier, fixed to `quarot`. | Fixed. | Not recommended to change. |
   | `online` | Whether to keep online rotation. `false` fuses offline as much as possible; `true` introduces runtime rotation and affects parallelism/operator requirements. | Default/prefer `false`. | Enable it only when the target deployment chain explicitly supports and requires online rotation. After enabling it, do not look only at accuracy; also evaluate inference latency and operator compatibility. |
   | `block_size` | Rotation block size; `-1` means the full hidden_dim, and a positive value must be a power of 2. In repository practices many models explicitly use `32`. | Start with `-1` when there is no model recipe; when a practice already uses a value such as `32`, follow it directly. | The smaller the block, the more local the rotation, which may reduce its ability to mix outliers across all dimensions, but it fits certain parallelism/hardware constraints better. |
   | `down_proj_online_layers` | Specifies which decoder layers' down_proj use online rotation; empty by default. | Default `[]`. | Add entries only when these layers cannot be fused offline or a dedicated recipe requires it. The more layers, the more extra runtime rotation. |
   | `max_tp_size` | The maximum TP supported by online rotation; must be 1 or a power of 2, default 4. | For online mode, set it to the actual maximum deployment TP; keep the default for offline. | Adjust it only when the actual deployment TP changes. |
   | `export_extra_info` | Whether to export the global rotation metadata; default true, consumed by specific savers/deployment chains. | Keep `true` if the target saver/backend needs the QuaRot metadata. | Disable it only when you are certain the downstream never reads these fields. |

4. **Algorithm orchestration guide**:
   - **Run the recommended configuration first, then tune single variables.** Do not change the bit width, granularity, algorithm parameters, and layer scope all at once.
   - **Prefer local fallback over raising accuracy globally.** If only a few layers are sensitive, preserve high precision via `exclude` or a mixed strategy.
   - **The model practice configuration and deployment capability are authoritative.** When the target model already has a `lab_practice` recipe, prefer reusing the verified combination.
   - QuaRot must be followed by low-bit quantization (such as `linear_quant`); QuaRot itself is usually not the final quantization format.

**Output**: The `process` algorithm chain fully orchestrated in the YAML.

### Step 4: Write the Quantization Configuration and Run the Command

**Goal**: Integrate the above steps into a complete YAML quantization configuration file and start the quantization workflow via the CLI.

#### Complete Example: QuaRot + W8A8 Dynamic Quantization (Recommended Starting Point)

##### Configuration File: `quarot_w8a8_dynamic.yaml`

```yaml
apiversion: modelslim_v1
spec:
  runner: auto                    # layer_wise for single-card, dp_layer_wise for multi-card
  process:
    - type: quarot                # QuaRot outlier suppression preprocessing
      online: False               # Prefer offline rotation fusion
      block_size: -1              # -1 means rotation over the full hidden_dim
      max_tp_size: 4              # Maximum TP effective when online rotation is enabled
      down_proj_online_layers: [] # No down_proj online rotation layers by default
      export_extra_info: True     # Export the global rotation matrix metadata
    - type: linear_quant          # connect to low-bit quantization
      qconfig:
        act:
          dtype: int8
          scope: per_token        # dynamic activation quantization, good accuracy
          symmetric: true
          method: minmax
        weight:
          dtype: int8
          scope: per_channel
          symmetric: true
          method: minmax
      include: ["*"]
  save:
    - type: ascendv1_saver        # Ascend inference standard save format
  dataset: mix_calib.jsonl        # built-in mixed calibration set
```

##### Run the command (single-card quantization)

```bash
msmodelslim quant \
  --model_path <float_model_dir> \
  --save_path <quantized_weight_output_dir> \
  --model_type <model_adapter_name> \
  --config ./quarot_w8a8_dynamic.yaml \
  --device npu \
  --device_id 0
```

**Output**: Complete quantized weight files and description files (including QuaRot rotation metadata) are generated in the directory specified by `--save_path`.

## 5. Terminology

| Term | Brief Description | Link |
| --- | --- | --- |
| QuaRot Rotation Quantization Algorithm | Explains the definition, core principle, key properties, applicable scenarios, and limitations of the algorithm. | [QuaRot Rotation Quantization Algorithm Terminology Entry](./term_quarot.md) |

## 6. Related Documents

| API or Document | Brief Description | Link |
| --- | --- | --- |
| `PipelineInterface` | Model pipeline adaptation interface (data preprocessing, model loading, module traversal). | [LLM Quantization Usage Guide, Step 1](../../ptq/llm/usage_large_language_model_quantization.md) |
| `QuaRotInterface` / `OnlineQuaRotInterface` / `LAOSOnlineRotationInterface` / `AdaptRotationInterface` | QuaRot algorithm model adaptation interfaces, all exported together from `msmodelslim.model.interface_hub`. | [Interface hub module](../../../../../msmodelslim/model/interface_hub.py) |
| quarot Configuration Description | Field types, default values, valid values, and complete configuration constraints. | [quarot Configuration Description](../../../api_reference/config/processor/quarot.md) |
| modelslim_v1 Configuration Description | Consult this when you need to explore task-level advanced configurations such as runner, prior, save, and dataset. | [modelslim_v1 Configuration Description](../../../api_reference/config/quant/modelslim_v1.md) |
| Weight Quantization Usage Guide | User guide: quantization command parameters and complete usage instructions. | [Weight Quantization Usage Guide](../../../user_guide/usage_weight_quantization.md) |
