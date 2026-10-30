# DeepSeek-V4-Pro New Model W4A8 Quantization Case

## 1. Case Background

**Objective**: integrate the newly released DeepSeek-V4-Pro model into the msModelSlim code repository, complete the development of the model adapter and the quantization configuration, run W4A8 one-click quantization and accuracy verification, and, if the accuracy does not meet the target, perform quantization tuning, completing the full closed loop from model integration to quantization deployment and quantization tuning.

**Comparison target**: compare the evaluation result of the quantized weights `${SAVE_PATH}` with the measured floating-point baseline or the baseline published in the paper or on the model card. Prefer the measured floating-point baseline obtained under identical evaluation conditions; when a published baseline is used, ensure that the evaluation dataset, service configuration, generation parameters, and post-processing configuration are consistent, and note the baseline source in the evaluation record.

**Accuracy acceptance criteria**: use the selected accuracy baseline (the measured floating-point baseline or the baseline published in the paper or on the model card) as the basis for acceptance, and ensure that the quantized result and the baseline use identical evaluation conditions. If the quantized result is not lower than the baseline, the case passes directly; if the quantized result is lower than the baseline, it passes only when the relative drop does not exceed 1%; if the relative drop exceeds 1%, it fails and proceeds to Step 8, quantization tuning.

**Covered steps**: analyze the new model structure in `${MODEL_PATH}/config.json` → design the `lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml` quantization scheme → complete the development of the `msmodelslim/model/deepseek_v4/` model adapter and the `lab_practice/deepseek_v4/` quantization YAML → environment preparation and tool installation → run the one-click quantization command → check the integrity of the `${SAVE_PATH}` quantized weights → inference deployment and accuracy evaluation with `${SAVE_PATH}` → quantization tuning (optional).

**Related process**: *[Weight Quantization Usage Guide](../user_guide/usage_weight_quantization.md)*, *[One-Click Quantization Usage Guide](../user_guide/usage_one_click_quantization.md)*, *[LLM Model Integration Quantization Workflow Guide](../knowledge_base/ptq/llm/integration_guide_large_language_model_quantization.md)*, *[Quantization Algorithm Overview](../knowledge_base/quantization_algorithms/README.md)*

**Model information**

| Item | Content |
| --- | --- |
| Model name | DeepSeek-V4-Pro |
| model_type | DeepSeek-V4-Pro |
| Parameter count | 685B |
| Modality | Text |
| Model structure | MoE × 256 + MLA + low-rank attention compression + MTP (multi-token prediction) |
| Quantization type | w4a8 (mixed quantization: routed experts W4A8, shared experts and attention layers W8A8) |
| Open-source weight source | [ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro) |
| Inference framework and parallelism | vLLM Ascend 0.22.1rc1, TP16 + DP2 (2 nodes × 16 devices/node), with Expert Parallel enabled |

## 2. Environment and Versions

| Item | Version or Configuration |
| --- | --- |
| Product form | Atlas 800I A3 products (restricted; this case is verified on the two-node 32-device environment of this server) |
| Runtime image | [quay.io/ascend/vllm-ascend:v0.22.1rc1-a3](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=v0.22.1rc1-a3) |
| CANN | CANN-9.0.0 (included in the image) |
| PyTorch | 2.10.0 (included in the image) |
| TorchNPU | 2.10.0 (included in the image) |
| vLLM Ascend | [0.22.1rc1](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=v0.22.1rc1-a3) (included in the image) |
| Evaluation tool | [AISBench](https://github.com/AISBench/benchmark) (for installation, see the [official documentation](https://github.com/AISBench/benchmark/blob/master/README.md)) |
| Other dependencies | `transformers==4.48.2` (downgrade required for quantization; the image defaults to 5.5.4) |

**Prerequisites for this case**:

- The DeepSeek-V4-Pro floating-point weights have been obtained (download link: [ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro)), the official vLLM Ascend image container has been started with the NPU devices and the model weight directory mounted, and it has been confirmed that the msModelSlim source code can be pulled normally (this case integrates the new model from source).

## 3. Inputs and Deliverables

| Type | Name | Source or Location | Format or Constraint | Acceptance Method |
| --- | --- | --- | --- | --- |
| Input | Floating-point model directory | `${MODEL_PATH}` (downloaded by the user) | Contains `config.json`, `tokenizer.json`, and model weight files (safetensors; the weights may be FP8); can be loaded by Transformers 4.48.2 | `transformers.AutoModel.from_pretrained` loads it without errors |
| Input | Calibration dataset | [Mixed calibration dataset (mix_calib.jsonl)](../../../lab_calib/mix_calib.jsonl) | JSONL format; each line contains the `inputs_pretokenized` field | The calibration set can be read normally by the quantization process |
| Deliverable | Model adapter code | [msmodelslim/model/deepseek_v4/](../../../msmodelslim/model/deepseek_v4/) (adapter + loader) | The adapter must implement the interfaces required by the quantization pipeline (QuaRot, FlexSmoothQuant, and so on) | After the msmodelslim tool is installed, the entry point registration takes effect, and `--model_type DeepSeek-V4-Pro` can hit the adapter |
| Deliverable | Quantization configuration file | [lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml](../../../lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml) | Contains the quantization strategy, qconfig, calibration dataset, and saver configuration | `msmodelslim quant --config ${CONFIG_PATH}` can load the configuration |
| Deliverable | Quantized weight directory | `${SAVE_PATH}` (output directory of the quantization workflow) | Contains a parseable `quant_model_description.json`, complete weight files (safetensors and the necessary shards/index), `config.json`, and auxiliary files such as the tokenizer | The quantized weights can be served by vLLM Ascend `--quantization ascend` and complete an inference smoke test |
| Deliverable | Accuracy test results | The AISBench output directory (`${QUANT_WORK_DIR}`) | Contains the selected accuracy baseline under identical evaluation conditions, the quantized result, and the acceptance calculation records; if a score published in the paper or on the model card is used, the baseline source must be noted and the evaluation conditions must be consistent | If the quantized result is not lower than the selected accuracy baseline, the case passes directly; if the quantized result is lower than the selected accuracy baseline and the relative drop does not exceed 1%, it passes; if the quantized result is lower than the selected accuracy baseline and the relative drop exceeds 1%, it fails |

## 4. Procedure

### Step 1: New Model Structure Analysis

**Objective**: analyze the model structure of DeepSeek-V4-Pro to provide the basis for the quantization scheme design and the adapter development.

**Inputs**: the floating-point weights have been downloaded to `${MODEL_PATH}`.

**Operations**:

1. Review the model structure configuration:

   ```bash
   ls ${MODEL_PATH}/
   cat ${MODEL_PATH}/config.json
   ```

   In `config.json`, confirm structural parameters such as `num_hidden_layers`, `n_routed_experts`, `n_shared_experts`, `compress_ratios` (the per-layer attention compression ratio), `q_lora_rank`/`o_lora_rank` (low-rank projections), and `n_mtp_layers` (the number of MTP layers).

2. Confirm the key structural characteristics based on the model structure:

   - **MoE architecture**: routed experts + shared experts (`n_routed_experts` / `n_shared_experts`). There are many routed experts and a single layer holds a large number of parameters, which is the focus of the quantization scheme design.
   - **Attention structure**: MLA low-rank projections (`q_lora_rank`/`o_lora_rank`), per-layer variable attention compression (`compress_ratios`), and the `compressor`/`indexer` submodules.
   - **MTP layers**: multi-token prediction (Multi-Token Prediction) structures that must be handled separately during loading and quantization.
   - **Weight precision**: the model weights may be in FP8 precision, so the adapter must support dequantization when loading.

**Output**:

- A list of the model structure parameters and a description of the key structural characteristics.

**Record**: a summary of the model structure parameters (number of layers, number of experts, compression ratios, and so on).

**Next step**: design the quantization scheme.

---

### Step 2: Design the Quantization Scheme

**Objective**: design a W4A8 mixed quantization scheme based on the model structure, and determine the quantization strategy and the quantization granularity of each module.

**Inputs**: the model structure characteristics confirmed in Step 1.

**Operations**:

1. Determine the overall quantization strategy. Given the characteristics of the large MoE model DeepSeek-V4-Pro, this case adopts a mixed quantization scheme dominated by W4A8, which reduces the model storage and inference memory overhead while maintaining accuracy.

2. Select the quantization algorithms. The expert layers and activations of a MoE model usually contain outliers, and direct low-bit quantization easily causes accuracy loss, so QuaRot and SSZ are selected for outlier handling:

   - **QuaRot rotation pre-processing** (`block_size=32`): improves the activation distribution through a Hadamard rotation, reducing the accuracy loss of the subsequent quantization.
   - **Flex AWQ SSZ outlier suppression**: acts on the `up-down` subgraph of the routed experts and uses the SSZ method to suppress outliers per channel, so that the routed experts can stably perform INT4 weight quantization.

3. Determine the quantization granularity of each module (corresponding to `spec.process` in [lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml](../../../lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml)):

   | Module | Match Rule | Quantization Strategy | Weight / Activation Configuration |
   | --- | --- | --- | --- |
   | Routed experts | `*ffn*` (excluding `*gate`, `*shared_experts*`) | W4A8 dynamic quantization | Weights per-channel INT4 (ssz), activations per-token INT8 (minmax) |
   | Shared experts | `*ffn.shared_experts*` | W8A8 dynamic quantization | Weights per-channel INT8 (minmax), activations per-token INT8 (minmax) |
   | Attention layers | `*attn*` (excluding low-rank branches such as `wo_a`/`wo_b`, `compressor.wgate`/`wkv`, and `indexer.weights_proj`) | W8A8 dynamic quantization | Weights per-channel INT8 (minmax), activations per-token INT8 (minmax) |

4. Determine the calibration dataset and the save method: the calibration dataset uses the built-in [lab_calib/mix_calib.jsonl](../../../lab_calib/mix_calib.jsonl); saving uses `ascendv1_saver` (natively compatible with vLLM Ascend, saving in 4GB shards).

**Output**:

- A list of the quantization strategy and quantization granularity for each module.

**Record**: the key points of the quantization strategy (QuaRot + Flex AWQ SSZ + mixed quantization).

**Next step**: complete the development of the model adapter and the quantization YAML.

---

### Step 3: Develop the Model Adapter and the Quantization YAML

**Objective**: complete the model adaptation, adapter registration, and W4A8 quantization configuration for DeepSeek-V4-Pro so that `msmodelslim quant` can recognize and quantize the model.

**Inputs**: the model structure information from Step 1 and the quantization scheme from Step 2.

**Operations**:

1. Implement the model structure and the adapter:

   - Define the structures required for MoE, MLA, MTP, and the quantization forward pass in [model.py](../../../msmodelslim/model/deepseek_v4/model.py), and set `USE_DP_MODE=True`.
   - Implement model loading, layer-by-layer forward, subgraph and rotation mapping, and weight saving in [model_adapter.py](../../../msmodelslim/model/deepseek_v4/model_adapter.py), and integrate it with the `modelslim_v1` pipeline. The adapter must support BF16 loading, `auto_dequant_state_dict` dequantization, and on-demand loading of the main model and the MTP layers, and must implement the following interfaces:

     | Interface | Purpose |
     | --- | --- |
     | `get_model_pedigree()` | Returns `'deepseek_v4'`. |
     | `init_model()`, `generate_decoder_layer()`, `generate_model_forward()` | Initialize the model, load it layer by layer, execute the forward pass, and hand the input to the quantization processors through `ProcessRequest`. |
     | `get_adapter_config_for_subgraph()` | Declares the expert `up-down`, MLA `linear-linear`, and attention `norm-linear` subgraphs. |
     | `get_ln_fuse_map()` | Configures the fusion of `ffn_norm`, `attn_norm`, `q_norm`, and the MTP normalization layers. |
     | `get_rotate_map()` | Configures a Hadamard rotation with a block size of 32, covering the Embedding, expert, attention, and MTP projections; the low-rank query projection is handled separately. |
     | `ascendv1_save_module_preprocess()` | Retains the module names and objects for `ascendv1_saver` to save the vLLM Ascend quantized weights. |

   During multi-device quantization, use `dist.barrier()` to synchronize the ranks and partition the local routed experts by rank; the shared experts, attention layers, and MTP layers are processed in a unified layer order. Finally, use the quantization log to confirm that each rank has been initialized and that there is no fallback to a single device.

2. Register the model adapter ([config/config.ini](../../../config/config.ini)):

   ```ini
   [ModelAdapter]
   deepseek_v4 = DeepSeek-V4-Flash, DeepSeek-V4-Pro

   [ModelAdapterEntryPoints]
   deepseek_v4 = msmodelslim.model.deepseek_v4.loader:DeepseekV4AdapterLoader
   ```

   `DeepseekV4AdapterLoader` points to `DeepSeekV4ModelAdapter`. After modifying the adapter or the registration configuration, reinstall msmodelslim for the entry point to take effect; `--model_type DeepSeek-V4-Pro` should hit this adapter.

3. Write the [quantization configuration](../../../lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml):

   - `spec.process` executes in the order `quarot` → `flex_awq_ssz` → `flex_smooth_quant` → `linear_quant`: QuaRot uses `block_size: 32`; Flex AWQ SSZ handles the routed expert `up-down` subgraph; Flex Smooth Quant handles the `norm-linear` subgraph; Linear Quant performs W4A8 on the routed experts and W8A8 on the shared experts and attention modules, and uses `include`/`exclude` to exclude non-matching modules such as the low-rank branches and the shared experts.
   - W4A8 uses per-token INT8 activations, per-channel INT4 weights, and SSZ; W8A8 uses per-token INT8 activations, per-channel INT8 weights, and minmax. The calibration set is `mix_calib.jsonl`, the saver is `ascendv1_saver`, and the output is saved in 4 GB shards.

4. Verify the adapter and the configuration:

   - Confirm that `--model_type DeepSeek-V4-Pro` matches the corresponding model adapter normally.
   - Check that the YAML loads normally with `--config ${CONFIG_PATH}`.
   - Run a small-scale or formal quantization as in Step 5, and confirm that the log contains the adapter loading, the processor execution, and the multi-device rank initialization information.
   - Check that the output directory contains `quant_model_description.json` and complete weight files, and then go to Step 6 to accept the artifacts.

**Output**: a model adapter and quantization configuration that can be recognized by `msmodelslim quant`.

**Record**: the adapter code, the registration configuration, and the quantization YAML path.

**Next step**: environment preparation and tool installation.

---

### Step 4: Environment Preparation and Tool Installation

**Objective**: confirm that the container environment meets the quantization requirements, and install the msModelSlim tool and its dependencies.

**Inputs**: the official vLLM Ascend image container has been started.

**Operations**:

1. If the environment has not been started yet, refer to the [official vLLM Ascend guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Pro.html) to pull the image and create the container (the NPU devices and the model weight directory must be mounted). The image used in this case is `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3`.

2. Check the NPU status and the CANN environment inside the container:

   ```bash
   pip list 2>/dev/null | grep -iE "torch|npu|cann"
   env | grep ASCEND
   npu-smi info
   ```

3. Record the current Transformers version so that it can be restored after quantization:

   ```bash
   pip list 2>/dev/null | grep transformers
   # The current version is 5.5.4
   ```

4. Download the msModelSlim source code and install the msmodelslim tool:

   ```bash
   git clone https://gitcode.com/Ascend/msmodelslim
   cd msmodelslim
   bash install.sh
   ```

   After installing the msmodelslim tool, exit the source directory and run `msmodelslim --help` to verify the installation.

   ```bash
   cd ..  # Leave the source directory; the msmodelslim command must be run outside it
   msmodelslim --help
   ```

5. Install the Transformers version adapted to DeepSeek-V4-Pro:

   ```bash
   pip install transformers==4.48.2
   ```

   > [!NOTE]
   >
   > msModelSlim does not require a specific Transformers version, but `transformers==4.48.2` is recommended for the DeepSeek-V4-Pro model. The downgrade is only for the quantization stage; after quantization you can run `pip install transformers==5.5.4` to restore it.

**Output**:

- `msmodelslim --help` runs successfully.
- The Transformers version has been switched to `4.48.2`.

**Record**: run `pip list | grep transformers` to confirm the version.

**Next step**: run the one-click quantization command.

---

### Step 5: Run the One-Click Quantization Command

**Objective**: use the `msmodelslim quant` command to generate the quantized weights. Based on `--model_type DeepSeek-V4-Pro`, `msmodelslim quant` hits the adapter developed in Step 3 through the entry point and loads the [deepseek_v4_pro_w4a8.yaml](../../../lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml) quantization configuration directly through `--config`, then executes each quantization processor in the order of `spec.process` and saves the artifacts.

**Inputs**: the floating-point weight directory `${MODEL_PATH}`.

**Operations**:

Let `${MODEL_PATH}` be the original floating-point weight directory, `${SAVE_PATH}` the quantization output directory, and `${CONFIG_PATH}` the quantization configuration file [deepseek_v4_pro_w4a8.yaml](../../../lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml).

```bash
msmodelslim quant \
    --model_path ${MODEL_PATH} \
    --save_path ${SAVE_PATH} \
    --model_type DeepSeek-V4-Pro \
    --config ${CONFIG_PATH} \
    --device npu \
    --device_id 0 1 2 3 4 5 6 7 \
    --trust_remote_code true
```

Parameter description:

- `--model_path`: path to the original floating-point model weight directory.
- `--save_path`: path to the directory where the quantized weights are saved.
- `--model_type`: model type, fixed to `DeepSeek-V4-Pro`.
- `--quant_type`: quantization strategy type; `w4a8` means weights INT4 + activations INT8.
- `--device`: specifies the NPU devices used for quantization. The W4A8 quantization algorithm is relatively complex and single-device quantization takes a long time, so it is recommended to use 8 devices (`npu --device_id 0 1 2 3 4 5 6 7`) for multi-device quantization to shorten the runtime.
- `--trust_remote_code`: trust custom model code (ensure that the code source is reliable).

DeepSeek-V4-Pro already sets `USE_DP_MODE=True` in `model.py` and implements distributed synchronization and EP local-expert scope handling in `model_adapter.py`, so multi-device quantization can be started with the command above without extra adaptation. The quantization log still needs to confirm that multi-device distributed startup and per-rank initialization information actually appear.

**Output**:

- the quantized weight directory `${SAVE_PATH}`, containing `quant_model_description.json`, weight files (safetensors format), `config.json`, and so on.
- The quantization log, which must record the multi-device distributed startup, the per-rank initialization, and the processor execution.

**Record**: the quantization log is printed to the terminal and can be redirected to a file for future reference; confirm that 8 NPUs are actually enabled in the log and that no single-device fallback message appears.

**Next step**: quantized weight integrity check.

---

### Step 6: Quantized Weight Integrity Check

**Objective**: confirm that the quantized weight artifacts are complete.

**Inputs**: the quantized weight output directory `${SAVE_PATH}`.

**Operations**:

1. Check the output directory structure:

   ```bash
   ls ${SAVE_PATH}/
   # Should contain quant_model_description.json, config.json, weight shards (*.safetensors and index), and auxiliary files such as tokenizer
   ```

2. Confirm that `quant_model_description.json` exists and its content is complete:

   ```bash
   cat ${SAVE_PATH}/quant_model_description.json | head -20
   ```

**Output**:

- The verified DeepSeek-V4-Pro W4A8 quantized weight directory `${SAVE_PATH}`.

**Pass conditions**: the quantization artifact directory complies with the AscendV1 export format conventions (`quant_model_description.json` + weight safetensors and the necessary shards/index); for details, see [AscendV1 Export Artifacts](../knowledge_base/quantization_format/ascendv1/term_ascendv1.md#export-artifacts). The quantization log contains no `ERROR`.

**Record**: the structure of the quantized weight directory and a summary of `quant_model_description.json`.

**Next step**: inference deployment and evaluation.

---

### Step 7: Inference Deployment and Evaluation

**Objective**: deploy the quantized weights to the vLLM Ascend inference engine, complete the loading smoke test and the accuracy evaluation, and compare the result with the selected accuracy baseline.

**Inputs**: the quantized weight directory `${SAVE_PATH}`, the selected accuracy baseline, and the quantized weight check record from Step 6.

**Operations**:

1. Restore the Transformers version (restore it to the initial version of the container after quantization):

   ```bash
   pip install transformers==5.5.4
   ```

2. Configure the environment variables and start the vLLM Ascend inference service (refer to the [official vLLM Ascend documentation](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Pro.html)):

   The DeepSeek-V4-Pro model is relatively large, and the official example uses multi-node deployment. The following is the startup command for node 0 of the Atlas 800I A3 products; for the specific commands, refer to the corresponding official documentation:

   ```bash
   export QUANT_HOST=${QUANT_HOST:-localhost}
   export QUANT_PORT=${QUANT_PORT:-8900}
   export HCCL_OP_EXPANSION_MODE="AIV"
   export HCCL_IF_IP=$local_ip
   export GLOO_SOCKET_IFNAME=$nic_name
   export TP_SOCKET_IFNAME=$nic_name
   export HCCL_SOCKET_IFNAME=$nic_name
   export HCCL_BUFFSIZE=2048
   export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
   export OMP_PROC_BIND=false
   export OMP_NUM_THREADS=10
   export TASK_QUEUE_ENABLE=1
   export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
   export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

   vllm serve ${SAVE_PATH} \
       --safetensors-load-strategy 'prefetch' \
       --max-model-len 135000 \
       --max-num-batched-tokens 4096 \
       --served-model-name dsv4-pro \
       --gpu-memory-utilization 0.9 \
       --max-num-seqs 32 \
       --data-parallel-size 2 \
       --data-parallel-size-local 1 \
       --data-parallel-start-rank 0 \
       --data-parallel-address $node0_ip \
       --data-parallel-rpc-port 13399 \
       --tensor-parallel-size 16 \
       --enable-expert-parallel \
       --quantization ascend \
       --port ${QUANT_PORT} \
       --host ${QUANT_HOST} \
       --block-size 128 \
       --async-scheduling \
       --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
       --tokenizer-mode deepseek_v4 \
       --tool-call-parser deepseek_v4 \
       --enable-auto-tool-choice \
       --reasoning-parser deepseek_v4 \
       --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
       --additional-config '
       {"ascend_compilation_config":{
           "enable_npugraph_ex":true,
           "enable_static_kernel":false
           },
       "enable_cpu_binding": true,
       "multistream_overlap_shared_expert":true}'
   ```

   The command above is used to load the quantized weights `${SAVE_PATH}` as a service. The selected accuracy baseline has been provided in advance, so this step does not start a floating-point service or repeat the test.

   > [!NOTE]
   >
   > DeepSeek-V4-Pro requires multi-node deployment, with at least 2 Atlas 800I A3 product nodes (16 devices/node). Node 1 requires the additional `--headless` parameter and must set `--data-parallel-start-rank` to 1. For detailed configuration, see the [official vLLM Ascend documentation](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Pro.html).

3. After the service starts, use curl to verify that the quantized service answers normally:

   ```bash
   curl "http://${QUANT_HOST}:${QUANT_PORT}/v1/chat/completions" \
       -H "Content-Type: application/json" \
       -d '{
           "model": "dsv4-pro",
           "messages": [{"role": "user", "content": "Who are you?"}],
           "max_tokens": 256,
           "temperature": 0
       }'
   ```

Use AISBench to run the accuracy test on the GPQA dataset. The test design is as follows:

| Item | Content |
| --- | --- |
| Evaluation tool | AISBench (for installation, see the [AISBench repository installation guide](https://github.com/AISBench/benchmark/blob/master/README.md)) |
| Evaluation target | W4A8 quantized weights `${SAVE_PATH}` |
| Comparison baseline | The selected accuracy baseline; a score published in the paper or on the model card may be used as the baseline, but the source must be noted and the evaluation conditions must be consistent |
| Dataset and task | [GPQA](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/gpqa) |
| Acceptance rule | If the quantized result is not lower than the selected accuracy baseline, the case passes directly; if the quantized result is lower than the selected accuracy baseline and the relative drop does not exceed 1%, it passes; if the quantized result is lower than the selected accuracy baseline and the relative drop exceeds 1%, it fails and proceeds to Step 8 |
| Configuration consistency | The quantization evaluation uses the same dataset, service configuration, generation parameters, and post-processing pipeline as the selected accuracy baseline |

During evaluation, use the same evaluation data, service configuration, generation parameters, and post-processing pipeline as the selected accuracy baseline, run the GPQA evaluation only on the quantized weights `${SAVE_PATH}`, record the quantized result, and compute the relative drop against the selected accuracy baseline.

**Key evaluation configuration points**:

| Config Item | Value |
| --- | --- |
| Configuration file path | `vllm_api_general_chat.py` (custom configuration, see below) |
| Service address / port | Quantized service `${QUANT_HOST}:${QUANT_PORT}` |
| served-model-name | Quantized service `dsv4-pro` |

**Evaluation configuration (`vllm_api_general_chat.py`)**:

```python
import os

from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="",
        model=os.environ["SERVE_MODEL"],
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip=os.environ.get("SERVE_HOST", "localhost"),
        host_port=int(os.environ["SERVE_PORT"]),
        url="",
        max_out_len=65536,
        batch_size=16,
        trust_remote_code=False,
        generation_kwargs=dict(
               temperature=1.0,
               top_p=1.0,
               chat_template_kwargs={"thinking":True}
            ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```

**Execution command**:

Use the same dataset, generation parameters, and post-processing configuration as the selected accuracy baseline, and run the evaluation only on the quantized service:

```bash
# Quantized service `${SAVE_PATH}`
ais_bench \
    --models vllm_api_general_chat \
    --datasets gpqa_gen \
    --work-dir ${QUANT_WORK_DIR} \
    --dump-eval-details
```

**Output and Record**:

- Quantized result directory: `${QUANT_WORK_DIR}` (for example, `AISBench outputs/{timestamp}/`).
- Must be retained: the quantization evaluation log, the GPQA score summary, and the complete AISBench artifacts, together with a record of the comparison between the quantized result and the selected accuracy baseline and the relative drop.

**Accuracy conclusion**: the measured GPQA score of the quantized weights is **89.9**, which meets the accuracy acceptance requirement when compared with the selected accuracy baseline.

**Note**: evaluation data may fluctuate. If a single evaluation result does not meet expectations, record the results of multiple evaluations and their average.

**Output**:

- The vLLM Ascend quantized inference service runs normally and the curl request returns a normal response; the AISBench quantized result and the record of its relative drop calculation against the selected accuracy baseline.

**Next step**: finish when the quantized result is not lower than the selected accuracy baseline; finish when the quantized result is lower than the selected accuracy baseline and the relative drop does not exceed 1%; run Step 8, rollback tuning, only when the relative drop exceeds 1%.

---

### Step 8: Rollback Tuning (Optional)

**Objective**: when the quantized result drops by more than 1% relative to the selected accuracy baseline, locate the sensitive layers, adjust the configuration, and re-run quantization and evaluation.

**Inputs**: the original floating-point weights `${MODEL_PATH}`, the quantization configuration `${CONFIG_PATH}`, the selected accuracy baseline, the quantized result, and the evaluation log. The underperforming quantized weights `${SAVE_PATH}` are used only for comparison and are not used as tuning input. The accuracy baseline and the quantized result must use identical evaluation conditions; a score published in the paper or on the model card may be used as the baseline, but the source must be noted and the evaluation conditions must be consistent.

**Operations**:

1. Use the [sensitive linear layer analysis tool](../user_guide/usage_sensitive_linear_analysis.md) to analyze the original floating-point weights and locate the quantization-sensitive layers. For the complete tuning workflow, see the *[Quantization Accuracy Tuning Guide](../user_guide/process_quantization_precision_tuning.md)*:

   ```bash
   msmodelslim analyze linear \
       --model_type DeepSeek-V4-Pro \
       --model_path ${MODEL_PATH} \
       --metrics kurtosis \
       --top_k 15
   ```

2. Adjust the [quantization configuration](../../../lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml) based on the sensitive layer results. You can choose to:

   - **Fall back to floating point**: exclude the sensitive layers in `exclude` of `linear_quant` so that they keep the floating-point path.
   - **Locally increase the bit width**: change only the sensitive layers from W4A8 to W8A8 and keep the remaining layers at W4A8.

3. Use the original floating-point weights `${MODEL_PATH}` and the adjusted configuration to run Step 5 again, and repeat Steps 6 and 7 to check the artifacts and the evaluation accuracy; if the accuracy still does not meet the target, you can further widen the fallback scope or adopt a more conservative quantization strategy.

**Output**: the regenerated quantized weights, the sensitive layer analysis results, and the accuracy comparison before and after tuning.

**Record**: the list of sensitive layers and fallbacks, the selected accuracy baseline, the quantized result, and the relative drop.

**Next step**: end.

---

## 5. Results and Lessons Learned

### 5.1 Summary of Key Results

| Step | Key Operation | Metric | Change | Note |
| --- | --- | --- | --- | --- |
| Steps 1-2 | Confirm the model structure and design the quantization scheme | Structural characteristics + mixed quantization scheme | - | Routed experts W4A8; shared experts and attention layers W8A8 |
| Step 3 | Develop the adapter, register the entry point, and write the quantization YAML | Adapter + YAML deliverables | - | `--model_type DeepSeek-V4-Pro` hits the adapter; the model already has DP+EP multi-device adaptation, and the YAML is specified through `--config` during quantization |
| Step 5 | Run `msmodelslim quant` W4A8 multi-device quantization | Quantized weights generated | Original floating-point weights -> W4A8 quantization | Uses `--device npu --device_id 0 1 2 3 4 5 6 7`; quantization strategy: QuaRot + Flex AWQ SSZ + mixed quantization |
| Step 6 | Check the quantization artifacts | Artifact completeness | - | The model files are complete and consistent with the DeepSeek-V4-Pro W4A8 scheme |
| Step 7 | AISBench GPQA quantization evaluation | Quantized result and relative drop | Quantized weights compared with the selected accuracy baseline | If the quantized result is not lower than the selected accuracy baseline, the case passes directly; if the quantized result is lower than the selected accuracy baseline and the relative drop does not exceed 1%, it passes; if the quantized result is lower than the selected accuracy baseline and the relative drop exceeds 1%, it fails |
| Step 8 | Sensitive layer analysis + quantization fallback / local bit-width increase (optional) | Accuracy re-test meets the target | Re-quantized output using the original floating-point weights | Run only when the quantized result is lower than the selected accuracy baseline and the relative drop exceeds 1% |

### 5.2 Lessons Learned

1. **The three elements of new-model integration**: confirm the model structure -> design the quantization scheme -> complete the development of the adapter and the quantization YAML. The model structural characteristics (MoE routed experts/shared experts, MLA low-rank attention, MTP layers, and so on) directly determine the implementation of the subgraph structure and the rotation mapping, and they are the input to the adapter development. Applicable boundary: the general workflow for integrating a new model into msModelSlim.

2. **The adapter must implement the interfaces required by the quantization pipeline**: `DeepSeekV4ModelAdapter` must inherit interfaces such as `ModelSlimPipelineInterfaceV1`, `FlexSmoothQuantInterface`, `QuaRotInterface`, and `AscendV1SaveInterface` at the same time. Whichever processors the quantization scheme uses, the adapter must provide the corresponding subgraph/mapping information; otherwise the corresponding processor cannot execute. Applicable boundary: model adapters supported by the msModelSlim quantization pipeline (modelslim_v1).

3. **Adapter registration relies on the entry point**: the model adapter is registered through `config/config.ini`, and the entry point is generated automatically by `setup.py` after the msmodelslim tool is installed; after adding or modifying an adapter, the msmodelslim tool must be reinstalled for the change to take effect. Applicable boundary: the source installation method.

4. **The quantization command runs in a work environment whose installation has been verified**: before quantization, confirm that `msmodelslim --help` runs successfully, and use explicit model and save paths. Applicable boundary: the source installation method.

5. **DeepSeek-V4-Pro already has DP+EP multi-device quantization adaptation**: `model.py` sets `USE_DP_MODE=True`, and `model_adapter.py` handles distributed synchronization and EP local routed-expert processing; the QuaRot, FlexAWQSSZ, FlexSmoothQuant, and LinearQuant processors used by the current configuration all support DP. Multi-device quantization can therefore be started directly with `--device npu --device_id 0 1 2 3 4 5 6 7`, and the log can be used to confirm that each rank has been initialized and that there is no fallback to a single device. Applicable boundary: the DeepSeek-V4-Pro W4A8 quantization stage.

6. **W4A8 mixed quantization strategy**: the W4A8 quantization of DeepSeek-V4-Pro uses a mixed strategy — routed experts use W4A8 dynamic quantization (the `ssz` method), while shared experts and attention layers use W8A8 dynamic quantization, which effectively reduces the model size and the inference memory footprint while maintaining accuracy. Applicable boundary: the DeepSeek-V4-Pro W4A8 quantization scheme (Atlas 800I A3 products).

7. **Quantization artifacts must pass both format and loading acceptance**: check `quant_model_description.json`, the weight files or shards and index, `config.json`, and auxiliary files such as the tokenizer against the [AscendV1 Export Format](../knowledge_base/quantization_format/ascendv1/term_ascendv1.md#export-artifacts); confirm that the description JSON is parseable and that the configuration identifier and model type are consistent; and complete at least one loading and inference smoke test through the target inference framework before starting the formal accuracy evaluation. Applicable boundary: vLLM Ascend 0.22.1rc1 and Atlas 800I A3 products.

8. **Accuracy tuning is based on the selected accuracy baseline**: only when the quantized result is lower than the selected accuracy baseline under identical conditions and the relative drop exceeds 1% should you perform a fallback or a local bit-width increase through sensitive layer analysis; the input to the analysis and the re-quantization is always the original floating-point weights `${MODEL_PATH}`, and `${SAVE_PATH}` serves only as the quantization output and result comparison directory and must not be used as tuning input. A score published in the paper or on the model card may be used as the baseline, but the source must be noted and the evaluation conditions must be consistent. Applicable boundary: the DeepSeek-V4-Pro W4A8 quantization scheme.

## 6. Exception Handling

- **The quantization command fails (OOM)**: confirm that all 8 NPUs are available and are not occupied by other tasks, and that the command uses `--device npu --device_id 0 1 2 3 4 5 6 7`.
- **Transformers fails to load the model**: confirm that `transformers==4.48.2` is installed and that the model weight path is correct; the original weights may be FP8 and must be dequantized automatically by the adapter.
- **The vLLM service fails to start**: check the CANN environment variables (`env | grep ASCEND`) and the NPU status (`npu-smi info`), and confirm that the quantized service port is not occupied and that `--tensor-parallel-size 16` and `--data-parallel-size 2` are consistent with the deployment configuration of two nodes with 16 devices each.

## 7. Appendix

- quantization configuration file: [lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml](../../../lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml)
- calibration dataset: [lab_calib/mix_calib.jsonl](../../../lab_calib/mix_calib.jsonl)
- Model adapter code: [msmodelslim/model/deepseek_v4/](../../../msmodelslim/model/deepseek_v4/) ([model_adapter.py](../../../msmodelslim/model/deepseek_v4/model_adapter.py), [model.py](../../../msmodelslim/model/deepseek_v4/model.py), [loader.py](../../../msmodelslim/model/deepseek_v4/loader.py))
- Adapter registration configuration: [config/config.ini](../../../config/config.ini)
- Quantized weight format and delivery check: [AscendV1 Usage Guide](../knowledge_base/quantization_format/ascendv1/ascendv1_usage.md)
- Official vLLM Ascend deployment documentation: [DeepSeek-V4-Pro tutorial](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Pro.html)
- AISBench evaluation tool and installation guide: [repository README](https://github.com/AISBench/benchmark/blob/master/README.md)
- GPQA accuracy reference: the score **89.1** published in the paper or on the model card is used as the public baseline score (see the [ModelScope model card](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro)).
