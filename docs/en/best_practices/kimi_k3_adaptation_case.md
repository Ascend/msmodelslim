# Kimi-K3 New Model W4A8 Quantization Case

## 1. Case Background

**Objective**: adapt and integrate the newly released Kimi-K3 (2026-07) into msModelSlim, and complete W4A8 quantization with the accuracy target met.

**Covered steps**: model adaptation → quantization scheme design → weight quantization → accuracy evaluation.

**Related process**: *[VLM Model Integration Quantization Workflow Guide](../knowledge_base/ptq/vlm/integration_guide_vision_transformer_quantization.md)*, *[One-Click Quantization Usage Guide](../user_guide/usage_one_click_quantization.md)*

## 2. Environment and Versions

| Item | Version or Configuration |
| --- | --- |
| Product form | Ascend A3 products (restricted; this case is verified on Ascend A3 products) |
| vLLM Ascend | [vllm-ascend:v0.23.0rc1-a3](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=v0.23.0rc1-a3) |
| CANN | 9.0.1 (included in the image) |
| PyTorch | 2.10.0 (included in the image) |
| TorchNPU | 2.10.0.post2 (included in the image) |
| transformers | 4.57.6 (downgrade required for quantization; the image defaults to 5.5.4) |
| compressed-tensors | 0.13.0 (downgrade required for quantization; the image defaults to 0.17.0) |
| fla-core | 0.5.1 |
| Evaluation tool | [AISBench](https://github.com/AISBench/benchmark) |

**Prerequisites for this case**:

- The [Kimi-K3 weights](https://huggingface.co/moonshotai/Kimi-K3) have been downloaded.
- The official vLLM Ascend image container has been started with reference to [Using Docker](https://docs.vllm.ai/projects/ascend/en/v0.23.0/installation.html#set-up-using-docker), with the NPU devices and the model weight directory mounted.
- The msModelSlim source code has been obtained as described in [Environment Preparation](../install_guide/install_guide.md#231-environment-preparation) (this case integrates a new model from source).

## 3. Inputs and Deliverables

| Type | Name | Source or Location | Format or Constraint | Acceptance Method |
| --- | --- | --- | --- | --- |
| Input | Floating-point weights | [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | The weight files are complete | The download is complete and the hash values match |
| Deliverable | Model adapter code | [`msmodelslim/model/kimi_k3/`](../../../msmodelslim/model/kimi_k3/) | The adapter must implement the interfaces required by the quantization pipeline | `--model_type Kimi-K3` on the command line hits the adapter |
| Deliverable | Quantization best practice | [`lab_practice/kimi_k3/kimi_k3_w4a8.yaml`](../../../lab_practice/kimi_k3/kimi_k3_w4a8.yaml) | Complies with the *[modelslim_v1 Configuration Reference](../api_reference/config/quant/modelslim_v1.md)* | The quantization configuration file is specified with the `--config` command-line parameter |
| Deliverable | Quantized weight directory | The save location is specified with the `--save_path` command-line parameter | Complies with the [AscendV1 format](../knowledge_base/quantization_format/ascendv1/ascendv1_usage.md) | The quantized weights meet the accuracy test target |

## 4. Procedure

### Step 1: Model Adaptation

**Objective**: complete the Kimi-K3 adaptation and integrate it into msModelSlim.

**Inputs**: the floating-point weights.

**Operations**:

1. Review the model structure configuration:

   ```bash
   cat ${MODEL_PATH}/config.json
   ```

   In `config.json`, confirm the following structural parameters in particular:

   | Parameter | Value | Description |
   | --- | --- | --- |
   | `auto_map.AutoModelForCausalLM` | `modeling_kimi_k3.KimiK3ForConditionalGeneration` | Indicates the actual class name that defines the model |
   | `format` | `mxfp4-pack-quantized` | The native weights contain MXFP4 quantized weights, so weight dequantization must be completed offline or online |
   | `mm_projector_type` | `patchmergerv2` | The fusion module class for visual features, which affects the adaptation of the rotation quantization algorithm |

2. Following the *[VLM Model Integration Quantization Workflow Guide](../knowledge_base/ptq/vlm/integration_guide_vision_transformer_quantization.md)*, develop the model adapter ([`msmodelslim/model/kimi_k3/model_adapter.py`](../../../msmodelslim/model/kimi_k3/model_adapter.py)). For Kimi-K3, even with layer-by-layer loading, the 64 GB of memory on a single device still overflows, so the expert parallelism adaptation must be completed with reference to the *[Expert Parallelism Usage Guide](../knowledge_base/parallel/expert_parallelism/expert_parallelism_guide.md)* ([`ep_patches.py`](../../../msmodelslim/model/kimi_k3/ep_patches.py)).

3. Register the model adapter ([`config/config.ini`](../../../config/config.ini)):

   ```ini
   [ModelAdapter]
   kimi_k3 = Kimi-K3

   [ModelAdapterEntryPoints]
   kimi_k3 = msmodelslim.model.kimi_k3.loader:KimiK3AdapterLoader

   [ModelAdapterDependencies]
   kimi_k3 = {"transformers": "==4.57.6", "compressed-tensors": "==0.13.0"}
   ```

**Output**:

- The code related to the model adaptation.

---

### Step 2: Quantization Scheme Design

**Objective**: design a W4A8 quantization scheme based on the model structure and the characteristics of the original factory weights.

**Inputs**: the model structure characteristics confirmed in Step 1.

**Operations**:

1. Determine the overall quantization strategy.

   Because the floating-point weights of Kimi-K3 already natively contain quantization, the quantization scheme design generally follows the principle of keeping the bit width consistent with the native weights. The native weights of the routed experts (`experts.*.w1/w2/w3`) in the floating-point weights are MXFP4, so W4A8 dynamic quantization is applied to the routed experts.

2. Locally adjust the quantization strategy.

   The shared experts (`shared_experts.{up,down,gate}_proj`) are natively BF16. Experience shows that applying W8A8 dynamic quantization to the shared experts keeps the accuracy under control, so W8A8 dynamic quantization is added for the shared experts to improve the performance after quantization.

   `routed_expert_up_proj` / `routed_expert_down_proj` are also natively BF16, but because the shapes of the weights of these two layers are large, W8A8 dynamic quantization is also added for these two layers to gain memory savings.

**Output**:

- The Kimi-K3 W4A8 quantization practice configuration.

---

### Step 3: Weight Quantization

**Objective**: once the environment is ready, use `msmodelslim quant` to generate the W4A8 quantized weights in one click.

**Inputs**: the floating-point weight directory `${MODEL_PATH}`; the quantization configuration `${YAML_PATH}` output by [Step 2](#step-2-quantization-scheme-design).

**Operations**:

1. Reinstall msmodelslim:

   In the source code path where the model adaptation was completed in [Step 1](#step-1-model-adaptation), run `bash install.sh` to reinstall msmodelslim so that the model adaptation code takes effect.

2. Run the one-click quantization:

   ```bash
   # Return to the parent directory; the msmodelslim command must be run outside the source directory
   cd ..
   # Replace ${MODEL_PATH} with the model path; ${SAVE_PATH} with the quantized weight save path; ${YAML_PATH} with the quantization configuration file path
   msmodelslim quant \
       --model_path ${MODEL_PATH} \
       --save_path ${SAVE_PATH} \
       --device npu \
       --device_id 0 1 2 3 4 5 6 7 \
       --model_type Kimi-K3 \
       --config ${YAML_PATH} \
       --trust_remote_code true
   ```

**Output**:

- The Kimi-K3 W4A8 quantized weights.

---

### Step 4: Accuracy Evaluation

**Objective**: deploy the quantized weights as a service and run the accuracy evaluation.

**Inputs**: the quantized weight directory `${SAVE_PATH}`.

**Operations**:

1. Before deployment, restore the Transformers version preset in the image (restore it to the container's initial version after quantization is complete):

   ```bash
   pip install transformers==5.5.4
   ```

2. Deploy the quantized model with reference to the [vLLM Ascend documentation](https://docs.vllm.ai/projects/ascend/en/v0.23.0/tutorials/models/Kimi-K3.html).

3. After the service starts, use curl to verify that the quantized model answers normally.

   ```bash
   # Replace <NODE0_LOCAL_IP> and <SERVICE_PORT> with the device IP and the service port respectively
   curl http://<NODE0_LOCAL_IP>:<SERVICE_PORT>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi-k3",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "The future of AI is"
            }]
        }],
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
   ```

4. Run the dataset accuracy evaluation with reference to the *[AISBench tutorial](https://github.com/AISBench/benchmark/blob/master/README.md)*.

   - Accuracy results:

     | Dataset | Baseline | Quantized Weights | Difference | Conclusion |
     | --- | --- | --- | --- | --- |
     | GPQA | 92.9 | 94.4 | +1.5 | Accuracy target met |
     | OCRBench | 89.1 | 91.7 | +2.6 | Accuracy target met |

   > [!NOTE]
   >
   > Evaluation data may fluctuate. If a single evaluation result does not meet the target, it is recommended to use the average of multiple evaluation results as the basis. If the results of multiple evaluations still do not meet the target, refer to the *[Quantization Accuracy Tuning Guide](../user_guide/process_quantization_precision_tuning.md)* for accuracy tuning.

**Output**:

- The vLLM Ascend inference service runs normally and the curl request returns a normal response; the AISBench accuracy evaluation result meets the accuracy target.

## 5. Results and Lessons Learned

### 5.1 Summary of Key Results

| Step | Key Operation | Metric | Change | Note |
| --- | --- | --- | --- | --- |
| Step 1 | Develop and register the model adapter | `--model_type Kimi-K3` hits the adapter | New model adapter | The code takes effect only after a reinstall is performed once development is complete |
| Step 2 | Design the mixed quantization scheme | [kimi_k3_w4a8.yaml](../../../lab_practice/kimi_k3/kimi_k3_w4a8.yaml) | New quantization practice configuration | The quantization practice configuration complies with the *[modelslim_v1 Configuration Reference](../api_reference/config/quant/modelslim_v1.md)* |
| Step 3 | Run the quantization command | W4A8 quantized weights | Quantized weights exported | The exported artifacts comply with the [AscendV1 format](../knowledge_base/quantization_format/ascendv1/ascendv1_usage.md) |
| Step 4 | vLLM Ascend deployment + AISBench accuracy evaluation | [GPQA](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/gpqa/README.md) / [OCRBench](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/ocrbench_v2/README.md) | Accuracy verified | Accuracy target met |

### 5.2 Lessons Learned

1. **Very large models require a combination of layer-by-layer and multi-device quantization**: as model size grows, when the memory of a single device is insufficient to load a single layer of weights, both layer-by-layer and multi-device quantization must be supported at the same time.

2. **EP can be implemented in several ways**: the EP of a MoE model can be implemented statically by modifying the model modeling source code, as in the [DeepSeek-V4 EP implementation](../../../msmodelslim/model/deepseek_v4/model.py); or dynamically by modifying the model modeling through monkey-patching, as in the [Kimi-K3 EP implementation](../../../msmodelslim/model/kimi_k3/ep_patches.py).

## 6. Exception Handling

- **The adapter does not take effect (`--model_type` is not recognized)**: confirm that `ModelAdapterEntryPoints` has been registered in [`config/config.ini`](../../../config/config.ini) as described in [Step 1](#step-1-model-adaptation), and re-run `bash install.sh` in the source code directory for the change to take effect.

## 7. Appendix

- Related PR: [[feature] support kimi_k3 model adapter](https://gitcode.com/Ascend/msmodelslim/pull/789)
