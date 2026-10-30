# DeepSeek-V4-Flash W8A8 One-Click Quantization Case

## 1. Case Background

**Objective**: perform W8A8 one-click quantization on the DeepSeek-V4-Flash model, deploy it to the vLLM Ascend inference engine for dataset-based accuracy evaluation, and complete the full closed loop from quantization to deployment.

**Covered steps**: environment preparation and tool installation → confirm the model and the quantization scheme → run the one-click quantization command → quantized weight acceptance → inference deployment and evaluation.

**Accuracy acceptance criteria**: use the selected accuracy baseline (this case uses the baseline published in the paper or on the model card) as the comparison reference, and obtain the quantized result score with AISBench. If the quantized result score is not lower than the selected accuracy baseline score, the case passes directly; if the quantized result score is lower than the selected accuracy baseline score, the relative drop must not exceed 1%; otherwise it fails.

**Related process**: *[One-Click Quantization Usage Guide](../user_guide/usage_one_click_quantization.md)*, *[Weight Quantization Usage Guide](../user_guide/usage_weight_quantization.md)*, *[msModelSlim Installation Guide](../install_guide/install_guide.md)*

## 2. Environment and Versions

| Item | Version or Configuration |
| --- | --- |
| Product form | Atlas 800I A3 products (restricted; this case is verified on the single-node 16-device environment of this server) |
| Runtime image | [quay.io/ascend/vllm-ascend:v0.22.1rc1-a3](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=v0.22.1rc1-a3) |
| CANN | CANN-9.0.0 (included in the image) |
| PyTorch | 2.10.0 (included in the image) |
| TorchNPU | 2.10.0 (included in the image) |
| vLLM Ascend | 0.22.1rc1 (included in the image) |
| Other dependencies | `transformers==4.48.2` (downgrade required for quantization; the image defaults to 5.5.4) |

**Prerequisites for this case**:

- The DeepSeek-V4-Flash floating-point weights have been obtained (download link: [ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash)).
- The target model has been confirmed to be in the support matrix, and `--model_type DeepSeek-V4-Flash` and `--quant_type w8a8` have been verified.
- The official vLLM Ascend image container has been started, with the NPU devices and the model weight directory mounted.

> [!NOTE]
>
> The floating-point weights used in this case are DeepSeek-V4-Flash preview (not 0731). The official release DeepSeek-V4-Flash-0731 has the same structure as DSpark; use `--model_type DeepSeek-V4-Flash-0731 --quant_type w4a4c8`. For details, see [DeepSeek quantization notes](../../../example/DeepSeek/README.md).

## 3. Inputs and Deliverables

| Type | Name | Source or Location | Format or Constraint | Acceptance Method |
| --- | --- | --- | --- | --- |
| Input | Floating-point model directory | `${MODEL_PATH}` (downloaded by the user) | Contains `config.json`, `tokenizer.json`, and model weight files (safetensors); can be loaded by Transformers 4.48.2 | `transformers.AutoModel.from_pretrained` loads it without errors |
| Input | Calibration dataset | [Mixed calibration dataset (mix_calib.jsonl)](../../../lab_calib/mix_calib.jsonl) | JSONL format; each line contains the `inputs_pretokenized` field | The calibration set can be read normally by the quantization process |
| Deliverable | Quantized weight directory | `${SAVE_PATH}` (output directory of the quantization workflow) | Contains `quant_model_description.json`, weight files (safetensors format), the index, `config.json`, and so on | Check the file list and the safetensors structure, confirm that the model structure matches the floating-point model, and confirm that it can be loaded by vLLM Ascend `--quantization ascend` |
| Deliverable | Accuracy test results | The AISBench output directory (`${EVAL_WORK_DIR}`) and the `${CASE_ROOT}` archive directory | Contains the evaluation configuration, service logs, curl responses, the selected accuracy baseline score, the quantized result score, and the acceptance calculation records | If the quantized result is not lower than the selected accuracy baseline, the case passes directly; if it is lower, the relative drop must not exceed 1% |

## 4. Procedure

### Step 1: Environment Preparation and Tool Installation

**Objective**: confirm that the container environment meets the quantization requirements, and install the msModelSlim tool and its dependencies.

**Inputs**: the official vLLM Ascend image container has been started.

**Operations**:

1. If the container has not been created yet, refer to the [official vLLM Ascend guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html) to pull the image and create the container; the NPU devices and the model weight directory must be mounted when it is created. The image used in this case is `m.daocloud.io/quay.io/ascend/vllm-ascend:v0.22.1rc1-a3`.

2. Check the NPU status and the CANN environment inside the container:

   ```bash
   pip list 2>/dev/null | grep -iE "torch|npu"
   env | grep ASCEND
   npu-smi info
   ```

3. Record the current Transformers version so that it can be restored after quantization:

   ```bash
   pip list 2>/dev/null | grep transformers
   # The current version is 5.5.4
   ```

4. Download the msModelSlim source code and install it:

   ```bash
   git clone https://gitcode.com/Ascend/msmodelslim
   cd msmodelslim
   bash install.sh
   cd ..  # Leave the source directory; the msmodelslim command must be run outside it
   msmodelslim --help
   ```

5. Install the Transformers version adapted to DeepSeek-V4-Flash:

   ```bash
   pip install transformers==4.48.2
   ```

   > [!NOTE]
   >
   > msModelSlim does not require a specific Transformers version, but `transformers==4.48.2` is recommended for the DeepSeek-V4-Flash model. The downgrade is only for the quantization stage; after quantization you can run `pip install transformers==5.5.4` to restore it.

**Output**:

- `msmodelslim --help` runs successfully.
- The Transformers version has been switched to `4.48.2`.

**Record**: run `pip list | grep transformers` to confirm the version.

**Next step**: confirm the model and the quantization scheme.

---

### Step 2: Confirm the Model and the Quantization Scheme

**Objective**: confirm that `model_type` and `quant_type` match the target model.

**Inputs**: the floating-point weights have been downloaded to `${MODEL_PATH}`.

**Operations**:

1. View the model quantization configuration:

   ```bash
   cat msmodelslim/lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml
   ```

   This configuration uses a QuaRot rotation pre-processing + Flex Smooth Quant outlier suppression + W8A8 dynamic quantization (per-token/per-channel) strategy.

2. Confirm the floating-point weight path:

   ```bash
   ls ${MODEL_PATH}/
   # Should contain files such as config.json, tokenizer.json, *.safetensors
   ```

**Output**:

- `--model_type DeepSeek-V4-Flash` and `--quant_type w8a8` have been confirmed.
- The floating-point weights are ready.

**Record**: the quantization configuration YAML file path is `lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml`.

**Next step**: run the one-click quantization command.

---

### Step 3: Run the One-Click Quantization Command

**Objective**: use the `msmodelslim quant` command to generate the quantized weights.

**Inputs**: the floating-point weight directory `${MODEL_PATH}`.

**Operations**:

```bash
msmodelslim quant \
    --model_path ${MODEL_PATH} \
    --save_path ${SAVE_PATH} \
    --model_type DeepSeek-V4-Flash \
    --quant_type w8a8 \
    --device npu \
    --device_id 0 \
    --trust_remote_code true
```

Parameter description:

- `--model_path`: path to the floating-point model weight directory.
- `--save_path`: path to the directory where the quantized weights are saved.
- `--model_type`: model type, fixed to `DeepSeek-V4-Flash`.
- `--quant_type`: quantization strategy type; `w8a8` means that both weights and activations are INT8.
- `--device npu --device_id 0`: `--device` specifies the device type as NPU, and `--device_id` specifies the device index.
- `--trust_remote_code`: trust custom model code (ensure that the code source is reliable).

**Output**:

- the quantized weight directory `${SAVE_PATH}`, containing `quant_model_description.json`, weight files (safetensors format), `config.json`, and so on.

**Record**: the quantization log is printed to the terminal and can be redirected to a file for later reference.

**Next step**: quantized weight acceptance.

---

### Step 4: Quantized Weight Acceptance

**Objective**: confirm that the quantized weight artifacts are complete.

**Inputs**: the quantized weight output directory `${SAVE_PATH}`.

**Operations**:

1. Check the output directory structure:

   ```bash
   ls ${SAVE_PATH}/
   # Should contain quant_model_description.json, config.json, *.safetensors, and so on
   ```

2. Confirm that `quant_model_description.json` exists and its content is complete:

   ```bash
   cat ${SAVE_PATH}/quant_model_description.json | head -20
   ```

**Output**:

- the verified quantized weight directory `${SAVE_PATH}`.

**Pass conditions**: the quantization output directory complies with the AscendV1 export format conventions (`quant_model_description.json` + weight safetensors); for details, see [AscendV1 Export Artifacts](../knowledge_base/quantization_format/ascendv1/term_ascendv1.md#export-artifacts). The quantization log contains no `ERROR`.

**Record**: the structure of the quantized weight directory and a summary of `quant_model_description.json`.

**Next step**: inference deployment and evaluation.

---

### Step 5: Inference Deployment and Evaluation

**Objective**: deploy the quantized weights to the vLLM Ascend inference engine for smoke testing and accuracy verification.

**Inputs**: the quantized weight directory `${SAVE_PATH}`.

**Operations**:

1. Restore the Transformers version (restore it to the initial version of the container after quantization):

   ```bash
   pip install transformers==5.5.4
   ```

2. Configure the environment variables and start the vLLM Ascend inference service (refer to the [official vLLM Ascend documentation](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html)):

   ```bash
   export OMP_PROC_BIND=false
   export OMP_NUM_THREADS=10
   export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
   export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
   export HCCL_BUFFSIZE=1024
   export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
   export TASK_QUEUE_ENABLE=1
   export HCCL_OP_EXPANSION_MODE="AIV"

   vllm serve ${SAVE_PATH} \
       --max-model-len 133120 \
       --max-num-batched-tokens 8192 \
       --served-model-name dsv4-flash \
       --gpu-memory-utilization 0.9 \
       --max-num-seqs 32 \
       --data-parallel-size 1 \
       --tensor-parallel-size 8 \
       --enable-expert-parallel \
       --tokenizer-mode deepseek_v4 \
       --tool-call-parser deepseek_v4 \
       --enable-auto-tool-choice \
       --reasoning-parser deepseek_v4 \
       --safetensors-load-strategy 'prefetch' \
       --no-enable-prefix-caching \
       --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
       --quantization ascend \
       --port 8900 \
       --block-size 128 \
       --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
       --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
       --additional-config '
       {"ascend_compilation_config":{
           "enable_npugraph_ex":true,
           "enable_static_kernel":false
           },
       "enable_cpu_binding": true,
       "enable_dsa_cp": true,
       "multistream_overlap_shared_expert":true}'
   ```

3. After the service starts, use curl to verify that the model answers normally:

   ```bash
   curl http://localhost:8900/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "dsv4-flash",
           "messages": [{"role": "user", "content": "Who are you?"}],
           "max_tokens": 256,
           "temperature": 0
       }'
   ```

4. Use [AISBench](https://github.com/AISBench/benchmark) to evaluate on the GPQA dataset. For installation and usage, see the [AISBench repository README](https://github.com/AISBench/benchmark/blob/master/README.md).

   When evaluating the quantized model, use the AISBench dataset, service parameters, generation parameters, and post-processing pipeline consistent with the baseline in the paper or on the model card, record the selected accuracy baseline score, and use the `${SAVE_PATH}` quantized model to obtain the quantized result score. This case does not require an additional test of the floating-point model.

   Create the model configuration file `vllm_api_general_chat.py`:

   ```python
   from ais_bench.benchmark.models import VLLMCustomAPIChat
   from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

   models = [
       dict(
           attr="service",
           type=VLLMCustomAPIChat,
           abbr="vllm-api-general-chat",
           path="",
           model="dsv4-flash",
           stream=False,
           request_rate=0,
           use_timestamp=False,
           retry=2,
           api_key="",
           host_ip="localhost",
           host_port=8900,
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

   Run the evaluation command (for reference):

   ```bash
   ais_bench \
       --models vllm_api_general_chat \
       --datasets gpqa_gen \
       --work-dir "${EVAL_WORK_DIR}" \
       --dump-eval-details
   ```

   `${EVAL_WORK_DIR}` is the output directory of this evaluation; also save the `vllm_api_general_chat.py` actually used, the AISBench version or commit, the dataset configuration, and the generation parameters.

   The public GPQA baseline score of DeepSeek-V4-Flash is **87.4** (see the [ModelScope model card](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash/summary)), which is used as the selected accuracy baseline.

   **Accuracy conclusion**: the measured GPQA score of the quantized weights is **87.88**, which meets the accuracy acceptance requirement when compared with the selected accuracy baseline. The evaluation configuration and the acceptance calculation records are those in the result files under `${EVAL_WORK_DIR}`.

   **Note**: evaluation data may fluctuate. If a single evaluation result does not meet expectations, record the results of multiple evaluations and their average.

**Output**:

- the vLLM Ascend inference service runs normally, and the curl request returns a normal response.
- A GPQA comparison record of the public baseline score and the quantized result score.

**Record**: the curl request response `${CURL_RESULT}`; the vLLM service log `${VLLM_LOG}`; the AISBench evaluation configuration and result directory `${EVAL_WORK_DIR}`; the selected accuracy baseline, the quantized result, and the acceptance calculation records.

**Next step**: end.

## 5. Results and Lessons Learned

### 5.1 Summary of Key Results

| Step | Key Operation | Metric | Change | Note |
| --- | --- | --- | --- | --- |
| Step 3: Run the One-Click Quantization Command | Run `msmodelslim quant` W8A8 quantization | Quantized weights generated | Floating-point -> W8A8 quantization | Quantization log `${QUANT_LOG}`; quantization strategy: QuaRot + Flex Smooth Quant + W8A8 dynamic quantization |
| Step 4: Quantized Weight Acceptance | Check the quantization artifacts | Structural completeness of the quantization artifacts | - | Check the description file, the configuration file, all weight shards, the structure comparison, and the inference loading result |
| Step 5: Inference Deployment and Evaluation | Start the inference service with vLLM Ascend + AISBench GPQA evaluation | Public baseline score: 87.4; measured quantized score: 87.88 | The quantized result is higher than the public baseline | The measured score is not lower than the selected accuracy baseline, so the accuracy acceptance requirement is met |

### 5.2 Lessons Learned

1. **The quantization command must run in an environment whose installation has been verified**: before quantization, confirm that `msmodelslim --help` runs successfully, and use explicit model and save paths. Applicable boundary: the source installation method (`git clone` + `bash install.sh`).

2. **The Transformers version must be compatible with the model**: `transformers==4.48.2` is recommended for DeepSeek-V4-Flash; a version mismatch may cause model loading to fail. msModelSlim does not require a specific version, but you are advised to install the version adapted to the model during quantization. Applicable boundary: this version is recommended only for the DeepSeek-V4-Flash quantization stage (the vLLM Ascend 0.22.1rc1 image).

3. **Quantized weights must pass both format and loading acceptance**: verify `quant_model_description.json`, the configuration file, all weight shards and the index, and the necessary auxiliary files against the [AscendV1 Export Format](../knowledge_base/quantization_format/ascendv1/term_ascendv1.md#export-artifacts), and complete loading and a smoke request through vLLM Ascend before starting the formal accuracy evaluation. Applicable boundary: vLLM Ascend 0.22.1rc1 and Atlas 800I A3 products.

4. **Accuracy acceptance is based on the selected accuracy baseline**: obtain the quantized result score with an evaluation configuration consistent with the paper or the model card, and compare it with the selected accuracy baseline score; if the quantized result score is not lower than the selected accuracy baseline score, the case passes directly, and if it is lower, it passes only when the relative drop does not exceed 1%. This case does not require an additional test of the floating-point model. Applicable boundary: the GPQA evaluation of this case.

## 6. Exception Handling

- **The quantization command fails (OOM)**: DeepSeek-V4-Flash W8A8 quantization can be completed on a single device, so OOM rarely occurs; if memory is insufficient, confirm that the NPU specified by `--device npu --device_id 0` is not occupied by another task.
- **Transformers fails to load the model**: confirm that `transformers==4.48.2` is installed and that the model weight path is correct. This case requires the preview weights; if you have DeepSeek-V4-Flash-0731, use `--model_type DeepSeek-V4-Flash-0731 --quant_type w4a4c8` instead.
- **The vLLM service fails to start**: in this case the actual troubleshooting actions were performed in the following order: check the CANN environment variables (`env | grep ASCEND`) and the NPU status (`npu-smi info`), and confirm that `--tensor-parallel-size 8` matches the actual number of devices.

## 7. Appendix

- quantization configuration file: [lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml](../../../lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml)
- GPQA accuracy reference: the score **87.4** published in the paper or on the model card is used as the public baseline score (see the [ModelScope model card](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash/summary)), and the measured quantized score of this case is **87.88**.
- Official vLLM Ascend deployment documentation: [DeepSeek-V4-Flash tutorial](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html)
- Official vLLM Ascend image: [quay.io/ascend/vllm-ascend:v0.22.1rc1-a3](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=v0.22.1rc1-a3)
- AISBench evaluation tool and installation guide: [repository README](https://github.com/AISBench/benchmark/blob/master/README.md)
