# Wan2.2 New Model W4A4F4 Quantization Case

## 1. Case Background

**Objective**: based on the MindIE SD inference stack, perform W4A4F4 quantization verification on the Wan2.2 series dual-expert DiT text-to-video model, and compare the generation quality of the original floating-point weights with that of the quantized weights. Relative to the original floating-point weights, the video quality loss of the quantized model must stay within an acceptable range, with no severe image degradation visible to the naked eye; and on the 1% subset of VBench-1.0-mini, the absolute score drops of both VBench Quality and VBench Semantic must be less than 1 percentage point (pp), that is, `|FP score − quantized score| < 1pp`.

**Covered steps**: model integration and adaptation → quantization scheme design → weight quantization → accuracy evaluation

**Related process**: *[DiT Model Integration Quantization Workflow Guide](../knowledge_base/ptq/dit/integration_guide_diffusion_transformer_quantization.md)*, *[One-Click Quantization Usage Guide](../user_guide/usage_one_click_quantization.md)*

## 2. Model Information

| Item | Content |
| ------------------ | ------------------------------------ |
| Model name | Wan2.2 (Alibaba Tongyi Wanxiang 2.2) |
| model_type         | Wan2.2-T2V-A14B |
| Parameter count | 14B (A14B version) |
| Modality | Multimodal generation (text-to-video) |
| Model structure | Dual-expert DiT (low_noise_model + high_noise_model) + GQA attention + text encoder + VAE |
| Quantization type | w4a4f4 |
| Open-source weight source | Official open-source addresses such as [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) |
| Inference framework and parallelism | MindIE SD; the parallelism strategy of this case: FSDP sharding enabled for DiT/T5, Ulysses Sequence Parallel (SP) enabled on the sequence dimension, CFG separated across two devices, and parallelism enabled for VAE; Expert Parallel is not enabled (the dual experts are scheduled automatically by the framework). Both sides of the comparison use exactly the same parallelism conventions; for the specific values, see Steps 6/7 in the "Procedure" section. |

**Validation scope statement**:

- Covered: W4A4F4 quantization of the dual experts in the DiT backbone network, and comparison of text-to-video generation quality
- Not covered: text encoder quantization, VAE quantization, audio output, long videos exceeding 81 frames or above 720P resolution, and specially style-fine-tuned versions

## 3. Environment and Versions

| Item | Version or Configuration |
| ----------- | -------------------------------------------------- |
| Product form | Ascend 950PR&950DT products (restricted: floating-point inference is limited to this hardware form) |
| Inference repository | The Modelers community Wan2.2 model inference repository. For quantization, the recommended version is `git checkout 38fb8eb13a018cca316678930720eafa446c4387`; for inference, the latest version is recommended. For the repository, see the *[Wan2.2 inference repository](https://modelers.cn/models/MindIE/wan2.2)* |
| CANN        | 9.1.0 |
| PyTorch     | 2.9.0 |
| TorchNPU    | 2.9.0 |
| MindIE SD   | 3.1.0 |
| Evaluation tool | [AISBench](https://github.com/AISBench/benchmark) |
| Other dependencies | The Wan2.2-T2V-A14B model weights and the [VBench-1.0-mini evaluation dataset](https://modelers.cn/datasets/AISBench/VBench-1.0-mini) |

**Prerequisites for this case**:

- The floating-point model can be loaded normally from the official Wan2.2 inference repository, and the generation quality of 720P/81-frame videos meets official expectations.
- The msModelSlim source code has been obtained as described in [Environment Preparation](../install_guide/install_guide.md#231-environment-preparation) (this case integrates a new model from source).

## 4. Inputs and Deliverables

| Type | Name | Source or Storage Location | Format or Constraints | Acceptance Method |
| --- | -------------------- | ------------------------ | -------------------------- | --------------------------------- |
| Input | Wan2.2 floating-point model weights | `${MODEL_PATH}` | Official open-source format, containing the weights of the dual DiT experts, the VAE, the text encoder, the tokenizer, and config.json | Can be loaded normally through the official inference repository, and 720P video generation is free of anomalies |
| Deliverable | Wan2.2 W4A4F4 quantized weights | `${SAVE_PATH}` | msModelSlim quantization export format, containing `quant_model_description.json` and the quantized weights of both experts | Can be loaded normally by MindIE SD and generates videos successfully |
| Deliverable | Quantization best practice | [`lab_practice/wan2_2/wan2_2_w4a4f4_mxfp_t2v.yaml`](../../../lab_practice/wan2_2/wan2_2_w4a4f4_mxfp_t2v.yaml) | Complies with the *[multimodal_sd_modelslim_v1 Configuration Reference](../api_reference/config/quant/multimodal_sd_modelslim_v1.md)* | The quantization configuration file is specified with the `--config` command-line parameter |
| Input | Accuracy evaluation set | `${EVAL_DATA_PATH}` | The standard VBench evaluation set | Both sides of the comparison use exactly the same data and random seed |
| Deliverable | Accuracy report | `./outputs/default/{timestamp}/` | Generated video samples (FP vs. quantized comparison) and a table of objective metric results such as VBench | Meets the preset acceptance threshold, with no severe image quality problems |

## 5. Procedure

### Step 1: Model Integration and Adaptation

**Objective**: complete the adaptation of Wan2.2-T2V-A14B and integrate it into msModelSlim.

**Inputs**: the floating-point weights.

**Operations**:

The Wan2.2 adapter base class `Wan2_2BaseModelAdapter` implements the common logic, and the scenario subclass `Wan2_2T2VModelAdapter` fills in the partition-specific differences.

1. Directory structure

    ```text
    msmodelslim/model/wan2_2/
    ├── base_model_adapter.py    # Base class
    ├── expert_sub_adapter.py    # Expert sub-adapter (composition relationship)
    ├── constants.py             # Constants
    └── t2v/
        ├── model_adapter.py     # T2V scenario subclass (scene_task=t2v-A14B)
        └── loader.py            # Wan2_2T2VAdapterLoader
    ```

2. Adapter class inheritance chain

    ```python
    class Wan2_2BaseModelAdapter(
        BaseModelAdapter,              # Base adapter
        ModelInfoInterface,            # get_model_type
        MultimodalPipelineInterface,   # Multimodal generation pipeline interface
        FA3QuantAdapterInterface,      # FA3 activation quantization
        OnlineQuaRotInterface,         # Online QuaRot
        IterSmoothInterface,           # iterative smoothing
    ):
        scene_task: ClassVar[str] = ""  # Fixed by the subclass

    class Wan2_2T2VModelAdapter(Wan2_2BaseModelAdapter):
        scene_task = "t2v-A14B"
    ```

3. Methods that the subclass must implement

    `MultimodalPipelineInterface` has 5 abstract methods in total, of which `configure_runtime`, `inference_dump_calib_data`, and `prepare_calib_data` are already implemented by the base class. The subclass must implement the following:

    | Member | Type | Description |
    |------|------|------|
    | `scene_task` | ClassVar | For example, `"t2v-A14B"`, in one-to-one correspondence with the `model_type` in config.ini |
    | `get_inference_config_class` | Method | Returns `Wan2_2T2VInferenceConfig` |
    | `quantization_context` | Method | Both experts enter the autocast + no_sync context at the same time |
    | `init_model` | Method | `_load_pipeline()` → `_bind_expert_sub_adapters(experts)`, returning `{"low_noise_model": ..., "high_noise_model": ...}` |
    | `_generate_video` | Method | Calls `wan_t2v.generate` to run inference, used to dump the calibration data |
    | `_build_wan_pipeline` | Method | Creates the WanT2V pipeline and mounts attention_cache |

4. Expert sub-adapter (composition relationship)

    `Wan2_2ExpertSubAdapter` must inherit `BaseModelAdapter`, `OnlineQuaRotInterface`, `FA3QuantAdapterInterface`, and `IterSmoothInterface`, so that `LayerWiseRunner` can schedule quantization per expert:

    ```python
    class Wan2_2ExpertSubAdapter(
        BaseModelAdapter,
        OnlineQuaRotInterface,
        FA3QuantAdapterInterface,
        IterSmoothInterface,
    ):
        def quantization_context(self):
            return self._parent._quantization_context_with_no_sync(self._module)
        def get_online_rotation_configs(self, model=None):
            return self._parent.get_online_rotation_configs(model or self._module)
        def inject_fa3_placeholders(self, root_name, root_module, should_inject):
            return self._parent.inject_fa3_placeholders(root_name, root_module, should_inject)
    ```

5. Register the model adapter and the Loader

    Register the model name in [`config/config.ini`](../../../config/config.ini), and implement [`t2v/loader.py`](../../../msmodelslim/model/wan2_2/t2v/loader.py) to point to the adapter class:

    ```ini
    [ModelAdapter]
    wan2_2_t2v = Wan2.2-T2V-A14B

    [ModelAdapterEntryPoints]
    wan2_2_t2v = msmodelslim.model.wan2_2.t2v.loader:Wan2_2T2VAdapterLoader
    ```

    ```python
    class Wan2_2T2VAdapterLoader(BaseModelAdapterLoader):
        ADAPTER_CLASS_PATH = "msmodelslim.model.wan2_2.t2v.model_adapter:Wan2_2T2VModelAdapter"
    ```

    > [!NOTE]
    >
    > After the model adaptation and registration are complete, run `bash install.sh` in the source code directory to reinstall msmodelslim so that the adapter code takes effect; otherwise `--model_type Wan2.2-T2V-A14B` cannot hit the adapter.

**Output**: the adapter code is written, the model name is registered, and msModelSlim can recognize `Wan2.2-T2V-A14B`.
**Record**: the list of adapter code files (`base_model_adapter.py`, `expert_sub_adapter.py`, `constants.py`, `t2v/model_adapter.py`, `t2v/loader.py`) and the `config.ini` registration configuration.

**Reference**: *[DiT Model Integration Quantization Workflow Guide](../knowledge_base/ptq/dit/integration_guide_diffusion_transformer_quantization.md)*

### Step 2: Quantization Scheme Design

**Objective**: design a W4A4F4 mixed quantization scheme based on the model structure and the characteristics of the original factory weights.

**Inputs**: the model structure characteristics confirmed in Step 1.

**Operations**:

1. Determine the overall quantization strategy.

    The dual-expert DiT backbone of Wan2.2 uses a W4A4F4 mixed quantization scheme, with the following key points:

    - Experience shows that the first five layers are generally more sensitive, so the first 5 layers (blocks.0 ~ blocks.4) are rolled back to W8A8 quantization (both activations and weights are mxfp8) to reduce the impact of the shallow-layer feature quantization error on the generation quality.
    - The vast majority of the backbone layers (`blocks.5` and later) undergo W4A4 quantization: both activations and weights are symmetrically quantized to mxfp4 at the `per_block` granularity (activations with `minmax`, weights with `ceil_x` and with `enable_search` enabled), obtaining memory and bandwidth savings while maintaining the generation quality.
    - Attention (`self_attn`) enables online QuaRot (`online_quarot`): attention activations usually contain outliers, and direct low-bit quantization easily causes accuracy loss, so a Hadamard rotation is used to improve the activation distribution and reduce the accuracy loss of quantization.
    - FA3 activation quantization is enabled: `self_attn` is quantized to mxfp4, of which `blocks.0.self_attn` is rolled back to fp8_e4m3 quantization.

2. Write the quantization practice configuration.

   The complete quantization configuration is written separately as a quantization practice configuration ([`lab_practice/wan2_2/wan2_2_w4a4f4_mxfp_t2v.yaml`](../../../lab_practice/wan2_2/wan2_2_w4a4f4_mxfp_t2v.yaml)) and is specified with the `--config` parameter during quantization. The `inference_config` field must be aligned with the CLI of the original repository, and `task` must be consistent with `scene_task`:

    ```yaml
    multimodal_sd_config:
      inference_config:
        size: "1280*720"
        frame_num: 81
        convert_model_dtype: True
        task: "t2v-A14B"
    ```

**Output**:

- The Wan2.2 W4A4F4 quantization practice configuration.

**Record**: the key points of the quantization scheme design and the decision basis (the quantization bit width/granularity/algorithm of each layer, and the enabled scope of QuaRot and FA3).

### Step 3: Environment Preparation and Path Configuration

**Objective**: set the environment variables, check the dependency versions, and confirm that the model and data paths are correct.
**Inputs**: the model path, the data path, and the output path.
**Operations**: configure the environment variables and run the version check commands.

```bash
# Set the environment variables (replace with the actual paths)
export MODEL_PATH=/path/to/Wan2.2-T2V-A14B                   # Floating-point model weight path
export SAVE_PATH=/path/to/output/wan22_w4a4f4                # Quantized weight export path
export EVAL_DATA_PATH=/path/to/eval/final_mini_dataset_0_01  # Evaluation dataset path; see Step 5 below for details
export OUTPUT_DIR=/path/to/output/wan22_verification         # Inference result output directory
export PYTHONPATH=/path/to/Wan2.2:$PYTHONPATH                # Inference repository path

# Check the environment versions (must be actually executed in the verification environment)
npu-smi info
python -V
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch_npu; print('TorchNPU:', torch_npu.__version__)"
pip show msmodelslim mindiesd
```

**Output**: the environment variables are configured, and all dependency version checks are on record.
**Record**: the CANN version, the PyTorch/TorchNPU version, the msmodelslim version, and the NPU model and driver.

### Step 4: Run Wan2.2 W4A4F4 Quantization

**Objective**: run the msModelSlim quantization workflow, complete the layer-by-layer quantization of both experts, and export the quantized weights.
**Inputs**: the floating-point model, the calibration dataset, and the quantization configuration (`${YAML_PATH}`).
**Operations**: run W4A4F4 quantization with the corresponding model_type.

```bash
# Floating-point model weight path: ${MODEL_PATH}
# Quantized weight export path: ${SAVE_PATH}
# Quantization configuration file path: ${YAML_PATH} (that is, wan2_2_w4a4f4_mxfp_t2v.yaml in Step 2)
# T2V text-to-video quantization example
msmodelslim quant \
    --model_path ${MODEL_PATH} \
    --save_path ${SAVE_PATH} \
    --device npu \
    --model_type Wan2.2-T2V-A14B \
    --config ${YAML_PATH} \
    --trust_remote_code true
```

**Output**: the quantized weights are saved to the `${SAVE_PATH}` directory, containing the quantized weights of both experts and the description file.
**Record**: the complete quantization log, the total quantization time, and the quantization status of each layer.

### Step 5: Prepare the VBench-1.0-mini Evaluation Subset

**Objective**: organize the 0.01 subset from the original VBench-1.0-mini dataset for use by the inference script.
**Inputs**: the downloaded original VBench-1.0-mini dataset.
**Operations**: adjust the directory structure and rename the files.

The original VBench-1.0-mini directory structure is as follows:

```text
VBench-1.0-mini/
├── VBench_kmeans_info.json              # Full mini set, 91 prompts
├── VBench_kmeans_info_0.01.json         # 11 prompts (the 0.01 subset)
├── VBench_kmeans_info_0.05.json         # 43 prompts
├── VBench_kmeans_info_0.10.json         # Same as the full mini set
└── README.md
```

> [!NOTE]
>
> The JSON file name is hard-coded as `VBench_kmeans_info.json` in the inference script `vbench.py`, so a file with that name must exist in the directory pointed to by `--vbench_mini_root`.

The goal is to organize the 0.01 subset into the following structure:

```text
final_mini_dataset_0_01/
└── VBench_kmeans_info.json          # Renamed from VBench_kmeans_info_0.01.json
```

Run the following commands to complete the organization (use `cp` to keep the original dataset unchanged):

```bash
mkdir -p final_mini_dataset_0_01
cp VBench-1.0-mini/VBench_kmeans_info_0.01.json \
   final_mini_dataset_0_01/VBench_kmeans_info.json
```

After that, pass `--vbench_mini_root ./final_mini_dataset_0_01` during inference to use the 0.01 subset. This subset contains 11 prompts covering 11 dimensions; counting 1 generated video per prompt, there are 11 videos in total.

**Output**: the `final_mini_dataset_0_01/` directory is ready and can be passed directly to `--vbench_mini_root`.
**Record**: the organized directory structure and the content of the `VBench_kmeans_info.json` file (a list of 11 prompts).

### Step 6: Run VBench Evaluation Inference for the Wan2.2 Floating-Point Model

**Objective**: run floating-point model inference and generate evaluation results on the VBench-mini dataset as the accuracy comparison baseline; this step only generates videos.
**Inputs**: the floating-point model weights, the VBench-mini evaluation dataset, and the inference hyperparameter configuration.
**Operations**: the specific values in this case are: `ulysses_size=2`, `dit_fsdp + t5_fsdp`, `cfg_size=2`, and `vae_parallel` enabled. The FP16 and W4A4F4 runs use exactly the same parallelism conventions. Here `ALGO` is the environment variable used by the inference repository to select the FA computation method, and its value must be chosen according to the device and the inference method (for the value description, see [Before You Start - ALGO Configuration](https://modelers.cn/models/MindIE/wan2.2#31-%E5%BC%80%E5%A7%8B%E5%89%8D%E5%BF%85%E8%AF%BB) in the inference repository documentation). The product form of this case is Ascend 950PR&950DT products: for floating-point inference, use `ALGO=0`; for quantized inference, because attention FP8 is enabled, use `ALGO=3`. This is the only difference between the two sides other than the model path.

```bash
# Wan2.2 floating-point model inference
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export FAST_LAYERNORM=1

# Model path: ${MODEL_PATH}         Floating-point model weight path
# Evaluation data: ${EVAL_DATA_PATH}     Evaluation dataset path
# Output directory: ${OUTPUT_DIR}         Inference result save path
torchrun --nproc_per_node=4 --master_port=23459 vbench.py \
--task t2v-A14B \
--ckpt_dir ${MODEL_PATH} \
--size "1280*720" \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 2 \
--ulysses_size 2 \
--vae_parallel \
--base_seed 0 \
--vbench_mini \
--vbench_mini_root ${EVAL_DATA_PATH} \
--save_path ${OUTPUT_DIR}/vbench_fp_output/ \
--num_samples 1 \
--temporal_flickering_samples 1
```

**Output**: inference completes normally, all evaluation videos are generated, and the results are saved to the `${OUTPUT_DIR}/vbench_fp_output/` directory; the generated videos have no image quality degradation.
**Record**: the complete inference run log.

### Step 7: Run VBench Evaluation Inference for the Wan2.2 Quantized Model

**Objective**: run W4A4F4 quantized model inference and generate evaluation videos for comparison with the floating-point baseline.
**Inputs**: the W4A4F4 quantized weights, the same VBench-mini evaluation dataset as the floating-point run, and the W4A4F4 inference hyperparameter configuration.
**Operations**: only change `ALGO` from 0 to 3 and add the `--quant_dit_path` parameter pointing to the quantized weight path; all other inference parameters are exactly the same as the floating-point run. Run inference to generate the evaluation videos.

```bash
# Wan2.2 quantized model inference and VBench evaluation command
export ALGO=3
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export FAST_LAYERNORM=1

# Model path: ${MODEL_PATH}         Floating-point model weight path
# Quantized weights: ${SAVE_PATH}          Quantized weight path
# Evaluation data: ${EVAL_DATA_PATH}     Evaluation dataset path
# Output directory: ${OUTPUT_DIR}         Inference result save path
torchrun --nproc_per_node=4 --master_port=23459 vbench.py \
--task t2v-A14B \
--ckpt_dir ${MODEL_PATH} \
--quant_dit_path ${SAVE_PATH} \
--size "1280*720" \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 2 \
--ulysses_size 2 \
--vae_parallel \
--base_seed 0 \
--vbench_mini \
--vbench_mini_root ${EVAL_DATA_PATH} \
--save_path ${OUTPUT_DIR}/vbench_quant_output/ \
--num_samples 1 \
--temporal_flickering_samples 1
```

**Output**: inference completes normally, all evaluation videos are generated, and the results are saved to the `${OUTPUT_DIR}/vbench_quant_output/` directory; the generated videos have no image quality degradation.
**Record**: the complete quantized inference run log.

## 6. Accuracy Testing

> For accuracy verification of the Wan2.2 generation model, it is recommended to combine **objective metrics with manual subjective evaluation**, focusing on: image clarity, color accuracy, temporal consistency, motion coherence, and whether the object structure degrades.

### 6.1 Test Design

| Item | Content |
| ---------- | ---------------------------- |
| Evaluation tool | AISBench |
| Comparison target | Wan2.2 FP16 floating-point inference results vs. Wan2.2 W4A4F4 quantized inference results |
| Dataset and task | The 1% subset of VBench-1.0-mini |
| Number of samples / subset strategy | The 1% subset of VBench-1.0-mini contains 11 prompts in total; 1 video is generated per prompt, giving 11 video samples in total |
| Metrics and direction | VBench Quality, VBench Semantic |
| Acceptance threshold | The absolute score drops of both VBench Quality and Semantic must be < 1pp (that is, `\|FP score − quantized score\| < 1pp`); the judgment is made item by item, and if any item exceeds the threshold, the case fails |
| Randomness control | Fixed random seed seed=0, fixed sampling steps=40, fixed resolution 720P, and fixed frame count 81 frames; all inference parameters are exactly identical between the two sides of the comparison |

### 6.2 Reproducible Procedure

**Data preparation**:

The VBench-1.0-mini dataset has been prepared in Step 5 (the `final_mini_dataset_0_01/` directory). For the VBench evaluation data preparation, the small-model weight cache download, and the AISBench environment installation, see the *[AISBench VBench evaluation documentation](https://github.com/AISBench/benchmark/blob/master/docs/source_en/extended_benchmark/lmm_generate/vbench.md)*.

**Key evaluation configuration points**:

The VBench metrics are computed through the `VBenchEvalTask` of aisbench. The core configuration items are as follows (taking `ais_bench/configs/vbench_examples/eval_vbench_standard.py` as an example).

**Note**: the values below must be written directly into the configuration file. Shell environment variables are not expanded in the configuration file, so fill in the expanded **absolute paths** (the table below uses `OUTPUT_DIR=/path/to/output/wan22_verification` and `EVAL_DATA_PATH=/path/to/eval/final_mini_dataset_0_01` from Step 3 as an example; replace them with the actual paths).

| Config Item | Value |
| ------ | ---- |
| `DATA_PATH` | Floating point: `/path/to/output/wan22_verification/vbench_fp_output/`; quantized: `/path/to/output/wan22_verification/vbench_quant_output/` (that is, the actual video directories after `--save_path` is expanded in Steps 6/7) |
| `full_json_dir` | `/path/to/eval/final_mini_dataset_0_01/VBench_kmeans_info.json` (the 0.01 subset JSON prepared in Step 5, consistent with the file under `--vbench_mini_root` in the inference stage) |
| `VBENCH_CACHE_DIR` | The cache directory path for the small-model weights |
| `dimension_list` | Leave empty to evaluate all 16 dimensions |
| `load_ckpt_from_local` | `True` (load the small-model weights from the local cache) |

**Execution command** (the videos were already generated in Steps 6/7 during inference; here the VBench metric computation is executed):

```bash
# 1) Floating-point result metric computation: first change DATA_PATH in the configuration file to .../vbench_fp_output/
cd /path/to/AIS_BENCH_ROOT  # Enter the AIS_BENCH evaluation tool root directory
ais_bench ais_bench/configs/vbench_examples/eval_vbench_standard.py \
    --mode eval \
    --max-num-workers 1

# 2) Quantized result metric computation: change DATA_PATH in the same configuration file to .../vbench_quant_output/, keep the other configuration items (full_json_dir, dimension_list, VBENCH_CACHE_DIR, etc.) unchanged, and run the same command again
cd /path/to/AIS_BENCH_ROOT  # Enter the AIS_BENCH evaluation tool root directory
ais_bench ais_bench/configs/vbench_examples/eval_vbench_standard.py \
    --mode eval \
    --max-num-workers 1
```

**Output and Record**:

- Result root directory: `./outputs/default/{timestamp}/` (under the current directory where the command is executed)
- Directory structure:

  ```text
  {timestamp}/
  ├── configs/                # Evaluation configuration snapshot
  │   └── {timestamp}_*.py
  ├── logs/eval/vbench_eval/  # Per-dimension score logs (.out)
  ├── results/vbench_eval/    # Per-dimension raw score JSON
  └── summary/                # Summary results (.csv/.md/.txt)
  ```

### 6.3 Accuracy Results

| Dataset | Samples | Baseline (FP16) (%) | Quantized (W4A4F4) (%) | Score Drop (pp) | Conclusion |
| -------------- | ------ | -------------- | -------------- | --------------- | ----------------- |
| VBench Quality | 11 | 85.80 | 86.05 | −0.25 | Pass |
| VBench Semantic | 11 | 72.01 | 71.06 | +0.95 | Pass |
| VBench Total | 11 | 83.04 | 83.05 | −0.01 | Pass |

**Accuracy conclusion**: on the 1% subset of VBench-1.0-mini (11 prompts in total, 1 video generated per prompt), the absolute score changes of the VBench Quality and VBench Semantic metrics of the Wan2.2-T2V-A14B W4A4F4 quantized inference results relative to the FP16 floating-point baseline (85.80% / 72.01%) are −0.25pp and +0.95pp respectively, both meeting the acceptance threshold of "an absolute score drop of < 1pp"; the relative change of VBench Total is −0.01pp, which also passes verification. In summary, the W4A4F4 quantization of this case reaches the preset accuracy target on the 1% subset of VBench-1.0-mini, and the acceptance passes.

## 7. Results and Lessons Learned

### 7.1 Summary of Key Results

| Step | Key Operation | Metric | Change | Note |
| ------------------ | -------------------- | -------------------- | ------------------- | ------------------ |
| Wan2.2 A14B quantization | W4A4F4 quantization | VBench Quality | −0.25pp | Meets the threshold (< 1pp) |
| Wan2.2 A14B quantization | W4A4F4 quantization | VBench Semantic | +0.95pp | Meets the threshold (< 1pp) |
| Wan2.2 A14B quantization | W4A4F4 quantization | VBench Total | −0.01pp | Meets the threshold (< 1pp) |

### 7.2 Lessons Learned

1. **Common issues**: when flickering/color blocks appear after quantization, try adjusting the activation quantization method or selectively rolling back some layers (a general principle, not limited to a specific device or quantization type).
2. **Comparison test**: floating-point and quantized inference must keep exactly the same hyperparameters, with only the model path differing (a general principle, not limited to a specific scenario).

## 8. Appendix

- Quantization configuration path: [W4A4F4 quantization YAML configuration](../../../lab_practice/wan2_2/wan2_2_w4a4f4_mxfp_t2v.yaml)
