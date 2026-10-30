# Auto Tuning Usage Guide

## 1. Overview

The auto tuning feature is designed for users to precisely control the accuracy of quantized models. Through automated quantization configuration search and evaluation, it helps you find a quantization solution that meets your accuracy requirements. This feature automatically attempts different quantization configurations based on your specified accuracy requirements, verifies the accuracy of the quantized model through the evaluation service, and outputs a quantized model that meets the requirements.

## 2. Preparations

Before using the auto tuning feature, install the following tools:

1. **msModelSlim**: used to launch and execute auto tuning tasks. For installation instructions, see the [Installation Guide](../install_guide/install_guide.md).
2. **vLLM-Ascend**: used to launch the quantized model as a service. When evaluating the accuracy of the quantized model by using the auto tuning feature, use `vLLM-Ascend` to start the quantized model as a service. You are advised to use the official image for installation. For details, see the [vLLM-Ascend Installation Guide](https://docs.vllm.ai/projects/ascend/en/latest/getting_started/installation.html).
3. **AISBench**: used for testing the quantized model. The auto tuning feature uses [AISBench](https://github.com/AISBench/benchmark) to evaluate and test the accuracy of the quantized model. After the installation is complete, you also need to prepare the corresponding dataset by referring to the [AISBench Dataset Preparation Guide](https://github.com/AISBench/benchmark/blob/master/docs/source_en/get_started/datasets.md).

Ensure that the preceding tools are correctly installed and configured. Otherwise, the auto tuning feature cannot perform model evaluation properly.

## 3. Function Introduction

### 3.1 Function Description

The auto tuning feature automatically searches for and verifies quantization configurations under a given accuracy target, helping you obtain quantization results that meet accuracy requirements and can be directly deployed and reused.

#### 3.1.1 Core Capabilities

- **Automatic search**: iteratively generates candidate quantization configurations based on the policies and search spaces defined in the tuning configuration file.
- **Automatic evaluation**: performs quantization for each candidate configuration, launches the quantized model service using `vLLM-Ascend`, and then evaluates the accuracy of the quantized model using `AISBench`.
- **Result output**: outputs the final quantization configurations and quantized weights that meet accuracy requirements. It retains the configuration and accuracy cache for each iteration, supporting interruption recovery and reproduction.
- **Target scenarios**: automated workflows that need to complete quantization configuration selection, fallback layers selection, and accuracy verification within a short timeframe, thereby reducing the cost of repeated manual parameter tuning.

#### 3.1.2 Tuning Process Description

The workflow of the auto tuning feature is as follows:

1. **Create a model adapter**: Create a model adapter based on the specified model type to ensure that the specified model is supported.

2. **Initializing the tuning service**: Load the tuning plan based on the specified configuration file path, and create a quantization configuration generator.

3. **Detect historical records**: Automatically detect whether a historical accuracy cache exists. If it exists, subsequent iterations will attempt to reuse the historical evaluation results to avoid repeatedly evaluating the same configuration.

4. **Start iterative tuning**: Record the tuning start time and timeout duration, and enter the iteration loop.

   Each iteration includes the following steps:
   - **Check for timeout**: Check whether the maximum iteration duration is exceeded at the beginning of each iteration. If it times out, stop the iteration.
   - **Generate a quantization configuration YAML**: Generate a quantization configuration (practice in YAML format) through the quantization configuration generator. In the first iteration, pass `None` to generate the configuration based on the initial policy and accuracy requirements. In subsequent iterations, pass the previous accuracy evaluation result to generate a new configuration based on that result.
   - **Attempting to recover the history**: If a historical accuracy cache exists, the system attempts to recover the evaluation result of the current iteration from the accuracy cache. The matching condition is: **both the quantization configuration and the evaluation configuration must be exactly the same**. If a fully matching evaluation result is found, directly reuse the historical evaluation result and skip the subsequent steps. Otherwise, perform the complete quantization and evaluation workflow again.
   - **Quantize the model**: If no history is recovered, quantize the model based on the generated quantization configuration YAML, and temporarily save the quantized weights to your specified path.
   - **Evaluate model accuracy**: Launch the quantized model as a service using `vLLM-Ascend`, and then evaluate and test the accuracy of the quantized model using `AISBench` to generate accuracy indicators. (Optionally, enable pre-check to detect obvious anomalies in advance.)
   - **Save tuning history**: Save the quantization configuration YAML and accuracy indicators of the current iteration to the tuning history.
   - **Determine whether to continue**: Use the evaluated accuracy indicators as the input for the next iteration. If the policy determines that the iteration stop conditions are met, stop the iteration. Otherwise, proceed to the next iteration.

5. **Stop conditions**: The iteration stops when any of the following conditions is met:
   - The tuning policy has no remaining search space (unable to generate new evaluable quantization configurations).
   - The maximum number of iterations is reached.
   - The maximum iteration duration is reached.
   - The tuning policy exits with an error proactively. For example, at the start of tuning, the Standing High policy first performs a floating-point pre-check (evaluating the accuracy of the fully rolled-back floating-point baseline). If even the floating-point model cannot meet the accuracy target configured by the user, the policy throws a `SpecError` (error code 302), exits, and prompts you to relax the accuracy expectation. For details, see [Standing High Tuning Algorithm](../knowledge_base/tuning_strategies/standing_high/standing_high.md)

6. **Save results**: After tuning is complete, save the quantized model weights, tuning history, accuracy cache, and evaluation and service logs. For details about the output location and directory structure, see the "Output Description" section below.

### 3.2 Precautions

**Important note**: Before using the auto tuning feature, ensure that your model is supported. The following conditions must be met simultaneously:

1. **Model type restrictions**: Currently, the auto tuning service supports only large language models (LLMs), and has limited support for MoE models (MoE models usually contain a large number of expert non-shared sublayers, resulting in a larger fallback and search space, more iterative optimization rounds, and a longer overall tuning duration).
2. **Quantization tool support**: The model must be in the support list of the quantization tool. You can check the `[ModelAdapter]` section in [config/config.ini](../../../config/config.ini) to determine whether the model is supported; it contains the currently supported model adapters and the model names they support. If the model is not in the support list, you must first perform model adaptation and implement the corresponding model interfaces before you can use the auto tuning feature. For model adaptation instructions, see the [LLM Integration Guide](../knowledge_base/ptq/llm/integration_guide_large_language_model_quantization.md).
3. **vLLM-Ascend support**: The model must be supported by vLLM-Ascend so that the quantized model can be launched as a service. Verify whether vLLM-Ascend supports launching your quantized model as a service before proceeding.
4. **transformers version compatibility**: The quantization tool and the inference engine (vLLM-Ascend) each have their own requirements for the `transformers` version, and you need to ensure that the version in the current environment can satisfy both. If, for a model in use, the `transformers` versions required by the quantization tool and the inference engine are inconsistent and **no single transformers version can satisfy both**, the auto tuning service cannot be started in that environment. Before use, confirm the `transformers` version compatibility in the current environment by referring to the dependency documentation of both sides: for the **quantization tool side**, see the `[ModelAdapterDependencies]` configuration item in [config/config.ini](../../../config/config.ini) for the version requirements of each model on `transformers` and other dependencies; for the **inference engine side**, see the Dependencies section of each version in the [vLLM-Ascend Release Notes](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/release_notes.html).
5. **Single-server deployment**: The auto tuning service does not support multi-server deployment. For ultra-large-scale models that require multi-server deployment to launch as a service, the auto tuning feature cannot be used.

### 3.3 Syntax

The auto tuning feature is started using the command line. After meeting the preceding prerequisites, you can run the following command:

``` bash
msmodelslim tune [ARGS]
```

The following is a command template containing all parameters. In the template, `${MODEL_PATH}` is the original floating-point weights path of the model, `${SAVE_PATH}` is your custom path for saving tuning results, and `${CONFIG}` is the tuning configuration file path. `${DEVICE}`, `${MODEL_TYPE}`, `${TIMEOUT}`, and `${TRUST_REMOTE_CODE}` correspond to the quantization device, model type, timeout duration, and whether to trust remote code, respectively.

``` bash
# Global command line invocation
msmodelslim tune --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --config ${CONFIG} --device ${DEVICE} --model_type ${MODEL_TYPE} --timeout ${TIMEOUT} --trust_remote_code ${TRUST_REMOTE_CODE}
```

**Interruption recovery**: If the tuning process is interrupted unexpectedly (such as by a system fault or manual termination), the system automatically checks whether a historical accuracy cache exists in the `${SAVE_PATH}/history` directory. If an accuracy cache is detected, the system restarts the iteration from the beginning but reuses the evaluation results of the already-evaluated quantization configurations from the cache to avoid repeated evaluations of the same configurations. You only need to run the tuning command again using the same `${SAVE_PATH}`.

**Notes**:

- The automatic recovery feature requires that the `${SAVE_PATH}` is identical to the path used before the interruption, and a valid historical accuracy cache exists in the `${SAVE_PATH}/history` directory. If the historical accuracy cache does not exist or is invalid, the system starts tuning from the beginning.
- **Accuracy cache reuse conditions**: The system reuses evaluation results from the historical accuracy cache only when both of the following conditions are met:
  1. **Identical evaluation configurations**: The `evaluation` field configurations (including evaluation datasets, evaluation tasks, and evaluation parameters) in the tuning configuration file are exactly the same as those in the historical record.
  2. **Identical quantization configurations**: The quantization configuration generated in the current iteration is exactly the same as that in the historical record.
- If there is any difference in either the evaluation configuration or quantization configuration, the system performs quantization, service launching, and accuracy evaluation again without reusing historical results.

**Environment Variables (Optional)**

To control the log level or configure the storage path for the "best practice library," set the following environment variables.

| Environment Variable| Purpose| Mandatory (Yes/No)| Value/Description|
|---|---|---|---|
| `MSMODELSLIM_LOG_LEVEL` | Outputs logs at the specified level and above.| No| Valid values: `INFO` (default) or `DEBUG`|
| `MSMODELSLIM_CUSTOM_PRACTICE_REPO` | Specifies the custom path for saving the best practice library configuration file.| No| Type: `str`. Once set, the final quantization configuration YAML is written to the best practice library directory under this path when the accuracy meets requirements. If not set, it will not be written to the best practice library. You can obtain the configurations from the YAML files of individual iterations in `${SAVE_PATH}/history`.|

### 3.4 **Parameters**

| Parameter             | Purpose       | Mandatory (Yes/No)             | Description                                                                                  |
|-------------------|-----------|-------------------|--------------------------------------------------------------------------------------|
| model_path        | Specifies the model path.     | Yes               | Type: `str`.                                                                              |
| save_path         | Specifies the storage path for the tuning results. | Yes               | Type: `str`.                                                                              |
| config            | Specifies the path to the tuning configuration file. | Yes               | 1. Type: `str` <br>2. The configuration file path, which must be a complete file path <br>3. The configuration file must be in YAML format. For the configuration protocol, see the [Auto Tuning Configuration Reference](../api_reference/config/tuning/README.md). For example configurations, see the directory of each policy (for example, the [Standing High Example](../knowledge_base/tuning_strategies/standing_high/standing_high.yaml)). |
| device            | Specifies the quantization device type. | No               | 1. Type: `str` <br>2. Valid values: 'npu', 'cpu' <br>3. Default: "npu". Specify multi-device indexes together with `device_id`. |
| device_id         | Specifies the quantization device index. | No               | 1. Type: a list of int, separated by spaces, for example, 0 1 2 3 <br>2. When multiple device indexes are specified (for example, 0 1 2 3), the system can start distributed layer-by-layer quantization (DP). For the range of supported algorithms and how to configure them, see [Multi-Device Quantization Parallel](../knowledge_base/parallel/README.md). |
| model_type        | Specifies the model name. | No               | 1. Type: `str` <br>2. Default: "default" <br>3. The value is case-sensitive. For details, see the [Foundation Model Support Matrix](../knowledge_base/model/README.md). |
| timeout           | Specifies the tuning timeout period. | No               | 1. Type: Int (seconds) <br>2. Examples: '7200' (2 hours), '3600' (1 hour) <br>3. Legacy duration strings are supported: '1D', '2H', '3D4H' <br>4. Default: None (no timeout limit). |
| trust_remote_code | Specifies whether to trust custom code. | No               | 1. Type: Bool. Default: false <br>2. Ensure the security of the loaded custom code files. Setting this parameter to true poses security risks. |
| h, help           | Displays help information for command-line options.| No               |               -            |

### 3.5 Example

The following example shows how to use the auto tuning feature to tune the Qwen3-32B model:

``` bash
msmodelslim tune --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --config ${CONFIG} --device npu --model_type Qwen3-32B --trust_remote_code true
```

After you run the command, the system automatically attempts different quantization configurations based on the accuracy requirements specified in the configuration file, and evaluates the quantized model until a quantization solution that meets the accuracy requirements is found or the maximum number of iterations or timeout duration is reached.

### 3.6 Output Description

After the auto tuning feature completes execution, the following outputs are generated in the `${SAVE_PATH}` directory. The directory structure is as follows:

```text
${SAVE_PATH}/
├── quant_model/          # Quantized model weights (final iteration)
├── history/              # Tuning history and accuracy cache
│   ├── history.yaml      # Tuning history index file
│   ├── accuracy.yaml     # Accuracy cache file
│   └── <config_id>.yaml  # Quantization configuration of each iteration (YAML)
├── vllm_server.log       # vLLM-Ascend service log (latest)
├── aisbench_logs/        # AISBench evaluation logs
└── aisbench_output/      # Detailed AISBench output
```

| Output | Path | Description |
| --- | --- | --- |
| Quantized model weights | `${SAVE_PATH}/quant_model` | Stores the quantized model weight file generated in the last iteration, based on the saver type specified in the `save` field of the quantization configuration. |
| Tuning history records | `${SAVE_PATH}/history` | Stores tuning history of all iterations, including `history.yaml` (index), `accuracy.yaml` (accuracy cache), and per-iteration `<config_id>.yaml` (quantization configuration). |
| vLLM service logs | `${SAVE_PATH}/vllm_server.log` | Logs from the latest vLLM-Ascend service startup and runtime, including model loading, service startup, and inference request processing. |
| AISBench evaluation logs | `${SAVE_PATH}/aisbench_logs` | AISBench evaluation log files for each iteration. |
| Detailed AISBench output | `${SAVE_PATH}/aisbench_output` | Detailed AISBench evaluation output for each iteration, including task configuration, inference/evaluation logs, results, and summary reports. |
| Final quantization configuration YAML | When accuracy meets requirements and is written to the best practice library: under `MSMODELSLIM_CUSTOM_PRACTICE_REPO` by model lineage (e.g. `qwen3/xxx.yaml`) | If `MSMODELSLIM_CUSTOM_PRACTICE_REPO` is not set, nothing is written to the best practice library; use `${SAVE_PATH}/history` to pick a `<config_id>.yaml` for reproduction. YAML written to the library can be reused for quantization; no final config is written if tuning stops due to max iterations or timeout. |

## 4. Appendix

### 4.1 References

- **Configuration protocol**: For detailed information about the tuning configuration file, see the [Auto Tuning Configuration Reference](../api_reference/config/tuning/README.md). For complete example configuration files, see the directory of each policy, for example, the [Standing High Example](../knowledge_base/tuning_strategies/standing_high/standing_high.yaml), the [Standing High With Experience Example](../knowledge_base/tuning_strategies/standing_high_with_experience/standing_high_with_experience.yaml), and the [Binary Fallback Example](../knowledge_base/tuning_strategies/binary_fallback/binary_fallback.yaml).
- **Tuning algorithms**: For an overview of the policies, see the [Auto Tuning Policy Overview](../knowledge_base/tuning_strategies/README.md). Details of each policy:
  - [Standing High Tuning Algorithm](../knowledge_base/tuning_strategies/standing_high/standing_high.md): an auto tuning algorithm based on quantization rollback layer selection and outlier suppression policies
  - [Standing High With Experience Tuning Algorithm](../knowledge_base/tuning_strategies/standing_high_with_experience/standing_high_with_experience.md): a standing-high algorithm policy based on expert experience
  - [Binary Fallback Tuning Algorithm](../knowledge_base/tuning_strategies/binary_fallback/binary_fallback.md): an auto tuning algorithm with a binary-search-based rollback layer selection policy
