---
toc_depth: 3
---
# Automatic Tuning Usage Guide

## Overview

The automatic tuning feature is designed for users to precisely control the accuracy of quantized models. Through automated quantization configuration search and evaluation, it helps you find a quantization solution that meets your accuracy requirements. This feature automatically attempts different quantization configurations based on your specified accuracy requirements, verifies the accuracy of the quantized model through the evaluation service, and outputs a quantized model that meets the requirements.

## Preparations

Before using the automatic tuning feature, install the following tools:

1. **msModelSlim**: used to start and execute automatic tuning tasks. For details about the installation method, see [msModelSlim Installation Guide](../../getting_started/install_guide.md).
2. **vLLM-Ascend**: used to launch the quantized model as a service. When evaluating the accuracy of the quantized model by using the automatic tuning feature, use `vLLM-Ascend` to start the quantized model as a service. You are advised to use the official image for installation. For details, see the [vLLM-Ascend Installation Guide](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/installation.html).
3. **AISBench**: used to test the quantized model. The automatic tuning feature uses [AISBench](https://github.com/AISBench/benchmark) to evaluate and test the accuracy of the quantized model. After completing the installation, you also need to prepare the corresponding dataset by referring to the [AISBench Dataset Preparation Guide](https://yh-ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/all_params/datasets.html).

Ensure that the preceding tools are correctly installed and configured. Otherwise, the automatic tuning feature cannot perform model evaluation properly.

## Function Introduction

### Function Description

The automatic tuning feature automatically searches for and verifies quantization configurations under a given accuracy target, helping you obtain quantization results that meet accuracy requirements and can be directly deployed and reused.

#### Core Capabilities

- **Automatic search**: iteratively generates candidate quantization configurations based on the policies and search spaces defined in the tuning configuration file.
- **Automatic evaluation**: performs quantization for each candidate configuration, launches the quantized model service using `vLLM-Ascend`, and then evaluates the accuracy of the quantized model using `AISBench`.
- **Result output**: outputs the final quantization configurations and quantized weights that meet accuracy requirements. It retains the configuration and accuracy cache for each iteration, supporting interruption recovery and reproduction.
- **Target scenarios**: automated workflows that need to complete quantization configuration selection, fallback layers selection, and accuracy verification within a short timeframe, thereby reducing the cost of repeated manual parameter tuning.

#### Tuning Process Description

The workflow of the automatic tuning feature is as follows:

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

6. **Save results**: After tuning is complete, save the quantized model weights, tuning history, accuracy cache, and evaluation and service logs. For details about the output location and directory structure, see the "Output Description" section below.

### Precautions

**Important note**: Before using the automatic tuning feature, ensure that your model is supported. The following conditions must be met simultaneously:

1. **Model type restrictions**: Currently, the automatic tuning service supports only large language models (LLMs), and has limited support for MoE models (MoE models usually contain a large number of expert non-shared sublayers, resulting in a larger fallback and search space, more iterative optimization rounds, and a longer overall tuning duration).
2. **Quantization tool support**: The model must be included in the support list of the quantization tool. You can check whether a model is supported by viewing the `[ModelAdapter]` section in [config/config.ini](https://gitcode.com/Ascend/msmodelslim/blob/master/config/config.ini), which lists the currently supported model adapters and their corresponding model names. If your model is not in the support list, perform model adaptation and implement the corresponding model interfaces before using the automatic tuning feature. For details about model adaptation, see the [LLM Model Integration Guide](../../developer_guide/integrating_models.md).
3. **vLLM-Ascend support**: The model must be supported by vLLM-Ascend so that the quantized model can be launched as a service. Verify whether vLLM-Ascend supports launching your quantized model as a service before proceeding.
4. **Transformers version compatibility**: The quantization tool and the inference engine (vLLM-Ascend) have their own version requirements for `transformers`. Ensure that the `transformers` version in your current environment satisfies requirements for both components. If a model requires inconsistent `transformers` versions for the quantization tool and the inference engine, and **no single transformers version can satisfy both requirements simultaneously**, the automatic tuning service cannot be started in this environment. Before use, verify the environment compatibility by referring to the dependency descriptions of both sides. **Quantization tool side** version requirements for dependencies such as `transformers` across different models can be found in the `[ModelAdapterDependencies]` configuration item in [config/config.ini](https://gitcode.com/Ascend/msmodelslim/blob/master/config/config.ini). **Inference engine side** dependency requirements for various vLLM-Ascend versions can be found in the "Dependencies" section of the corresponding version in the [vLLM-Ascend Release Notes](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/user_guide/release_notes.html#id5).
5. **Single-server deployment**: The automatic tuning service does not support multi-server deployment. For ultra-large-scale models that require multi-server deployment to launch as a service, the automatic tuning feature cannot be used.

### Syntax

The automatic tuning feature is started using the command line. After meeting the preceding prerequisites, you can run the following command:

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

### **Parameters**

| Parameter             | Purpose       | Mandatory (Yes/No)             | Description                                                                                  |
|-------------------|-----------|-------------------|--------------------------------------------------------------------------------------|
| model_path        | Specifies the model path.     | Yes               | Type: `str`.                                                                              |
| save_path         | Specifies the storage path for the tuning results. | Yes               | Type: `str`.                                                                              |
| config            | Specifies the path to the tuning configuration file. | Yes               | 1. Type: `str`.<br>2. The path must be a complete file path.<br>3. The configuration file must be in YAML format. For details about the configuration protocol, see Automatic Tuning Configuration Protocols. For a sample configuration, see [example](example).|
| device            | Specifies the quantization device.     | No               | 1. Type: `str`.<br>2. Example values: `'npu'`, `'npu:0,1,2,3'`, `'cpu'`<br>3. Default value: `'npu'` (single device).<br>4. When multiple devices are specified (such as, `'npu:0,1,2,3'`), the system can start distributed layer-wise quantization (DP layer-wise). For details about the algorithm support scope and configuration method, see [Layer-wise and DP Layer-wise Quantization](../quick_quantization_v1/usage.md#layer-wise-and-dp-layer-wise-quantization).|
| model_type        | Specifies the model name.     | No               | 1. Type: `str`.<br>2. Default value: `default`.<br>3. The value is case-sensitive. For details, see [Foundation Model Support Matrix](../../model_support/foundation_model_support_matrix.md).|
| timeout           | Specifies the tuning timeout duration.   | No               | 1. Type: `str`.<br>2. Format: `D`, `H`, or `DH`.<br>3. Examples: `'1D'`, `'2H'`, and `'3D4H'`.<br>4. Default value: `None` (no timeout limit).|
| trust_remote_code | Specifies whether to trust custom code.| No               | 1. Type: `bool`. Default value: `False`.<br>2. Setting this parameter to `True` enables the execution of custom code, which may pose security risks. Ensure the loaded custom code file is secure.                         |
| h, help           | Displays help information for command-line options.| No               |               -            |

### Example

The following example shows how to use the automatic tuning feature to tune the Qwen3-32B model:

``` bash
msmodelslim tune --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --config ${CONFIG} --device npu --model_type Qwen3-32B --trust_remote_code True
```

After you run the command, the system automatically attempts different quantization configurations based on the accuracy requirements specified in the configuration file, and evaluates the quantized model until a quantization solution that meets the accuracy requirements is found or the maximum number of iterations or timeout duration is reached.

### Output Description

After the automatic tuning feature completes execution, the following outputs are generated in the `${SAVE_PATH}` directory. The directory structure is as follows:

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

## Appendix

### References

- **Configuration protocols**: For detailed descriptions of the tuning configuration file, see the [Automatic Tuning Configuration Protocol](configuration_protocols.md). For complete configuration file examples, see [example](example).
- **Tuning algorithms**: For details about the strategies and algorithms supported by automatic tuning, see the following documents:
  - [Standing High Tuning Algorithm](../../quantization_algorithms/auto_tuning_strategies/standing_high.md): an automatic tuning algorithm based on fallback layers selection and outlier suppression strategies
  - [Standing High With Experience Tuning Algorithm](../../quantization_algorithms/auto_tuning_strategies/standing_high_with_experience.md): a standing high algorithm strategy based on expert experience
