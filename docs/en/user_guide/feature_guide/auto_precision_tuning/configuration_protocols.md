# Automatic Tuning Configuration Protocol

## 1. Overview

### 1.1 Configuration Protocol Introduction

The automatic tuning configuration protocol uses a hierarchical structure design. The top layer contains two core fields:

- `strategy`: Tuning policy configuration, which defines the quantization configuration generation policy and basic quantization configuration.
- `evaluation`: Evaluation service configuration, which defines the model accuracy evaluation method and parameters related to the service-based startup of the quantized model.

### 1.2 Configuration File Location

Customize the tuning configuration file based on your environment. For a reference template, see the [example](./example) directory.

### 1.3 Basic Configuration Structure

```yaml
strategy:
  type: <strategy_type>  # Tuning strategy type, such as standing_high, binary_fallback, standing_high_with_experience.
  # Different strategies have different configuration fields and semantics. For details, see the corresponding strategy documentation.

evaluation:
  type: service_oriented
  demand:
    expectations:
      # Accuracy expectation list: describes the specific datasets and corresponding accuracy requirements that must be met.
  evaluation:
    type: aisbench
    # For details about the fields, see section evaluation.evaluation.
  inference_engine:
    # Inference engine configuration
```

## 2. Configuration Fields

### 2.1 `strategy` - Tuning Strategy Configuration

**Purpose**: Defines the tuning strategy type, core parameters, and basic quantization configuration.

**`type` - Strategy Type**

**Purpose**: Specifies the type of tuning algorithm to execute. Different strategy types correspond to different optimization algorithms.

**Type**: `string`

**Value**: Determined by the currently implemented tuning strategy. Valid values: `standing_high`, `standing_high_with_experience`, or `binary_fallback`.

**Model Adapter (by Strategy)**

When a tuning strategy needs **automatic layer sensitivity analysis** (to build rollback candidates), the model adapter must implement **`ModelSlimPipelineInterfaceV1`** (`PipelineInterface` in `core/runner/pipeline_interface.py`, consistent with the CLI `msmodelslim analyze` command and [Sensitive Layer Analysis](../sensitive_layer_analysis/usage.md)). The strategy calls `PipelineAnalysisService`, which invokes `init_model`, `handle_dataset`, and visit/forward pipeline methods; the strategy **does not** call `load_model` upfront.

**Additional** requirements per strategy:

| Strategy | Model adapter requirement |
|----------|-------------------------|
| `standing_high` | **`ModelSlimPipelineInterfaceV1`** (always runs automatic sensitivity analysis) |
| `standing_high_with_experience` | Above + **`StandingHighWithExperienceInterface`** (`load_model`, outlier suppression capability probe) |
| `binary_fallback` | **`ModelSlimPipelineInterfaceV1`** (V1 architecture) |

See each strategy document and [Integrating LLM Models](../../../development_guide/integrating_models.md#auto-tuning-and-sensitive-layer-analysis).

**Strategy-specific Fields**

The configuration fields vary significantly depending on the selected tuning strategy. For details about specific configuration items, see the "YAML Configuration Fields" in the corresponding algorithm document. The currently supported algorithms are as follows:

- [Standing High Tuning Algorithm](../../quantization_algorithms/auto_tuning_strategies/standing_high.md)
- [Standing High With Experience Tuning Algorithm](../../quantization_algorithms/auto_tuning_strategies/standing_high_with_experience.md)
- [Binary Fallback Tuning Algorithm](../../quantization_algorithms/auto_tuning_strategies/binary_fallback.md)
 
### 2.2 `evaluation` - Evaluation Service Configuration

**Purpose**: Defines the configuration for model accuracy evaluation, including the evaluation service type, evaluation tool configuration, and inference engine configuration.

**Core Fields**

| Field| Purpose| Type| Mandatory (Yes/No)| Description|
|--------|------|------|----------|------|
| type | Specifies the evaluation service type.| string | Yes| The value is fixed to `service_oriented` (service-oriented evaluation, which launches the model as a service to perform evaluation).|
| demand | Specifies the accuracy requirement configuration.| object | Yes| Defines the specific accuracy requirements for model evaluation, including the target dataset, target accuracy, and tolerance.|
| evaluation | Specifies the evaluation tool configuration.| object | Yes| Defines configuration parameters for the underlying evaluation tool.|
| inference_engine | Specifies the inference engine configuration.| object | Yes| Defines configuration parameters for the inference engine, which is utilized to launch the quantized model in service-oriented mode.|

#### 2.2.1 `type` - Evaluation Service Type

**Purpose**: Specifies the type of the evaluation service.

**Type**: `string`

**Value**: `service_oriented` (service-oriented evaluation, which launches the model as a service to perform evaluation).

#### 2.2.2 `demand` - Accuracy Requirement Configuration

**Purpose**: Defines the accuracy requirements for model evaluation, including the target dataset, target accuracy, and tolerance.

**Field Description**

| Field| Purpose| Type| Mandatory (Yes/No)| Description|
|--------|------|------|----------|------|
| expectations | Specifies the accuracy expectation list.| list | Yes| A list of defined accuracy requirements. Each element contains the target dataset, target accuracy, and tolerance. At least one element must be included.|

**`expectations` Fields**

| Field| Purpose| Type| Mandatory (Yes/No)| Description|
|--------|------|------|----------|------|
| dataset | Specifies the target dataset to be evaluated.| string | Yes| The dataset name, which must match the dataset name specified in `evaluation.evaluation.datasets`.|
| target | Sets the target accuracy value.| number | Yes| The expected target accuracy, which must be greater than 0. The value can be configured as a number (such as `0.95`) or a string (such as `"0.95"`).|
| tolerance | Sets the accuracy tolerance.| number | Yes| The error range allowed for accuracy evaluation, which must be greater than or equal to 0. The value can be configured as a number (such as `0.95`) or a string (such as `"0.95"`).|

**Configuration Example**

```yaml
# Accuracy requirements for a single dataset
demand:
  expectations:
    - dataset: gsm8k
      target: 83  # Target accuracy: 83%
      tolerance: 2  # Tolerance: ±2%

# Accuracy requirements for multiple datasets
demand:
  expectations:
    - dataset: gsm8k
      target: 83  # Target accuracy: 83%
      tolerance: 2  # Tolerance: ±2%
    - dataset: aime25
      target: 85  # Target accuracy: 85%
      tolerance: 1  # Tolerance: ±1%
    - dataset: bfcl-simple
      target: 80  # Target accuracy: 80%
      tolerance: 2  # Tolerance: ±2%
```

**Notes**

- **Accuracy metric units**: The accuracy format returned by different datasets may vary. Some datasets return accuracy values in a decimal format (ranging from 0.0 to 1.0, where 0.83 represents 83%), while others return values in a percentage format (ranging from 0 to 100, where 83 represents 83%). The units configured for `target` and `tolerance` must match the precision format returned by the corresponding dataset. Always configure `target` and `tolerance` based on the actual accuracy format output by the evaluation tool for that dataset.
- **Accuracy numerical types**: The `target` and `tolerance` values are processed internally by the system using the `Decimal` data type. In the YAML configuration file, you can write these parameters directly as numbers (such as `0.95`) or as strings (such as `"0.95"`). For scenarios that require strict control over decimal point precision, you are advised to use the string format.
- **Accuracy target guidelines**: The accuracy metrics provided in this document are for reference only. Configure these fields based on the baseline accuracy of your actual floating-point model. In theory, the accuracy of a quantized model will not exceed that of the original floating-point model. Therefore, you are advised to set the accuracy target slightly lower than or equal to the accuracy of the floating-point model.
- **Multi-dataset support**: The framework supports configuring accuracy requirements for multiple datasets simultaneously. You can define distinct target accuracy and error tolerance for each individual dataset.

#### 2.2.3 `evaluation` - Evaluation Tool Configuration

**Purpose**: Defines configuration parameters for the evaluation tool.

**Core Fields**

| Field| Purpose| Type| Mandatory (Yes/No)| Description|
|--------|------|------|----------|------|
| type | Specifies the evaluation tool type.| string | Yes| The value is fixed to `aisbench`.|
| precheck | Specifies the pre-check configuration.| list | No| Defines the pre-check configuration executed before formal evaluation. An empty list skips the pre-check. Default value: `[]`|
| aisbench | Specifies the AISBench configuration.| object | Yes| Detailed configuration parameters for the AISBench evaluation tool, such as `timeout`, `batch_size`, and `generation_kwargs`.|
| datasets | Specifies the dataset configuration.| dict | Yes| Defines the targeted datasets to be evaluated and their configurations. This configuration must include all datasets specified in `demand.expectations`.|
| host | Specifies the service host address.| string | No| The IP address or hostname used by the evaluation client to connect to the inference server. **This value must match the configuration on the inference engine side.** Default value: `"localhost"`|
| port | Specifies the service port.| int | No| The port number used by the evaluation client to connect to the inference server. **This value must match the configuration on the inference engine side.** Default value: `1234`|
| served_model_name | Specifies the service-oriented model name.| string | No| The model name identifier used by the evaluation client to send requests. **This value must match the configuration on the inference engine side**. Default value: `"served_model_name"`|

**Configuration Example**

```yaml
evaluation:
  type: aisbench
  aisbench:
    binary: ais_bench
    mode: all
    timeout: 7200
    request_rate: 1.0
    retry: 2
    batch_size: 32
    max_out_len: 512
    trust_remote_code: false
    pred_postprocessor: extract_non_reasoning_content
    generation_kwargs:
      temperature: 0.5
      top_k: 10
      top_p: 0.9
      seed: null
      repetition_penalty: 1.03
    model_meta:
      base_name: vllm_api_general_chat
      subdir: vllm_api
      abbr: vllm-api-general-chat
      attr: service
    default_metric_keys:
      - final_accuracy
      - accuracy
      - score
  datasets:
    gsm8k:
      config_name: "gsm8k_gen_0_shot_cot_str"
      mode: all
    aime25:
      config_name: "aime2025_gen_0_shot_chat_prompt"
      mode: all
    bfcl-simple:
      config_name: "BFCL_gen_simple"
      mode: all
  host: localhost
  port: 1234
  served_model_name: served_model_name
```

##### 2.2.3.1 `aisbench` - AISBench Configuration

**Purpose**: Configures the command-line and evaluation parameters (such as `timeout`, `batch_size` and `generation_kwargs`) for AISBench.

**`aisbench` Fields**

The following table describes configuration fields of the AISBench evaluation tool.

| Field| Purpose| Type| Mandatory (Yes/No)| Description|
|--------|------|------|----------|------|
| binary | Specifies the AISBench startup command.| string | No| Fixed value: `ais_bench`. Default value: `"ais_bench"`|
| mode | Specifies the evaluation mode.| string | No| The specific evaluation mode to execute. Default value: `"all"`|
| timeout | Specifies the timeout duration for command execution.| int | No| The timeout duration (in seconds) must be greater than 0. Default value: `7200` (2 hours)|
| cleanup_model_config | Specifies whether to clear the model configuration.| bool | No| Specifies whether to clear the generated model configuration files. Default value: `true`|
| model_meta | Specifies the model metadata configuration.| object | No| For details about the model metadata configuration, see the following section. Default value: `ModelConfigMeta()`|
| request_rate | Specifies the default request rate.| float | No| The default request rate must be greater than 0. Default value: `1.0`|
| pred_postprocessor | Specifies the prediction post-processor.| string | No| The name of the prediction post-processor. Default value: `"extract_non_reasoning_content"`|
| retry | Specifies the maximum number of request retries.| int | No| The number of request retries must be greater than or equal to 0. Default value: `2`|
| batch_size | Specifies the batch size.| int | No| The batch size for data processing must be greater than 0. Default value: `1`|
| max_out_len | Specifies the maximum output length.| int | No| The maximum output length must be greater than 0. Default value: `512`|
| trust_remote_code | Specifies whether to trust remote code.| bool | No| Specifies whether to trust remote code. Default value: `false`|
| generation_kwargs | Specifies generation parameters for the inference backend.| dict | No| A dictionary containing generation configuration parameters. Default value: `{}`|
| extra_args | Specifies extra command-line arguments.| list | No| A list of additional command-line arguments to append. Default value: `[]`|
| log_dir | Specifies the log directory path.| string | No| An empty string indicates that the tool uses the default system path. Default value: `""`|

**`model_meta` Fields**

The following table describes parameters used to obtain the service-oriented inference backend model configuration.

| Field| Purpose| Type| Mandatory (Yes/No)| Description|
|--------|------|------|----------|------|
| directory | Specifies the model configuration directory path.| string | No| The explicit path to the model configuration directory. An empty string indicates that the tool uses the default system path. Default value: `""`|
| subdir | Specifies the subdirectory for the service-oriented model backend configuration.| string | No| The subdirectory name for the service-oriented model backend configuration. Default value: `"vllm_api"`|
| base_name | Specifies the base name for the service-oriented model backend configuration.| string | No| The base name for the service-oriented model backend configuration. Default value: `"vllm_api_general_chat"`|
| name_suffix | Specifies the name suffix for the service-oriented model backend configuration.| string | No| The value `auto` triggers automatic generation. Default value: `"auto"`|
| abbr | Specifies the model configuration abbreviation.| string | No| The abbreviated name used for the model configuration. Default value: `"vllm-api-general-chat"`|
| attr | Specifies the model configuration attribute.| string | No| The attribute tag for the model configuration. Default value: `"service"`|

**Note**: Most of the preceding parameters correspond directly to AISBench command-line parameters and service-oriented inference backend settings. For comprehensive configuration options, see the [AISBench Detailed Parameter Description](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/all_params/index.html).

##### 2.2.3.2 `datasets` - Dataset Configuration

**Purpose**: Configures the datasets to be evaluated and the parameters of each dataset in AISBench. The configuration must contain all datasets that appear in `demand.expectations`.

**`datasets` Fields**

This field specifies the mapping between different dataset keys and the dataset configurations in AISBench. The following example lists only three datasets (`gsm8k`, `aime25`, and `bfcl-simple`). For details about more datasets, see [Supported Dataset Types](https://github.com/AISBench/benchmark/blob/master/docs/source_en/get_started/datasets.md) in the AISBench documentation.

The following table describes the configuration fields of each dataset.

| Field| Purpose| Type| Mandatory (Yes/No)| Description|
|--------|------|------|----------|------|
| config_name | Specifies the configuration name in AISBench.| string | Yes| This field indicates the configuration name of the dataset in AISBench, which must be a non-empty string.|
| mode | Specifies the evaluation mode for the dataset.| string | No| An empty string enables global mode. Default value: `""`|
| request_rate | Specifies the request rate for the dataset.| float | No| A value of `0.0` applies the global default value. The value must be greater than or equal to 0. Default value: `0.0`|
| max_out_len | Specifies the maximum output length for this dataset.| int | No| `None`: applies the global default value. If this parameter is specified, the value must be greater than 0. Default value: `None`|
| returns_tool_calls | Specifies whether to return tool calls.| bool | No| `None` indicates that the field is not written. Default value: `None`|
| api_chat_type | Specifies the API chat type used by the dataset.| string | No| This field must be consistent with the API or request format required by the corresponding AISBench dataset configuration. Default value: `"VLLMCustomAPIChat"`|
| extra_args | Specifies additional command-line arguments for the dataset.| list | No| A list of additional command-line arguments to append. Default value: `[]`|

##### 2.2.3.3 (Optional) `precheck` - Pre-check Configuration

**Purpose**: Defines the pre-check configuration before formal evaluation, which is used to pre-verify the quantized model before each model evaluation iteration.

**Type**: `list`

**Note**: The `precheck` field contains a list where each element is a pre-check item. Each item includes a `type` field that specifies the pre-check type. If this field is configured and is not an empty list, the system executes a pre-check before starting formal evaluation.

**Supported pre-check type**: `expected_answer`

**`expected_answer` - Expected Answer Verification**

**Purpose**: Verifies whether the model output contains the expected answer content.

**Field Description**

| Field| Purpose| Type| Mandatory (Yes/No)| Description|
|--------|------|------|----------|------|
| type | Specifies the pre-check type.| string | Yes| Fixed value: `expected_answer`.|
| test_cases | Specifies the test case list.| list | No| This field contains a list of test cases in key-value pairs representing questions and answers. If omitted, the configuration applies a default test case: `{"What is 2+2?": "4"}`. Default value: `[{"What is 2+2?": "4"}]`|
| max_tokens | Specifies the maximum number of tokens.| int | No| The value must be greater than 0. Default value: `512`|
| timeout | Specifies the timeout duration.| float | No| This field defines the timeout duration in seconds, which must be greater than 0. Default value: `60.0`|

**`test_cases` Field**

The `test_cases` field uses a dictionary format of key-value pairs, where the key represents the question and the value represents the answer:

```yaml
test_cases:
  - "What is 2+2?": ["4", "four"] # Required content in the response: "4" or "four"
  - "What is the capital of China?": "Beijing" # Required content in the response: "Beijing"
```

**Format Description**

- Key (question): a string that contains the test message.
- Value (answer): a string or a list of strings.
  - String: A string value like `"4"` indicates that the expected response must contain "4".
  - List of strings: A list like `["4", "four"]` indicates that the expected response must contain either "4" or "four".

**Configuration Example**

```yaml
precheck:
  - type: expected_answer
    test_cases:
      - "What is 2+2?": ["4", "four"]
      - "What is the capital of China?": "Beijing"
    max_tokens: 1024
    timeout: 60.0
```

**Notes:**

- The system executes the pre-check function after starting the service and before evaluating the model in each iteration. The system runs all configured pre-check rules in sequence. If any pre-check fails, the system skips the formal evaluation for the current iteration, returns zeroed-out dataset results, and proceeds directly to the next iteration.
- Pre-checks quickly identify obvious issues to prevent wasting time on a full evaluation. If all pre-checks pass, the system proceeds to the formal accuracy evaluation.
- **English Q&A support only**: The pre-check function currently supports only English Q&A. You must provide test messages and expected answers in English.
- If you omit the `precheck` field or configure it as an empty list, the system skips the pre-check phase and proceeds directly to the formal evaluation.

#### 2.2.4 `inference_engine` - Inference Engine Configuration

**Purpose**: Defines the configuration parameters of the inference engine, which are used to start the quantized model as a service.

**Field Description**

| Field| Purpose| Type| Mandatory (Yes/No)| Description|
|--------|------|------|----------|------|
| type | Specifies the inference engine type.| string | Yes| Currently, only `vllm-ascend` is supported.|
| entrypoint | Specifies the service entry point.| string | No| This field requires a non-empty string. The default value `vllm.entrypoints.openai.api_server` designates the service entry point for the OpenAI-compatible API of vLLM. Alternative entries must match an executable module using the `-m` flag within your installed vLLM or vLLM-Ascend deployment.|
| env_vars | Specifies environment variables.| dict | No| This field configures the required environment variables. Default value: `{}`|
| served_model_name | Specifies the external service model name.| string | No| This field requires a non-empty string that acts as the model identifier for the external inference service. **This value must match the identifier specified in `evaluation.evaluation`.** Default value: `"served_model_name"`|
| host | Specifies the service host address.| string | No| This field defines the listening address of the inference service. Supported formats: `localhost`, IPv4, and IPv6. **This value must match the value specified on the evaluation side.** Default value: `"localhost"`|
| port | Specifies the service port.| int | No| This field defines the listening port of the inference service. Value range: 1 to 65535. **This value must match the value specified on the evaluation side.** Default value: `1234`|
| health_check_endpoint | Specifies the health check endpoint.| string | No| This field defines the **HTTP path** requested during readiness probing. The value must be identical to the URL that returns a success response from the active inference process. The default value `"/v1/models"` maps to the standard model list interface of typical OpenAI-compatible services. The value must start with a forward slash `/`. You can customize this value based on the actual routes of your deployed vLLM-Ascend cluster.|
| startup_timeout | Specifies the startup timeout duration.| int | No| The timeout duration (in seconds) must be greater than 0. Default value: `600`|
| args | Specifies startup arguments for the inference engine.| dict | No| This field allows you to append additional arguments for vLLM-Ascend. Default value: `{}`|

**Note**: The parameters required to launch a service vary depending on the model. You must tune these parameters based on your actual model requirements. To configure specific parameters, see [Configuration Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/configuration/index.html) in the vLLM-Ascend documentation. You can add custom startup arguments to the `args` field and define system environment variables within the `env_vars` dictionary.

**Configuration Example**

```yaml
inference_engine:
  type: vllm-ascend
  entrypoint: vllm.entrypoints.openai.api_server
  env_vars:
    HCCL_BUFFSIZE: 1024
    VLLM_VERSION: 0.11.0
    ASCEND_RT_VISIBLE_DEVICES: 0
  served_model_name: served_model_name
  host: localhost
  port: 1234
  health_check_endpoint: /v1/models
  startup_timeout: 600
  args:
    enforce-eager: true
    served-model-name: served_model_name
    trust-remote-code: true
    tensor-parallel-size: 1
    data-parallel-size: 1
    quantization: ascend
    enable-prefix-caching: false
    max-model-len: 8192
    max-num-batched-tokens: 8192
    gpu-memory-utilization: 0.9
    additional_config:
      ascend_scheduler_config:
        enable: true
      enable_weight_nz_layout: true
```

### 2.3 Examples

Refer to the following files for complete automatic tuning configuration examples:

- Configuration for the `standing_high` tuning strategy: [`standing_high.yaml`](./example/standing_high.yaml)
- Configuration for the `standing_high_with_experience` tuning strategy: [`standing_high_with_experience.yaml`](./example/standing_high_with_experience.yaml)
- Configuration for the `binary_fallback` tuning strategy: [binary_fallback.yaml](./example/binary_fallback.yaml)
