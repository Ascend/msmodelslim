<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.infra.service_oriented_evaluate_service.ServiceOrientedEvaluateServiceConfig -->
# evaluation_service_oriented Configuration Reference

## 1. Configuration Overview

Service-oriented evaluation service configuration: evaluation demand + aisbench evaluation + vLLM-Ascend inference engine.

| Item | Content |
|------|------|
| Config Class | `ServiceOrientedEvaluateServiceConfig` |
| Source | [service_oriented_evaluate_service.py](../../../../../msmodelslim/infra/service_oriented_evaluate_service.py) |

## 2. Parameter List

<h3 id="2-1-evaluation-service-oriented">2.1 ServiceOrientedEvaluateServiceConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `service_oriented` | — | Evaluation service type, fixed to `service_oriented` | None |
| `demand` | `object` | Required | None | — | Evaluation demand (dataset accuracy expectations) | This page <a href="#2-2-evaluate-demand">§2.2</a> |
| `evaluation` | `object` | Required | None | — | AISBench evaluation service configuration | This page <a href="#2-3-aisbench">§2.3</a> |
| `inference_engine` | `object` | Required | None | — | vLLM-Ascend inference engine configuration | This page <a href="#2-4-vllm-ascend">§2.4</a> |

**Configuration Constraints**

- Validates that every dataset in expectations is configured in evaluation.datasets

<h3 id="2-2-evaluate-demand">2.2 EvaluateDemand</h3>

Evaluation demand: declares which accuracy expectations must be met on which datasets.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `expectations` | `list[object]` | Required | None | — | List of accuracy expectations, at least 1; each item declares a dataset and the target accuracy (including tolerance) | This page <a href="#2-2-1-accuracy-expectation">§2.2.1</a> |

**Configuration Constraints**

- None.

<h4 id="2-2-1-accuracy-expectation">2.2.1 AccuracyExpectation</h4>

Accuracy expectation: the target accuracy (including tolerance) that the model must reach on the specified dataset.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `dataset` | `string` | Required | None | — | Dataset name | None |
| `target` | `string` | Required | None | `^(?!^[-+.]*$)[+-]?0*\d*\.?\d*$` | Target accuracy; must be > 0 | None |
| `tolerance` | `string` | Required | None | `^(?!^[-+.]*$)[+-]?0*\d*\.?\d*$` | Deviation tolerated relative to the target accuracy; must be >= 0 | None |

**Configuration Constraints**

- None.

<h3 id="2-3-aisbench">2.3 AisbenchServerConfig</h3>

AISBench evaluation service configuration

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `aisbench` | `aisbench` | Evaluation service type, fixed to `aisbench` | None |
| `aisbench` | `object` | Optional | See the nested configuration default | — | AISBench evaluation configuration | This page <a href="#2-3-1-aisbench-config">§2.3.1</a> |
| `datasets` | `object` | Optional | `{}` | — | Dataset configuration dictionary, keyed by dataset name | This page <a href="#2-3-3-dataset-config">§2.3.3</a> |
| `host` | `string` | Optional | `localhost` | — | Listening address of the evaluation service; must be a valid host name/IP | None |
| `port` | `int` | Optional | `1234` | — | Listening port of the evaluation service; must be a valid port number | None |
| `served_model_name` | `string` | Optional | `served_model_name` | — | Name of the deployed model | None |
| `precheck` | `list[object]` | Optional | `[]` | — | List of model pre-check configurations; each element is a dictionary containing a 'type' field ('garbled_text' or 'expected_answer') | This page <a href="#2-3-4-base-precheck-config">§2.3.4</a> |

**Configuration Constraints**

- None.

<h4 id="2-3-1-aisbench-config">2.3.1 AisbenchConfig</h4>

AISBench evaluation configuration

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `binary` | `string` | Optional | `ais_bench` | `ais_bench` | aisbench startup command, fixed to 'ais_bench' | None |
| `mode` | `string` | Optional | `all` | — | Evaluation mode | None |
| `timeout` | `int` | Optional | `7200` | — | Command execution timeout in seconds; the default is 2 hours | None |
| `cleanup_model_config` | `bool` | Optional | `true` | — | Whether to clean up the generated model configuration files | None |
| `model_meta` | `object` | Optional | See the nested configuration default | — | Model configuration metadata | This page <a href="#2-3-2-model-config-meta">§2.3.2</a> |
| `request_rate` | `float` | Optional | `1.0` | — | Default request rate; must be > 0 | None |
| `pred_postprocessor` | `string` | Optional | `extract_non_reasoning_content` | — | Name of the prediction postprocessor | None |
| `retry` | `int` | Optional | `2` | ≥0 | Number of request retries; must be >= 0 | None |
| `batch_size` | `int` | Optional | `1` | — | Batch size; must be > 0 | None |
| `max_out_len` | `int` | Optional | `512` | — | Maximum output length; must be > 0 | None |
| `trust_remote_code` | `bool` | Optional | `false` | — | Whether to trust remote code | None |
| `generation_kwargs` | `object` | Optional | `{}` | — | Dictionary of generation parameters | None |
| `extra_args` | `list[string]` | Optional | `[]` | — | Additional command-line argument list; the default is an empty list | None |
| `log_dir` | `string` | Optional | `` | — | Log directory path; an empty string means the default path is used | None |

**Configuration Constraints**

- None.

<h4 id="2-3-2-model-config-meta">2.3.2 ModelConfigMeta</h4>

Model configuration metadata

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `directory` | `string` | Optional | `` | — | Explicit path of the model configuration directory; an empty string means the default path is used | None |
| `subdir` | `string` | Optional | `vllm_api` | — | Model configuration subdirectory | None |
| `base_name` | `string` | Optional | `vllm_api_general_chat` | — | Base name of the model configuration | None |
| `name_suffix` | `string` | Optional | `auto` | — | Name suffix of the model configuration; 'auto' means generated automatically | None |
| `abbr` | `string` | Optional | `vllm-api-general-chat` | — | Abbreviation of the model configuration | None |
| `attr` | `string` | Optional | `service` | — | Model configuration attribute | None |

**Configuration Constraints**

- None.

<h4 id="2-3-3-dataset-config">2.3.3 DatasetConfig</h4>

Evaluation configuration of a single dataset

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `config_name` | `string` | Required | None | — | Configuration name of the dataset in ais_bench (required) | None |
| `mode` | `string` | Optional | `` | — | Evaluation mode of this dataset; an empty string means the global mode is used | None |
| `request_rate` | `float` | Optional | `0.0` | ≥0.0 | Request rate of this dataset; 0.0 means the global default is used | None |
| `max_out_len` | `int / null` | Optional | `null` | — | Maximum output length of this dataset; None means the global default is used | None |
| `returns_tool_calls` | `bool / null` | Optional | `null` | — | Whether to return tool calls; None means the field is not written | None |
| `api_chat_type` | `string` | Optional | `VLLMCustomAPIChat` | — | API Chat type used by this dataset | None |
| `chat_template_kwargs` | `object` | Optional | `{}` | — | Additional parameters for chat_template; refer to the AISBench documentation for the specific parameters | None |
| `extra_args` | `list[string]` | Optional | `[]` | — | Additional command-line argument list for this dataset; the default is an empty list | None |

**Configuration Constraints**

- None.

<h4 id="2-3-4-base-precheck-config">2.3.4 BasePrecheckConfig</h4>

Pre-check configuration base class

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Required | None | — | Pre-check type, dispatched by the `type` field, such as `expected_answer` | None |
| `max_tokens` | `int` | Optional | `512` | — | Maximum number of generated tokens; must be greater than 0 | None |
| `timeout` | `float` | Optional | `60.0` | — | API call timeout in seconds; must be greater than 0 | None |

**Configuration Constraints**

- None.

**Derived Classes**

| Config Class | `type` | Description | Document |
|--------|----------|------|------|
| `ExpectedAnswerPrecheckConfig` | `expected_answer` | Expected-answer verification pre-check configuration | This page <a href="#2-3-5-expected-answer">§2.3.5</a> |

<h4 id="2-3-5-expected-answer">2.3.5 ExpectedAnswerPrecheckConfig</h4>

Expected-answer verification pre-check configuration

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `expected_answer` | `expected_answer` | — | None |
| `max_tokens` | `int` | Optional | `512` | — | Maximum number of generated tokens; must be greater than 0 | None |
| `timeout` | `float` | Optional | `60.0` | — | API call timeout in seconds; must be greater than 0 | None |
| `test_cases` | `list[object]` | Optional | `[TestCase(message='What is 2+2?', expected_answer='4')]` | At least 1 item | List of test cases in dictionary (key-value) format; the content must be in English and contain at least 1 item | This page <a href="#2-3-6-test-case">§2.3.6</a> |

**Configuration Constraints**

- None.

<h4 id="2-3-6-test-case">2.3.6 TestCase</h4>

Pre-check test case

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `message` | `string` | Required | None | — | Test message; must be a non-empty string | None |
| `expected_answer` | `string / list[string] / null` | Optional | `null` | — | Content expected to appear in the model response; can be a string or a list of strings | None |

**Configuration Constraints**

- None.

<h3 id="2-4-vllm-ascend">2.4 VllmAscendConfig</h3>

vLLM-Ascend inference engine configuration: used to start an OpenAI-compatible server and perform health checks.

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `vllm-ascend` | `vllm-ascend` | Inference engine type, fixed to `vllm-ascend` | None |
| `entrypoint` | `string` | Optional | `vllm.entrypoints.openai.api_server` | — | vLLM service startup entry point; defaults to the OpenAI API server | None |
| `env_vars` | `object` | Optional | `{}` | — | Dictionary of additional environment variables (key-value pairs) passed to the vLLM process; refer to the vLLM-Ascend documentation for the specific parameters | None |
| `served_model_name` | `string` | Optional | `served_model_name` | — | Name of the deployed/exposed model | None |
| `host` | `string` | Optional | `localhost` | — | Service listening address | None |
| `port` | `int` | Optional | `1234` | — | Service listening port | None |
| `health_check_endpoint` | `string` | Optional | `/v1/models` | — | Health check endpoint path (vLLM OpenAI-compatible) | None |
| `startup_timeout` | `int` | Optional | `600` | — | Service startup timeout in seconds; must be > 0 | None |
| `args` | `object` | Optional | `{}` | — | Additional vLLM startup command-line arguments (key-value pairs); refer to the vLLM-Ascend documentation for the specific parameters | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
strategy:
  type: standing_high
  anti_outlier_strategies:
  - - type: iter_smooth
      alpha: 0.5
  - - type: flex_smooth_quant
  template:
    runner: auto
    process:
    - type: linear_quant
      qconfig:
        act:
          scope: per_tensor
          dtype: int8
          symmetric: false
          method: minmax
        weight:
          scope: per_channel
          dtype: int8
          symmetric: true
          method: minmax
      include:
      - '*'
      exclude: []
    save:
    - type: ascendv1_saver
      part_file_size: 4
    dataset: mix_calib.jsonl
  metadata:
    config_id: standing_high
    label:
      w_bit: 8
      a_bit: 8
      is_sparse: false
      kv_cache: false
evaluation:
  type: service_oriented
  demand:
    expectations:
    - dataset: gsm8k
      target: '83'
      tolerance: '2'
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
    datasets:
      gsm8k:
        config_name: gsm8k_gen_0_shot_cot_str
        mode: all
      aime25:
        config_name: aime2025_gen_0_shot_chat_prompt
        mode: all
        chat_template_kwargs:
          thinking: true
      bfcl-simple:
        config_name: BFCL_gen_simple
        mode: all
        max_out_len: 1024
        returns_tool_calls: true
        api_chat_type: VLLMFunctionCallAPIChat
    host: localhost
    port: 1234
    served_model_name: served_model_name
  inference_engine:
    type: vllm-ascend
    entrypoint: vllm.entrypoints.openai.api_server
    env_vars:
      HCCL_BUFFSIZE: 1024
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
      enable-auto-tool-choice: true
      tool-call-parser: hermes
      additional_config:
        ascend_scheduler_config:
          enable: true
        enable_weight_nz_layout: true
```
