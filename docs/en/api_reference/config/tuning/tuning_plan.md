<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.app.auto_tuning.plan_manager_infra.TuningPlanConfig -->
# tuning_plan Configuration Reference

## 1. Configuration Overview

Auto-tuning plan configuration: the top level contains only two fields, strategy (the tuning strategy) and evaluation (the evaluation service).

| Item | Content |
|------|------|
| Config Class | `TuningPlanConfig` |
| Source | [plan_manager_infra.py](../../../../../msmodelslim/app/auto_tuning/plan_manager_infra.py) |

## 2. Parameter List

<h3 id="2-1-tuning-plan">2.1 TuningPlanConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `strategy` | `object` | Required | None | — | Tuning strategy configuration, dispatched by the `type` field, such as `standing_high`, `standing_high_with_experience`, `binary_fallback` | This page <a href="#2-2-strategyconfig">§2.2</a> |
| `evaluation` | `object` | Required | None | — | Evaluation service configuration, dispatched by the `type` field, such as `service_oriented` | This page <a href="#2-3-evaluateserviceconfig">§2.3</a> |

**Configuration Constraints**

- None.

<h3 id="2-2-strategyconfig">2.2 StrategyConfig</h3>

**Derived Classes**

| Config Class | `type` | Description | Document |
|--------|----------|------|------|
| `StandingHighStrategyConfig` | `standing_high` | Standing high algorithm strategy configuration (V1 framework): runs sensitive layer analysis first, then rolls back the unsatisfactory layers progressively and tries different outlier suppression strategies. | [standing_high Configuration Reference](strategy_standing_high.md) |
| `StandingHighWithExperienceStrategyConfig` | `standing_high_with_experience` | Experience-based Standing high algorithm strategy configuration | [standing_high_with_experience Configuration Reference](strategy_standing_high_with_experience.md) |
| `BinaryFallbackStrategyConfig` | `binary_fallback` | Binary fallback tuning strategy configuration. | [binary_fallback Configuration Reference](strategy_binary_fallback.md) |

<h3 id="2-3-evaluateserviceconfig">2.3 EvaluateServiceConfig</h3>

**Derived Classes**

| Config Class | Description | Document |
|--------|------|------|
| `ServiceOrientedEvaluateServiceConfig` | Service-oriented evaluation service configuration: evaluation demand + aisbench evaluation + vLLM-Ascend inference engine. | [evaluation_service_oriented Configuration Reference](evaluation_service_oriented.md) |

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
