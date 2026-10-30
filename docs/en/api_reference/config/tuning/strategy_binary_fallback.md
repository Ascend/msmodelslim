<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.core.tune_strategy.binary_fallback.strategy.BinaryFallbackStrategyConfig -->
# binary_fallback Configuration Reference

## 1. Configuration Overview

Binary fallback tuning strategy configuration.

| Item | Content |
|------|------|
| Config Class | `BinaryFallbackStrategyConfig` |
| Source | [strategy.py](../../../../../msmodelslim/core/tune_strategy/binary_fallback/strategy.py) |

## 2. Parameter List

<h3 id="2-1-strategy-binary-fallback">2.1 BinaryFallbackStrategyConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `binary_fallback` | `binary_fallback` | Strategy type, fixed to `binary_fallback` | None |
| `template` | `object` | Required | None | — | Complete best-practice PracticeConfig; apiversion must be modelslim_v1 | [PracticeConfig Configuration Reference](../quant/practice_config.md) |
| `rollback_path` | `string` | Required | None | — | Dot-separated path pointing to a field inside template that must be a list | None |
| `rollback_candidates` | `list[string] / null` | Optional | `null` | — | Ordered fallback candidates; when non-empty, sensitive layer analysis is skipped | None |
| `analysis_dataset` | `string / null` | Optional | `null` | — | Name of the calibration set for sensitive layer analysis; when empty, template.spec.dataset is used | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
strategy:
  type: binary_fallback
  rollback_path: spec.process.1.exclude
  rollback_candidates: []
  analysis_dataset: mix_calib.jsonl
  template:
    apiversion: modelslim_v1
    metadata:
      config_id: binary_fallback_tune
      label:
        w_bit: 8
        a_bit: 8
        is_sparse: false
        kv_cache: false
    spec:
      runner: auto
      process:
      - type: iter_smooth
        alpha: 0.5
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
