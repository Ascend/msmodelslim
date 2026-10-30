<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.core.tune_strategy.standing_high_with_experience.strategy.StandingHighWithExperienceStrategyConfig -->
# standing_high_with_experience Configuration Reference

## 1. Configuration Overview

Experience-based Standing high algorithm strategy configuration

| Item | Content |
|------|------|
| Config Class | `StandingHighWithExperienceStrategyConfig` |
| Source | [strategy.py](../../../../../msmodelslim/core/tune_strategy/standing_high_with_experience/strategy.py) |

## 2. Parameter List

<h3 id="2-1-strategy-standing-high-with-experience">2.1 StandingHighWithExperienceStrategyConfig</h3>

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Optional | `standing_high_with_experience` | `standing_high_with_experience` | Strategy type, fixed to `standing_high_with_experience` | None |
| `structure_configs` | `list[object]` | Required | None | — | List of structure configurations; each configuration contains a structure type and the corresponding include/exclude, for example [{'type': 'GQA', 'include': ['*self_attn*'], 'exclude': ['*kv_b_proj']}] (required, no default) | This page <a href="#2-2-structure-config">§2.2</a> |
| `quant_type` | `string` | Optional | `w8a8` | `w4a4`, `w4a8`, `w4a4c8`, `w4a4f8`, `w4a8c8`, `w8a16`, `w8a8`, `w8a8s`, `w8a8c8`, `w8a8f8`, `w4a4f4`, `w16a16s` | Quantization type (QuantType enum); must be within the expert experience's supported_quant_types. | None |

**Configuration Constraints**

- None.

<h3 id="2-2-structure-config">2.2 StructureConfig</h3>

Structure configuration; defines the quantization configuration of a model structure

| Field Path | Type | Required/Optional | Default | Value Range or Format | Description | Referenced Config |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | Required | None | Length ≥ 1 | Structure type; non-empty, for example 'GQA', 'MoE', 'FFN' | None |
| `include` | `list[string]` | Required | None | At least 1 item | List of patterns to include; required, must not be empty and must not contain an empty string, for example ['*self_attn*'] | None |
| `exclude` | `list[string] / null` | Optional | `null` | — | List of patterns to exclude, for example ['*kv_b_proj', '*wq_b'] | None |

**Configuration Constraints**

- None.

## 3. Complete Configuration Reference

```yaml
strategy:
  type: standing_high_with_experience
  quant_type: w8a8
  structure_configs:
  - type: GQA
    include:
    - '*self_attn*'
  - type: FFN
    include:
    - '*mlp*'
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
