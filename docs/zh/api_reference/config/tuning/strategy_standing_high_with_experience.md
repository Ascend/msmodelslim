<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.core.tune_strategy.standing_high_with_experience.strategy.StandingHighWithExperienceStrategyConfig -->
# standing_high_with_experience 配置说明

## 1. 配置概述

基于专家经验的摸高算法策略配置

| 项目 | 内容 |
|------|------|
| 配置类 | `StandingHighWithExperienceStrategyConfig` |
| 源码 | [strategy.py](../../../../../msmodelslim/core/tune_strategy/standing_high_with_experience/strategy.py) |

## 2. 参数列表

<h3 id="2-1-strategy-standing-high-with-experience">2.1 StandingHighWithExperienceStrategyConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `standing_high_with_experience` | `standing_high_with_experience` | 策略类型，固定为 `standing_high_with_experience` | 无 |
| `structure_configs` | `list[object]` | 必选 | 无 | — | 结构配置列表，每个配置包含结构类型和对应的 include/exclude，例如 [{'type': 'GQA', 'include': ['*self_attn*'], 'exclude': ['*kv_b_proj']}]（必填，无默认配置） | 本页 <a href="#2-2-structure-config">§2.2</a> |
| `quant_type` | `string` | 可选 | `w8a8` | `w4a4`、`w4a8`、`w4a4c8`、`w4a4f8`、`w4a8c8`、`w8a16`、`w8a8`、`w8a8s`、`w8a8c8`、`w8a8f8`、`w4a4f4`、`w16a16s` | 量化类型（QuantType 枚举），须在专家经验 supported_quant_types 范围内。 | 无 |

**配置约束**

- 无。

<h3 id="2-2-structure-config">2.2 StructureConfig</h3>

结构配置，定义模型结构的量化配置

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 必选 | 无 | 长度 ≥ 1 | 结构类型，非空，例如 'GQA', 'MoE', 'FFN' | 无 |
| `include` | `list[string]` | 必选 | 无 | 最少1项 | 包含的模式列表，必选，不可为空且不能包含空字符串，例如 ['*self_attn*'] | 无 |
| `exclude` | `list[string] / null` | 可选 | `null` | — | 排除的模式列表，例如 ['*kv_b_proj', '*wq_b'] | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

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
        chat_template_kwargs:
          thinking: true
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
