<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.core.tune_strategy.binary_fallback.strategy.BinaryFallbackStrategyConfig -->
# binary_fallback 配置说明

## 1. 配置概述

二分回退调优策略配置。

| 项目 | 内容 |
|------|------|
| 配置类 | `BinaryFallbackStrategyConfig` |
| 源码 | [strategy.py](../../../../../msmodelslim/core/tune_strategy/binary_fallback/strategy.py) |

## 2. 参数列表

<h3 id="2-1-strategy-binary-fallback">2.1 BinaryFallbackStrategyConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `binary_fallback` | `binary_fallback` | 策略类型，固定为 `binary_fallback` | 无 |
| `template` | `object` | 必选 | 无 | — | 完整最佳实践 PracticeConfig，apiversion 须为 modelslim_v1 | 《[PracticeConfig 配置说明](../task/practice_config.md)》 |
| `rollback_path` | `string` | 必选 | 无 | — | 点分路径，指向 template 内必须为 list 的回退字段 | 无 |
| `rollback_candidates` | `list[string] / null` | 可选 | `null` | — | 有序回退候选；非空则跳过敏感层分析 | 无 |
| `analysis_dataset` | `string / null` | 可选 | `null` | — | 敏感层分析校准集名称；未填则使用 template.spec.dataset | 无 |

**配置约束**

- 无。

## 3. 完整配置参考

```yaml
strategy:
  type: binary_fallback
  rollback_path: spec.process.1.exclude
  rollback_candidates: []
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
