<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.app.auto_tuning.plan_manager_infra.TuningPlanConfig -->
# tuning_plan 配置说明

## 1. 配置概述

自动调优计划配置：顶层只含 strategy（调优策略）与 evaluation（评估服务）两个字段。

| 项目 | 内容 |
|------|------|
| 配置类 | `TuningPlanConfig` |
| 源码 | [plan_manager_infra.py](../../../../../msmodelslim/app/auto_tuning/plan_manager_infra.py) |

## 2. 参数列表

<h3 id="2-1-tuning-plan">2.1 TuningPlanConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `strategy` | `object` | 必选 | 无 | — | 调优策略配置，按 `type` 字段分派，如 `standing_high`、`standing_high_with_experience`、`binary_fallback` | 本页 <a href="#2-2-strategyconfig">§2.2</a> |
| `evaluation` | `object` | 必选 | 无 | — | 评估服务配置，按 `type` 字段分派，如 `service_oriented` | 本页 <a href="#2-3-evaluateserviceconfig">§2.3</a> |

**配置约束**

- 无。

<h3 id="2-2-strategyconfig">2.2 StrategyConfig</h3>

**派生类**

- `StandingHighStrategyConfig`（`type: standing_high`） — 摸高算法策略配置（V1框架）：先跑敏感层分析，再逐步回退不满意的层并尝试不同离群值抑制策略。 《[standing_high 配置说明](strategy_standing_high.md)》
- `StandingHighWithExperienceStrategyConfig`（`type: standing_high_with_experience`） — 基于专家经验的摸高算法策略配置 《[standing_high_with_experience 配置说明](strategy_standing_high_with_experience.md)》
- `BinaryFallbackStrategyConfig`（`type: binary_fallback`） — 二分回退调优策略配置。 《[binary_fallback 配置说明](strategy_binary_fallback.md)》

<h3 id="2-3-evaluateserviceconfig">2.3 EvaluateServiceConfig</h3>

**派生类**

- `ServiceOrientedEvaluateServiceConfig` — 面向服务的评估服务配置：评估需求 + aisbench 评测 + vLLM-Ascend 推理引擎。 《[evaluation_service_oriented 配置说明](evaluation_service_oriented.md)》

## 3. 完整配置参考

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
