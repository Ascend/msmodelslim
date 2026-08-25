<!-- generated-by: skills/docs-management/scripts/gen_quant_config_docs.py ; class: msmodelslim.core.tune_strategy.standing_high.strategy.StandingHighStrategyConfig -->
# standing_high 配置说明

## 1. 配置概述

摸高算法策略配置（V1框架）：先跑敏感层分析，再逐步回退不满意的层并尝试不同离群值抑制策略。

| 项目 | 内容 |
|------|------|
| 配置类 | `StandingHighStrategyConfig` |
| 源码 | [strategy.py](../../../../../msmodelslim/core/tune_strategy/standing_high/strategy.py) |

## 2. 参数列表

<h3 id="2-1-strategy-standing-high">2.1 StandingHighStrategyConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `standing_high` | `standing_high` | 策略类型，固定为 `standing_high` | 无 |
| `anti_outlier_strategies` | `list[list[object]]` | 必选 | 无 | — | 离群值抑制处理器链列表，至少1个；每个元素是一条处理器链，链内每个处理器是 `type` 分派的处理器配置（如 smooth_quant 等） | 本页 <a href="#2-2-autoprocessorconfig">§2.2</a> |
| `template` | `object` | 可选 | 由工厂函数生成 | — | 完整的PracticeConfig模板，用于提取所有配置（包括线性层量化）。如果未提供，将使用默认的V1模板 | 《[modelslim_v1_spec 配置说明](../task/modelslim_v1.md#2-2-modelslim-v1-spec)》 |
| `metadata` | `object` | 可选 | 见嵌套配置默认值 | — | 量化配置元数据（config_id / label 等） | 本页 <a href="#2-3-metadata">§2.3</a> |

**配置约束**

- 校验template中至少有一个linear_quant配置

<h3 id="2-2-autoprocessorconfig">2.2 AutoProcessorConfig</h3>

**派生类**

- `AdaptRotationProcessorConfig`（`type: adapt_rotation`） — 自适应旋转（adapt_rotation）处理器配置。 《[adapt_rotation 配置说明](../processor/adapt_rotation.md)》
- `AutoroundProcessorConfig`（`type: autoround_quant`） — autoround 量化处理器配置。 《[autoround_quant 配置说明](../processor/autoround_quant.md)》
- `AWQProcessorConfig`（`type: awq`） — AWQ（Activation-aware Weight Quantization）处理器配置。 《[awq 配置说明](../processor/awq.md)》
- `BinaryAnalysisProcessorConfig`（`type: binary_analysis`） — 二值（有/无量化）敏感度分析处理器配置。 《[binary_analysis 配置说明](../processor/binary_analysis.md)》
- `BinaryOperatorLayerWiseProcessorConfig`（`type: binary_operator_layer_wise`） — 逐层敏感度分析处理器配置（对比逐块浮点与量化输出）。 《[binary_operator_layer_wise 配置说明](../processor/binary_operator_layer_wise.md)》
- `BinaryOperatorModelWiseProcessorConfig`（`type: binary_operator_model_wise`） — 模型级敏感度分析配置（对比模型最终输出，使用 MSE 指标） 《[binary_operator_model_wise 配置说明](../processor/binary_operator_model_wise.md)》
- `DynamicCacheProcessorConfig`（`type: dynamic_cache`） — KV cache 量化处理器配置。 《[dynamic_cache 配置说明](../processor/dynamic_cache.md)》
- `FA3QuantProcessorConfig`（`type: fa3_quant`） — FA3（FlashAttention-3）量化处理器配置。 《[fa3_quant 配置说明](../processor/fa3_quant.md)》
- `FlatQuantProcessorConfig`（`type: flatquant`） — FlatQuant处理器配置：定义量化训练参数、策略、混合精度等 《[flatquant 配置说明](../processor/flatquant.md)》
- `FlexAWQSSZProcessorConfig`（`type: flex_awq_ssz`） — FlexAWQSSZ 平滑+AWQ 处理器配置。 《[flex_awq_ssz 配置说明](../processor/flex_awq_ssz.md)》
- `FlexSmoothQuantProcessorConfig`（`type: flex_smooth_quant`） — FlexSmoothQuant 平滑量化处理器配置。 《[flex_smooth_quant 配置说明](../processor/flex_smooth_quant.md)》
- `FloatSparseProcessorConfig`（`type: float_sparse`） — 浮点稀疏处理器配置。 《[float_sparse 配置说明](../processor/float_sparse.md)》
- `GroupProcessorConfig`（`type: group`） — 处理器合并器配置。 《[group 配置说明](../processor/group.md)》
- `IterSmoothProcessorConfig`（`type: iter_smooth`） — 迭代平滑（IterativeSmooth）处理器配置。 《[iter_smooth 配置说明](../processor/iter_smooth.md)》
- `KVSmoothProcessorConfig`（`type: kv_smooth`） — KV cache 平滑处理器配置。 《[kv_smooth 配置说明](../processor/kv_smooth.md)》
- `LinearProcessorConfig`（`type: linear_quant`） — 线性层（Linear）量化处理器配置。 《[linear_quant 配置说明](../processor/linear_quant.md)》
- `LoadProcessorConfig`（`type: load`） — 模块加载/卸载处理器配置。 《[load 配置说明](../processor/load.md)》
- `OASQProcessorConfig`（`type: oasq`） — OASQ（Outlier-Aware Smooth Quantization）处理器配置。 《[oasq 配置说明](../processor/oasq.md)》
- `OnlineQuaRotProcessorConfig`（`type: online_quarot`） — 在线 QuaRot 旋转处理器配置。 《[online_quarot 配置说明](../processor/online_quarot.md)》
- `QuaRotProcessorConfig`（`type: quarot`） — QuaRot（离线旋转）处理器配置。 《[quarot 配置说明](../processor/quarot.md)》
- `QuantSaveProcessorConfig`（`type: saver`） — 统一保存处理器配置。 《[saver 配置说明](../processor/saver.md)》
- `SmoothQuantProcessorConfig`（`type: smooth_quant`） — SmoothQuant 平滑量化处理器配置。 《[smooth_quant 配置说明](../processor/smooth_quant.md)》
- `SVDResidualProcessorConfig`（`type: svd_res`） — SVD 残差（低秩补偿）处理器配置。 《[svd_res 配置说明](../processor/svd_res.md)》
- `TrainableLinearQuantProcessorConfig`（`type: trainable_linear_quant`） — 可训练线性量化（TLQ）处理器配置。 《[trainable_linear_quant 配置说明](../processor/trainable_linear_quant.md)》
- `UnaryAnalysisProcessorConfig`（`type: unary_analysis`） — 一元（无量化）敏感度分析处理器配置。 《[unary_analysis 配置说明](../processor/unary_analysis.md)》

<h3 id="2-3-metadata">2.3 Metadata</h3>

量化配置元数据：标识配置的 ID、评分、标签与已验证的模型/场景。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `config_id` | `string` | 可选 | `Unknown` | — | 量化配置 ID，例如 'Qwen3-32B W8A8' | 无 |
| `score` | `float` | 可选 | `100.0` | — | 量化配置评分，用于排序，必须 >= 0 | 无 |
| `label` | `object` | 可选 | `{}` | — | 量化配置标签，用于过滤，例如 {'w_bit': 8, 'a_bit': 8, 'is_sparse': True, 'kv_cache': True} | 无 |
| `verified_model_types` | `list[string]` | 可选 | `[]` | — | 已验证的模型类型列表，例如 ['LLaMa3.1-70B', 'Qwen2.5-72B'] | 无 |
| `verified_tags` | `object` | 可选 | `{}` | — | 已验证场景标签：键为模型类型，值为场景标签列表（每个场景是一组标签，如 ['MindIE','Atlas_A2_Inference']） | 无 |

**配置约束**

- 无。

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
