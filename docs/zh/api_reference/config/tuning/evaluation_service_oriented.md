<!-- generated-by: tools.config_docgen ; class: msmodelslim.infra.service_oriented_evaluate_service.ServiceOrientedEvaluateServiceConfig -->
> **自动生成文档，请勿手工修改。**
>
> 本页由 `tools.config_docgen` 根据源码注解生成。
> 修改字段描述、类概述或校验逻辑后，请重新生成并提交生成稿。

# evaluation_service_oriented 配置说明

## 1. 配置概述

面向服务的评估服务配置：评估需求 + aisbench 评测 + vLLM-Ascend 推理引擎。

| 项目 | 内容 |
|------|------|
| 配置类 | `ServiceOrientedEvaluateServiceConfig` |
| 源码 | [service_oriented_evaluate_service.py](../../../../../msmodelslim/infra/service_oriented_evaluate_service.py) |

## 2. 参数列表

<h3 id="2-1-evaluation-service-oriented">2.1 ServiceOrientedEvaluateServiceConfig</h3>

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `service_oriented` | — | 评估服务类型，固定为 `service_oriented` | 无 |
| `demand` | `object` | 必选 | 无 | — | 评估需求（数据集精度期望） | 本页 <a href="#2-2-evaluate-demand">§2.2</a> |
| `evaluation` | `object` | 必选 | 无 | — | AISBench 评测服务配置 | 本页 <a href="#2-3-aisbench">§2.3</a> |
| `inference_engine` | `object` | 必选 | 无 | — | vLLM-Ascend 推理引擎配置 | 本页 <a href="#2-4-vllm-ascend">§2.4</a> |

**配置约束**

- 校验 expectations 中的所有 dataset 都在 evaluation.datasets 中配置了

<h3 id="2-2-evaluate-demand">2.2 EvaluateDemand</h3>

评估需求：声明需要在哪些数据集上达到哪些精度期望。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `expectations` | `list[object]` | 必选 | 无 | 列表至少包含 1 项 | 精度期望列表，至少1个；每项声明数据集与目标精度（含容差） | 本页 <a href="#2-2-1-accuracy-expectation">§2.2.1</a> |

**配置约束**

- 无。

<h4 id="2-2-1-accuracy-expectation">2.2.1 AccuracyExpectation</h4>

精度期望：要求模型在指定数据集上达到的目标精度（含容差）。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `dataset` | `string` | 必选 | 无 | 字符串长度 ≤ 200 | 数据集名称 | 无 |
| `target` | `string` | 必选 | 无 | `^(?!^[-+.]*$)[+-]?0*\d*\.?\d*$`；>0 | 目标精度，必须 > 0 | 无 |
| `tolerance` | `string` | 必选 | 无 | `^(?!^[-+.]*$)[+-]?0*\d*\.?\d*$`；≥0 | 相对目标精度可容忍的偏差，必须 >= 0 | 无 |

**配置约束**

- 无。

<h3 id="2-3-aisbench">2.3 AisbenchServerConfig</h3>

AISBench 评测服务配置

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `aisbench` | `aisbench` | 评测服务类型，固定为 `aisbench` | 无 |
| `aisbench` | `object` | 可选 | 见嵌套配置默认值 | — | AISBench 评测配置 | 本页 <a href="#2-3-1-aisbench-config">§2.3.1</a> |
| `datasets` | `object` | 可选 | `{}` | — | 数据集配置字典，键为数据集名称 | 本页 <a href="#2-3-3-dataset-config">§2.3.3</a> |
| `host` | `string` | 可选 | `localhost` | 合法主机名/IP，仅允许本机、回环或私有/链路本地地址 | 评测服务监听地址，须为合法主机名/IP | 无 |
| `port` | `int` | 可选 | `1234` | 1–65535 | 评测服务监听端口，须为合法端口号 | 无 |
| `served_model_name` | `string` | 可选 | `served_model_name` | 非空字符串；字符串长度 ≤ 200 | 已部署的模型名称 | 无 |
| `precheck` | `list[object]` | 可选 | `[]` | — | 模型预检配置列表，每个元素是一个字典，包含 'type' 字段（'garbled_text' 或 'expected_answer'） | 本页 <a href="#2-3-4-base-precheck-config">§2.3.4</a> |

**配置约束**

- 无。

<h4 id="2-3-1-aisbench-config">2.3.1 AisbenchConfig</h4>

AISBench 评测配置

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `binary` | `string` | 可选 | `ais_bench` | `ais_bench` | aisbench 启动命令，固定为 'ais_bench' | 无 |
| `mode` | `string` | 可选 | `all` | 字符串长度 ≤ 200 | 评测模式 | 无 |
| `timeout` | `int` | 可选 | `7200` | >0 | 命令执行超时时间（秒），默认2小时 | 无 |
| `cleanup_model_config` | `bool` | 可选 | `true` | — | 是否清理生成的模型配置文件 | 无 |
| `model_meta` | `object` | 可选 | 见嵌套配置默认值 | — | 模型配置元数据 | 本页 <a href="#2-3-2-model-config-meta">§2.3.2</a> |
| `request_rate` | `float` | 可选 | `1.0` | >0 | 默认请求速率，必须 > 0 | 无 |
| `pred_postprocessor` | `string` | 可选 | `extract_non_reasoning_content` | 字符串长度 ≤ 200 | 预测后处理器名称 | 无 |
| `retry` | `int` | 可选 | `2` | ≥0 | 请求重试次数，必须 >= 0 | 无 |
| `batch_size` | `int` | 可选 | `1` | >0 | 批处理大小，必须 > 0 | 无 |
| `max_out_len` | `int` | 可选 | `512` | >0 | 最大输出长度，必须 > 0 | 无 |
| `trust_remote_code` | `bool` | 可选 | `false` | — | 是否信任远程代码 | 无 |
| `generation_kwargs` | `object` | 可选 | `{}` | — | 生成参数配置字典 | 无 |
| `extra_args` | `list[string]` | 可选 | `[]` | — | 额外的命令行参数列表，默认为空列表 | 无 |
| `log_dir` | `string` | 可选 | `` | 字符串长度 ≤ 200 | 日志目录路径，空字符串表示使用默认路径 | 无 |

**配置约束**

- 无。

<h4 id="2-3-2-model-config-meta">2.3.2 ModelConfigMeta</h4>

模型配置元数据

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `directory` | `string` | 可选 | `` | 字符串长度 ≤ 200 | 模型配置目录的显式路径，空字符串表示使用默认路径 | 无 |
| `subdir` | `string` | 可选 | `vllm_api` | 字符串长度 ≤ 200 | 模型配置子目录 | 无 |
| `base_name` | `string` | 可选 | `vllm_api_general_chat` | 字符串长度 ≤ 200 | 模型配置基础名称 | 无 |
| `name_suffix` | `string` | 可选 | `auto` | 字符串长度 ≤ 200 | 模型配置名称后缀，'auto'表示自动生成 | 无 |
| `abbr` | `string` | 可选 | `vllm-api-general-chat` | 字符串长度 ≤ 200 | 模型配置缩写 | 无 |
| `attr` | `string` | 可选 | `service` | 字符串长度 ≤ 200 | 模型配置属性 | 无 |

**配置约束**

- 无。

<h4 id="2-3-3-dataset-config">2.3.3 DatasetConfig</h4>

单个数据集的评测配置

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `config_name` | `string` | 必选 | 无 | 非空字符串；字符串长度 ≤ 200 | 数据集在 ais_bench 中的配置名称（必需） | 无 |
| `mode` | `string` | 可选 | `` | 字符串长度 ≤ 200 | 该数据集的评测模式，空字符串表示使用全局模式 | 无 |
| `request_rate` | `float` | 可选 | `0.0` | ≥0.0 | 该数据集的请求速率，0.0 表示使用全局默认值 | 无 |
| `max_out_len` | `int / null` | 可选 | `null` | >0；或 null | 该数据集的最大输出长度，None 表示使用全局默认值 | 无 |
| `returns_tool_calls` | `bool / null` | 可选 | `null` | — | 是否返回工具调用，None 表示不写入该字段 | 无 |
| `api_chat_type` | `string` | 可选 | `VLLMCustomAPIChat` | 非空字符串；字符串长度 ≤ 200 | 该数据集使用的 API Chat 类型 | 无 |
| `chat_template_kwargs` | `object` | 可选 | `{}` | — | chat_template 的额外参数，最终写入 AISBench 模型配置；具体参数请参考 AISBench 文档，例如 aime25 需要 {"thinking": true}。 | 无 |
| `extra_args` | `list[string]` | 可选 | `[]` | — | 该数据集额外的命令行参数列表，默认为空列表 | 无 |

**配置约束**

- 无。

<h4 id="2-3-4-base-precheck-config">2.3.4 BasePrecheckConfig</h4>

预检查配置基类

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 必选 | 无 | — | 预检查类型，按 `type` 字段分派，如 `garbled_text`、`expected_answer` | 无 |
| `max_tokens` | `int` | 可选 | `512` | >0 | 最大生成 token 数，必须大于 0 | 无 |
| `timeout` | `float` | 可选 | `60.0` | >0 | API 调用超时时间（秒），必须大于 0 | 无 |

**配置约束**

- 无。

**派生类**

| 配置类 | `type` | 说明 | 文档 |
|--------|----------|------|------|
| `ExpectedAnswerPrecheckConfig` | `expected_answer` | 期望答案验证预检查配置 | 本页 <a href="#2-3-5-expected-answer">§2.3.5</a> |
| `GarbledTextPrecheckConfig` | `garbled_text` | 基本乱码检测预检查配置 | 本页 <a href="#2-3-7-garbled-text">§2.3.7</a> |

<h4 id="2-3-5-expected-answer">2.3.5 ExpectedAnswerPrecheckConfig</h4>

期望答案验证预检查配置

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `expected_answer` | `expected_answer` | 预检查类型，固定为 `expected_answer`。 | 无 |
| `max_tokens` | `int` | 可选 | `512` | >0 | 最大生成 token 数，必须大于 0 | 无 |
| `timeout` | `float` | 可选 | `60.0` | >0 | API 调用超时时间（秒），必须大于 0 | 无 |
| `test_cases` | `list[object]` | 可选 | `[TestCase(message='What is 2+2?', expected_answer='4')]` | 最少1项 | 测试用例列表，字典（键值对）格式，内容须为英文，至少包含 1 条。 | 本页 <a href="#2-3-6-test-case">§2.3.6</a> |

**配置约束**

- 将单条或列表形式的测试用例统一转换为列表。
- 测试用例列表必须至少包含一项。

<h4 id="2-3-6-test-case">2.3.6 TestCase</h4>

预检查测试用例

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `message` | `string` | 必选 | 无 | 非空字符串；字符串长度 ≤ 4096 | 测试消息，须为非空字符串。 | 无 |
| `expected_answer` | `string / list[string] / null` | 可选 | `null` | 字符串长度 ≤ 200；列表元素：字符串长度 ≤ 200；或 null | 期望模型回复中包含的内容，可为字符串或字符串列表。 | 无 |

**配置约束**

- 无。

<h4 id="2-3-7-garbled-text">2.3.7 GarbledTextPrecheckConfig</h4>

基本乱码检测预检查配置

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `garbled_text` | `garbled_text` | 预检查类型，固定为 `garbled_text`。 | 无 |
| `max_tokens` | `int` | 可选 | `512` | >0 | 最大生成 token 数，必须大于 0 | 无 |
| `timeout` | `float` | 可选 | `60.0` | >0 | API 调用超时时间（秒），必须大于 0 | 无 |
| `test_cases` | `list[object] / null` | 可选 | `null` | — | List of test cases. Each case contains a message and enabled check items. | 本页 <a href="#2-3-8-test-case-config">§2.3.8</a> |

**配置约束**

- 将单条、列表或空值形式的测试用例统一转换为非空列表。
- 测试用例列表必须至少包含一项。

<h4 id="2-3-8-test-case-config">2.3.8 TestCaseConfig</h4>

测试用例配置

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `message` | `string` | 必选 | 无 | 非空字符串；字符串长度 ≤ 200 | Test message, must be a non-empty string | 无 |
| `items` | `list[string]` | 可选 | `['empty_text', 'repeated_char', 'normal_char_ratio', 'control_char', 'repeated_pattern']` | — | List of check item names to enable for this test case. If not specified, all items are enabled. | 无 |

**配置约束**

- 检查项名称必须来自内置检查项注册表。

<h3 id="2-4-vllm-ascend">2.4 VllmAscendConfig</h3>

vLLM-Ascend 推理引擎配置：用于拉起 OpenAI 兼容的服务端并做健康检查。

| 字段路径 | 类型 | 必选/可选 | 默认值 | 取值范围或格式 | 含义 | 引用配置 |
|----------|------|-----------|--------|----------------|------|----------|
| `type` | `string` | 可选 | `vllm-ascend` | `vllm-ascend` | 推理引擎类型，固定为 `vllm-ascend` | 无 |
| `entrypoint` | `string` | 可选 | `vllm.entrypoints.openai.api_server` | 非空字符串；字符串长度 ≤ 200 | vLLM 服务启动入口，默认 OpenAI API server | 无 |
| `env_vars` | `object` | 可选 | `{}` | — | 传递给 vLLM 进程的额外环境变量字典（键值对），键和值会作为环境变量注入；可配置参数见 vLLM-Ascend 文档。 | 无 |
| `served_model_name` | `string` | 可选 | `served_model_name` | 非空字符串；字符串长度 ≤ 200 | 已部署/对外暴露的模型名称 | 无 |
| `host` | `string` | 可选 | `localhost` | 合法主机名/IP，仅允许本机、回环或私有/链路本地地址 | 服务监听地址 | 无 |
| `port` | `int` | 可选 | `1234` | 1–65535 | 服务监听端口 | 无 |
| `health_check_endpoint` | `string` | 可选 | `/v1/models` | 以 `/` 开头的安全路径，禁止 `..`、`//` 和非安全字符 | 健康检查接口路径（vLLM OpenAI 兼容） | 无 |
| `startup_timeout` | `int` | 可选 | `600` | >0 | 服务启动超时（秒），必须 > 0 | 无 |
| `args` | `object` | 可选 | `{}` | — | 追加的 vLLM 启动命令行参数（键值对），键会转换为 `--<key>` 选项；布尔值 `true` 作为开关，`false`/`null` 忽略，对象值序列化为紧凑 JSON；可配置参数见 vLLM-Ascend 文档。 | 无 |

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
