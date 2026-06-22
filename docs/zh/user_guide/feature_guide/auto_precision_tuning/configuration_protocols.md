# 自动调优配置协议说明

## 1. 概述

### 1.1 配置协议简介

自动调优配置协议采用分层结构设计，顶层包含两个核心字段。

- **strategy**: 调优策略配置，定义量化配置生成策略和量化基础配置。
- **evaluation**: 评估服务配置，定义模型精度评估的方式和量化模型服务化拉起相关参数。

### 1.2 配置文件位置

用户需要自定义调优配置文件，可以参考 [example](./example) 目录下的配置文件格式进行自定义。

### 1.3 基础配置结构

```yaml
strategy:
  type: <strategy_type>  # 调优策略类型，如 standing_high、standing_high_with_experience、binary_fallback
  # 不同策略有不同的配置字段与语义，详见各策略文档

evaluation:
  type: service_oriented
  demand:
    expectations:
      # 精度期望列表：用于描述“需要在哪些数据集上达到什么精度要求”
  evaluation:
    type: aisbench
    # 具体字段详见下文 evaluation.evaluation 小节
  inference_engine:
    # 推理引擎配置
```

## 2. 配置字段详解

### 2.1 strategy - 调优策略配置

**作用**: 定义调优策略的类型、参数和量化基础配置。

**type - 策略类型**

**作用**: 指定调优算法的类型，不同的策略类型对应不同的调优算法。

**类型**: `string`

**可选值**: 根据已实现的调优策略而定，例如 `standing_high`、`standing_high_with_experience`、`binary_fallback`。

**模型适配（按策略）**

若调优策略需**自动运行敏感层分析**（生成回退候选），模型适配器须实现 **`ModelSlimPipelineInterfaceV1`**（即 `core/runner/pipeline_interface.py` 中的 `PipelineInterface`，与 CLI `msmodelslim analyze` 及 《[敏感层分析使用说明](../sensitive_layer_analysis/usage.md)》 要求相同）。策略内由领域层 `PipelineAnalysisService` 调用 `init_model`、`handle_dataset` 及 visit/forward pipeline；**不会在策略侧预先 `load_model`**。

各策略在自动敏感层分析之上的**额外**要求如下：

| 策略 | 模型适配器要求 |
|------|----------------|
| `standing_high` | **`ModelSlimPipelineInterfaceV1`**（始终自动敏感层分析） |
| `standing_high_with_experience` | 上表接口 + **`StandingHighWithExperienceInterface`**（`load_model`，离群值抑制能力探测） |
| `binary_fallback` | 配置了非空 `rollback_candidates` 时无额外要求（跳过敏感层分析）；否则须 **`ModelSlimPipelineInterfaceV1`** |

详见各策略文档「适用要求 / 模型适配」章节及 《[LLM 大模型接入指南](../../../development_guide/integrating_models.md#自动调优与敏感层分析)》。

**策略特有配置字段**

不同调优策略的配置字段差异较大，详细的配置项请参考相应算法文档的「YAML 配置字段详解」章节。目前已支持的算法如下：

- 《[Standing High 调优算法](../../quantization_algorithms/auto_tuning_strategies/standing_high.md)》
- 《[Standing High With Experience 调优算法](../../quantization_algorithms/auto_tuning_strategies/standing_high_with_experience.md)》
- 《[Binary Fallback 调优算法](../../quantization_algorithms/auto_tuning_strategies/binary_fallback.md)》

### 2.2 evaluation - 评估服务配置

**作用**: 定义模型精度评估的配置，包括评估服务类型、测评工具配置和推理引擎配置。

**核心字段**:

| 字段名 | 作用 | 类型 | 必选/可选 | 说明 |
|--------|------|------|----------|------|
| type | 评估服务类型 | string | 必选 | 当前支持 `service_oriented`（面向服务的评估，通过服务化方式启动模型进行评估）。 |
| demand | 精度需求配置 | object | 必选 | 定义模型精度评估的精度需求，包括数据集、目标精度和容差 |
| evaluation | 测评工具配置 | object | 必选 | 定义测评工具的配置参数 |
| inference_engine | 推理引擎配置 | object | 必选 | 定义推理引擎的配置参数，用于将量化后的模型以服务化方式启动 |

#### 2.2.1 type - 评估服务类型

**作用**: 指定评估服务的类型。

**类型**: `string`

**可选值**: `service_oriented`（面向服务的评估，通过服务化方式启动模型进行评估）

#### 2.2.2 demand - 精度需求配置

**作用**: 定义模型精度评估的精度需求，包括数据集、目标精度和容差。

**字段说明**:

| 字段名 | 作用 | 类型 | 必选/可选 | 说明 |
|--------|------|------|----------|------|
| expectations | 精度期望列表 | list | 必选 | 精度需求的列表，每个元素包含数据集、目标精度和容差，至少包含一个元素 |

**expectations 字段说明**:

| 字段名 | 作用 | 类型 | 必选/可选 | 说明 |
|--------|------|------|----------|------|
| dataset | 指定要评估的数据集 | string | 必选 | 数据集名称，必须是在 evaluation.evaluation.datasets 中配置的数据集名称 |
| target | 设置精度目标值 | number | 必选 | 期望达到的精度目标值，必须大于 0；可写数字（如 `0.95`）或字符串（如 `"0.95"`） |
| tolerance | 设置精度容差 | number | 必选 | 精度允许的误差范围，必须大于等于 0；可写数字（如 `0.95`）或字符串（如 `"0.95"`） |

**配置示例**:

```yaml
# 单个数据集的精度需求
demand:
  expectations:
    - dataset: gsm8k
      target: 83  # 目标精度 83%
      tolerance: 2  # 容差 ±2%

# 多个数据集的精度需求
demand:
  expectations:
    - dataset: gsm8k
      target: 83  # 目标精度 83%
      tolerance: 2  # 容差 ±2%
    - dataset: aime25
      target: 85  # 目标精度 85%
      tolerance: 1  # 容差 ±1%
    - dataset: bfcl-simple
      target: 80  # 目标精度 80%
      tolerance: 2  # 容差 ±2%
```

**注意**:

- **精度单位说明**：不同数据集返回的精度格式可能不同，有些数据集返回小数形式（0.0-1.0，如 0.83 表示 83%），有些数据集返回百分制（0-100，如 83 表示 83%）。`target` 和 `tolerance` 的单位必须与对应数据集返回的精度格式保持一致。请根据测评工具实际数据集返回的精度格式来配置 `target` 和 `tolerance`。
- **精度数值设置说明**：`target` 和 `tolerance` 在系统内部按 Decimal 进行计算。YAML 中可直接写数字（如 `0.95`），也可写字符串（如 `"0.95"`）；在需要严格控制小数精度的场景，建议使用字符串写法。
- **精度目标设置说明**：文档中给出的精度数据仅供参考，请根据实际浮点模型的精度进行配置。理论上量化后模型不会超过原始浮点模型的精度，因此建议将精度目标设置为略低于或等于浮点模型的精度。
- 支持配置多个数据集的精度需求，每个数据集可以设置不同的目标精度和容差。

#### 2.2.3 evaluation - 测评工具配置

**作用**: 定义测评工具的配置参数。

**核心字段**:

| 字段名 | 作用 | 类型 | 必选/可选 | 说明 |
|--------|------|------|----------|------|
| type | 测评工具类型 | string | 必选 | 当前支持 `aisbench`。 |
| precheck | 预检查配置 | list | 可选 | 定义正式评估前的预检查配置，空列表表示跳过预检查。默认值：`[]` |
| aisbench | AISBench 测评工具配置 | object | 必选 | AISBench 测评工具的详细配置参数（如 `timeout`、`batch_size`、`generation_kwargs` 等）。 |
| datasets | 数据集配置 | dict | 必选 | 定义需要评估的数据集及其配置，必须至少包含 `demand.expectations` 中指定的所有数据集。 |
| host | 服务主机地址 | string | 可选 | 测评端连接推理服务的地址。**需与推理引擎侧保持一致**。默认值：`"localhost"` |
| port | 服务端口 | int | 可选 | 测评端连接推理服务的端口。**需与推理引擎侧保持一致**。默认值：`1234` |
| served_model_name | 服务化模型名称 | string | 可选 | 测评端请求时使用的模型名称标识，**需与推理引擎侧保持一致**。默认值：`"served_model_name"` |

**配置示例**:

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

##### 2.2.3.1 aisbench - AISBench 测评工具配置

**作用**：配置 AISBench 的命令行与测评运行参数（如 `timeout`、`batch_size`、`generation_kwargs` 等）。

**aisbench 字段说明**:

AISBench 测评工具的详细配置参数：

| 字段名 | 作用 | 类型 | 必选/可选 | 说明 |
|--------|------|------|----------|------|
| binary | 指定 AISBench 启动命令 | string | 可选 | 固定值：`ais_bench`。默认值：`"ais_bench"` |
| mode | 设置评测模式 | string | 可选 | 评测模式。默认值：`"all"` |
| timeout | 设置命令执行超时时间 | int | 可选 | 超时时间（秒），必须大于 0。默认值：`7200`（2小时） |
| cleanup_model_config | 控制是否清理模型配置 | bool | 可选 | 是否清理生成的模型配置文件。默认值：`true` |
| model_meta | 配置模型元数据 | object | 可选 | 模型配置元数据，详见下方说明。默认值：`ModelConfigMeta()` |
| request_rate | 设置默认请求速率 | float | 可选 | 默认请求速率，必须大于 0。默认值：`1.0` |
| pred_postprocessor | 指定预测后处理器 | string | 可选 | 预测后处理器名称。默认值：`"extract_non_reasoning_content"` |
| retry | 设置请求重试次数 | int | 可选 | 请求重试次数，必须大于等于 0。默认值：`2` |
| batch_size | 设置批处理大小 | int | 可选 | 批处理大小，必须大于 0。默认值：`1` |
| max_out_len | 设置最大输出长度 | int | 可选 | 最大输出长度，必须大于 0。默认值：`512` |
| trust_remote_code | 控制是否信任远程代码 | bool | 可选 | 是否信任远程代码。默认值：`false` |
| generation_kwargs | 服务化推理后端配置生成参数 | dict | 可选 | 生成参数配置字典。默认值：`{}` |
| extra_args | 添加额外命令行参数 | list | 可选 | 额外的命令行参数列表。默认值：`[]` |
| log_dir | 指定日志目录路径 | string | 可选 | 日志目录路径，空字符串表示使用默认路径。默认值：`""` |

**model_meta 字段说明**:

下面参数主要用于获得服务化推理后端模型配置：

| 字段名 | 作用 | 类型 | 必选/可选 | 说明 |
|--------|------|------|----------|------|
| directory | 指定模型配置目录路径 | string | 可选 | 模型配置目录的显式路径，空字符串表示使用默认路径。默认值：`""` |
| subdir | 指定模型服务化后端配置子目录 | string | 可选 | 模型服务化后端配置子目录。默认值：`"vllm_api"` |
| base_name | 指定模型服务化后端配置基础名称 | string | 可选 | 模型服务化后端配置基础名称。默认值：`"vllm_api_general_chat"` |
| name_suffix | 指定模型服务化后端配置名称后缀 | string | 可选 | 模型服务化后端配置名称后缀，`auto` 表示自动生成。默认值：`"auto"` |
| abbr | 指定模型配置缩写 | string | 可选 | 模型配置缩写。默认值：`"vllm-api-general-chat"` |
| attr | 指定模型配置属性 | string | 可选 | 模型配置属性。默认值：`"service"` |

**注意**: 上面大部分参数来自于 AISBench 命令行参数与服务化推理后端参数，可以参考 [AISBench 详细参数说明](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/index.html) 进行配置。

##### 2.2.3.2 datasets - 数据集配置

**作用**：配置待评估数据集及每个数据集在 AISBench 中的参数；必须至少包含 `demand.expectations` 中出现的所有数据集。

**datasets 字段说明**:

该字段指定了不同的数据集键与 AISBench 侧数据集配置的对应关系。下文示例中仅列出三个数据集（gsm8k、aime25、bfcl-simple）；更多数据集可参考 [AISBench 文档数据集支持列表](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/get_started/datasets.html) 自行扩展。

每个数据集的配置字段说明：

| 字段名 | 作用 | 类型 | 必选/可选 | 说明 |
|--------|------|------|----------|------|
| config_name | 指定 AISBench 中的配置名称 | string | 必选 | 数据集在 AISBench 中的配置名称，非空字符串 |
| mode | 设置该数据集的评测模式 | string | 可选 | 评测模式，空字符串表示使用全局模式。默认值：`""` |
| request_rate | 设置该数据集的请求速率 | float | 可选 | 请求速率，0.0 表示使用全局默认值。必须大于等于 0。默认值：`0.0` |
| max_out_len | 设置该数据集的最大输出长度 | int | 可选 | 最大输出长度，None 表示使用全局默认值。如果指定则必须大于 0。默认值：`None` |
| returns_tool_calls | 控制是否返回工具调用 | bool | 可选 | 是否返回工具调用，None 表示不写入该字段。默认值：`None` |
| api_chat_type | 指定该数据集使用的 API Chat 类型 | string | 可选 | 该字段需与 AISBench 对应数据集配置所要求的 API/请求格式保持一致。默认值：`"VLLMCustomAPIChat"` |
| extra_args | 添加该数据集的额外命令行参数 | list | 可选 | 额外的命令行参数列表。默认值：`[]` |

##### 2.2.3.3 precheck - 预检查配置（可选）

**作用**: 定义正式评估前的预检查配置，用于在每次迭代的模型评估前对量化后的模型进行预验证。

**类型**: `list`

**说明**: precheck 是一个列表，每个元素是一个预检查规则配置，包含 `type` 字段用于指定预检查类型。如果配置了 precheck 且不为空列表，系统会在正式评估前执行预检查。

**支持的预检查类型**: `expected_answer`

**expected_answer - 期望答案验证**

**作用**: 验证模型输出是否包含预期的答案内容。

**字段说明**:

| 字段名 | 作用 | 类型 | 必选/可选 | 说明 |
|--------|------|------|----------|------|
| type | 指定预检查类型 | string | 必选 | 固定值：`expected_answer`。 |
| test_cases | 配置测试用例列表 | list | 可选 | 测试用例列表，使用键值对格式（问题: 答案）。如果不配置，默认使用一个测试用例（"What is 2+2?": "4"）。默认值：`[{"What is 2+2?": "4"}]` |
| max_tokens | 设置最大 token 数 | int | 可选 | 必须大于零。默认值：`512` |
| timeout | 设置超时时间 | float | 可选 | 超时时间（秒），必须大于零。默认值：`60.0` |

**test_cases 字段说明**:

`test_cases` 使用字典格式（键值对），键为问题，值为答案：

```yaml
test_cases:
  - "What is 2+2?": ["4", "four"]  # 期望响应中包含 "4" 或 "four"
  - "What is the capital of China?": "Beijing"  # 期望响应中包含 "Beijing"
```

**格式说明**：

- 键（问题）：字符串类型，测试消息
- 值（答案）：可以是字符串、字符串列表
  - 字符串：如 `"4"`，表示期望响应中包含 "4"
  - 字符串列表：如 `["4", "four"]`，表示期望响应中包含 "4" 或 "four" 中的任意一个

**配置示例**:

```yaml
precheck:
  - type: expected_answer
    test_cases:
      - "What is 2+2?": ["4", "four"]
      - "What is the capital of China?": "Beijing"
    max_tokens: 1024
    timeout: 60.0
```

**注意**:

- 预检查功能会在每次迭代的模型评估前执行（在服务化启动后）。系统会依次执行所有配置的预检查规则，如果任何一个预检查失败，系统会跳过本次迭代的正式评估，返回数据集全零结果，直接开启下一次迭代。
- 预检查的目的是快速发现明显的问题，避免浪费时间进行完整的评估。如果所有预检查都通过，系统会继续进行正式的精度评估。
- **仅支持英文问答**：预检查功能目前仅支持英文问答，测试消息和期望答案应使用英文。
- 如果不配置 `precheck` 字段或配置为空列表，将跳过预检查直接进行正式评估。

#### 2.2.4 inference_engine - 推理引擎配置

**作用**: 定义推理引擎的配置参数，用于将量化后的模型以服务化方式启动。

**字段说明**:

| 字段名 | 作用 | 类型 | 必选/可选 | 说明 |
|--------|------|------|----------|------|
| type | 指定推理引擎类型 | string | 必选 | 当前只支持 `vllm-ascend`。 |
| entrypoint | 指定服务入口点 | string | 可选 | 服务入口点，非空字符串。默认 `vllm.entrypoints.openai.api_server` 为 vLLM OpenAI 兼容 API 服务入口；若需其他入口，须与当前安装的 vLLM/vLLM-Ascend 中可 `-m` 运行的模块一致。 |
| env_vars | 配置环境变量 | dict | 可选 | 环境变量配置。默认值：`{}` |
| served_model_name | 指定服务化模型名称 | string | 可选 | 推理服务对外使用的模型名称标识，非空字符串。**需与测评端（`evaluation.evaluation`）保持一致**。默认值：`"served_model_name"` |
| host | 指定服务主机地址 | string | 可选 | 推理服务监听地址，支持 localhost、IPv4、IPv6 格式。**需与测评端保持一致**。默认值：`"localhost"` |
| port | 指定服务端口 | int | 可选 | 推理服务监听端口，范围：1-65535。**需与测评端保持一致**。默认值：`1234` |
| health_check_endpoint | 指定健康检查端点 | string | 可选 | 就绪探测时请求的 **HTTP 路径**，须与推理进程实际提供的、可返回成功响应的 URL 一致。默认 `"/v1/models"` 对应常见 OpenAI 兼容服务的模型列表接口。必须以 `/` 开头，其它取值无统一列表，可按部署的 vLLM-Ascend 真实路由自定义。 |
| startup_timeout | 设置启动超时时间 | int | 可选 | 启动超时时间（秒），必须大于 0。默认值：`600` |
| args | 配置推理引擎启动参数 | dict | 可选 | 推理引擎启动参数，用于添加其他 vLLM-Ascend 参数配置。默认值：`{}` |

**注意**: 不同模型拉起服务化时需要的参数可能不同，用户需要根据实际模型调整服务化参数。可以参考 [vLLM-Ascend 教程](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/user_guide/configuration/index.html) 按模型需求配置参数，启动参数可以在 `args` 中进行添加，环境变量可以在 `env_vars` 中添加。

**配置示例**:

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

### 2.3 使用示例

完整的自动调优配置示例请参考：

- standing_high 调优策略配置：[standing_high.yaml](./example/standing_high.yaml)
- standing_high_with_experience 调优策略配置：[standing_high_with_experience.yaml](./example/standing_high_with_experience.yaml)
- binary_fallback 调优策略配置：[binary_fallback.yaml](./example/binary_fallback.yaml)
