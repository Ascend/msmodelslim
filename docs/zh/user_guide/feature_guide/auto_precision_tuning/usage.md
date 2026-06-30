---
toc_depth: 3
---
# 自动调优使用说明

## 1. 简介

自动调优功能面向需要精确控制量化后模型精度的用户，通过自动化的量化配置搜索和评估流程，帮助用户找到满足精度要求的量化方案。本功能支持根据用户指定的精度需求，自动尝试不同的量化配置，并通过评估服务验证量化后的模型精度，最终输出满足需求的量化模型。

## 2. 使用前准备

使用自动调优功能前，需要完成以下工具的安装：

1. **msModelSlim**：用于自动调优任务的启动与执行。安装方式请参考 《[安装指南](../../../install_guide/install_guide.md)》。
2. **vLLM-Ascend**：用于量化后模型服务化启动。自动调优功能在评估量化后模型精度时，需要使用 vLLM-Ascend 将量化后的模型以服务化方式启动。建议直接使用官方提供的镜像进行安装，详细安装说明请参考 [vLLM-Ascend 安装文档](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/installation.html)。
3. **AISBench**：用于量化后模型测试。自动调优功能使用 [AISBench](https://github.com/AISBench/benchmark) 对量化后的模型进行精度评估和测试。完成安装后，还需要参考 [AISBench 数据集准备指南](https://yh-ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/datasets.html) 准备对应的数据集。

请确保上述工具已正确安装并配置，否则自动调优功能将无法正常进行模型评估。

## 3. 功能介绍

### 3.1 功能说明

自动调优功能用于在给定精度目标下，自动搜索并验证量化配置，帮助用户获得“精度达标、可落地复用”的量化结果。

#### 3.1.1 核心能力

- **自动搜索**：基于调优配置文件定义的策略与搜索空间，迭代生成候选量化配置。
- **自动评估**：对每个候选配置执行量化，并通过 vLLM-Ascend 拉起量化模型服务，再使用 AISBench 对量化后模型精度进行评估。
- **结果产出**：输出满足精度要求的最终量化配置与量化权重，并保留每次迭代的配置与精度缓存，支持中断恢复与复现。
- **面向场景**：适用于需要在较短时间内完成“量化配置选择/回退层选择/精度验证”的自动化流程，减少人工反复试参成本。

#### 3.1.2 调优流程说明

自动调优功能的工作流程如下：

1. **创建模型适配器**：根据指定的模型类型创建模型适配器，确保指定的模型可支持。

2. **初始化调优服务**：根据指定的配置文件路径加载调优计划，创建量化配置生成器。

3. **检测历史记录**：自动检测是否存在历史精度缓存；若存在，后续迭代会尝试复用历史评估结果，避免重复评估相同配置。

4. **开始迭代调优**：记录调优开始时间和超时时间，进入迭代循环：

   每次迭代包含以下步骤：
   - **检查超时**：在每次迭代开始时检查是否超过最大迭代耗时，如果超时则停止迭代。
   - **生成量化配置 YAML**：通过量化配置生成器生成量化配置（YAML 格式的 practice）。首次迭代时传入 None，基于初始策略和精度需求生成配置；后续迭代时传入上一次的精度评估结果，根据评估结果生成新的配置。
   - **尝试恢复历史**：如果存在历史精度缓存，系统会尝试从精度缓存中恢复当前迭代的评估结果。匹配条件为：**量化配置和评估配置都必须完全一致**。如果找到完全匹配的评估结果，则直接复用历史评估结果，跳过后续步骤；否则将重新进行完整的量化-评估流程。
   - **量化模型**：如果未从历史恢复，根据生成的量化配置 YAML，对模型进行量化，量化权重临时存入用户指定路径。
   - **评估模型精度**：使用 vLLM-Ascend 将量化后的模型以服务化方式启动，然后使用 AISBench 对量化后模型精度进行评估与测试，产生精度指标（可选启用预检查以提前发现明显异常）。
   - **保存调优历史**：将本次迭代的量化配置 YAML 和精度指标保存到调优历史。
   - **判断是否继续**：将评估得到的精度指标作为下一次迭代的输入。若策略判断已满足迭代停止条件，则停止迭代；否则继续下一次迭代。

5. **停止条件**：迭代会在满足以下任一条件时停止：
   - 调优策略已无搜索空间（无法生成新的可评估量化配置）
   - 达到最大迭代次数
   - 达到最大迭代耗时

6. **保存结果**：调优结束后，会保存量化模型权重、调优历史与精度缓存，以及评估与服务日志等产出件；具体产出位置与目录结构请参考下文「输出说明」。

### 3.2 注意事项

**重要提示**：在使用自动调优功能之前，请确保使用的模型是支持的。需要同时满足以下条件：

1. **模型类型限制**：当前自动调优服务仅支持**大语言模型（LLM）**，且对 **MoE** 模型支持有限（MoE 通常包含大量专家非共享子层，回退/搜索空间更大，导致迭代优化轮次更多，整体调优耗时更长）。
2. **量化工具支持**：模型需要在量化工具的支持列表中，可以通过查看 [config/config.ini](https://gitcode.com/Ascend/msmodelslim/blob/master/config/config.ini) 中的 `[ModelAdapter]` 确定模型是否支持，其中包含了当前支持的模型适配器及其支持的模型名。如果模型不在支持列表中，需要先进行模型适配，实现相应的模型接口后才能使用自动调优功能。模型适配说明可参考 《[LLM 大模型接入指南](../../../development_guide/integrating_models.md)》。
3. **vLLM-Ascend 支持**：模型需要被 vLLM-Ascend 支持，能够将量化后的模型以服务化方式启动。请先确定 vLLM-Ascend 是否支持量化后模型服务化启动。
4. **transformers 版本兼容**：量化工具与推理引擎（vLLM-Ascend）对 `transformers` 的版本有各自的要求，需确保当前环境中的版本能同时满足二者。若某一模型在使用的过程中，量化工具与推理引擎所需的 `transformers` 版本不一致，且**不存在一个 transformers 版本能同时满足两边要求**，则无法在该环境下启动自动调优服务。使用前，请参照双方依赖说明确认当前环境中的 `transformers` 版本兼容：**量化工具侧**各模型对 transformers 等依赖的版本要求见 [config/config.ini](https://gitcode.com/Ascend/msmodelslim/blob/master/config/config.ini) 中的 `[ModelAdapterDependencies]` 配置项；**推理引擎侧**各版本 vLLM-Ascend 的依赖要求见 [vLLM-Ascend 发布说明](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/user_guide/release_notes.html#id5) 中各版本的 Dependencies 小节。
5. **单机服务化**：当前自动调优服务不支持跨机部署。对于需要跨机才能完成服务化启动的超大规模模型，暂时无法使用自动调优功能。

### 3.3 命令格式

自动调优功能通过命令行方式启动。完成上述前置依赖后，可以通过如下命令运行：

``` bash
msmodelslim tune [ARGS]
```

下面给出一个包含所有参数的命令模板。其中 `${MODEL_PATH}` 为模型原始浮点权重路径，`${SAVE_PATH}` 为用户自定义的调优结果保存路径，`${CONFIG}` 为调优配置文件路径；`${DEVICE}`、`${MODEL_TYPE}`、`${TIMEOUT}`、`${TRUST_REMOTE_CODE}` 分别对应量化设备、模型类型、超时时间与是否信任远程代码。

``` bash
# 全局调用命令行
msmodelslim tune --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --config ${CONFIG} --device ${DEVICE} --model_type ${MODEL_TYPE} --timeout ${TIMEOUT} --trust_remote_code ${TRUST_REMOTE_CODE}
```

**中断恢复**：如果调优过程因为意外中断（如系统故障、手动停止等），系统会自动检测 `${SAVE_PATH}/history` 目录中是否存在历史精度缓存。如果检测到精度缓存，系统会在重新启动时从头开始迭代，但会复用精度缓存中已评估过的量化配置的评估结果，避免重复评估相同的配置，只需使用相同的 `${SAVE_PATH}` 重新运行调优命令即可。

**重要说明**：

- 自动恢复功能要求 `${SAVE_PATH}` 与之前中断时的路径一致，且 `${SAVE_PATH}/history` 目录中存在有效的历史精度缓存。如果历史精度缓存不存在或无效，系统将从头开始调优。
- **精度缓存复用条件**：系统只有在同时满足以下两个条件时才会复用历史精度缓存中的评估结果：
  1. **评估配置完全一致**：启动调优时调优配置文件中的 evaluation 字段配置（包括评估数据集、评估任务、评估参数等）与历史记录中的评估配置完全相同。
  2. **量化配置完全一致**：当前迭代生成的量化配置与历史记录中的量化配置完全相同。
- 如果评估配置或量化配置有任何差异，系统将重新进行量化、服务化启动和精度评估，不会复用历史结果。

**环境变量（可选）**

如需控制日志级别或配置“最佳实践库”保存路径，可通过以下环境变量进行设置。

| 环境变量 | 解释 | 是否可选 | 取值/说明 |
|---|---|---|---|
| `MSMODELSLIM_LOG_LEVEL` | 打印同级及以上日志 | 可选 | `INFO`（默认）、`DEBUG` |
| `MSMODELSLIM_CUSTOM_PRACTICE_REPO` | 自定义最佳实践库配置文件保存路径 | 可选 | 类型：`str`。设置后，精度达标时会将最终量化配置 YAML 写入该路径下的最佳实践库目录；**不设置则不会写入最佳实践库**，可从 `${SAVE_PATH}/history` 中各次迭代的 YAML 获取配置 |

### 3.4 参数说明

| 参数名称              | 解释        | 是否可选              | 范围                                                                                   |
|-------------------|-----------|-------------------|--------------------------------------------------------------------------------------|
| model_path        | 模型路径      | 必选                | 类型：Str                                                                               |
| save_path         | 调优结果保存路径  | 必选                | 类型：Str                                                                               |
| config            | 调优配置文件路径  | 必选                | 1. 类型：Str <br>2. 配置文件路径，必须为完整的文件路径 <br>3. 配置文件格式为 YAML，配置协议说明见 《[自动调优配置协议说明](configuration_protocols.md)》，示例配置见 [example](example) |
| device            | 量化设备      | 可选                | 1. 类型：Str <br>2. 参考值：'npu','npu:0,1,2,3','cpu' <br>3. 默认值为"npu"（单设备）<br>4. 指定多个设备时（如：'npu:0,1,2,3'），系统可启动分布式逐层量化（DP）。算法支持范围与配置方式详见《[一键量化使用说明](../quick_quantization_v1/usage.md#51-逐层量化及分布式逐层量化)》 |
| model_type        | 模型名称      | 可选                | 1. 类型：Str <br>2. 默认值为"default" <br>3. 大小写敏感，请参考《[大模型支持矩阵](../../model_support/foundation_model_support_matrix.md)》 |
| timeout           | 调优超时时间    | 可选                | 1. 类型：Str <br>2. 格式：`<天数>D`、`<小时数>H` 或 `<天数>D<小时数>H` <br>3. 示例：'1D'、'2H'、'3D4H' <br>4. 默认值：None（无超时限制） |
| trust_remote_code | 是否信任自定义代码 | 可选                | 1. 类型：Bool，默认值：False <br>2. 请确保加载的自定义代码文件的安全性，设置为True有安全风险。                          |
| h, help           | 命令行参数帮助信息 | 可选                |               -            |

### 3.5 使用示例

例如，使用自动调优功能对 Qwen3-32B 模型进行调优，调优命令如下。

``` bash
msmodelslim tune --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --config ${CONFIG} --device npu --model_type Qwen3-32B --trust_remote_code True
```

用户输入命令后，系统将根据配置文件中指定的精度需求，自动尝试不同的量化配置，并对量化后的模型进行评估，直到找到满足精度要求的量化方案或达到最大迭代次数/超时时间。

### 3.6 输出说明

自动调优功能执行完成后，会在 `${SAVE_PATH}` 目录下生成以下输出件。目录结构示例如下：

```text
${SAVE_PATH}/
├── quant_model/          # 量化模型权重（最终一次迭代）
├── history/              # 调优历史与精度缓存
│   ├── history.yaml      # 调优历史索引文件
│   ├── accuracy.yaml     # 精度缓存文件
│   └── <config_id>.yaml  # 每次迭代的量化配置（YAML）
├── vllm_server.log       # vLLM-Ascend 服务日志（最近一次）
├── aisbench_logs/        # AISBench 评估日志
└── aisbench_output/      # AISBench 详细输出
```

| 目录 | 路径 | 说明 |
| --- | --- | --- |
| 量化模型权重 | `${SAVE_PATH}/quant_model` | 保存最后一次迭代生成的量化模型权重文件。根据量化配置中 `save` 字段指定的保存器类型。 |
| 调优历史记录 | `${SAVE_PATH}/history` | 保存所有迭代的调优历史记录，包括 `history.yaml`（调优历史索引）、`accuracy.yaml`（精度缓存）及各次迭代的 `<config_id>.yaml`（量化配置）。 |
| vLLM 服务日志 | `${SAVE_PATH}/vllm_server.log` | 记录最近一次 vLLM-Ascend 服务启动和运行过程中的日志信息，包括模型加载、服务启动、推理请求处理等日志。 |
| AISBench 评估日志 | `${SAVE_PATH}/aisbench_logs` | 保存每次迭代的 AISBench 评估日志文件，记录每次迭代的评估过程日志。 |
| AISBench 详细输出 | `${SAVE_PATH}/aisbench_output` | 保存每次迭代的 AISBench 详细评估输出，包含评估任务配置、推理和评估日志、推理结果、精度评估结果及汇总报告等文件。 |
| 最终量化配置 YAML | 精度达标且写入最佳实践库时：由 `MSMODELSLIM_CUSTOM_PRACTICE_REPO` 指定路径下按模型谱系生成（如 `qwen3/xxx.yaml`） | 未设置 `MSMODELSLIM_CUSTOM_PRACTICE_REPO` 时**不会**写入最佳实践库；可在 `${SAVE_PATH}/history` 中查看各次迭代配置与精度并选择 `<config_id>.yaml` 复现。写入最佳实践库的 YAML 可直接用于后续量化；若因达到最大迭代次数或超时而停止，则不会写入最终配置。 |

## 4. 附录

### 4.1 相关资料

- **配置协议**：调优配置文件的详细说明，可以参考 《[自动调优配置协议说明](configuration_protocols.md)》。完整的配置文件示例，可以参考 [example](example)。
- **调优算法**：对于自动调优支持的多种策略和算法，可以参考相关文档：
  - 《[Standing High 调优算法](../../quantization_algorithms/auto_tuning_strategies/standing_high.md)》：基于量化回退层选择和离群值抑制策略的自动调优算法
  - 《[Standing High With Experience 调优算法](../../quantization_algorithms/auto_tuning_strategies/standing_high_with_experience.md)》：基于专家经验的摸高算法策略
  - 《[Binary Fallback 调优算法](../../quantization_algorithms/auto_tuning_strategies/binary_fallback.md)》：基于二分搜索的回退层选择策略的自动调优算法
