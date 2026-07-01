---
toc_depth: 3
---

# 一键量化完整指南

## 1. 目录

- [简介](#2-简介)
- [使用前准备](#3-使用前准备)
- [快速开始](#4-快速开始)
  - [命令格式](#41-命令格式)
  - [参数说明](#42-参数说明)
  - [使用示例](#43-使用示例)
- [高级特性](#5-高级特性)
  - [逐层量化及分布式逐层量化](#51-逐层量化及分布式逐层量化)
- [量化配置协议详解](#6-量化配置协议详解)
  - [配置协议概述](#61-量化配置协议概述)
  - [modelslim_v1 配置详解](#62-modelslim_v1-配置详解)
  - [multimodal_sd_modelslim_v1 配置详解](#63-multimodal_sd_modelslim_v1-配置详解)
  - [multimodal_vlm_modelslim_v1 配置详解](#64-multimodal_vlm_modelslim_v1-配置详解)
  - [modelslim_v0 配置说明](#65-modelslim_v0-配置说明)
- [附录](#7-附录)

## 2. 简介

一键量化功能面向零基础用户，集成热门开源模型量化功能，具备“开箱即用”的特性。本功能支持全局调用量化命令，用户指定必要参数后，即可对目标原始权重执行指定的量化操作。

一键量化提供了两种使用方式：

1. **方式1（推荐）**：适用于工具已经支持且用户无特殊量化诉求的主流模型量化场景，可通过指定 `quant_type` 参数，工具在最佳实践库中自动匹配最适合的量化配置进行量化。量化配置请参考[量化配置协议详解](#6-量化配置协议详解)。
2. **方式2**：适用于模型或模型量化方式未收录最佳实践库或用户有特殊量化诉求场景，可通过指定 `config_path` 参数，工具直接使用用户指定的自定义量化配置进行量化。量化配置请参考[量化配置协议详解](#6-量化配置协议详解)。

## 3. 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../install_guide/install_guide.md)。

## 4. 快速开始

### 4.1 命令格式

一键量化功能通过命令行方式启动，可以通过如下命令运行：

```bash
msmodelslim quant [ARGS]
```

指定`--quant_type`参数时，系统将根据指定需求，在最佳实践库中匹配到最佳配置从而实施量化；指定`--config_path`参数时，将直接使用用户指定的配置，不会匹配最佳实践库。

**注意事项**：

1.最佳实践库中的配置文件放在 `msmodelslim/lab_practice` 中。

2.一键量化搜索最佳实践策略yaml的搜索优先级如下（优先级从高到低）：

   - 优先级1：采用模型指定量化方式和指定场景标签（'tag'参数）的最佳实践策略yaml
   - 优先级2：采用模型指定量化方式和忽略场景标签（'tag'参数）的最佳实践策略yaml，询问用户是否采用
   - 优先级3：采用模型指定量化方式和忽略场景标签（'tag'参数）的默认实践策略yaml（默认实践策略不保证精度正常），询问用户是否采用
   - 优先级4：采用模型推荐量化方式(W8A8)和指定场景标签（'tag'参数）的最佳实践策略yaml，询问用户是否采用
   - 优先级5：采用模型推荐量化方式(W8A8)和忽略场景标签（'tag'参数）的最佳实践策略yaml，询问用户是否采用
   - 优先级6：采用模型推荐量化方式(W8A8)和忽略场景标签（'tag'参数）的默认实践策略yaml（默认实践策略不保证精度正常），询问用户是否采用

3.如果需要打印量化运行日志，可通过以下环境变量进行设置：

   | 环境变量                  | 解释        | 是否可选 | 范围             |
   |-----------------------|-----------|------|----------------|
   | MSMODELSLIM_LOG_LEVEL | 打印同级及以上日志 | 可选   | INFO(默认),DEBUG |

### 4.2 参数说明

|参数名称|可选/必选| 参数说明                                                                                                                                                                                                                                                                                                      |
|-------------------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|model_path|必选| 模型路径。<br>类型：Str。                                                                                                                                                                                                                                                                                          |
|save_path|必选| 量化权重保存路径。<br>类型：Str。                                                                                                                                                                                                                                                                                      |
|device|可选| 量化设备。<br>1. 类型：Str。 <br>2. 参考值：'npu','npu:0,1,2,3','cpu'。 <br>3. 默认值为"npu"（单设备）。<br>4. 当配置文件启用分布式逐层量化，且指定多个设备时（如：'npu:0,1,2,3'），系统启动DP逐层量化，请确定配置的算法是否支持分布式执行，配置方式及算法支持详见[逐层量化及分布式逐层量化](#51-逐层量化及分布式逐层量化)。                                                                                                    |
|model_type|必选| 模型名称。<br>1. 类型：Str。 <br>2. 大小写敏感，请参考《[大模型支持矩阵](../../model_support/foundation_model_support_matrix.md)》。                                                                                                                                                                                                  |
|config_path|与"quant_type"不共存。| 指定配置路径。<br>1. 类型：Str。 <br>2. 配置文件格式为yaml。<br>3. 当前只支持最佳实践库中已验证的配置，若自定义配置，msModelSlim不为量化结果负责。配置指导可参考[量化配置协议详解](#6-量化配置协议详解)。 <br> 4. 用户选用config_path后，tag参数无效。                                                                                                                                              |
|quant_type|与"config_path"不共存| 量化类型。<br>w4a4, w4a8, w4a4c8, w4a4f8, w4a8c8, w8a16, w8a8, w8a8s, w8a8c8, w8a8f8, w4a4f4, w16a16s，请参考《[大模型支持矩阵](../../model_support/foundation_model_support_matrix.md)》。                                                                                                                                                                |
|tag|可选| 校验指定场景标签。<br>1. 类型：Str。<br> 2. 大小写不敏感，支持多个标签，用空格分割；支持用户确定地指定一种场景。<br> 3. 当前支持两类标签，每一类别可指定一种场景：指定使用的推理引擎，包含MindIE、vLLM-Ascend、SGLang等；指定推理用的硬件形态，包含Atlas_A2_Inference、Atlas_A3_Inference、Atlas_A2_Training、Atlas_A3_Training、Atlas_300I_Duo、CPU等。 <br> 4. 如果未找到已验证当前场景的配置，则与用户交互，询问是否采用匹配 quant_type 或 model_type 的量化配置。 |
|debug|可选| 启用调试模式。<br>1. 类型：Bool，默认值：False。 <br>2. 启用后会在量化完成时自动保存量化过程中的上下文信息到 `save_path/debug_info` 目录，用于问题排查和算法分析，详见《[调试模式使用指南](debug_mode.md)》。                                                                                                                                                                   |
|trust_remote_code|可选| 是否信任自定义代码。<br>1. 类型：Bool，默认值：False。 <br>2. 请确保加载的自定义代码文件的安全性，设置为True有安全风险。                                                                                                                                                                                                                                |
|h, help|可选| 命令行参数帮助信息                                                                                                                                                                                                                                                                                                 |

### 4.3 使用示例

#### 4.3.1 示例1：使用量化类型参数（推荐方式）

使用一键量化功能量化 Qwen2.5-7B-Instruct 模型，量化方式采用 w8a8：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type Qwen2.5-7B-Instruct \
  --quant_type w8a8 \
  --trust_remote_code True
```

其中：

- `${MODEL_PATH}` 为 Qwen2.5-7B-Instruct 原始浮点权重路径
- `${SAVE_PATH}` 为用户自定义的量化权重保存路径

#### 4.3.2 示例2：使用配置文件参数

使用自定义配置文件进行量化：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type ${MODEL_TYPE} \
  --config_path ${CONFIG_PATH} \
  --trust_remote_code ${TRUST_REMOTE_CODE}
```

#### 4.3.3 示例3：多卡分布式量化

使用4张NPU卡进行分布式量化：

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu:0,1,2,3 \
  --model_type ${MODEL_TYPE} \
  --quant_type w8a8 \
  --trust_remote_code True
```

>[!NOTE]
>
>在配置DP逐层量化之前，请首先确保配置的算法支持分布式执行，详见[逐层量化及分布式逐层量化](#51-逐层量化及分布式逐层量化)。

### 4.4 输出说明

一键量化生成的结果文件及其说明详见[一键量化生成结果](./quantization_result.md)。

## 5. 高级特性

### 5.1 逐层量化及分布式逐层量化

#### 5.1.1 简介

逐层量化（Layer-wise Quantization）是 modelslim_v1 量化服务的重要特性。通过逐层处理模型，显著降低内存占用，使得大模型量化成为可能。

在此基础上，**分布式逐层量化（DP Layer-wise Quantization）** 通过在多设备上进行数据并行（Data Parallel）来显著提升量化效率，同时保持逐层量化的低内存优势。

#### 5.1.2 适用场景

| 场景类型 | 场景描述 | 推荐方案 | 说明 |
|----------|----------|----------|------|
| 大模型量化 | 32B 及以上规模的模型 | 逐层量化 | 内存优势明显 |
| 内存受限环境 | NPU 内存不足以加载整网 | 逐层量化 | 大幅降低内存需求 |
| 追求量化速度 | 超大模型或大规模校准集 | 分布式逐层量化 | 多卡并行显著加速 |

#### 5.1.3 工作原理与优势对比

| 特性 | 传统量化 (Model-wise) | 逐层量化 (单卡) | 分布式逐层量化 (多卡 DP) |
|----------|----------|----------|----------|
| **处理方式** | 模型级整网处理 | 单设备分层顺序处理 | 多设备并行分层处理 |
| **内存占用** | 模型大小的 2-3 倍 | **单层大小的 2-3 倍** | **单层大小的 2-3 倍** |
| **量化效率** | 快 (针对小模型) | 较慢 (针对大模型) | **显著提升 (多卡并行)** |
| **适用模型** | 小模型 (<32B) | 大模型 (≥32B) | 超大模型或大规模校准集 |

#### 5.1.4 配置方法

**1. 通过配置文件指定 runner**

可以在 YAML 配置文件中指定 `runner` 类型，逐层量化和分布式逐层量化分别指定为 `layer_wise` 和 `dp_layer_wise`。如果设置为 `auto`（默认），系统会根据设备数量自动选择 `layer_wise` 或 `dp_layer_wise`。

```yaml
apiversion: "modelslim_v1"       # 协议版本
spec:
  runner: "dp_layer_wise"       # 量化调度器类型: DP逐层调度器
  process:
    - type: "linear_quant"
      qconfig:
        act:                    # 激活值量化配置
          scope: "per_tensor"   # 静态量化标识：整个张量共用量化参数
          dtype: "int8"         # 量化数据类型。默认：int8
          symmetric: false      # 是否对称量化。默认：false
          method: "minmax"      # 量化方法。默认：minmax
        weight:                 # 权重量化配置
          scope: "per_channel"  # 权重量化粒度：逐通道量化
          dtype: "int8"         # 量化数据类型。默认：int8
          symmetric: true       # 是否对称量化。默认：true
          method: "minmax"      # 量化方法。默认：minmax
      include: ["*"]            # 包含的层，支持通配符。默认：["*"]
```

**2. 通过命令行参数配置设备**

```bash
# 单卡逐层量化
msmodelslim quant --device npu:0 ...

# 多卡分布式逐层量化（自动启用 DP）
msmodelslim quant --device npu:0,1,2,3 ...
```

#### 5.1.5 注意事项

1. **分布式算法支持**：使用 `dp_layer_wise` 时，必须确保所有处理器（如 `linear_quant`）和算法（如 `minmax`, `ssz`, `iter_smooth`）均支持分布式执行。
2. **加速比说明**：多卡加速效果受校准集大小影响。若校准集过小，通信开销可能导致加速效果不明显。
3. **多模态限制**：分布式逐层量化暂不支持多模态模型，多模态场景请使用单卡 `layer_wise`。

#### 5.1.6 模型适配

逐层量化支持范围参考[大模型支持矩阵](../../model_support/foundation_model_support_matrix.md) 中支持一键量化的模型。
分布式逐层量化继承自逐层量化，因此支持所有逐层量化适配的大语言模型。

**注意**：DP逐层量化暂不支持多模态模型。多模态模型请使用单卡逐层量化（`layer_wise`）。

#### 5.1.7 算法适配

逐层量化（`layer_wise`）已支持 modelslim_v1 架构下的所有算法。

分布式逐层量化（`dp_layer_wise`）目前支持以下算法：

**离群值抑制算法**

| 算法名称 | 处理器类型 | 支持状态 | 说明 |
|----------|----------|---------|------|
| Iterative Smooth | iter_smooth | ✅ 支持 | 完全支持分布式执行 |
| Flex Smooth Quant | flex_smooth | ✅ 支持 | 完全支持分布式执行 |
| Flex AWQ SSZ | flex_awq_ssz | ✅ 支持 | 完全支持分布式执行 |
| QuaRot | quarot | ✅ 支持 | 完全支持分布式执行 |
| Online QuaRot | online_quarot | ✅ 支持 | 完全支持分布式执行 |
| Adapt Rotation | adapt_rotation | ✅ 支持 | 完全支持分布式执行 |

**量化算法**

| 算法名称 | 量化方法 | 支持状态 | 说明 |
|----------|---------|---------|------|
| MinMax | minmax | ✅ 支持 | 完全支持分布式执行 |
| SSZ | ssz | ✅ 支持 | 完全支持分布式执行 |
| CeilX | ceil_x | ✅ 支持 | 完全支持分布式执行 |
| FA3 Quant | fa3_quant | ✅ 支持 | 完全支持分布式执行 |

## 6. 量化配置协议详解

### 6.1 量化配置协议概述

一键量化配置协议采用分层结构设计思想，通过YAML把整条量化流水线抽象成配置：使用的量化服务版本、流水线类型、量化处理方式、保存策略以及量化校准集等。开发者只关心“策略和流程”，无需在 Python 里硬编码这些细节。

#### 6.1.1 基本结构

YAML配置文件的基本结构如下：

```yaml
apiversion: "modelslim_v1"   # 协议版本：用于选择后端量化服务的版本
spec:                         # 具体的量化服务配置字段
  runner: "auto"              # 量化调度器类型。默认：auto
  prior: [ ]                  # 前置阶段列表（可选）：先于主流程执行的阶段，每阶段含 process 与 dataset
  process: [ ]                # 处理器配置列表：按顺序执行每个处理器
  save: [ ]                   # 保存器配置列表：定义量化结果的保存方式
  dataset: "mix_calib.jsonl"  # 校准数据集配置：将会从lab_calib目录下匹配使用的校准集
```

#### 6.1.2 协议版本说明

| 参数           | 可选/必选 | 说明                                                                                                                                                                  | 作用                                |
|--------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| apiversion   | 必选    | 1. 当前支持列表：`"modelslim_v0"`、`"modelslim_v1"`、`"multimodal_vlm_modelslim_v1"`、`"multimodal_sd_modelslim_v1"`。<br> 2. 工具根据此字段选择对应的量化服务后端。<br> 3. 不同版本的量化服务可能有不同的配置字段和参数要求。 | 用于选择后端量化服务的版本，不同的量化服务有着不同的具体配置协议。 |
| spec         | 必选    | 1. **流水线定义**：指定量化处理的流水线类型。<br> 2. **处理器配置**：定义各种量化处理器的参数。<br>3. **保存策略**：指定量化结果的保存方式和格式 <br>4. **数据集配置**：指定校准数据集                            | 具体的量化服务配置字段，包含量化策略、处理流程和保存方式等所有具体参数。                                 |

**协议版本维护策略**：

| 协议版本         | 维护策略 | 状态  |
|--------------|------|-----|
| modelslim_v0 | 即将废弃 | 不推荐 |
| modelslim_v1 | 逐步完善 | 推荐  |
| multimodal_vlm_modelslim_v1 | 逐步完善 | 推荐  |
| multimodal_sd_modelslim_v1 | 逐步完善 | 推荐  |

### 6.2 modelslim_v1 配置详解

#### 6.2.1 功能说明

modelslim_v1是量化工具推出的新一代量化处理框架，目前正在快速演进中。

相较于modelslim_v0版本，modelslim_v1有以下优势：

- 算法独立实现，配置自由组合。
- 支持逐层量化，大幅降低资源消耗。
- 不依赖特定版本的CANN。

#### 6.2.2 runner - 量化调度器类型

**作用**: 定义量化处理的调度器类型。
**类型**: `string`。
**默认值**: `"auto"`。

| 可选值 | 说明 | 适用场景 | 特点 |
|--------|------|----------|------|
| auto | 工具自动选择 | 大多数场景 | 根据模型大小、可用内存、设备配置自动选择最优策略 |
| layer_wise | 逐层量化 | 大模型（≥32B） | 内存占用低，可能需要适配 |
| dp_layer_wise | 分布式逐层量化 | 大模型（≥32B）多卡场景 | 多设备并行，显著提升量化效率|
| model_wise | 非逐层量化 | 小模型（<32B） | 内存占用较高，兼容性好 |

#### 6.2.3 prior - 前置阶段配置

**作用**: 在**主流程**（`spec.process`）之前执行的一个或多个前置阶段，用于需要多阶段的算法（例如'adapt_rotation'算法）。

**类型**: 列表，每项为一个阶段配置对象。
**默认值**: `[]`（不配置时无前置阶段）。

**每个阶段包含**:

| 字段 | 类型 | 说明 |
|------|------|------|
| process | 列表 | 该阶段要执行的处理器列表，写法与 `spec.process` 一致。 |
| dataset | 字符串，可选 | 该阶段使用的校准数据集文件名（从 lab_calib 目录匹配）。不填时使用 `spec.dataset`。 |

**执行顺序**: 按 `prior` 中阶段顺序依次执行，全部完成后再执行 `spec.process` 主流程。前置阶段与主流程共享同一模型实例，前置阶段可通过上下文（Context）将结果传给主流程中的处理器。

**典型用法**: 使用 [Adapt Rotation](../../quantization_algorithms/outlier_suppression_algorithms/adapt_rotation.md) 时，将 Stage1 放在 `prior` 的某一阶段的 `process` 中并配置该阶段的 `dataset`，将 Stage2 与后续量化放在 `spec.process` 中。示例见 Adapt Rotation 文档中的 YAML 配置示例。

#### 6.2.4 process - 处理器配置字段

**作用**: 定义量化处理的处理器列表，按顺序执行每个处理器。

**特点**:

- **列表结构**: process 是处理器列表，包含多个处理器配置，不同的处理器配置以type字段为区分。
- **顺序执行**: 处理器按照在列表中的顺序依次执行。
- **灵活组合**: 可以组合不同类型的处理器实现复杂的量化策略，但并非所有配置组合都能正常运行。组合时应遵循以下原则（若缺乏相关使用经验，请参考本文后续提供的配置示例，或咨询专业人员获取指导）：
  - 先离群值抑制后量化，例如，当结合Iterative Smooth与W8A8量化时，需先进行Iterative Smooth，后进行W8A8量化。
  - 避免对同一个层进行多种量化设置，如果在配置文件中多次定义同一层的量化参数，可能导致运行时报错，或出现不符合预期的量化效果（如精度异常、量化功能失效等）。

##### 6.2.4.1 支持处理器表

| 处理器 | 处理器类型 | 配置示例 | 配置字段详解 |
| :--- | :--- | :--- | :--- |
| SmoothQuant | 离群值抑制 | [SmoothQuant 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/smooth_quant.md#yaml配置示例) | [配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/smooth_quant.md#yaml配置字段详解) |
| Iterative Smooth | 离群值抑制 | [Iterative Smooth 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/iterative_smooth.md#yaml配置示例) | [配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/iterative_smooth.md#yaml配置字段详解) |
| Flex Smooth Quant | 离群值抑制 | [Flex Smooth Quant 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/flex_smooth_quant.md#yaml配置示例) | [配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/flex_smooth_quant.md#yaml配置字段详解) |
| Flex AWQ SSZ | 离群值抑制 | [Flex AWQ SSZ 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/flex_awq_ssz.md#yaml配置示例) | [配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/flex_awq_ssz.md#yaml配置字段详解) |
| AWQ | 离群值抑制 | [AWQ 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/awq_smooth.md#yaml-配置示例) | [配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/awq_smooth.md#yaml-配置字段详解) |
| KV Smooth | 离群值抑制 | [KV Smooth 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/kv_smooth.md#yaml配置示例) | [KV Smooth 配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/kv_smooth.md#yaml配置字段详解) |
| QuaRot | 离群值抑制 | [QuaRot 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/quarot.md#yaml配置示例) | [QuaRot 配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/quarot.md#yaml配置字段详解) |
| Online QuaRot | 离群值抑制 | [Online QuaRot 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/quarot.md#yaml配置示例) | [Online QuaRot 配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/quarot.md#yaml配置字段详解) |
| Adapt Rotation | 离群值抑制 | [Adapt Rotation 配置示例](../../quantization_algorithms/outlier_suppression_algorithms/adapt_rotation.md#yaml-配置示例) | [配置字段详解](../../quantization_algorithms/outlier_suppression_algorithms/adapt_rotation.md#yaml-配置字段详解) |
| linear_quant | 量化 | [线性量化配置示例](../../quantization_algorithms/quantization_algorithms/linear_quant.md#yaml配置示例) | [线性量化配置字段详解](../../quantization_algorithms/quantization_algorithms/linear_quant.md#yaml配置字段详解) |
| group | 量化 | [group 配置示例](group.md#42-yaml配置示例) | [group 配置字段详解](group.md#43-yaml配置字段详解) |
| KVCache Quant | 量化 | [KVCache Quant 配置示例](../../quantization_algorithms/quantization_algorithms/kvcache_quant.md#yaml配置示例) | [KVCache Quant 配置字段详解](../../quantization_algorithms/quantization_algorithms/kvcache_quant.md#yaml配置字段详解) |
| FA3 Quant | 量化 | [FA3 Quant 配置示例](../../quantization_algorithms/quantization_algorithms/fa3_quant.md#yaml配置示例) | [FA3 Quant 配置字段详解](../../quantization_algorithms/quantization_algorithms/fa3_quant.md#yaml配置字段详解) |
| Float Sparse | 量化 | [Float Sparse 配置示例](../../quantization_algorithms/quantization_algorithms/float_sparse.md#yaml配置示例) | [Float Sparse 配置字段详解](../../quantization_algorithms/quantization_algorithms/float_sparse.md#yaml配置字段详解) |
| AutoRound | 量化 | [AutoRound 配置示例](../../quantization_algorithms/quantization_algorithms/autoround.md#yaml配置示例) | [AutoRound 配置字段详解](../../quantization_algorithms/quantization_algorithms/autoround.md#yaml配置字段详解) |

#### 6.2.5 save - 保存器配置字段

**作用**: 定义量化结果的保存器列表。modelslim_v1 当前支持 `ascendv1_saver`（昇腾推理，默认）、`compressed_tensors`（vLLM 等 HF 生态）两种保存器，格式对比请参见《[格式支持矩阵](../../quantization_formats/README.md)》。

##### 6.2.5.1 ascendv1_saver

**作用**: 保存为ascendv1格式。

**配置示例**:

```yaml
spec:
  save:
    - type: "ascendv1_saver"        # 保存器类型：保存为ascendv1格式
      part_file_size: 4            # 分片文件大小（GB）
```

**字段说明**:

| 字段名 | 作用 | 说明 |
|--------|------|------|
| type | 保存器类型标识 | 固定值"ascendv1_saver"，用于标识这是一个Ascend格式保存器 |
| part_file_size | 分片文件大小 | 分片文件的大小，单位为GB |

##### 6.2.5.2 compressed_tensors

**作用**: 保存为 [compressed-tensors](../../quantization_formats/compressed_tensors.md) 格式，面向 HuggingFace 生态推理框架（如 vLLM）。量化元数据写入 `config.json` 的 `quantization_config` 字段，权重写入 `model*.safetensors`。

**适用场景**:

- 目标推理框架为 vLLM 等支持 HF `quantization_config` 的引擎。
- 当前支持 **W8A8 静态量化**（`act.scope: "per_tensor"`）与 **W8A8 动态量化**（`act.scope: "per_token"`）。

**配置示例**:

```yaml
spec:
  save:
    - type: "compressed_tensors"   # 保存器类型：保存为 compressed-tensors 格式
      part_file_size: 4            # 分片文件大小（GB），0 表示不分片
```

**字段说明**:

| 字段名 | 作用 | 说明 |
|--------|------|------|
| type | 保存器类型标识 | 固定值 `"compressed_tensors"` |
| part_file_size | 分片文件大小 | 分片文件的大小，单位为 GB；`0` 表示不分片 |

**使用限制**:

- 当前仅支持线性层量化，不支持 KV Cache 量化。
- 不支持分布式导出。

格式协议与张量命名请参见《[compressed-tensors 格式说明](../../quantization_formats/compressed_tensors.md)》。

**完整配置示例**（W8A8 静态量化 + compressed-tensors 保存）:

```yaml
apiversion: modelslim_v1

spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_tensor"      # W8A8 Static：激活静态量化
          dtype: "int8"
          symmetric: False
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: True
          method: "minmax"
      include: ["*"]

  save:
    - type: "compressed_tensors"
      part_file_size: 4
```

#### 6.2.6 dataset - 校准数据集配置

**作用**: 指定校准数据集文件名，将会从lab_calib目录下匹配使用的校准集。
**类型**: `string`。
**默认值**: `"mix_calib.jsonl"`。

| 属性 | 说明 |
|------|------|
| 文件位置 | lab_calib目录下 |
| 文件格式 | JSONL格式 |
| 用途 | 用于激活值量化的校准过程 |

#### 6.2.7 使用示例

在一键量化中，通过 `qconfig.act.scope` 字段来区分 **静态量化** 与 **动态量化**：

- **静态量化 (`per_tensor`)**：在校准阶段统计并固定量化参数，推理时不再计算。特点是**推理性能最优**，硬件兼容性好。
- **动态量化 (`per_token`)**：在推理时为每个 token 动态计算量化参数。特点是**精度更高**，能有效应对激活值中的离群值分布。
以下示例展示了一份稠密模型的静态量化配置：

```yaml
apiversion: modelslim_v1       # 协议版本

spec:                          # 规格定义
  process:                     # 处理器执行列表
    - type: "linear_quant"     # 处理器类型：线性层量化
      qconfig:
        act:                   # 激活值量化配置
          scope: "per_tensor"  # 静态量化标识：整个张量共用量化参数
          dtype: "int8"        # 数据类型：int8
          symmetric: False     # 非对称量化：False（静态量化常用）
          method: "minmax"     # 量化方法：minmax
        weight:                # 权重量化配置
          scope: "per_channel" # 权重量化粒度：逐通道量化
          dtype: "int8"        # 数据类型：int8
          symmetric: True      # 对称量化：True
          method: "minmax"     # 量化方法：minmax
      include: ["*"]           # 包含的层，支持通配符。默认：["*"]
      exclude: ["*down_proj*"] # 排除的层，支持通配符。默认：[]

  save:                        # 保存器配置列表
    - type: "ascendv1_saver"   # 使用标准 ascendv1 格式保存
      part_file_size: 4        # 权重分片大小：4GB（建议大模型开启）
```

以下示例展示了如何针对MOE模型不同层使用不同的量化策略（混合量化）：

```yaml
apiversion: modelslim_v1       # 协议版本
                               #
# 定义 W8A8 动态量化配置模板
default_w8a8_dynamic: &default_w8a8_dynamic
  act:                         # 激活值配置
    scope: "per_token"         # 动态量化标识：每个 token 独立量化参数
    dtype: "int8"              # 数据类型：int8
    symmetric: True            # 对称量化：True
    method: "minmax"           # 量化算法：minmax
  weight:                      # 权重量化配置
    scope: "per_channel"       # 权重量化粒度：逐通道量化
    dtype: "int8"              # 数据类型：int8
    symmetric: True            # 对称量化：True
    method: "minmax"           # 量化算法：minmax
                               #
# 定义 W8A8 静态量化配置模板
default_w8a8: &default_w8a8    # 静态量化模板定义
  act:                         # 激活值配置
    scope: "per_tensor"        # 静态量化标识：整个张量共用量化参数
    dtype: "int8"              # 数据类型：int8
    symmetric: False           # 非对称量化：False（静态量化常用）
    method: "minmax"           # 量化算法：minmax
  weight:                      # 权重量化配置
    scope: "per_channel"       # 权重量化粒度：逐通道量化
    dtype: "int8"              # 数据类型：int8
    symmetric: True            # 对称量化：True
    method: "minmax"           # 量化算法：minmax
                               #
spec:                          # 规格定义
  process:                     # 处理器执行列表
    - type: "group"            # 使用组合处理器，支持对不同层应用不同配置
      configs:                 # 组合内的子处理器配置列表
        - type: "linear_quant" # 线性层量化子处理器 1
          qconfig: *default_w8a8 # 引用静态量化模板：对 Attention 层使用静态量化以优化性能
          include: ["*self_attn*"] # 匹配包含 self_attn 的层
                               #
        - type: "linear_quant" # 线性层量化子处理器 2
          qconfig: *default_w8a8_dynamic # 引用动态量化模板：对 MLP 层使用动态量化以保障精度
          include: ["*mlp*"]   # 匹配包含 mlp 的层
          exclude: ["*gate"]   # 从上述匹配中排除门控层
                               #
  save:                        # 保存器配置列表
    - type: "ascendv1_saver"   # 使用标准 ascendv1 格式保存
      part_file_size: 4        # 权重分片大小：4GB（建议大模型开启）
```

### 6.3 multimodal_sd_modelslim_v1 配置详解

#### 6.3.1 功能说明

multimodal_sd_modelslim_v1 面向文生视频 / 图生视频等多模态**生成**模型，基于 modelslim_v1 框架构建，默认采用逐层量化（`layer_wise`）。

**核心特性**:

- **多模态生成**：支持文本、图像等校准样本，经适配器桥接原推理仓完成浮点重放与 dump。
- **双编排路径**：按模型适配器接口自动分发——新模型走 `MultimodalPipelineInterface`（`inference_config` + `prepare_calib_data`）；主仓已接入模型保留 `LegacyMultimodalPipelineInterface`（`model_config` + `run_calib_inference`）。
- **多专家量化**：双 DiT 专家（如 Wan2.2）按专家名分别 dump、量化与保存。
- **校准数据缓存**：按专家命名 pth 文件，存在则加载，缺失则触发浮点推理生成。

**适用模型类型**（`--model_type` 须与 [`config/config.ini`](../../../../../config/config.ini) 一致）:

| 编排 | 典型 `model_type` | 说明 |
|------|-------------------|------|
| 重构 | `Wan2.2-T2V-A14B`、`Wan2.2-I2V-A14B`、`Wan2.2-TI2V-5B`、`HunyuanVideo` | 使用 `inference_config`；详见[《多模态生成模型接入指南》](../../../development_guide/integrating_multimodal_generation_model.md) |
| Legacy | `Wan2_1` / `Wan2.1`、`Wan2_2` / `Wan2.2`（单体）、`flux1`、`qwen_image_edit` 等 | 使用迁移期 `model_config`；行为与主仓历史版本兼容 |

**配置特点**:

- `spec.dataset`：校准样本（短名称 / 路径），重构路径下由适配器 `handle_dataset` 加载为 `VlmCalibSample` 列表。
- `multimodal_sd_config.dump_config`：校准 pth 的目录与捕获模式。
- `multimodal_sd_config.inference_config`：**推荐**；推理参数经 Pydantic 强校验后桥接到原推理仓 CLI。
- `multimodal_sd_config.model_config`：**即将废弃**（仅 Legacy）；与 `inference_config` 不可同时配置。

#### 6.3.2 runner - 量化调度器类型

当前多模态生成模型考虑到显存占用问题，默认且仅支持layer_wise（逐层量化）形式。runner默认无需配置，配置为非'layer_wise'值时，会警告提示并自动转换为layer_wise（逐层量化）形式。

#### 6.3.3 process - 处理器配置字段

此配置字段与 modelslim_v1 保持一致，参考[modelslim_v1 配置详解/process - 处理器配置字段](#624-process---处理器配置字段)

#### 6.3.4 <span id="save---保存器配置字段-sd">save - 保存器配置字段</span>

**作用**: 定义量化结果的保存器列表。

##### 6.3.4.1 mindie_format_saver

**作用**: 保存为MindIE-SD格式，专为多模态生成模型设计。

**配置示例**:

```yaml
spec:
  save:
    - type: "mindie_format_saver"   # 保存器类型：保存为MindIE-SD格式
      part_file_size: 0            # 分片文件大小（GB），0表示不分片
```

**字段说明**:

| 字段名 | 作用 | 说明 |
|--------|------|------|
| type | 保存器类型标识 | 固定值"mindie_format_saver"，用于标识这是一个MindIE-SD格式保存器 |
| part_file_size | 分片文件大小 | 分片文件的大小，单位为GB，0表示不分片 |

#### 6.3.5 multimodal_sd_config - 多模态生成特有配置字段

**作用**: 多模态生成模型特有的配置参数，包含校准数据捕获和模型加载与推理配置。

##### 6.3.5.1 <span id="dump_config---校准数据捕获配置">dump_config - 校准数据捕获配置</span>

**作用**: 配置校准数据的捕获方式和存储路径。

**配置示例**:

```yaml
spec:
  multimodal_sd_config:
    dump_config:
      enable_dump: True            # 是否启用校准数据的 load/dump。默认：True
      capture_mode: "args"         # 数据捕获模式。当前仅支持"args"
      dump_data_dir: ""            # 校准数据保存目录。空字符串时自动处理为权重保存路径
```

**字段说明**:

| 字段名 | 作用 | 说明                                                                                                            | 可选值 |
|--------|------|---------------------------------------------------------------------------------------------------------------|--------|
| enable_dump | 是否启用 dump | **Legacy 路径**：为 **False** 时不 load/dump，各专家 `calib_data` 直接置为 `None`（YAML 已显式配置，无需二次确认）。**重构路径**：量化服务将校准准备委托给适配器 `prepare_calib_data`；通常配合 `dump_data_dir` 下按专家命名的 pth——文件齐全则加载，缺失则浮点 dump。纯动态量化（如 W8A8 MXFP8）可设 `False` 且不依赖 pth，但量化时仍须为每个专家保留 `calib_data` 的 key | True（默认）/ False |
| capture_mode | 数据捕获模式 | 指定如何捕获模型的输入数据                                                                                                 | 当前仅支持"args"，其他模式待后续扩展 |
| dump_data_dir | 校准数据目录 | 检索与保存 pth 的根目录；空字符串时使用权重 `save_path`。重构路径下按专家生成文件名 `calib_data_<task_config>_<expert_name>.pth`（如 `calib_data_t2v-A14B_low_noise_model.pth`）。**全部 pth 已存在则直接加载；任一缺失则执行浮点推理 dump 后写入** | 字符串路径 |

**捕获模式说明**:

- **args**: 捕获位置参数，适用于大多数多模态生成模型。

**校准数据文件命名（重构路径）**:

- 单 DiT（如 HunyuanVideo）：`calib_data_<task_config>_.pth`（`expert_name` 为空字符串）
- 双专家 Wan2.2：`calib_data_<task_config>_low_noise_model.pth`、`calib_data_<task_config>_high_noise_model.pth`
- 量化阶段要求 `init_model()` 返回的每个专家在 `calib_data` 中均有对应 **key**；缺 key 将 fail-fast。`calib_data[expert]=None` 表示该专家无 dump 张量（如全动态量化），仍须保留 key。

**Legacy 路径补充**：`enable_dump=False` 时不会 load/dump，各专家 `calib_data` 直接置为 `None`；Legacy 缓存文件名仍为 `calib_data_<task_config>_<expert>.pth`（与重构命名一致），不再使用单一 `calib_data.pth`。

##### 6.3.5.2 inference_config - 推理参数配置（推荐）

**作用**: 配置浮点推理重放与量化桥接参数。由量化服务调用 `validate_inference_config`，使用适配器声明的 `InferenceConfig`（Pydantic，`extra="forbid"`）校验后，再经 `configure_runtime` 合并为原推理仓的 `model_args`。

**配置示例**（Wan2.2-T2V-A14B）:

```yaml
spec:
  dataset: wan2_2_t2v
  multimodal_sd_config:
    inference_config:
      size: "1280*720"
      frame_num: 81
      sample_steps: 40
      convert_model_dtype: True
      task: "t2v-A14B"   # 须与当前 --model_type 绑定，勿用于切换 T2V/I2V/TI2V
```

**字段说明**:

| 要点 | 说明 |
|------|------|
| 合法字段 | 以各模型适配器 `*InferenceConfig` 及原推理仓 CLI 为准；非法字段校验失败 |
| 与 `model_type`的关系 | 场景由 CLI `--model_type` 固定（如 `Wan2.2-T2V-A14B`），勿仅依赖 YAML 中的 `task` 切换场景 |
| 互斥 | 与 `model_config` 不可同时出现在同一 YAML 中 |

##### 6.3.5.3 model_config - 模型加载与推理配置（Legacy，将废弃）

**作用**: 仅 **Legacy** 适配器（`LegacyMultimodalPipelineInterface`）在 `set_model_args` 阶段读取；字段以原推理仓为准。

**迁移**: 新接入与重构后的模型请改用 `inference_config`。单独配置 `model_config` 时会打印废弃告警；与 `inference_config` 同时配置将报错。

**配置示例**（Legacy Wan2.1）:

```yaml
spec:
  multimodal_sd_config:
    model_config:
      prompt: "A stylish woman walks down a Tokyo street..." # 校准提示词
      offload_model: True          # 是否在推理后卸载模型到CPU
      frame_num: 121               # 视频生成的帧数
      task: "t2v-14B"              # 任务类型
      size: "1280*720"             # 生成尺寸规格
      sample_steps: 50             # 采样步数
```

#### 6.3.6 dataset - 校准数据集配置

**作用**: 指定校准样本，供重构路径 `handle_dataset` / Legacy 路径 `run_calib_inference` 使用。

**类型**: `string`（短名称、绝对路径或相对路径）。短名称在 [`lab_calib`](../../../../../lab_calib/) 下解析。

**样本格式**: 推荐 **`index.json` / `index.jsonl`**，每条至少包含非空 **`text`**；图生视频场景需按模型要求提供 **`image`**。字段约定与多模态理解类似，详见 [dataset - 校准数据路径配置](#dataset---校准数据路径配置)。

**各场景样本要求**（重构 Wan2.2）:

| `model_type` | 样本要求 |
|--------------|----------|
| `Wan2.2-T2V-A14B` | 必须有 `text`，不得带 `image` |
| `Wan2.2-I2V-A14B` | 必须有 `text` 与可访问的 `image` |
| `Wan2.2-TI2V-5B` | 必须有 `text`；`image` 可选 |

校准 pth 的生成与复用逻辑见 [dump_config - 校准数据捕获配置](#dump_config---校准数据捕获配置)。

#### 6.3.7 使用示例

- Wan2.1 Legacy W8A8 动态量化：[wan2_1_w8a8_dynamic.yaml](../../../../../lab_practice/wan2_1/wan2_1_w8a8_dynamic.yaml)
- Wan2.2-T2V W8A8 MXFP8 + QuaRot + FA3：[wan2_2_w8a8f8_mxfp_t2v.yaml](../../../../../lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_t2v.yaml)
- Wan2.2-I2V W8A8 MXFP8 + QuaRot + FA3：[wan2_2_w8a8f8_mxfp_i2v.yaml](../../../../../lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_i2v.yaml)
- Wan2.2-TI2V W8A8 MXFP8 + QuaRot + FA3：[wan2_2_w8a8f8_mxfp_ti2v.yaml](../../../../../lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_ti2v.yaml)
- HunyuanVideo：[hunyuan_video_w8a8f8_mxfp.yaml](../../../../../lab_practice/hunyuan_video/hunyuan_video_w8a8f8_mxfp.yaml)

### 6.4 multimodal_vlm_modelslim_v1 配置详解

#### 6.4.1 功能说明

multimodal_vlm_modelslim_v1是专门为多模态视觉语言模型（VLM）设计的量化服务，基于modelslim_v1框架构建。

**核心特性**:

- **多模态VLM支持**：针对图像-文本多模态理解模型优化。
- **逐层处理**：采用逐层量化策略，显著减少大模型量化时的显存消耗。
- **多种数据集格式**：支持图像目录和通过JSON/JSONL格式自定义文本prompt的多模态校准数据集。

**适用模型类型**:

- Qwen2.5-Omni系列：Qwen2.5-Omni-7B 等多模态端到端模型（文本/图像/音频/视频）
- Qwen3-VL-MoE系列：Qwen3-VL-235B-A22B、Qwen3-VL-30B-A3B等多模态模型
- 其他多模态VLM模型：待后续逐步支持

**配置特点**:

- 支持`dataset`字段配置校准数据集，支持三种使用方式：方式一 index.json/index.jsonl（推荐，支持多模态）、方式二 纯图像目录（后续不再演进）、方式三 图像目录+单个 json/jsonl（后续不再演进），详见下方 [dataset - 校准数据路径配置](#dataset---校准数据路径配置)
- 支持`default_text`字段配置默认文本 prompt（方式二必填；方式一在条目缺 text 字段时使用）
- 默认使用 layer_wise（逐层量化）模式，针对大规模多模态模型优化

#### 6.4.2 runner - 量化调度器类型

当前多模态VLM模型考虑到显存占用问题，默认且仅支持layer_wise（逐层量化）形式。runner默认无需配置，配置为非'layer_wise'字段时，会警告提示并自动转换为layer_wise（逐层量化）形式。

#### 6.4.3 <span id="process---处理器配置字段-vlm">process - 处理器配置字段</span>

此配置字段与 modelslim_v1 保持一致，参考[modelslim_v1 配置详解/process - 处理器配置字段](#624-process---处理器配置字段)

#### 6.4.4 default_text - 默认文本prompt配置

**作用**: 统一指定所有校准图像的默认文本prompt。
**类型**: `string`
**默认值**: `"Describe this image in detail."`
**限制**：不能使用空字符串作为文本prompt，当dataset字段配置为包含JSON/JSONL文件（用于描述每个图像的自定义文本prompt）的图像目录时，此字段失效。

#### 6.4.5 <span id="save---保存器配置字段-vlm">save - 保存器配置字段</span>

此配置字段与 modelslim_v1 保持一致，参考[modelslim_v1 配置详解/save - 保存器配置字段](#634-save---保存器配置字段)

**推荐配置**:

```yaml
spec:
  save:
    - type: "ascendv1_saver"    # 保存器类型：保存为ascendv1格式
      part_file_size: 4        # 分片文件大小（GB）。建议大模型进行分片保存
```

#### 6.4.6 <span id="dataset---校准数据路径配置">dataset - 校准数据路径配置</span>

**作用**: 指定校准数据集的路径。

**类型**: `string`

支持以下三种使用方式。`dataset` 可配置为**短名称**（在 lab_calib 等 dataset_dir 下查找）、**绝对路径**或**相对路径**。

---

**方式一：index.json / index.jsonl（推荐）**

指向 **index.json** 或 **index.jsonl** 文件，或指向**仅包含一个 index.json 或 index.jsonl** 的目录。支持多模态（图像、音频、视频），格式规范，后续功能会在此方式上演进。

- 每条为 JSON 对象，**至少包含 `text`**（非空字符串）；缺省时使用配置中的 `default_text`。
- 可选字段（若提供则路径必须存在）：`image`（.jpg/.jpeg/.png）、`audio`（.wav/.mp3）、`video`（.mp4）；路径相对 index 文件所在目录。

目录示例：

```text
calib_dir/
├── index.jsonl
├── img1.jpg
├── img2.png
└── a.wav
```

index.jsonl 示例：

```json
{"image": "img1.jpg", "text": "Describe this image."}
{"image": "img2.jpg", "audio": "a.wav", "text": "What is in this picture?"}
```

配置示例：`dataset: "/path/to/calib_dir"` 或 `dataset: "/path/to/index.jsonl"` 或短名称解析到上述路径。

---

**方式二：纯图像目录**

目录内仅包含图像文件，无 .json/.jsonl。所有图像使用配置中的 `default_text` 作为统一文本 prompt。

**说明**：此方式后续不再演进，新场景请使用方式一。

目录示例：

```text
calibImages/
├── img1.jpg
├── img2.png
└── img3.jpeg
```

配置示例：

```yaml
spec:
  dataset: "calibImages"   # 或绝对/相对路径
  default_text: "Describe this image in detail."
```

---

**方式三：图像目录 + 单个 .json/.jsonl（任意文件名）**

目录内包含图像及**一个**任意文件名的 .json 或 .jsonl 文件（文件名不为 index.json/index.jsonl），用于为每张图指定自定义文本。仅支持图像字段，不支持 audio/video。

**说明**：此方式后续不再演进，新场景请使用方式一。

目录示例：

```text
calibImages/
├── img1.jpg
├── img2.png
├── img3.jpeg
└── calib_data.jsonl
```

calib_data.jsonl 示例：

```json
{"image": "img1.jpg", "text": "What objects are in this image?"}
{"image": "img2.png", "text": "Describe the scene."}
```

配置示例：`dataset: "calibImages"` 或对应路径。

#### 6.4.7 使用示例

- Qwen2.5-Omni模型W8A8量化：[qwen2_5_omni_thinker_w8a8.yaml](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/qwen2_5_omni_thinker/qwen2_5_omni_thinker_w8a8.yaml)
- Qwen3-VL-MoE模型W8A8混合量化：[qwen3_vl_moe_w8a8.yaml](https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/qwen3_vl_moe/qwen3_vl_moe_w8a8.yaml)

### 6.5 modelslim_v0 配置说明

#### 6.5.1 功能说明

modelslim_v0量化服务主要由Calibrator、AntiOutlier等旧版接口组成，其配置协议（YAML）基本与原有的Python-API接口保持一致，便于从旧版本平滑迁移。

**相关文档**:

- [Calibrator.md](../../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)
- [AntiOutlier.md](../../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/AntiOutlier.md)

**注意**：modelslim_v0协议版本即将废弃，不推荐使用。建议使用modelslim_v1或更新的协议版本。

## 7. 附录

### 7.1 相关资料

- 对于过大的模型，可以参考[逐层量化及分布式逐层量化](#51-逐层量化及分布式逐层量化)使用逐层量化，能够明显降低显存使用。
- 对于一键量化支持的多种算法，可以参考[一键量化V1架构支持的算法](../../quantization_algorithms/README.md)。

### 7.2 常见问题

**Q1: 如何选择合适的量化调度器？**

- **auto**：适用于大多数场景，工具会自动根据模型大小、可用内存、设备配置选择最优策略
- **layer_wise**：适用于32B及以上规模的模型，内存受限环境
- **dp_layer_wise**：适用于超大模型或大规模校准集的多卡场景
- **model_wise**：适用于小模型（<32B），兼容性最好

**Q2: 逐层量化与分布式逐层量化的区别？**

- **逐层量化**：单设备逐层处理，内存占用为单层大小的 2-3 倍。
- **分布式逐层量化**：多设备并行逐层处理，在保持低内存优势的同时显著提升量化速度。详见[逐层量化及分布式逐层量化](#51-逐层量化及分布式逐层量化)。

**Q3: 如何判断是否需要使用逐层量化？**

如果遇到以下情况，建议使用逐层量化：

- 模型规模 > 32B
- NPU 内存不足
- 量化过程中出现内存溢出错误

**Q4: 多卡量化一定能加速吗？**

不一定。多卡量化的加速效果受多种因素影响：

- 校准集大小：校准集较小时，通信开销可能超过并行收益。
- 卡数选择：并非卡数越多速度越快，需要根据实际情况合理选择。
- 算法支持：只有支持分布式执行的算法才能在多卡环境下正常工作。详见[逐层量化及分布式逐层量化](#51-逐层量化及分布式逐层量化)。

**Q5: 如何做量化精度调优？**

量化精度调优可参考[量化精度调优指导](../../../best_practices/quantization_precision_tuning_guide.md)
