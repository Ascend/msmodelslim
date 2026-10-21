# Multi-Device Quantization Parallel

> **Scope note**: the actual content of this directory covers only the multi-device quantization mechanisms (DP / EP / DTS). **The multi-device quantization mechanisms do not cover weight conversion (Convert); weight conversion follows its own independent conversion parallelism scheme** (NPU multi-device uses `--device npu --device_id` to start one independent compute process per device, and CPU parallelism uses `parallel.cpu_workers`). For details, see *[Weight Conversion Usage Guide](../ptq/convert/usage_weight_conversion.md)*.

## 1. Brief Description of the Parallel Mechanisms

The parallel mechanisms on the msModelSlim quantization side are based on data parallelism (Data Parallel, DP). When every quantization algorithm involved in a quantization workflow already implements DP support, that quantization can be completed on multiple devices; we therefore also call DP the completeness support of multi-device quantization. Expert parallelism (Expert Parallel, EP) and the distributed task scheduler (Distributed Task Scheduler, DTS) are both optimizations of multi-device quantization for specific scenarios, built on top of that completeness support. EP targets MoE models and reduces single-device memory usage by loading experts in shards, while also speeding up quantization. DTS targets compute-intensive algorithms: it splits a complex workflow into schedulable atomic subtasks and replaces every device repeatedly executing the whole task chain with a task queue shared between devices, thereby improving the execution efficiency of complex algorithms.

The principles of the parallel mechanisms determine that their adaptation surfaces in different concept domains of msModelSlim: DP and DTS are adapted at the algorithm level, and the algorithms of msModelSlim are carried by Processors; EP is adapted at the model level, and msModelSlim carries the modification and adaptation of models through model adapters. The prerequisite for enabling multi-device quantization for a quantization task is completing DP support for the algorithms involved in the quantization scheme; EP and DTS are both optimizations on top of that. Because the parameter count of foundation models keeps growing, for models with an MoE structure it is recommended to complete EP support during model adaptation to prevent memory overflow. DTS support must be analyzed case by case together with the algorithm itself: it produces a significant speedup only for compute-intensive complex algorithms.

## 2. Parallel Mechanism Index

This directory collects the workflow guides and terminology entries related to msModelSlim **multi-device quantization**.

| Document | Description |
|------|------|
| *[Data Parallelism](data_parallelism/term_data_parallelism.md)* | DP terminology entry |
| *[Data Parallelism Usage Guide](data_parallelism/data_parallelism_guide.md)* | DP usage guide |
| *[Expert Parallelism](expert_parallelism/term_expert_parallelism.md)* | EP terminology entry |
| *[Expert Parallelism Usage Guide](expert_parallelism/expert_parallelism_guide.md)* | EP usage guide |
| *[Distributed Task Scheduler](distributed_task_scheduler/term_distributed_task_scheduler.md)* | DTS terminology entry |
| *[Distributed Task Scheduler Usage Guide](distributed_task_scheduler/distributed_task_scheduler_guide.md)* | DTS usage guide |

## 3. Parallel Mechanism Support List

The following table summarizes the current parallel mechanism support of the quantization algorithms.

| Processor | DP Support | DTS Support |
|-----------|:----------:|:--------:|
| `AdaptRotationProcessor` | ✓ | — |
| `FA3QuantProcessor` | ✓ | — |
| `FlexSmoothQuantProcessor` | ✓ | ✓ |
| `FlexAWQSSZProcessor` | ✓ | ✓ |
| `IterSmoothProcessor` | ✓ | — |
| `LinearQuantProcessor` | ✓ | ✓ |
| `OnlineQuaRotProcessor` | ✓ | — |
| `QuaRotProcessor` | ✓ | — |

Summarized from the algorithm-name perspective, `dp_layer_wise` (distributed layer-wise quantization) currently supports the following algorithms:

**Outlier Suppression Algorithms**

| Algorithm | Processor Type | Support Status | Description |
|----------|----------|---------|------|
| Iterative Smooth | iter_smooth | ✅ Supported | Fully supports distributed execution |
| Flex Smooth Quant | flex_smooth | ✅ Supported | Fully supports distributed execution |
| Flex AWQ SSZ | flex_awq_ssz | ✅ Supported | Fully supports distributed execution |
| QuaRot | quarot | ✅ Supported | Fully supports distributed execution |
| Online QuaRot | online_quarot | ✅ Supported | Fully supports distributed execution |
| Adapt Rotation | adapt_rotation | ✅ Supported | Fully supports distributed execution |

**Quantization Algorithms**

| Algorithm | Quantization Method | Support Status | Description |
|----------|---------|---------|------|
| MinMax | minmax | ✅ Supported | Fully supports distributed execution |
| SSZ | ssz | ✅ Supported | Fully supports distributed execution |
| CeilX | ceil_x | ✅ Supported | Fully supports distributed execution |
| FA3 Quant | fa3_quant | ✅ Supported | Fully supports distributed execution |

The following table summarizes the models that currently support EP.

| Model Adapter | Specific Model |
|-----------|---------|
| `DeepSeekV3ModelAdapter` | `DeepSeek-V3`, `DeepSeek-V3-0324`, `DeepSeek-R1`, `DeepSeek-R1-0528`, `DeepSeek-V3.1`, `DeepSeek-V3.1-Terminus` |
| `DeepSeekV32ModelAdapter` | `DeepSeek-V3.2-Exp`, `DeepSeek-V3.2` |
| `DeepSeekV4ModelAdapter` | `DeepSeek-V4-Flash`, `DeepSeek-V4-Pro` |
| `DeepSeekV4DSparkModelAdapter` | `DeepSeek-V4-Flash-DSpark`, `DeepSeek-V4-Pro-DSpark`, `DeepSeek-V4-Flash-0731`, `DeepSeek-V4-Pro-0813` |
| `GLM5ModelAdapter` | `GLM-5`, `GLM-5.1` |
| `GLM52ModelAdapter` | `GLM-5.2`, `GLM-5.3` |
| `GLM5NextModelAdapter` | `GLM-5.3-Flash` |
| `KimiK2ModelAdapter` | `Kimi-K2-Instruct-0905`, `Kimi-K2-Thinking` |
| `KimiK3ModelAdapter` | `Kimi-K3` |

## 4. Multi-Device Quantization Configuration and Usage

Multi-device quantization can be started by passing multiple device indexes through the command-line `--device npu --device_id 0 1 ...` parameters:

```shell
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --model_type ${MODEL_TYPE} \
  --quant_type ${QUANT_TYPE} \
  --device npu \
  --device_id 0 1 2 3 4 5 6 7
```

**Note**: under the quantization path, DP, EP, and DTS are enabled according to what has been adapted. The multi-device quantization mechanisms do not cover weight conversion (Convert); weight conversion follows its own scheme. For details, see *[Weight Conversion Usage Guide](../ptq/convert/usage_weight_conversion.md)*.
