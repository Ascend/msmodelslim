# 多卡并行

## 并行机制简述

msModelSlim 的并行机制以数据并行（Data Parallel, DP）为基础，当一个量化流程中涉及的量化算法均已实现 DP 支持时，则该量化可使用多卡完成，因而我们也称 DP 为多卡量化的完备性支持。专家并行（Expert Parallel, EP）和分布式任务调度器（Distributed Task Scheduler, DTS）均是在完备性支持的基础上，在特定场景下对多卡量化进行的优化。EP 面向 MoE 模型，通过专家分片加载降低单卡显存占用，同时也起到量化提速的作用。DTS 面向计算密集型算法，将复杂流程拆为可调度的原子子任务，用卡间共享任务队列替代各卡重复执行完整任务链，以提升复杂算法的执行效率。

并行机制的原理决定了它们的适配体现在 msModelSlim 的不同概念域，DP 与 DTS 的适配体现在算法层面，msModelSlim 的算法则以 Processor 承载；EP 的适配体现在模型层面，msModelSlim 通过模型适配器承载对模型的改造与适配。使能一个量化任务的多卡量化的必要条件是完成量化方案中涉及到的算法的 DP 支持，EP 和 DTS 均在此基础上进行优化。由于当前基础模型的演进存在参数量持续上升的趋势，因而对于 MoE 结构的模型，建议在模型适配时完成 EP 支持以防止显存溢出。DTS 的支持则需要结合算法本身进行具体分析，只有对计算密集型的复杂算法它才会带来较为显著的加速效果。

## 并行机制目录

本目录收录 msModelSlim 多卡量化相关的流程指南与术语百科。

| 文档 | 说明 |
|------|------|
| 《[数据并行](data_parallelism/term_data_parallelism.md)》 | DP 术语百科 |
| 《[数据并行机制使用指南](data_parallelism/data_parallelism_guide.md)》 | DP 使用指南 |
| 《[专家并行](expert_parallelism/term_expert_parallelism.md)》 | EP 术语百科 |
| 《[专家并行机制使用指南](expert_parallelism/expert_parallelism_guide.md)》 | EP 使用指南 |
| 《[分布式任务调度器](distributed_task_scheduler/term_distributed_task_scheduler.md)》 | DTS 术语百科 |
| 《[分布式任务调度器使用指南](distributed_task_scheduler/distributed_task_scheduler_guide.md)》 | DTS 使用指南 |

## 并行机制支持列表

下表汇总当前量化算法的并行机制支持情况。

| Processor | DP 支持 | DTS 支持 |
|-----------|:----------:|:--------:|
| `AdaptRotationProcessor` | ✓ | — |
| `FA3QuantProcessor` | ✓ | — |
| `FlexSmoothQuantProcessor` | ✓ | ✓ |
| `FlexAWQSSZProcessor` | ✓ | ✓ |
| `IterSmoothProcessor` | ✓ | — |
| `LinearQuantProcessor` | ✓ | ✓ |
| `OnlineQuaRotProcessor` | ✓ | — |
| `QuaRotProcessor` | ✓ | — |

## 多卡量化配置与使用

通过命令行 `--device` 参数传入多个设备索引即可启动多卡量化：

```shell
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --model_type ${MODEL_TYPE} \
  --quant_type ${QUANT_TYPE} \
  --device npu:0,1,2,3,4,5,6,7
```

**说明**：多卡量化启用后会自动调用已适配支持的 DP、EP、DTS 等并行机制。
