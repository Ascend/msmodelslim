# Kimi-K3 新模型 W4A8 量化案例

## 1. 案例背景

**目标**：将 2026-07 新发布的 Kimi-K3 适配接入 msModelSlim，完成 W4A8 量化且精度达标。

**覆盖流程**：模型适配 → 量化方案设计 → 权重量化 → 精度测评。

**关联流程**：《[多模态理解模型接入指南](../knowledge_base/model/integrating_multimodal_understanding_model.md)》、《[一键量化使用说明](../user_guide/usage_quick_quantization.md)》

## 2. 环境与版本

| 项 | 版本或配置 |
| --- | --- |
| 产品形态 | 昇腾 A3 系列产品（限定，本案例基于昇腾 A3 系列产品完成验证） |
| vLLM Ascend | [vllm-ascend:v0.23.0rc1-a3](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=v0.23.0rc1-a3) |
| CANN | 9.0.1（随镜像预置） |
| PyTorch | 2.10.0（随镜像预置） |
| TorchNPU | 2.10.0.post2（随镜像预置） |
| transformers | 4.57.6（量化时需要降级，镜像预置版本为 5.5.4） |
| compressed-tensors | 0.13.0（量化时需要降级，镜像预置版本为 0.17.0） |
| fla-core | 0.5.1 |
| 评测工具 | [AISBench](https://github.com/AISBench/benchmark) |

**本案例前置条件**：

- 已完成[Kimi-K3 权重](https://huggingface.co/moonshotai/Kimi-K3)下载。
- 已参考[使用 Docker](https://docs.vllm.ai/projects/ascend/zh-cn/v0.23.0/installation.html#set-up-using-docker)启动 vLLM Ascend 官方镜像容器并挂载 NPU 设备与模型权重目录。
- 已参考[环境准备](../install_guide/install_guide.md#231-环境准备)拉取 msModelSlim 源码（本案例以源码方式适配接入新模型）。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点权重 | [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | 权重文件完整 | 下载完成且哈希值匹配 |
| 交付件 | 模型适配器代码 | [`msmodelslim/model/kimi_k3/`](../../../msmodelslim/model/kimi_k3/) | 适配器需实现量化流水线所需接口 | 命令行指定 `--model_type Kimi-K3` 可命中适配器 |
| 交付件 | 量化最佳实践 | [`lab_practice/kimi_k3/kimi_k3_w4a8.yaml`](../../../lab_practice/kimi_k3/kimi_k3_w4a8.yaml) | 遵循[量化配置协议](../user_guide/usage_quick_quantization.md#5-量化配置协议详解) | 命令行参数 `--config_path` 指定量化配置文件 |
| 交付件 | 量化权重目录 | 命令行参数 `--save_path` 指定保存位置 | 遵循[AscendV1 格式](../knowledge_base/quantization_format/ascendv1/ascendv1_usage.md) | 量化权重精度测试达标 |

## 4. 操作步骤

### 步骤 1：模型适配

**目标**：完成 Kimi-K3 适配接入 msModelSlim。

**输入**：浮点权重。

**操作**：

1. 查看模型结构配置：

   ```bash
   cat ${MODEL_PATH}/config.json
   ```

   `config.json` 中需重点确认以下结构参数：

   | 参数 | 取值 | 说明 |
   | --- | --- | --- |
   | `auto_map.AutoModelForCausalLM` | `modeling_kimi_k3.KimiK3ForConditionalGeneration` | 指明了模型的实际定义类名 |
   | `format` | `mxfp4-pack-quantized` | 原生权重中包含 MXFP4 量化权重，需要离线或在线完成权重反量化 |
   | `mm_projector_type` | `patchmergerv2` | 视觉特征的融合模块类，影响旋转量化算法适配 |

2. 按照《[多模态理解模型接入指南](../knowledge_base/model/integrating_multimodal_understanding_model.md)》开发模型适配器（[`msmodelslim/model/kimi_k3/model_adapter.py`](../../../msmodelslim/model/kimi_k3/model_adapter.py)）。对于 Kimi-K3，即使采用逐层加载方式，单卡 64GB 显存仍会溢出，需参考《[专家并行机制使用指南](../knowledge_base/parallel/expert_parallelism/expert_parallelism_guide.md)》完成专家并行的适配（[`ep_patches.py`](../../../msmodelslim/model/kimi_k3/ep_patches.py)）。

3. 注册模型适配器（[`config/config.ini`](../../../config/config.ini)）：

   ```ini
   [ModelAdapter]
   kimi_k3 = Kimi-K3

   [ModelAdapterEntryPoints]
   kimi_k3 = msmodelslim.model.kimi_k3.loader:KimiK3AdapterLoader

   [ModelAdapterDependencies]
   kimi_k3 = {"transformers": "==4.57.6", "compressed-tensors": "==0.13.0"}
   ```

**输出**：

- 模型适配相关代码。

---

### 步骤 2：量化方案设计

**目标**：结合模型结构与原厂权重特性，设计 W4A8 量化方案。

**输入**：步骤 1 确认的模型结构特性。

**操作**：

1. 确定整体量化策略。

   由于 Kimi-K3 的浮点权重已原生包含量化，量化方案设计时总体上遵循与原生权重的 bit 数一致的原则。浮点权重的路由专家（`experts.*.w1/w2/w3`）权重原生为 MXFP4，因而对路由专家做 W4A8 动态量化。

2. 局部调整量化策略。

   共享专家（`shared_experts.{up,down,gate}_proj`）原生为 BF16，经验上对共享专家做 W8A8 动态量化精度可控，因而为提升量化后的性能，共享专家增加 W8A8 动态量化。

   `routed_expert_up_proj` / `routed_expert_down_proj`原生也为 BF16，但由于这两层权重的 shape 较大，为取得显存收益，对这两层也增加 W8A8 动态量化。

**输出**：

- Kimi-K3 W4A8 量化实践配置。

---

### 步骤 3：权重量化

**目标**：环境就绪后，使用 `msmodelslim quant` 一键生成 W4A8 量化权重。

**输入**：浮点权重目录 `${MODEL_PATH}`；[步骤 2](#步骤-2量化方案设计)输出的量化配置 `${YAML_PATH}`。

**操作**：

1. 重新安装 msmodelslim：

   在[步骤 1](#步骤-1模型适配)完成模型适配的源代码路径执行 `bash install.sh` 重新安装 msmodelslim，使模型适配代码生效。

2. 执行一键量化：

   ```bash
   cd ..  # 返回上级目录，msmodelslim 命令需在源码目录外执行
   msmodelslim quant \
       --model_path ${MODEL_PATH} \  # ${MODEL_PATH} 替换为模型路径
       --save_path ${SAVE_PATH} \  # ${SAVE_PATH} 替换为量化权重保存路径
       --device npu --device_id 0 1 2 3 4 5 6 7 \
       --model_type Kimi-K3 \
       --config ${YAML_PATH} \  # ${YAML_PATH} 替换为量化配置文件路径
       --trust_remote_code True
   ```

**输出**：

- Kimi-K3 W4A8 量化权重。

---

### 步骤 4：精度测评

**目标**：将量化权重部署拉起服务化，进行精度测评。

**输入**：量化权重目录 `${SAVE_PATH}`。

**操作**：

1. 部署前还原镜像预置的 Transformers 版本（量化完成后还原为容器初始版本）：

   ```bash
   pip install transformers==5.5.4
   ```

2. 参考[vLLM Ascend指导文档](https://docs.vllm.ai/projects/ascend/zh-cn/v0.23.0/tutorials/models/Kimi-K3.html)进行量化模型部署。

3. 服务启动后，使用 curl 验证量化模型对话正常

   ```bash
   # 分别替换 <NODE0_LOCAL_IP> 和 <SERVICE_PORT> 为设备 IP 和服务化端口
   curl http://<NODE0_LOCAL_IP>:<SERVICE_PORT>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi-k3",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "The future of AI is"
            }]
        }],
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
   ```

4. 参考《[AISBench教程](https://github.com/AISBench/benchmark/blob/master/README.md)》进行数据集精度测评。

   - 精度结果：

     | 数据集 | 基线 | 量化权重 | 差值 | 结论 |
     | --- | --- | --- | --- | --- |
     | GPQA | 92.9 | 94.4 | +1.5 | 精度达标 |
     | OCRBench | 89.1 | 91.7 | +2.6 | 精度达标 |

   > [!NOTE]
   >
   > 评测数据可能存在波动，若单次测评结果不达标，建议以多次测评的平均结果为准。若多次测评结果仍不达标，可参考《[量化精度调优指南](../user_guide/process_quantization_precision_tuning.md)》进行精度调优。

**输出**：

- vLLM Ascend 推理服务正常运行，curl 请求返回正常响应；AISBench 精度测评结果精度达标。

## 5. 结果与经验

### 5.1 关键结果汇总

| 步骤 | 关键操作 | 指标 | 变化 | 备注 |
| --- | --- | --- | --- | --- |
| 步骤 1 | 开发并注册模型适配器 | `--model_type Kimi-K3` 可命中适配器 | 新增模型适配器 | 开发完成后需重新安装以使代码生效 |
| 步骤 2 | 设计混合量化方案 | [kimi_k3_w4a8.yaml](../../../lab_practice/kimi_k3/kimi_k3_w4a8.yaml) | 新增量化实践配置 | 量化实践配置遵循[量化配置协议](../user_guide/usage_quick_quantization.md#5-量化配置协议详解) |
| 步骤 3 | 执行量化命令 | W4A8 量化权重 | 导出量化权重 | 导出件遵循[AscendV1 格式](../knowledge_base/quantization_format/ascendv1/ascendv1_usage.md) |
| 步骤 4 | vLLM Ascend 部署 + AISBench 精度测评 | [GPQA](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/gpqa/README.md) / [OCRBench](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/ocrbench_v2/README.md) | 完成精度验证 | 精度达标 |

### 5.2 经验总结

1. **超大模型需结合逐层和多卡量化**：随着模型体量增长，对于单卡显存不足以支撑单层权重加载的情况，需要同时适配支持逐层与多卡量化。

2. **EP实现存在多种方式**：对于 MoE 模型的 EP 实现，可以通过修改模型建模的源码静态完成，如[DeepSeek-V4 的 EP 实现](../../../msmodelslim/model/deepseek_v4/model.py)；也可以通过 monkey-patch 的方式动态修改模型建模完成，如[Kimi-K3 的 EP 实现](../../../msmodelslim/model/kimi_k3/ep_patches.py)。

## 6. 异常处理

- **适配器未生效（`--model_type` 无法识别）**：确认 [`config/config.ini`](../../../config/config.ini) 已参照[步骤 1](#步骤-1模型适配)注册 `ModelAdapterEntryPoints`，并在源代码目录重新执行 `bash install.sh` 使修改生效。

## 7. [OPTIONAL] 附录

- 相关 PR：[[feature] support kimi_k3 model adapter](https://gitcode.com/Ascend/msmodelslim/pull/789)
