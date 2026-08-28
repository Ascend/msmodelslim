# DeepSeek-V4-Flash W8A8 一键量化案例

## 1. 案例背景

**目标**：对 DeepSeek-V4-Flash 模型进行 W8A8 一键量化，并部署到 vLLM Ascend 推理引擎进行数据集评测精度验证，完成从量化到部署的全流程闭环。

**覆盖流程**：环境准备与工具安装 → 确认模型与量化方案 → 执行一键量化命令 → 量化权重验收 → 推理部署与评测。

**精度验收口径**：以选定的精度基线（本案例采用论文或模型卡片公开基线）作为对比基准，使用 AISBench 获得量化结果分数。当量化结果分数不低于选定的精度基线分数时直接通过；当量化结果分数低于选定的精度基线分数时，要求相对下降比例不超过 1%，否则不通过。

**关联流程**：[《一键量化完整指南》](../user_guide/usage_quick_quantization.md)、[《权重量化使用指南》](../user_guide/usage_weight_quantization.md)、[《msModelSlim 安装指南》](../install_guide/install_guide.md)

## 2. 环境与版本

| 项 | 版本或配置 |
| --- | --- |
| 产品形态 | Atlas 800I A3 推理服务器（限定，本案例基于该服务器单节点 16 卡环境验证） |
| 环境镜像 | [quay.io/ascend/vllm-ascend:v0.22.1rc1-a3](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=v0.22.1rc1-a3) |
| CANN | CANN-9.0.0（随镜像预置） |
| PyTorch | 2.10.0（随镜像预置） |
| TorchNPU | 2.10.0（随镜像预置） |
| vLLM Ascend | 0.22.1rc1（随镜像预置） |
| 其他依赖 | `transformers==4.48.2`（量化时需要降级，镜像内默认 5.5.4） |

**本案例前置条件**：

- 已获取 DeepSeek-V4-Flash 浮点权重（下载地址：[ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash)）。
- 已确认目标模型在支持矩阵中，`--model_type DeepSeek-V4-Flash`、`--quant_type w8a8` 已验证通过。
- 已启动 vLLM Ascend 官方镜像容器，并挂载了 NPU 设备与模型权重目录。

> [!NOTE]
>
> 本案例使用的浮点权重为非 0731 版本，0731 版本的支持将随后续版本更新。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型目录 | `${MODEL_PATH}`（用户下载） | 含 `config.json`、`tokenizer.json`、模型权重文件（safetensors）；可被 Transformers 4.48.2 加载 | 使用 `transformers.AutoModel.from_pretrained` 加载无报错 |
| 输入 | 校准数据集 | [混合校准数据集（mix_calib.jsonl）](../../../lab_calib/mix_calib.jsonl) | JSONL 格式，每行含 `inputs_pretokenized` 字段 | 校准集可以在量化流程被正常读取 |
| 交付件 | 量化权重目录 | `${SAVE_PATH}`（量化流程输出目录） | 含 `quant_model_description.json`、权重文件（safetensors 格式）、index、`config.json` 等 | 检查文件清单、safetensors 结构及模型结构与浮点模型一致，并可被 vLLM Ascend `--quantization ascend` 加载 |
| 交付件 | 精度测试结果 | AISBench 输出目录（`${EVAL_WORK_DIR}`）及 `${CASE_ROOT}` 归档目录 | 含评测配置、服务日志、curl 响应、选定的精度基线分数、量化结果分数与验收计算记录 | 量化结果不低于选定的精度基线时直接通过；低于选定的精度基线时相对下降不超过 1% |

## 4. 操作步骤

### 步骤 1：环境准备与工具安装

**目标**：确认容器环境满足量化要求，安装 msModelSlim 工具及依赖。

**输入**：vLLM Ascend 官方镜像容器已启动。

**操作**：

1. 若尚未创建容器，请参考 [《vLLM Ascend 官方指导》](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Flash.html) 拉取镜像并创建容器，创建时需挂载 NPU 设备与模型权重目录。本案例使用的镜像为 `m.daocloud.io/quay.io/ascend/vllm-ascend:v0.22.1rc1-a3`。

2. 在容器内检查 NPU 状态与 CANN 环境：

   ```bash
   pip list 2>/dev/null | grep -iE "torch|npu"
   env | grep ASCEND
   npu-smi info
   ```

3. 记录当前 Transformers 版本，以便量化完成后还原：

   ```bash
   pip list 2>/dev/null | grep transformers
   # 当前版本为 5.5.4
   ```

4. 下载 msModelSlim 源码并安装：

   ```bash
   git clone https://gitcode.com/Ascend/msmodelslim
   cd msmodelslim
   bash install.sh
   cd ..  # 退出源码目录，msmodelslim 命令需在源码目录外执行
   msmodelslim --help
   ```

5. 安装与 DeepSeek-V4-Flash 适配的 Transformers 版本：

   ```bash
   pip install transformers==4.48.2
   ```

   > [!NOTE]
   >
   > msModelSlim 不强求特定 Transformers 版本，但 DeepSeek-V4-Flash 模型建议使用 `transformers==4.48.2`。降级仅用于量化阶段，量化完成后可执行 `pip install transformers==5.5.4` 还原。

**输出**：

- `msmodelslim --help` 可正常执行。
- Transformers 版本已切换为 `4.48.2`。

**记录**：执行 `pip list | grep transformers` 确认版本。

**下一步**：确认模型与量化方案。

---

### 步骤 2：确认模型与量化方案

**目标**：确认 `model_type`、`quant_type` 与目标模型匹配。

**输入**：浮点权重已下载至 `${MODEL_PATH}`。

**操作**：

1. 查看模型量化配置：

   ```bash
   cat msmodelslim/lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml
   ```

   该配置采用 QuaRot 旋转预处理 + Flex Smooth Quant 离群值抑制 + W8A8 动态量化（per-token/per-channel）策略。

2. 确认浮点权重路径：

   ```bash
   ls ${MODEL_PATH}/
   # 应包含 config.json、tokenizer.json、*.safetensors 等文件
   ```

**输出**：

- 已确认 `--model_type DeepSeek-V4-Flash`、`--quant_type w8a8`。
- 浮点权重已就绪。

**记录**：量化配置 YAML 文件路径为 `lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml`。

**下一步**：执行一键量化命令。

---

### 步骤 3：执行一键量化命令

**目标**：使用 `msmodelslim quant` 命令生成量化权重。

**输入**：浮点权重目录 `${MODEL_PATH}`。

**操作**：

```bash
msmodelslim quant \
    --model_path ${MODEL_PATH} \
    --save_path ${SAVE_PATH} \
    --model_type DeepSeek-V4-Flash \
    --quant_type w8a8 \
    --device npu --device_id 0 \
    --trust_remote_code true
```

参数说明：

- `--model_path`：浮点模型权重目录路径。
- `--save_path`：量化权重保存目录路径。
- `--model_type`：模型类型，固定为 `DeepSeek-V4-Flash`。
- `--quant_type`：量化策略类型，`w8a8` 表示权重和激活均为 INT8。
- `--device npu --device_id 0`：`--device` 指定设备类型为 NPU，`--device_id` 指定设备编号。
- `--trust_remote_code`：信任自定义模型代码（请确保代码来源可靠）。

**输出**：

- 量化权重目录 `${SAVE_PATH}`，包含 `quant_model_description.json`、权重文件（safetensors 格式）、`config.json` 等。

**记录**：量化日志输出至终端，可重定向至文件备查。

**下一步**：量化权重验收。

---

### 步骤 4：量化权重验收

**目标**：确认量化权重产物完整。

**输入**：量化权重生成目录 `${SAVE_PATH}`。

**操作**：

1. 检查输出目录结构：

   ```bash
   ls ${SAVE_PATH}/
   # 应包含 quant_model_description.json、config.json、*.safetensors 等
   ```

2. 确认 `quant_model_description.json` 存在且内容完整：

   ```bash
   cat ${SAVE_PATH}/quant_model_description.json | head -20
   ```

**输出**：

- 已验证的量化权重目录 `${SAVE_PATH}`。

**通过条件**：量化产物目录符合 AscendV1 导出格式约定（`quant_model_description.json` + 权重 safetensors），详见 [AscendV1 导出产物](../knowledge_base/quantization_format/ascendv1/term_ascendv1.md#export-artifacts)；且量化日志中无 `ERROR`。

**记录**：量化权重目录结构及 `quant_model_description.json` 摘要。

**下一步**：推理部署与评测。

---

### 步骤 5：推理部署与评测

**目标**：将量化权重部署到 vLLM Ascend 推理引擎，进行冒烟测试和精度验证。

**输入**：量化权重目录 `${SAVE_PATH}`。

**操作**：

1. 还原 Transformers 版本（量化完成后还原为容器的初始版本）：

   ```bash
   pip install transformers==5.5.4
   ```

2. 配置环境变量并拉起 vLLM Ascend 推理服务（参考[《vLLM Ascend 官方文档》](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Flash.html)）：

   ```bash
   export OMP_PROC_BIND=false
   export OMP_NUM_THREADS=10
   export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
   export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
   export HCCL_BUFFSIZE=1024
   export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
   export TASK_QUEUE_ENABLE=1
   export HCCL_OP_EXPANSION_MODE="AIV"

   vllm serve ${SAVE_PATH} \
       --max-model-len 133120 \
       --max-num-batched-tokens 8192 \
       --served-model-name dsv4-flash \
       --gpu-memory-utilization 0.9 \
       --max-num-seqs 32 \
       --data-parallel-size 1 \
       --tensor-parallel-size 8 \
       --enable-expert-parallel \
       --tokenizer-mode deepseek_v4 \
       --tool-call-parser deepseek_v4 \
       --enable-auto-tool-choice \
       --reasoning-parser deepseek_v4 \
       --safetensors-load-strategy 'prefetch' \
       --no-enable-prefix-caching \
       --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
       --quantization ascend \
       --port 8900 \
       --block-size 128 \
       --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
       --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
       --additional-config '
       {"ascend_compilation_config":{
           "enable_npugraph_ex":true,
           "enable_static_kernel":false
           },
       "enable_cpu_binding": true,
       "enable_dsa_cp": true,
       "multistream_overlap_shared_expert":true}'
   ```

3. 服务启动后，使用 curl 验证模型是否能正常回答：

   ```bash
   curl http://localhost:8900/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "dsv4-flash",
           "messages": [{"role": "user", "content": "Who are you?"}],
           "max_tokens": 256,
           "temperature": 0
       }'
   ```

4. 使用 [AISBench](https://github.com/AISBench/benchmark) 基于 GPQA 数据集进行评测。安装和使用方法请参考 [AISBench 代码仓 README](https://github.com/AISBench/benchmark/blob/master/README.md)。

   评测量化模型时，应使用与论文或模型卡片基线一致的 AISBench 数据集、服务参数、生成参数和后处理流程，记录选定的精度基线分数，并使用 `${SAVE_PATH}` 量化模型获得量化结果分数。本案例不要求额外测试浮点模型。

   创建模型配置文件 `vllm_api_general_chat.py`：

   ```python
   from ais_bench.benchmark.models import VLLMCustomAPIChat
   from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

   models = [
       dict(
           attr="service",
           type=VLLMCustomAPIChat,
           abbr="vllm-api-general-chat",
           path="",
           model="dsv4-flash",
           stream=False,
           request_rate=0,
           use_timestamp=False,
           retry=2,
           api_key="",
           host_ip="localhost",
           host_port=8900,
           url="",
           max_out_len=65536,
           batch_size=16,
           trust_remote_code=False,
           generation_kwargs=dict(
               temperature=1.0,
               top_p=1.0,
               chat_template_kwargs={"thinking":True}
           ),
           pred_postprocessor=dict(type=extract_non_reasoning_content),
       )
   ]
   ```

   执行评测命令（参考）：

   ```bash
   ais_bench \
       --models vllm_api_general_chat \
       --datasets gpqa_gen \
       --work-dir "${EVAL_WORK_DIR}" \
       --dump-eval-details
   ```

   `${EVAL_WORK_DIR}` 为本次评测结果输出目录；同时保存实际使用的 `vllm_api_general_chat.py`、AISBench 版本或 commit、数据集配置和生成参数。

   DeepSeek-V4-Flash 的 GPQA 公开基线分数为 **87.4**（参考[《ModelScope 模型卡片》](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash/summary)），作为选定的精度基线。

   **精度结论**：量化权重 GPQA 实测分数为 **87.88**，与选定的精度基线对比后，满足精度验收要求。评测配置和验收计算记录以 `${EVAL_WORK_DIR}` 中的结果文件为准。

   **注意**：评测数据可能存在波动，若单次测评结果不符合预期，应记录多次测评结果及平均值。

**输出**：

- vLLM Ascend 推理服务正常运行，curl 请求返回正常响应。
- AISBench 公开基线分数与量化结果分数的 GPQA 对比记录。

**记录**：curl 请求的响应结果 `${CURL_RESULT}`；vLLM 服务日志 `${VLLM_LOG}`；AISBench 评测配置和结果目录 `${EVAL_WORK_DIR}`；选定的精度基线、量化结果及验收计算记录。

**下一步**：结束。

## 5. 结果与经验

### 5.1 关键结果汇总

| 步骤 | 关键操作 | 指标 | 变化 | 备注 |
| --- | --- | --- | --- | --- |
| 步骤 3：执行一键量化命令 | 执行 `msmodelslim quant` W8A8 量化 | 量化权重生成 | 浮点 -> W8A8 量化 | 量化日志 `${QUANT_LOG}`，量化策略：QuaRot + Flex Smooth Quant + W8A8 动态量化 |
| 步骤 4：量化权重验收 | 检查量化产物 | 量化产物结构完整性 | - | 检查描述文件、配置文件、全部权重分片、结构对比及推理加载结果 |
| 步骤 5：推理部署与评测 | vLLM Ascend 拉起推理服务 + AISBench GPQA 评测 | 公开基线分数：87.4；量化实测分数：87.88 | 量化结果高于公开基线 | 实测分数不低于选定的精度基线，满足精度验收要求 |

### 5.2 经验总结

1. **量化命令需在已完成安装验证的环境中执行**：执行量化前确认 `msmodelslim --help` 可正常调用，并使用明确的模型路径和保存路径。适用边界：源码安装方式（`git clone` + `bash install.sh`）。

2. **Transformers 版本需与模型适配**：DeepSeek-V4-Flash 建议使用 `transformers==4.48.2`，版本不匹配可能导致模型加载失败。msModelSlim 不强求特定版本，但建议量化时安装模型适配的版本。适用边界：该版本建议仅用于 DeepSeek-V4-Flash 量化阶段（vLLM Ascend 0.22.1rc1 镜像）。

3. **量化权重需同时通过格式和加载验收**：按 [AscendV1 导出格式](../knowledge_base/quantization_format/ascendv1/term_ascendv1.md#export-artifacts) 核对 `quant_model_description.json`、配置文件、全部权重分片及 index、必要辅助文件，并通过 vLLM Ascend 完成加载和冒烟请求后，再进入正式精度评测。适用边界：vLLM Ascend 0.22.1rc1 及 Atlas 800I A3 推理服务器。

4. **精度验收以选定的精度基线为准**：使用与论文或模型卡片一致的评测配置获得量化结果分数，并与选定的精度基线分数对比；量化结果分数不低于选定的精度基线分数时直接通过，低于选定的精度基线分数时相对下降不超过 1% 才通过。本案例不要求额外测试浮点模型。适用边界：本案例 GPQA 评测。

## 6. 异常处理

- **量化命令执行失败（OOM）**：DeepSeek-V4-Flash W8A8 量化单卡即可完成，一般不会出现 OOM；若遇显存不足，请确认 `--device npu --device_id 0` 指定的 NPU 未被其他任务占用。
- **Transformers 加载模型报错**：确认是否安装了 `transformers==4.48.2`，且模型权重路径正确。注意使用非 0731 版本的权重。
- **vLLM 服务启动失败**：本次实际排查动作依次为：检查 CANN 环境变量（`env | grep ASCEND`）、NPU 状态（`npu-smi info`），并确认 `--tensor-parallel-size 8` 与实际卡数一致。

## 7. 附录

- 量化配置文件：[lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml](../../../lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml)
- GPQA 精度参考：论文或模型卡片公开分数 **87.4** 作为公开基线分数（参考[《ModelScope 模型卡片》](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash/summary)），本案例量化实测分数为 **87.88**。
- vLLM Ascend 官方部署文档：[《DeepSeek-V4-Flash 教程》](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Flash.html)
- vLLM Ascend 官方镜像：[quay.io/ascend/vllm-ascend:v0.22.1rc1-a3](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=v0.22.1rc1-a3)
- AISBench 评测工具及安装指南：[代码仓 README](https://github.com/AISBench/benchmark/blob/master/README.md)
