# DeepSeek-V4-Pro W4A8 量化新接入案例

## 1. 案例背景

**目标**：将新发布的 DeepSeek-V4-Pro 模型接入 msModelSlim 代码仓库，完成模型适配器与量化配置的开发，并进行 W4A8 一键量化，然后部署到 vLLM Ascend 推理引擎进行精度验证，完成从模型接入到量化部署的全流程闭环。

**对比对象**：浮点模型原始权重（论文基线） vs W4A8 量化后部署权重。

**精度验收口径**：相对原始权重，GPQA 评测集得分损失 < 1%（以 ModelScope 模型卡片论文基线 89.1 为参考）。

**覆盖流程**：新模型结构分析 → 设计量化方案 → 完成模型适配器以及量化 yaml 的开发 → 环境准备与工具安装 → 执行一键量化命令 → 量化权重完整性检查 → 推理部署与评测 → 精度测试 → 回退调优（可选）。

**关联流程**：[《权重量化使用指南》](../user_guide/usage_weight_quantization.md)、[《一键量化完整指南》](../user_guide/usage_quick_quantization.md)、[《LLM 大模型接入指南》](../knowledge_base/model/integrating_models.md)、[《量化算法总览》](../knowledge_base/quantization_algorithms/README.md)

## 2. 模型相关信息

| 项 | 内容 |
| --- | --- |
| 模型名称 | DeepSeek-V4-Pro |
| model_type | DeepSeek-V4-Pro |
| 参数量 | 685B |
| 模态 | 文本 |
| 模型结构 | MoE × 256 + MLA + 低秩注意力压缩 + MTP（多 token 预测） |
| 量化类型 | w4a8（混合量化：路由专家 W4A8，共享专家与注意力层 W8A8） |
| 开源权重来源 | [ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro) |
| 推理框架与并行 | vLLM Ascend 0.22.1rc1，TP16 + DP2（2 节点 × 16 卡/节点），启用 Expert Parallel |

**验证范围声明**：

- 覆盖：文本输入与文本输出精度，GPQA 评测集。
- 不覆盖：性能测试（TTFT / TPOT / 吞吐 / 显存容量）。

## 3. 环境与版本

| 项 | 版本或配置 |
| --- | --- |
| 产品形态 | Atlas 800I A3（限定，本案例基于 Atlas 800I A3 多节点 16 卡/节点环境验证） |
| 环境镜像 | `m.daocloud.io/quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` |
| CANN | CANN-9.0.0（随镜像预置） |
| PyTorch | 2.10.0（随镜像预置） |
| TorchNPU | 2.10.0（随镜像预置） |
| vLLM Ascend | 0.22.1rc1（随镜像预置） |
| 评测工具 | [AISBench](https://github.com/AISBench/benchmark)（安装指南见 [《官方文档》](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/)） |
| 其他依赖 | `transformers==4.48.2`（量化时需要降级，默认 5.5.4） |

**本次前置事实**：

- 已获取 DeepSeek-V4-Pro 浮点权重（下载地址：[ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro)），已启动 vLLM Ascend 官方镜像容器并挂载 NPU 设备与模型权重目录，已确认 msModelSlim 源码可正常拉取（本案例以源码方式接入新模型）。

## 4. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型目录 | `${MODEL_PATH}`（用户下载） | 含 `config.json`、`tokenizer.json`、模型权重文件（safetensors，可能为 FP8 权重）；可被 Transformers 4.48.2 加载 | 使用 `transformers.AutoModel.from_pretrained` 加载无报错 |
| 输入 | 校准数据集 | 内置（`msmodelslim/lab_calib/mix_calib.jsonl`） | JSONL 格式，每行含 `inputs_pretokenized` 字段 | 校准集可以在量化流程被正常读取 |
| 交付件 | 模型适配器代码 | `msmodelslim/model/deepseek_v4/`（适配器 + 模型结构 + 加载器） | 适配器需实现量化流水线所需接口（QuaRot、FlexSmoothQuant 等） | `bash install.sh` 后 entry point 注册生效，`--model_type DeepSeek-V4-Pro` 可命中适配器 |
| 交付件 | 量化配置文件 | `lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml` | 含量化策略、qconfig、校准数据集与保存器配置 | `msmodelslim quant --quant_type w4a8` 可匹配到该配置 |
| 交付件 | 量化权重目录 | `${SAVE_PATH}`（量化流程输出目录） | 含 `quant_model_description.json`、权重文件（safetensors 格式）、`config.json` 等 | 量化权重可被 vLLM Ascend `--quantization ascend` 拉起服务化，对话正常 |
| 交付件 | 精度测试结果 | AISBench 输出目录（`${WORK_DIR}`） | 含量化权重 GPQA 分数与基线对比 | 使用 AISBench 基于 GPQA 数据集评测，量化误差 < 1% |

## 5. 操作步骤

### 步骤 1：新模型结构分析

**目标**：梳理 DeepSeek-V4-Pro 的模型结构，为量化方案设计和适配器开发提供依据。

**输入**：浮点权重已下载至 `${MODEL_PATH}`。

**操作**：

1. 查看模型结构配置：

   ```bash
   ls ${MODEL_PATH}/
   cat ${MODEL_PATH}/config.json
   ```

   `config.json` 中需重点确认 `num_hidden_layers`、`n_routed_experts`、`n_shared_experts`、`compress_ratios`（逐层注意力压缩比）、`q_lora_rank`/`o_lora_rank`（低秩投影）、`n_mtp_layers`（MTP 层数）等结构参数。

2. 结合模型结构代码确认关键结构特性（`msmodelslim/model/deepseek_v4/model.py`）：

   - **MoE 架构**：路由专家 + 共享专家（`n_routed_experts` / `n_shared_experts`），路由专家数量多且单层参数量大，是量化方案设计的重点。
   - **注意力结构**：MLA 低秩投影（`q_lora_rank`/`o_lora_rank`）、逐层可变的注意力压缩（`compress_ratios`）以及 `compressor`/`indexer` 子模块。
   - **MTP 层**：多 token 预测（Multi-Token Prediction）结构，加载与量化时需单独处理。
   - **权重精度**：模型权重可能为 FP8 精度，适配器加载时需支持反量化。

**输出**：

- 模型结构参数清单及关键结构特性说明。

**记录**：模型结构参数（层数、专家数、压缩比等）摘要。

**下一步**：设计量化方案。

---

### 步骤 2：设计量化方案

**目标**：结合模型结构设计 W4A8 混合量化方案，确定各模块的量化策略与量化粒度。

**输入**：步骤 1 确认的模型结构特性。

**操作**：

1. 确定整体量化策略。DeepSeek-V4-Pro 采用 QuaRot 旋转预处理 + Flex AWQ SSZ 离群值抑制 + 混合量化策略：

   - **QuaRot 旋转预处理**（`block_size=32`）：通过 Hadamard 旋转消除激活离群值，降低后续量化的精度损失。
   - **Flex AWQ SSZ 离群值抑制**：作用于路由专家的 `up-down` 子图，采用 SSZ 方法按通道抑制离群值，配合 INT4 权重量化。
   - **混合量化**：路由专家参数量大且精度敏感，采用 W4A8 动态量化以最大限度压缩模型体积；共享专家与注意力层采用 W8A8 动态量化以兼顾精度。

2. 确定各模块的量化粒度（对应 `lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml` 中 `spec.process`）：

   | 模块 | 匹配规则 | 量化方案 | 权重/激活配置 |
   | --- | --- | --- | --- |
   | 路由专家 | `*ffn*`（排除 `*gate`、`*shared_experts*`） | W4A8 动态量化 | 权重 per-channel INT4（ssz），激活 per-token INT8（minmax） |
   | 共享专家 | `*ffn.shared_experts*` | W8A8 动态量化 | 权重 per-channel INT8（minmax），激活 per-token INT8（minmax） |
   | 注意力层 | `*attn*`（排除 `wo_a`/`wo_b`、`compressor.wgate`/`wkv`、`indexer.weights_proj` 等低秩分支） | W8A8 动态量化 | 权重 per-channel INT8（minmax），激活 per-token INT8（minmax） |

3. 确定校准数据集与保存方式：校准数据集使用内置 `mix_calib.jsonl`；保存使用 `ascendv1_saver`（与 vLLM Ascend 原生兼容，按 4GB 分片保存）。

**输出**：

- 各模块量化策略与量化粒度清单。

**记录**：量化策略要点（QuaRot + Flex AWQ SSZ + 混合量化）。

**下一步**：完成模型适配器以及量化 yaml 的开发。

---

### 步骤 3：完成模型适配器以及量化 yaml 的开发

**目标**：开发 DeepSeek-V4-Pro 模型适配器并编写量化配置文件，使 `msmodelslim quant` 能识别并量化该模型。

**输入**：步骤 1 的模型结构信息、步骤 2 的量化方案。

**操作**：

1. 开发模型适配器（`msmodelslim/model/deepseek_v4/model_adapter.py`）：

   `DeepSeekV4ModelAdapter` 需继承并实现量化流水线所需的接口：

   - `get_model_pedigree()`：返回 `'deepseek_v4'`，将适配器与 `lab_practice/deepseek_v4/` 配置目录关联。
   - `init_model()`：以 BF16 加载模型，支持 FP8 权重自动反量化（`auto_dequant_state_dict`）；主模型按单层加载省显存，MTP 层惰性加载。
   - `generate_model_forward()`：逐层前向传播并交给量化流水线处理，MTP 层走独立预处理流程。
   - `get_adapter_config_for_subgraph()`：声明 `up-down`、`linear-linear`、`norm-linear` 子图结构，供 Flex AWQ SSZ / Flex Smooth Quant / 线性量化处理器使用。
   - `get_rotate_map()` / `get_ln_fuse_map()`：提供 QuaRot 旋转映射与 LN-Linear 融合映射。

2. 注册模型适配器（`config/config.ini`）：

   ```ini
   [ModelAdapterEntryPoints]
   deepseek_v4 = msmodelslim.model.deepseek_v4.loader:DeepseekV4AdapterLoader

   [ModelAdapter]
   deepseek_v4 = DeepSeek-V4-Flash, DeepSeek-V4-Pro
   ```

   执行 `bash install.sh` 后，`setup.py` 会根据上述配置自动生成 entry point，`--model_type DeepSeek-V4-Pro` 即可通过插件工厂命中 `DeepSeekV4ModelAdapter`。

3. 编写量化配置（`lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml`）：

   按步骤 2 的方案编写量化配置，关键字段包括：

   - `metadata`：声明 `config_id`、`score`、`verified_model_types`（`DeepSeek-V4-Pro`）及 `label`（`w_bit: 4`、`a_bit: 8`），用于与 `--quant_type w4a8` 匹配。
   - `spec.process`：依次声明 QuaRot、Flex AWQ SSZ、Flex Smooth Quant 与各模块 `linear_quant` 处理器及其 `qconfig`。
   - `spec.dataset` / `spec.save`：指定校准数据集与 `ascendv1_saver` 保存器。

**输出**：

- 可被 `msmodelslim quant` 识别的模型适配器与量化配置。

**记录**：适配器代码路径、注册配置及量化配置 YAML 文件路径。

**下一步**：环境准备与工具安装。

---

### 步骤 4：环境准备与工具安装

**目标**：确认容器环境满足量化要求，安装 msModelSlim 工具及依赖。

**输入**：vLLM Ascend 官方镜像容器已启动。

**操作**：

1. 若尚未拉起环境，请参考 [《vLLM Ascend 官方指导》](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Pro.html) 拉取镜像并创建容器（需挂载 NPU 设备与模型权重目录），本案例使用的镜像为 `m.daocloud.io/quay.io/ascend/vllm-ascend:v0.22.1rc1-a3`。

2. 在容器内检查 NPU 状态与 CANN 环境：

   ```bash
   pip list 2>/dev/null | grep -iE "torch|npu|cann"
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

   > [!NOTE]
   >
   > 新模型适配器与量化配置开发完成后，需重新执行 `bash install.sh` 使注册的 entry point 生效。

5. 安装与 DeepSeek-V4-Pro 适配的 Transformers 版本：

   ```bash
   pip install transformers==4.48.2
   ```

   > [!NOTE]
   >
   > msModelSlim 不强求特定 Transformers 版本，但 DeepSeek-V4-Pro 模型建议使用 `transformers==4.48.2`。降级仅用于量化阶段，量化完成后可执行 `pip install transformers==5.5.4` 还原。

**输出**：

- `msmodelslim --help` 可正常执行。
- Transformers 版本已切换为 `4.48.2`。

**记录**：执行 `pip list | grep transformers` 确认版本。

**下一步**：执行一键量化命令。

---

### 步骤 5：执行一键量化命令

**目标**：使用 `msmodelslim quant` 命令生成量化权重。`msmodelslim quant` 根据 `--model_type DeepSeek-V4-Pro` 通过 entry point 命中步骤 3 开发的适配器，并根据 `model_pedigree`（`deepseek_v4`）+ `--quant_type w4a8` 在 `lab_practice/deepseek_v4/` 下匹配到 `deepseek_v4_pro_w4a8.yaml` 量化配置，随后按 `spec.process` 依次执行各量化处理器并保存产物。

**输入**：浮点权重目录 `${MODEL_PATH}`。

**操作**：

设 `${MODEL_PATH}` 为浮点权重目录，`${SAVE_PATH}` 为量化输出目录，然后执行：

```bash
cd ..  # 退出 msmodelslim 目录，msmodelslim 命令需在源码目录外执行
msmodelslim quant \
    --model_path ${MODEL_PATH} \
    --save_path ${SAVE_PATH} \
    --model_type DeepSeek-V4-Pro \
    --quant_type w4a8 \
    --device npu --device_id 0 1 2 3 4 5 6 7 \
    --trust_remote_code True
```

参数说明：

- `--model_path`：浮点模型权重目录路径。
- `--save_path`：量化权重保存目录路径。
- `--model_type`：模型类型，固定为 `DeepSeek-V4-Pro`。
- `--quant_type`：量化策略类型，`w4a8` 表示权重 INT4 + 激活 INT8。
- `--device`：指定量化使用的 NPU 设备，W4A8 量化算法较复杂，单卡量化耗时较长，建议使用 8 卡（`npu:0,1,2,3,4,5,6,7`）进行多卡量化以缩短耗时。
- `--trust_remote_code`：信任自定义模型代码（请确保代码来源可靠）。

**输出**：

- 量化权重目录 `${SAVE_PATH}`，包含 `quant_model_description.json`、权重文件（safetensors 格式）、`config.json` 等。

**记录**：量化日志输出至终端，可重定向至文件备查。

**下一步**：量化权重完整性检查。

---

### 步骤 6：量化权重完整性检查

**目标**：确认量化权重产物完整。

**输入**：量化权重目录 `${SAVE_PATH}`。

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

**通过条件**：目录包含 `quant_model_description.json` 且日志输出无 ERROR。

**记录**：量化权重目录结构及 `quant_model_description.json` 摘要。

**下一步**：推理部署与评测。

---

### 步骤 7：推理部署与评测

**目标**：将量化权重部署到 vLLM Ascend 推理引擎，进行冒烟测试，为后续精度验证准备好服务环境。

**输入**：量化权重目录 `${SAVE_PATH}`。

**操作**：

1. 还原 Transformers 版本（量化完成后还原为容器的初始版本）：

   ```bash
   pip install transformers==5.5.4
   ```

2. 配置环境变量并拉起 vLLM Ascend 推理服务（参考 [《vLLM Ascend 官方文档》](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Pro.html)）：

   DeepSeek-V4-Pro 模型较大，这里官方示例是多节点部署。以下为 A3 系列节点 0 的拉起命令，具体命令可以参考对应的官方文档：

   ```bash
   export HCCL_OP_EXPANSION_MODE="AIV"
   export HCCL_IF_IP=$local_ip
   export GLOO_SOCKET_IFNAME=$nic_name
   export TP_SOCKET_IFNAME=$nic_name
   export HCCL_SOCKET_IFNAME=$nic_name
   export HCCL_BUFFSIZE=2048
   export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
   export OMP_PROC_BIND=false
   export OMP_NUM_THREADS=10
   export TASK_QUEUE_ENABLE=1
   export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
   export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

   vllm serve ${SAVE_PATH} \
       --safetensors-load-strategy 'prefetch' \
       --max-model-len 135000 \
       --max-num-batched-tokens 4096 \
       --served-model-name dsv4-pro \
       --gpu-memory-utilization 0.9 \
       --max-num-seqs 32 \
       --data-parallel-size 2 \
       --data-parallel-size-local 1 \
       --data-parallel-start-rank 0 \
       --data-parallel-address $node0_ip \
       --data-parallel-rpc-port 13399 \
       --tensor-parallel-size 16 \
       --enable-expert-parallel \
       --quantization ascend \
       --port 8900 \
       --host 0.0.0.0 \
       --block-size 128 \
       --async-scheduling \
       --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
       --tokenizer-mode deepseek_v4 \
       --tool-call-parser deepseek_v4 \
       --enable-auto-tool-choice \
       --reasoning-parser deepseek_v4 \
       --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
       --additional-config '
       {"ascend_compilation_config":{
           "enable_npugraph_ex":true,
           "enable_static_kernel":false
           },
       "enable_cpu_binding": true,
       "multistream_overlap_shared_expert":true}'
   ```

   > [!NOTE]
   >
   > DeepSeek-V4-Pro 需要多节点部署，至少 2 个 A3 节点（16 卡/节点）。节点 1 需额外添加 `--headless` 参数，并将 `--data-parallel-start-rank` 设为 1。详细配置请参考 [《vLLM Ascend 官方文档》](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Pro.html)。

3. 服务启动后，使用 curl 验证模型是否能正常回答：

   ```bash
   curl http://localhost:8900/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "dsv4-pro",
           "messages": [{"role": "user", "content": "Who are you?"}],
           "max_tokens": 256,
           "temperature": 0
       }'
   ```

**输出**：

- vLLM Ascend 推理服务正常运行，curl 请求返回正常响应。

**记录**：curl 请求的响应结果。

**下一步**：精度测试（见第 6 节）；若精度不达标，执行步骤 8 回退调优。

---

### 步骤 8：回退调优（可选）

**目标**：当第 6 节精度测试结果不满足验收标准（量化误差 ≥ 1%）时，通过敏感层分析定位量化敏感层，采用量化回退或局部提位宽的方式调整量化配置，重新量化并复测，直到精度达标。

**输入**：精度测试未达标的量化权重目录 `${SAVE_PATH}`。

**操作**：

1. 使用敏感层分析工具定位量化敏感层（详细用法见《[线性层敏感层分析使用指南](../user_guide/usage_sensitive_linear_analysis.md)》）：

   ```bash
   msmodelslim analyze linear \
       --model_type DeepSeek-V4-Pro \
       --model_path ${MODEL_PATH} \
       --metrics kurtosis \
       --top_k 15
   ```

   分析结果按敏感度从高到低输出敏感层列表，日志中会附带可直接粘贴到量化配置的 YAML 片段。

2. 根据敏感层排序结果调整量化配置（`lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml`），常见做法二选一：

   - **回退到浮点**：对排序靠前的敏感层，在对应 `linear_quant` 处理器中通过 `exclude` 使其不参与量化，推理时保持浮点路径：

     ```yaml
     - type: "linear_quant"
       qconfig:
         # ...（其余配置保持不变）
       include:
         - "*"
       exclude:
         - "model.layers.3.mlp.down_proj"
         - "model.layers.4.mlp.down_proj"
     ```

   - **局部提位宽**：在 W4A8 方案中仅对敏感层将权重 4 bit 提升至 8 bit（W8A8），其余层保持 W4A8，以局部精度换取整体压缩收益。

3. 使用调整后的配置重新执行一键量化（重复步骤 5），并重新部署评测（重复步骤 7 及第 6 节），对比回退前后精度。若精度仍未达标，可进一步扩大回退层范围或改用更保守的量化策略后重试。

**输出**：

- 精度达标的量化权重目录。

**记录**：敏感层分析结果、回退层清单、回退前后精度对比。

**下一步**：结束。

---

## 6. 精度测试

### 6.1 测试设计

| 项 | 内容 |
| --- | --- |
| 评测工具 | AISBench（安装指南见 [《AISBench 官方文档》](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/)） |
| 对比对象 | 论文基线（GPQA 89.1，参考 [《ModelScope 模型卡片》](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro/summary)） vs W4A8 量化后部署权重 |
| 数据集与任务 | GPQA |
| 验收阈值 | 相对论文基线，量化后得分损失 < 1% |
| 随机性控制 | 参考论文或者模型卡片中的推荐生成参数 |

### 6.2 可复现过程

**评测配置要点**：

| 配置项 | 取值 |
| --- | --- |
| 配置文件路径 | `vllm_api_general_chat.py`（自定义配置，见下方） |
| 服务地址 / 端口 | `localhost:8900` |
| served-model-name | `dsv4-pro` |

**评测配置（`vllm_api_general_chat.py`）**：

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="",
        model="dsv4-pro",
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

**执行命令**：

```bash
ais_bench \
    --models vllm_api_general_chat \
    --datasets gpqa_gen \
    --work-dir ${WORK_DIR} \
    --dump-eval-details
```

**输出与记录**：

- 结果目录：`${WORK_DIR}`（如 `AISBench outputs/{timestamp}/`）。
- 须保留：评测日志、GPQA 分数摘要、`AISBench outputs/{timestamp}/` 完整产物。

### 6.3 精度结果

| 数据集 / 任务 | 样本数 | 基线（论文） | 待验（W4A8 量化） | 差值 | 结论 |
| --- | --- | --- | --- | --- | --- |
| GPQA | 全量 | 89.1 (Accuracy) | 89.9 (Accuracy) | +0.8 | 量化误差 < 1%，通过 |

**精度结论**：DeepSeek-V4-Pro W4A8 混合量化后 GPQA 实测 **89.9**，相对论文基线 89.1，得分提升 0.8，**满足精度验收标准**（相对损失 < 1%）。

> **说明**：本案例以论文基线作为对比基准，非浮点服务实测值。量化接入案例聚焦模型接入与量化流程验证，不单独拉起浮点推理服务做对比评测。
>
> **声明**：评测数据可能存在波动，若单次测评结果不符合预期，建议以多次测评的平均结果为准。

## 7. 结果与经验

### 7.1 关键结果汇总

| 步骤 | 关键操作 | 指标 | 变化 | 备注 |
| --- | --- | --- | --- | --- |
| 步骤 1-2 | 确认模型结构、设计量化方案 | 结构特性 + 混合量化方案 | - | 路由专家 W4A8，共享专家与注意力层 W8A8 |
| 步骤 3 | 开发适配器、注册 entry point、编写量化 yaml | 适配器 + yaml 交付 | - | `--model_type DeepSeek-V4-Pro` 可命中适配器并匹配配置 |
| 步骤 5 | 执行 `msmodelslim quant` W4A8 量化 | 量化权重生成 | 浮点 -> W4A8 量化 | 量化策略：QuaRot + Flex AWQ SSZ + 混合量化 |
| 步骤 6 | 检查量化产物 | 产物完整性 | - | 包含 `quant_model_description.json`、safetensors 权重文件 |
| 步骤 7 | AISBench GPQA 评测 | GPQA：论文基线 89.1 / 量化实测 89.9 | 量化误差 < 1% | 精度验收标准通过 |
| 步骤 8 | 敏感层分析 + 量化回退/局部提位宽（可选） | 精度复测达标 | W4A8 -> 回退后权重 | 仅当精度测试不达标时执行 |

### 7.2 经验总结

1. **新模型接入的三要素**：确认模型结构 -> 设计量化方案 -> 完成适配器与量化 yaml 的开发。其中，模型结构特性（MoE 路由专家/共享专家、MLA 低秩注意力、MTP 层等）直接决定子图结构与旋转映射的实现，是适配器开发的输入。适用边界：新模型接入 msModelSlim 的通用流程。

2. **适配器需实现量化流水线所需接口**：`DeepSeekV4ModelAdapter` 需同时继承 `ModelSlimPipelineInterfaceV1`、`FlexSmoothQuantInterface`、`QuaRotInterface`、`AscendV1SaveInterface` 等接口，量化方案中使用了哪些处理器，适配器就要提供对应的子图/映射信息，否则对应 processor 无法执行。适用边界：msModelSlim 量化流水线（modelslim_v1）支持的模型适配器。

3. **适配器注册依赖 entry point**：模型适配器通过 `config/config.ini` 注册，执行 `bash install.sh` 后 `setup.py` 自动生成 entry point；新增或修改适配器后需重新安装才生效。适用边界：源码安装方式（`git clone` + `bash install.sh`）。

4. **量化命令需在 msmodelslim 目录外执行**：`msmodelslim quant` 命令需要在 `msmodelslim` 目录的父目录执行，否则可能因路径解析问题导致执行失败。适用边界：源码安装方式（`git clone` + `bash install.sh`）。

5. **DeepSeek-V4-Pro W4A8 量化需指定多卡**：W4A8 量化算法较为复杂，单卡量化时间较久，推荐通过 `--device npu --device_id 0 1 2 3 4 5 6 7` 指定 8 卡进行多卡量化，以缩短量化耗时。适用边界：DeepSeek-V4-Pro 等大参数量模型在 Atlas 800I A3 形态下的 W4A8 量化阶段（vLLM Ascend 0.22.1rc1 镜像）。

6. **W4A8 混合量化策略**：DeepSeek-V4-Pro 的 W4A8 量化采用混合策略——路由专家使用 W4A8 动态量化（`ssz` 方法），共享专家和注意力层使用 W8A8 动态量化，在保证精度的同时有效降低模型大小和推理显存占用。适用边界：DeepSeek-V4-Pro W4A8 量化方案（Atlas 800I A3 形态）。

7. **量化权重可直接用于 vLLM Ascend 推理**：一键量化输出的权重格式与 vLLM Ascend 原生兼容，部署时指定 `--quantization ascend` 即可加载量化权重，无需额外转换。适用边界：vLLM Ascend 0.22.1rc1 及 Atlas 800I A3 形态。

8. **精度不达标时优先回退敏感层**：当精度测试结果不满足验收标准时，先通过 `msmodelslim analyze` 定位量化敏感层，再在量化配置中通过 `exclude` 回退敏感层或对敏感层局部提位宽，重新量化复测即可收敛精度；回退会带来一定的模型体积与性能开销，应优先回退敏感度排序靠前的少量层。适用边界：DeepSeek-V4-Pro W4A8 量化方案（Atlas 800I A3 形态）。

## 8. 异常处理

- **适配器未生效（`--model_type` 无法识别）**：确认 `config/config.ini` 已注册 `deepseek_v4 = DeepSeek-V4-Flash, DeepSeek-V4-Pro`，并在修改后重新执行 `bash install.sh`，检查 `msmodelslim.egg-info/entry_points.txt` 中是否已生成 `DeepSeek-V4-Pro = msmodelslim.model.deepseek_v4.loader:DeepseekV4AdapterLoader`。
- **量化配置未匹配**：确认 `lab_practice/deepseek_v4/` 下 yaml 的 `metadata.verified_model_types` 包含 `DeepSeek-V4-Pro`，且 `metadata.label` 与 `--quant_type w4a8` 对应（`w_bit: 4`、`a_bit: 8`）。
- **量化命令执行失败（OOM）**：DeepSeek-V4-Pro W4A8 量化单卡即可完成，实际工程场景下会开启多卡量化加速，一般不会出现 OOM；若遇显存不足，请确认 `--device npu --device_id 0 1 2 3 4 5 6 7` 指定的 NPU 未被其他任务占用。
- **Transformers 加载模型报错**：确认是否安装了 `transformers==4.48.2`，且模型权重路径正确。
- **vLLM 服务启动失败**：本次实际排查动作依次为：检查 CANN 环境变量（`env | grep ASCEND`）、NPU 状态（`npu-smi info`），并确认 `--tensor-parallel-size 16` 与多节点卡数一致；DeepSeek-V4-Pro 模型较大，需使用多节点部署（至少 2 个 A3 节点，16 卡/节点）。

## 9. 附录

- 量化配置文件：`lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml`
- 模型适配器代码：`msmodelslim/model/deepseek_v4/`（`model_adapter.py`、`model.py`、`loader.py`）
- 适配器注册配置：`config/config.ini`
- vLLM Ascend 官方部署文档：[《DeepSeek-V4-Pro 教程》](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Pro.html)
- AISBench 评测工具：[GitHub](https://github.com/AISBench/benchmark) | [《文档》](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/)
- GPQA 精度：论文基线 89.1，本案例实测 89.9（参考 [《ModelScope 模型卡片》](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro/summary)）
