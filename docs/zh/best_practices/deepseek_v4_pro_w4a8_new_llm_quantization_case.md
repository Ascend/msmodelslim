# DeepSeek-V4-Pro 新模型 W4A8 量化案例

## 1. 案例背景

**目标**：将新发布的 DeepSeek-V4-Pro 模型接入 msModelSlim 代码仓库，完成模型适配器与量化配置的开发，进行 W4A8 一键量化和精度验证，并在精度不达标时进行量化调优，完成从模型接入到量化部署、量化调优的全流程闭环。

**对比对象**：将量化权重 `${SAVE_PATH}` 的评测结果与浮点实测基线或论文、模型卡片公开的基线进行对比。优先使用相同评测条件下的浮点实测基线；使用公开基线时，应确保评测数据集、服务配置、生成参数和后处理配置一致，并在评测记录中注明基线来源。

**精度验收口径**：以选定的精度基线（浮点实测基线或论文、模型卡片公开基线）作为验收依据，并确保量化结果与基线使用一致的评测条件。量化结果不低于基线时直接通过；量化结果低于基线时，仅当相对下降比例不超过 1% 才通过；当相对下降比例超过 1% 时不通过，并进入步骤 8 量化调优。

**覆盖流程**：分析 `${MODEL_PATH}/config.json` 中的新模型结构 → 设计 `lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml` 量化方案 → 完成 `msmodelslim/model/deepseek_v4/` 模型适配器及 `lab_practice/deepseek_v4/` 量化 YAML 开发 → 环境准备与工具安装 → 执行一键量化命令 → 检查 `${SAVE_PATH}` 量化权重完整性 → 使用 `${SAVE_PATH}` 推理部署与精度评测 → 量化调优（可选）。

**关联流程**：[《权重量化使用指南》](../user_guide/usage_weight_quantization.md)、[《一键量化完整指南》](../user_guide/usage_quick_quantization.md)、[《LLM 大模型接入指南》](../knowledge_base/model/integrating_models.md)、[《量化算法总览》](../knowledge_base/quantization_algorithms/README.md)

**模型相关信息**

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

## 2. 环境与版本

| 项 | 版本或配置 |
| --- | --- |
| 产品形态 | Atlas 800I A3 推理服务器（限定，本案例基于该服务器两节点 32 卡环境验证） |
| 环境镜像 | [quay.io/ascend/vllm-ascend:v0.22.1rc1-a3](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=v0.22.1rc1-a3) |
| CANN | CANN-9.0.0（随镜像预置） |
| PyTorch | 2.10.0（随镜像预置） |
| TorchNPU | 2.10.0（随镜像预置） |
| vLLM Ascend | [0.22.1rc1](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=v0.22.1rc1-a3)（随镜像预置） |
| 评测工具 | [AISBench](https://github.com/AISBench/benchmark)（安装指南见[《官方文档》](https://github.com/AISBench/benchmark/blob/master/README.md)） |
| 其他依赖 | `transformers==4.48.2`（量化时需要降级，镜像内默认 5.5.4） |

**本案例前置条件**：

- 已获取 DeepSeek-V4-Pro 浮点权重（下载地址：[ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro)），已启动 vLLM Ascend 官方镜像容器并挂载 NPU 设备与模型权重目录，已确认 msModelSlim 源码可正常拉取（本案例以源码方式接入新模型）。

## 3. 输入和交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
| --- | --- | --- | --- | --- |
| 输入 | 浮点模型目录 | `${MODEL_PATH}`（用户下载） | 含 `config.json`、`tokenizer.json`、模型权重文件（safetensors，可能为 FP8 权重）；可被 Transformers 4.48.2 加载 | 使用 `transformers.AutoModel.from_pretrained` 加载无报错 |
| 输入 | 校准数据集 | [混合校准数据集（mix_calib.jsonl）](../../../lab_calib/mix_calib.jsonl) | JSONL 格式，每行含 `inputs_pretokenized` 字段 | 校准集可以在量化流程被正常读取 |
| 交付件 | 模型适配器代码 | [msmodelslim/model/deepseek_v4/](../../../msmodelslim/model/deepseek_v4/)（适配器 + 加载器） | 适配器需实现量化流水线所需接口（QuaRot、FlexSmoothQuant 等） | 安装 msmodelslim 工具后 entry point 注册生效，`--model_type DeepSeek-V4-Pro` 可命中适配器 |
| 交付件 | 量化配置文件 | [lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml](../../../lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml) | 含量化策略、qconfig、校准数据集与保存器配置 | `msmodelslim quant --config ${CONFIG_PATH}` 可加载该配置 |
| 交付件 | 量化权重目录 | `${SAVE_PATH}`（量化流程输出目录） | 含可解析的 `quant_model_description.json`、完整权重文件（safetensors 及必要的分片/index）、`config.json`、tokenizer 等辅助文件 | 量化权重可被 vLLM Ascend `--quantization ascend` 拉起服务化，并完成推理冒烟 |
| 交付件 | 精度测试结果 | AISBench 输出目录（`${QUANT_WORK_DIR}`） | 含相同评测条件下的选定精度基线、量化结果及验收计算记录；若使用论文或模型卡片公开分数，需注明基线来源并确保评测条件一致 | 量化结果不低于选定的精度基线时直接通过；量化结果低于选定的精度基线且相对下降比例不超过 1% 时通过；量化结果低于选定的精度基线且相对下降比例超过 1% 时不通过 |

## 4. 操作步骤

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

2. 结合模型结构确认关键结构特性：

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

1. 确定整体量化策略。针对 DeepSeek-V4-Pro 大参数量 MoE 模型的特点，本案例采用以 W4A8 为主的混合量化方案，在降低模型存储和推理显存开销的同时兼顾精度。

2. 选择量化算法。MoE 模型的专家层和激活通常存在离群值，直接进行低比特量化容易产生精度损失，因此选择 QuaRot 和 SSZ 进行离群值处理：

   - **QuaRot 旋转预处理**（`block_size=32`）：通过 Hadamard 旋转改善激活分布，降低后续量化的精度损失。
   - **Flex AWQ SSZ 离群值抑制**：作用于路由专家的 `up-down` 子图，采用 SSZ 方法按通道抑制离群值，使路由专家能够稳定执行 INT4 权重量化。

3. 确定各模块的量化粒度（对应 [lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml](../../../lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml) 中 `spec.process`）：

   | 模块 | 匹配规则 | 量化方案 | 权重/激活配置 |
   | --- | --- | --- | --- |
   | 路由专家 | `*ffn*`（排除 `*gate`、`*shared_experts*`） | W4A8 动态量化 | 权重 per-channel INT4（ssz），激活 per-token INT8（minmax） |
   | 共享专家 | `*ffn.shared_experts*` | W8A8 动态量化 | 权重 per-channel INT8（minmax），激活 per-token INT8（minmax） |
   | 注意力层 | `*attn*`（排除 `wo_a`/`wo_b`、`compressor.wgate`/`wkv`、`indexer.weights_proj` 等低秩分支） | W8A8 动态量化 | 权重 per-channel INT8（minmax），激活 per-token INT8（minmax） |

4. 确定校准数据集与保存方式：校准数据集使用内置 [lab_calib/mix_calib.jsonl](../../../lab_calib/mix_calib.jsonl)；保存使用 `ascendv1_saver`（与 vLLM Ascend 原生兼容，按 4GB 分片保存）。

**输出**：

- 各模块量化策略与量化粒度清单。

**记录**：量化策略要点（QuaRot + Flex AWQ SSZ + 混合量化）。

**下一步**：完成模型适配器以及量化 YAML 的开发。

---

### 步骤 3：完成模型适配器以及量化 YAML 的开发

**目标**：完成 DeepSeek-V4-Pro 的模型适配、适配器注册和 W4A8 量化配置，使 `msmodelslim quant` 能够识别并量化该模型。

**输入**：步骤 1 的模型结构信息、步骤 2 的量化方案。

**操作**：

1. 实现模型结构和适配器：

   - 在 [model.py](../../../msmodelslim/model/deepseek_v4/model.py) 中定义 MoE、MLA、MTP 及量化前向所需结构，并设置 `USE_DP_MODE=True`。
   - 在 [model_adapter.py](../../../msmodelslim/model/deepseek_v4/model_adapter.py) 中实现模型加载、逐层前向、子图与旋转映射及权重保存，接入 `modelslim_v1` 流水线。适配器需支持 BF16 加载、`auto_dequant_state_dict` 反量化、主模型和 MTP 层按需加载，并实现以下接口：

     | 接口 | 作用 |
     | --- | --- |
     | `get_model_pedigree()` | 返回 `'deepseek_v4'`。 |
     | `init_model()`、`generate_decoder_layer()`、`generate_model_forward()` | 初始化模型、逐层加载并执行前向，将输入通过 `ProcessRequest` 交给量化处理器。 |
     | `get_adapter_config_for_subgraph()` | 声明专家 `up-down`、MLA `linear-linear` 及注意力 `norm-linear` 子图。 |
     | `get_ln_fuse_map()` | 配置 `ffn_norm`、`attn_norm`、`q_norm` 及 MTP 归一化层融合。 |
     | `get_rotate_map()` | 配置 block size 为 32 的 Hadamard 旋转，覆盖 Embedding、专家、注意力和 MTP 投影；低秩查询投影单独处理。 |
     | `ascendv1_save_module_preprocess()` | 保留模块名称和对象，供 `ascendv1_saver` 保存 vLLM Ascend 量化权重。 |

   多卡量化时使用 `dist.barrier()` 同步 rank，并按 rank 划分本地路由专家；共享专家、注意力层和 MTP 层按统一层序处理。最终以量化日志确认各 rank 已初始化且未回退到单卡。

2. 注册模型适配器（[config/config.ini](../../../config/config.ini)）：

   ```ini
   [ModelAdapter]
   deepseek_v4 = DeepSeek-V4-Flash, DeepSeek-V4-Pro

   [ModelAdapterEntryPoints]
   deepseek_v4 = msmodelslim.model.deepseek_v4.loader:DeepseekV4AdapterLoader
   ```

   `DeepseekV4AdapterLoader` 指向 `DeepSeekV4ModelAdapter`。修改适配器或注册配置后需重新安装 msmodelslim，使 entry point 生效；使用 `--model_type DeepSeek-V4-Pro` 应能命中该适配器。

3. 编写[量化配置](../../../lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml)：

   - `spec.process` 按 `quarot` → `flex_awq_ssz` → `flex_smooth_quant` → `linear_quant` 顺序执行：QuaRot 使用 `block_size: 32`；Flex AWQ SSZ 处理路由专家 `up-down` 子图；Flex Smooth Quant 处理 `norm-linear` 子图；Linear Quant 对路由专家执行 W4A8，对共享专家和注意力模块执行 W8A8，并通过 `include`/`exclude` 排除低秩分支和共享专家等不匹配模块。
   - W4A8 使用 per-token INT8 激活、per-channel INT4 权重和 SSZ；W8A8 使用 per-token INT8 激活、per-channel INT8 权重和 minmax。校准集为 `mix_calib.jsonl`，保存器为 `ascendv1_saver`，按 4 GB 分片保存。

4. 验证适配器和配置：

   - 确认 `--model_type DeepSeek-V4-Pro` 能正常匹配到对应的模型适配器。
   - 使用 `--config ${CONFIG_PATH}` 检查 YAML 可正常加载。
   - 按步骤 5 执行小规模或正式量化，确认日志包含适配器加载、处理器执行及多卡 rank 初始化信息。
   - 检查输出目录包含 `quant_model_description.json` 和完整权重文件，再进入步骤 6 验收产物。

**输出**：可被 `msmodelslim quant` 识别的模型适配器与量化配置。

**记录**：适配器代码、注册配置和量化 YAML 路径。

**下一步**：环境准备与工具安装。

---

### 步骤 4：环境准备与工具安装

**目标**：确认容器环境满足量化要求，安装 msModelSlim 工具及依赖。

**输入**：vLLM Ascend 官方镜像容器已启动。

**操作**：

1. 若尚未拉起环境，请参考 [《vLLM Ascend 官方指导》](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Pro.html) 拉取镜像并创建容器（需挂载 NPU 设备与模型权重目录），本案例使用的镜像为 `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3`。

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

4. 下载 msModelSlim 源码并安装 msmodelslim 工具：

   ```bash
   git clone https://gitcode.com/Ascend/msmodelslim
   cd msmodelslim
   bash install.sh
   ```

   安装 msmodelslim 工具后，退出源码目录并执行 `msmodelslim --help` 验证安装结果。

   ```bash
   cd ..  # 退出源码目录，msmodelslim 命令需在源码目录外执行
   msmodelslim --help
   ```

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

**目标**：使用 `msmodelslim quant` 命令生成量化权重。`msmodelslim quant` 根据 `--model_type DeepSeek-V4-Pro` 通过 entry point 命中步骤 3 开发的适配器，并通过 `--config` 直接加载 [deepseek_v4_pro_w4a8.yaml](../../../lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml) 量化配置，随后按 `spec.process` 依次执行各量化处理器并保存产物。

**输入**：浮点权重目录 `${MODEL_PATH}`。

**操作**：

设 `${MODEL_PATH}` 为原始浮点权重目录，`${SAVE_PATH}` 为量化输出目录，`${CONFIG_PATH}` 为量化配置文件 [deepseek_v4_pro_w4a8.yaml](../../../lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml)。

```bash
msmodelslim quant \
    --model_path ${MODEL_PATH} \
    --save_path ${SAVE_PATH} \
    --model_type DeepSeek-V4-Pro \
    --config ${CONFIG_PATH} \
    --device npu --device_id 0 1 2 3 4 5 6 7 \
    --trust_remote_code true
```

参数说明：

- `--model_path`：原始浮点模型权重目录路径。
- `--save_path`：量化权重保存目录路径。
- `--model_type`：模型类型，固定为 `DeepSeek-V4-Pro`。
- `--quant_type`：量化策略类型，`w4a8` 表示权重 INT4 + 激活 INT8。
- `--device`：指定量化使用的 NPU 设备，W4A8 量化算法较复杂，单卡量化耗时较长，建议使用 8 卡（`npu --device_id 0 1 2 3 4 5 6 7`）进行多卡量化以缩短耗时。
- `--trust_remote_code`：信任自定义模型代码（请确保代码来源可靠）。

DeepSeek-V4-Pro 已在 `model.py` 中设置 `USE_DP_MODE=True`，并在 `model_adapter.py` 中实现分布式同步和 EP 本地专家范围处理，因此无需额外适配即可按上述命令启动多卡量化。量化日志需进一步确认实际出现多卡分布式启动和各 rank 初始化信息。

**输出**：

- 量化权重目录 `${SAVE_PATH}`，包含 `quant_model_description.json`、权重文件（safetensors 格式）、`config.json` 等。
- 量化日志，需记录多卡分布式启动、各 rank 初始化及处理器执行情况。

**记录**：量化日志输出至终端，可重定向至文件备查；确认日志中实际启用了 8 张 NPU，且未出现单卡回退提示。

**下一步**：量化权重完整性检查。

---

### 步骤 6：量化权重完整性检查

**目标**：确认量化权重产物完整。

**输入**：量化权重生成目录 `${SAVE_PATH}`。

**操作**：

1. 检查输出目录结构：

   ```bash
   ls ${SAVE_PATH}/
   # 应包含 quant_model_description.json、config.json、权重分片（*.safetensors 及 index）和 tokenizer 等辅助文件
   ```

2. 确认 `quant_model_description.json` 存在且内容完整：

   ```bash
   cat ${SAVE_PATH}/quant_model_description.json | head -20
   ```

**输出**：

- 已验证的 DeepSeek-V4-Pro W4A8 量化权重目录 `${SAVE_PATH}`。

**通过条件**：量化产物目录符合 AscendV1 导出格式约定（`quant_model_description.json` + 权重 safetensors 及必要分片/index），详见 [AscendV1 导出产物](../knowledge_base/quantization_format/ascendv1/term_ascendv1.md#export-artifacts)；且量化日志中无 `ERROR`。

**记录**：量化权重目录结构及 `quant_model_description.json` 摘要。

**下一步**：推理部署与评测。

---

### 步骤 7：推理部署与评测

**目标**：将量化权重部署到 vLLM Ascend 推理引擎，完成加载冒烟测试和精度评测，并与选定的精度基线进行对比。

**输入**：量化权重目录 `${SAVE_PATH}`、选定的精度基线、步骤 6 的量化权重检查记录。

**操作**：

1. 还原 Transformers 版本（量化完成后还原为容器的初始版本）：

   ```bash
   pip install transformers==5.5.4
   ```

2. 配置环境变量并拉起 vLLM Ascend 推理服务（参考[《vLLM Ascend 官方文档》](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Pro.html)）：

   DeepSeek-V4-Pro 模型较大，这里官方示例是多节点部署。以下为 Atlas 800I A3 推理服务器节点 0 的拉起命令，具体命令可以参考对应的官方文档：

   ```bash
   export QUANT_HOST=${QUANT_HOST:-localhost}
   export QUANT_PORT=${QUANT_PORT:-8900}
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
       --port ${QUANT_PORT} \
       --host ${QUANT_HOST} \
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

   上述命令用于量化权重 `${SAVE_PATH}` 的服务化加载。选定的精度基线已提前提供，本步骤不再启动浮点服务或重复测试。

   > [!NOTE]
   >
   > DeepSeek-V4-Pro 需要多节点部署，至少 2 个 Atlas 800I A3 推理服务器节点（16 卡/节点）。节点 1 需额外添加 `--headless` 参数，并将 `--data-parallel-start-rank` 设为 1。详细配置请参考[《vLLM Ascend 官方文档》](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Pro.html)。

3. 服务启动后，对量化服务使用 curl 验证模型是否能正常回答：

   ```bash
   curl "http://${QUANT_HOST}:${QUANT_PORT}/v1/chat/completions" \
       -H "Content-Type: application/json" \
       -d '{
           "model": "dsv4-pro",
           "messages": [{"role": "user", "content": "Who are you?"}],
           "max_tokens": 256,
           "temperature": 0
       }'
   ```

使用 AISBench 基于 GPQA 数据集进行精度测试。测试设计如下：

| 项 | 内容 |
| --- | --- |
| 评测工具 | AISBench（安装指南见[《AISBench 代码仓安装指南》](https://github.com/AISBench/benchmark/blob/master/README.md)） |
| 评测对象 | W4A8 量化权重 `${SAVE_PATH}` |
| 对比基准 | 选定的精度基线，论文或模型卡片公开分数可作为基线，但需注明来源并确保评测条件一致 |
| 数据集与任务 | [GPQA](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/gpqa) |
| 验收规则 | 量化结果不低于选定的精度基线时直接通过；量化结果低于选定的精度基线且相对下降比例不超过 1% 时通过；量化结果低于选定的精度基线且相对下降比例超过 1% 时不通过并进入步骤 8 |
| 配置一致性 | 量化评测使用与选定的精度基线一致的数据集、服务配置、生成参数和后处理流程 |

评测时，使用与选定的精度基线一致的评测数据、服务配置、生成参数和后处理流程，仅对量化权重 `${SAVE_PATH}` 完成 GPQA 评测，记录量化结果，并与选定的精度基线计算相对下降比例。

**评测配置要点**：

| 配置项 | 取值 |
| --- | --- |
| 配置文件路径 | `vllm_api_general_chat.py`（自定义配置，见下方） |
| 服务地址 / 端口 | 量化服务 `${QUANT_HOST}:${QUANT_PORT}` |
| served-model-name | 量化服务 `dsv4-pro` |

**评测配置（`vllm_api_general_chat.py`）**：

```python
import os

from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="",
        model=os.environ["SERVE_MODEL"],
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip=os.environ.get("SERVE_HOST", "localhost"),
        host_port=int(os.environ["SERVE_PORT"]),
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

使用与选定的精度基线一致的数据集、生成参数和后处理配置，仅对量化服务执行评测：

```bash
# 量化服务 `${SAVE_PATH}`
ais_bench \
    --models vllm_api_general_chat \
    --datasets gpqa_gen \
    --work-dir ${QUANT_WORK_DIR} \
    --dump-eval-details
```

**输出与记录**：

- 量化结果目录：`${QUANT_WORK_DIR}`（如 `AISBench outputs/{timestamp}/`）。
- 须保留：量化评测日志、GPQA 分数摘要和完整 AISBench 产物，并记录量化结果与选定的精度基线的对比及相对下降比例。

**精度结论**：量化权重 GPQA 实测分数为 **89.9**，与选定的精度基线对比后，满足精度验收要求。

**注意**：评测数据可能存在波动，若单次测评结果不符合预期，应记录多次测评结果及平均值。

**输出**：

- vLLM Ascend 量化推理服务正常运行，curl 请求返回正常响应；AISBench 量化结果及其与选定的精度基线的相对下降计算记录。

**下一步**：当量化结果不低于选定的精度基线时结束；当量化结果低于选定的精度基线且相对下降比例不超过 1% 时结束；仅当相对下降比例超过 1% 时执行步骤 8 回退调优。

---

### 步骤 8：回退调优（可选）

**目标**：当量化结果相对选定的精度基线下降超过 1% 时，定位敏感层并调整配置后重新量化、评测。

**输入**：原始浮点权重 `${MODEL_PATH}`、量化配置 `${CONFIG_PATH}`、选定的精度基线、量化结果及评测日志。未达标的量化权重 `${SAVE_PATH}` 仅用于对比，不作为调优输入。精度基线和量化结果应使用相同的评测条件，论文或模型卡片公开分数可作为基线，但需注明来源并确保评测条件一致。

**操作**：

1. 使用[线性层敏感层分析工具](../user_guide/usage_sensitive_linear_analysis.md)分析原始浮点权重，定位量化敏感层。完整调优流程参见[《量化精度调优指南》](../user_guide/process_quantization_precision_tuning.md)：

   ```bash
   msmodelslim analyze linear \
       --model_type DeepSeek-V4-Pro \
       --model_path ${MODEL_PATH} \
       --metrics kurtosis \
       --top_k 15
   ```

2. 根据敏感层结果调整[量化配置](../../../lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml)，可选择：

   - **回退到浮点**：在 `linear_quant` 的 `exclude` 中排除敏感层，使其保留浮点路径。
   - **局部提位宽**：仅将敏感层从 W4A8 调整为 W8A8，其余层保持 W4A8。

3. 使用原始浮点权重 `${MODEL_PATH}` 和调整后的配置重新执行步骤 5，并重复步骤 6、步骤 7 检查产物和评测精度；若仍未达标，可继续扩大回退范围或采用更保守的量化策略。

**输出**：重新生成的量化权重、敏感层分析结果及调优前后的精度对比。

**记录**：敏感层及回退清单、选定的精度基线、量化结果和相对下降比例。

**下一步**：结束。

---

## 5. 结果与经验

### 5.1 关键结果汇总

| 步骤 | 关键操作 | 指标 | 变化 | 备注 |
| --- | --- | --- | --- | --- |
| 步骤 1-2 | 确认模型结构、设计量化方案 | 结构特性 + 混合量化方案 | - | 路由专家 W4A8，共享专家与注意力层 W8A8 |
| 步骤 3 | 开发适配器、注册 entry point、编写量化 YAML | 适配器 + YAML 交付 | - | `--model_type DeepSeek-V4-Pro` 可命中适配器；模型已具备 DP+EP 多卡适配，量化时通过 `--config` 指定 YAML |
| 步骤 5 | 执行 `msmodelslim quant` W4A8 多卡量化 | 量化权重生成 | 原始浮点权重 -> W4A8 量化 | 使用 `--device npu --device_id 0 1 2 3 4 5 6 7`；量化策略：QuaRot + Flex AWQ SSZ + 混合量化 |
| 步骤 6 | 检查量化产物 | 产物完整性 | - | 模型文件齐全且与 DeepSeek-V4-Pro W4A8 方案一致 |
| 步骤 7 | AISBench GPQA 量化评测 | 量化结果及相对下降 | 量化权重与选定的精度基线对比 | 量化结果不低于选定的精度基线时直接通过；量化结果低于选定的精度基线且相对下降比例不超过 1% 时通过；量化结果低于选定的精度基线且相对下降比例超过 1% 时不通过 |
| 步骤 8 | 敏感层分析 + 量化回退/局部提位宽（可选） | 精度复测达标 | 使用原始浮点权重重新量化输出 | 仅当量化结果低于选定的精度基线且相对下降比例超过 1% 时执行 |

### 5.2 经验总结

1. **新模型接入的三要素**：确认模型结构 -> 设计量化方案 -> 完成适配器与量化 YAML 的开发。其中，模型结构特性（MoE 路由专家/共享专家、MLA 低秩注意力、MTP 层等）直接决定子图结构与旋转映射的实现，是适配器开发的输入。适用边界：新模型接入 msModelSlim 的通用流程。

2. **适配器需实现量化流水线所需接口**：`DeepSeekV4ModelAdapter` 需同时继承 `ModelSlimPipelineInterfaceV1`、`FlexSmoothQuantInterface`、`QuaRotInterface`、`AscendV1SaveInterface` 等接口，量化方案中使用了哪些处理器，适配器就要提供对应的子图/映射信息，否则对应 processor 无法执行。适用边界：msModelSlim 量化流水线（modelslim_v1）支持的模型适配器。

3. **适配器注册依赖 entry point**：模型适配器通过 `config/config.ini` 注册，安装 msmodelslim 工具后由 `setup.py` 自动生成 entry point；新增或修改适配器后需重新安装 msmodelslim 工具才生效。适用边界：源码安装方式。

4. **量化命令在已完成安装验证的工作环境中执行**：执行量化前确认 `msmodelslim --help` 可正常调用，并使用明确的模型路径和保存路径。适用边界：源码安装方式。

5. **DeepSeek-V4-Pro 已具备 DP+EP 多卡量化适配**：`model.py` 中设置 `USE_DP_MODE=True`，`model_adapter.py` 负责分布式同步和 EP 本地路由专家处理，当前配置使用的 QuaRot、FlexAWQSSZ、FlexSmoothQuant 和 LinearQuant 均支持 DP。因此可直接通过 `--device npu --device_id 0 1 2 3 4 5 6 7` 启动多卡量化，并以日志确认各 rank 已初始化且未回退到单卡。适用边界：DeepSeek-V4-Pro W4A8 量化阶段。

6. **W4A8 混合量化策略**：DeepSeek-V4-Pro 的 W4A8 量化采用混合策略——路由专家使用 W4A8 动态量化（`ssz` 方法），共享专家和注意力层使用 W8A8 动态量化，在保证精度的同时有效降低模型大小和推理显存占用。适用边界：DeepSeek-V4-Pro W4A8 量化方案（Atlas 800I A3 推理服务器）。

7. **量化产物需同时通过格式和加载验收**：按 [AscendV1 导出格式](../knowledge_base/quantization_format/ascendv1/term_ascendv1.md#export-artifacts) 核对 `quant_model_description.json`、权重文件或分片及 index、`config.json`、tokenizer 等辅助文件，确认描述 JSON 可解析且配置标识、模型类型一致，并通过目标推理框架完成至少一次加载和推理冒烟后，才能进入正式精度评测。适用边界：vLLM Ascend 0.22.1rc1 及 Atlas 800I A3 推理服务器。

8. **精度调优以选定的精度基线为依据**：当量化结果低于相同条件下的选定精度基线且相对下降比例超过 1% 时，才通过敏感层分析进行回退或局部提位宽；分析和重新量化的输入始终是原始浮点权重 `${MODEL_PATH}`，`${SAVE_PATH}` 仅作为量化输出和结果对比目录，不能作为调优输入。论文或模型卡片公开分数可作为基线，但需注明来源并确保评测条件一致。适用边界：DeepSeek-V4-Pro W4A8 量化方案。

## 6. 异常处理

- **量化命令执行失败（OOM）**：确认 8 张 NPU 均可用且未被其他任务占用，命令使用了 `--device npu --device_id 0 1 2 3 4 5 6 7`。
- **Transformers 加载模型报错**：确认是否安装了 `transformers==4.48.2`，模型权重路径正确；原始权重可能为 FP8，需由适配器执行自动反量化。
- **vLLM 服务启动失败**：检查 CANN 环境变量（`env | grep ASCEND`）、NPU 状态（`npu-smi info`），确认量化服务端口未被占用，以及 `--tensor-parallel-size 16`、`--data-parallel-size 2` 与两节点各 16 卡的部署配置一致。

## 7. 附录

- 量化配置文件：[lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml](../../../lab_practice/deepseek_v4/deepseek_v4_pro_w4a8.yaml)
- 校准数据集：[lab_calib/mix_calib.jsonl](../../../lab_calib/mix_calib.jsonl)
- 模型适配器代码：[msmodelslim/model/deepseek_v4/](../../../msmodelslim/model/deepseek_v4/)（[model_adapter.py](../../../msmodelslim/model/deepseek_v4/model_adapter.py)、[model.py](../../../msmodelslim/model/deepseek_v4/model.py)、[loader.py](../../../msmodelslim/model/deepseek_v4/loader.py)）
- 适配器注册配置：[config/config.ini](../../../config/config.ini)
- 量化权重格式与交付检查：[AscendV1 使用指南](../knowledge_base/quantization_format/ascendv1/ascendv1_usage.md)
- vLLM Ascend 官方部署文档：[《DeepSeek-V4-Pro 教程》](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Pro.html)
- AISBench 评测工具及安装指南：[代码仓 README](https://github.com/AISBench/benchmark/blob/master/README.md)
- GPQA 精度参考：论文或模型卡片公开分数 **89.1** 作为公开基线分数（参考[《ModelScope 模型卡片》](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro)）。
