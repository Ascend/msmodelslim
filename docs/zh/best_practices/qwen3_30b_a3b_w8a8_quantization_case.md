# Qwen3-30B-A3B-W8A8 量化实践案例

本案例完整记录了 Qwen3-30B-A3B 模型从 BF16 到 W8A8 的端到端量化部署与评测实践过程。我们在 Ascend 910B 硬件环境下，使用 msModelSlim 26.1.0 提供的一键量化 CLI 工具成功完成了模型量化及权重完整性校验，随后基于 vLLM-Ascend 完成了服务化部署，并通过 AISBench 工具对模型进行了严格的精度和性能压测。本实践的全部步骤与结论均源自真实的部署与测算记录，旨在为开发者提供一套严谨、可复现的参考指南

## 1. 案例背景

### 1.1 目标与使用场景

本案例目标如下：

- 将 Qwen3-30B-A3B 从 BF16 权重量化为 W8A8（权重 INT8、激活 INT8）。
- 验证量化产物能够在昇腾 NPU 上完成服务化推理。
- 使用 AISBench 对 BF16 与 W8A8 进行 GPQA diamond、AIME 2024 精度对比，并完成轻量负载下的性能对比。
- 沉淀 MoE 模型量化、服务化和评测过程中可迁移的排障经验。

适用场景：希望通过 msModelSlim 一键量化减少 Qwen3-30B-A3B 权重体积，并参考历史独立评测进行后续部署验证。

### 1.2 本案例覆盖的流程步骤

本文覆盖以下步骤：前 3 步在新 CLI 环境实测，后续服务化和评测为历史脚本产物的独立实测：

1. 环境与模型准备。
2. W8A8 量化。
3. 量化产物完整性核对。
4. W8A8 / BF16 服务化启动。
5. AISBench 精度评测。
6. AISBench 性能评测。
7. 结果汇总与异常排查。

### 1.3 关联流程

- [主流模型量化部署流程指南](../user_guide/process_mainstream_model_deployment.md)
- [一键量化使用指南](../user_guide/usage_one_click_quantization.md)
- [一键量化完整指南](../user_guide/usage_quick_quantization.md)
- [大模型支持矩阵](../knowledge_base/model/README.md)

本案例属于“已获得可部署量化权重后进行服务化与精度/性能测评”的实际案例，场景落在《主流模型量化部署流程指南》的适用范围内；量化部分同时参考一键量化相关指南。本案例仅记录 Qwen3-30B-A3B W8A8 的实际取值、操作与结果，不重复流程文档中的通用门禁、安装规范或异常处置规则。

## 2. 环境与版本

### 2.1 实测环境

**本次一键量化环境**

| 项目 | 实测配置 |
|---|---|
| 平台及 NPU | HiDevLab A2，4×Ascend 910B4，每卡 32GiB HBM |
| CPU / 内存 / 磁盘 | 128 核 / 256GB / 300GB（创建页规格） |
| 镜像 | TorchNPU 2.1.0，Ubuntu 22.04 |
| Python | 3.11.15，虚拟环境 `/workspace/venv_quant` |
| PyTorch / torch_npu | 2.10.0+cpu / 2.10.0；NPU 计算通过 |
| CANN | 安装目录 `/usr/local/Ascend/cann-9.0.0` |
| msModelSlim | **26.1.0**，`msmodelslim quant` |
| Transformers | **4.57.6** |
| 输入权重 | `/workspace/shared_assets/models/Qwen/Qwen3-30B-A3B`（约 62GB） |
| 输出权重 | `/workspace/qwen3_w8a8_cli_output`（约 30GB） |

## 3. 输入与交付件

### 3.1 输入

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
|---|---|---|---|---|
| 模型权重 | Qwen3-30B-A3B BF16 | `${MODEL_PATH}`；模型标识 `Qwen/Qwen3-30B-A3B` | Hugging Face / safetensors 格式，可被 transformers 正常加载 | 能够完成模型加载；权重文件完整 |
| 校准数据 | anti prompt 50 条 | `example/common/qwen3-moe_anti_prompt_50.json` | JSON；使用仓库自带数据 | 文件存在且可被量化脚本读取 |
| 校准数据 | calib prompt 50 条 | `example/common/qwen3-moe_calib_prompt_50.json` | JSON；使用仓库自带数据 | 文件存在且可被量化脚本读取 |
| 精度评测配置 | GPQA diamond | AISBench `configs/datasets/gpqa/` 下的 `gpqa_gen_0_shot_str.py` | 0-shot 生成式评测；`max_out_len` 需给足，避免长推理输出被截断 | AISBench 任务成功并生成 GPQA summary |
| 精度评测配置 | AIME 2024 | AISBench `configs/datasets/aime2024/` 下的官方配置 `aime2024_gen_0_shot_str.py` | 0-shot 字符串格式，accuracy(pass@1)；`max_out_len` 需给足 | AISBench 任务成功并生成 AIME 2024 summary |
| 性能评测配置 | GSM8K perf | AISBench `configs/datasets/gsm8k/` 下的 `gsm8k_gen_0_shot_cot_str_perf.py` | 本案例使用 50 请求，服务端 `max-model-len=4096` | Success 50 / Failed 0，生成性能 summary |

### 3.2 交付件

| 类型 | 名称 | 来源或保存位置 | 格式或约束 | 验收方式 |
|---|---|---|---|---|
| 量化权重 | Qwen3-30B-A3B W8A8 量化目录 | `${SAVE_PATH}` | 本案例历史权重为 8 个 safetensors、合计约 30GB，并覆盖 48 层 | 量化日志正常结束，输出 `SUCCESS`；生成 8 个 safetensors 权重分片，总体积约 30GB；通过 safetensors 元数据检查，共包含 57,220 个张量，覆盖全部 48 层（0–47），无缺失层。 |
| 服务化结果 | W8A8 OpenAI 兼容服务 | `${W8A8_LOG}` | vLLM-Ascend，TP=1 | 日志出现 `Application startup complete`，并可正常生成文本 |
| 服务化结果 | BF16 OpenAI 兼容服务 | `${BF16_LOG}` | vLLM-Ascend，TP=4；MoE 开启专家并行 | 服务正常启动，可被 AISBench 调用 |
| 精度结果 | GPQA / AIME summary | `${AISBENCH_WORK_DIR}` | BF16 与 W8A8 使用一致评测口径 | GPQA、AIME 均获得可核对分数 |
| 性能结果 | GSM8K perf summary | `${PERF_WORK_DIR}` | 50 请求；双方 `max-model-len=4096` | 两组均 Success 50 / Failed 0，并记录吞吐与 token 数 |

## 4. 操作步骤

### 4.1 步骤 1：准备路径与一键量化环境

**目标**：确认昇腾 NPU、原始 BF16 权重和 msModelSlim CLI 均可用。复现者请按实际部署路径调整下列变量。

```bash
source /workspace/venv_quant/bin/activate
cd /workspace
export MODEL_PATH=/workspace/shared_assets/models/Qwen/Qwen3-30B-A3B
export SAVE_PATH=/workspace/qwen3_w8a8_cli_output
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
mkdir -p /workspace/quant_logs
msmodelslim quant --help
npu-smi info
```

本次使用 msModelSlim 26.1.0 和 Transformers 4.57.6，原始模型目录约 62GB。工具从最佳实践自动选择 Qwen3-30B W8A8 配置，而非人工编辑专用 Python 量化脚本。

### 4.2 步骤 2：执行 W8A8 一键量化

**本次成功运行的完整命令**：

```bash
cd /workspace
source /workspace/venv_quant/bin/activate
export MODEL_PATH=/workspace/shared_assets/models/Qwen/Qwen3-30B-A3B
export SAVE_PATH=/workspace/qwen3_w8a8_cli_output
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
mkdir -p /workspace/quant_logs
set -o pipefail
msmodelslim quant \
  --model_path "$MODEL_PATH" \
  --save_path "$SAVE_PATH" \
  --device npu \
  --model_type Qwen3-30B \
  --quant_type w8a8 \
  --trust_remote_code True \
  2>&1 | tee /workspace/quant_logs/qwen3_w8a8_cli.log
```

**真实运行结果**：日志显示 `Get best practice ... success`、使用 `mix_calib.jsonl` 准备校准数据成功、模型加载成功、校准成功、权重及 `quant_model_description.json` 保存成功，最终出现 `QUANTIZATION: END` 与 `SUCCESS`。从约 07:13 至 08:36，耗时约 **1 小时 23 分钟**（按终端时间估计）。

工具自动匹配 Qwen3-30B W8A8 最佳实践，本次运行未通过 `--config_path` 手动传入 YAML；导出的量化描述包含 `W8A8`、`W8A8_DYNAMIC` 与保留 `FLOAT` 的项目。虽然四张 NPU 对进程可见，但不应据此推断所有层完全均匀分配。`modelslim_v0` 不适用 v1 专属的多卡索引 CLI 写法。

### 4.3 步骤 3：核对量化产物

新产物输出目录为 `/workspace/qwen3_w8a8_cli_output`，约 30GB，包含 8 个 safetensors、权重索引、配置及 tokenizer 文件和 `quant_model_description.json`。描述文件中 `model_quant_type` 为 `W8A8`。

下列检查已在本次终端实际运行，通过 safetensors header 读取 tensor keys；不会把完整权重加载到 NPU：

```bash
python3 - <<'PYCODE'
from pathlib import Path
from safetensors import safe_open
import re

folder = Path('/workspace/qwen3_w8a8_cli_output')
files = sorted(folder.glob('*.safetensors'))
layers = set()
tensor_count = 0
for file in files:
    with safe_open(str(file), framework='pt', device='cpu') as f:
        keys = list(f.keys())
        tensor_count += len(keys)
        for key in keys:
            m = re.search(r'model\.layers\.(\d+)\.', key)
            if m:
                layers.add(int(m.group(1)))
print('权重文件数量:', len(files))
print('张量总数:', tensor_count)
print('实际覆盖层数:', len(layers))
print('缺失层:', sorted(set(range(48)) - layers))
assert len(files) == 8
assert layers == set(range(48))
print('PASS: 8个权重分片可读取，48层模型完整')
PYCODE
```

### 4.4 步骤 4：启动 W8A8 服务

**目标**：将 W8A8 权重以 OpenAI 兼容 API 方式启动，为 AISBench 提供服务。

**输入**：量化产物路径 `${SAVE_PATH}`

**操作**：性能评测配置如下：

```bash
export SAVE_PATH=/workspace/qwen3_w8a8_cli_output
export LOG_DIR=/workspace/quant_logs
mkdir -p "${LOG_DIR}"
export ASCEND_RT_VISIBLE_DEVICES=0
export W8A8_LOG="${LOG_DIR}/vllm_w8a8_accuracy.log"

python3 -m vllm.entrypoints.openai.api_server \
    --model "${SAVE_PATH}" \
    --served-model-name qwen3-30b-a3b-w8a8 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --enforce-eager \
    --gpu-memory-utilization 0.9 \
    --max-model-len 40960 \
    2>&1 | tee "${W8A8_LOG}"
```

精度评测时，本案例将服务端 `--max-model-len` 调整为 `40960`，避免长推理题被上下文长度限制截断。

**输出**：OpenAI 兼容推理服务监听 8000 端口。

**记录**：服务就绪日志包含 `Application startup complete`；本案例还观察到量化格式被识别为 Ascend 量化格式。

### 4.5 步骤 5：启动 BF16 基线服务

**目标**：启动与 W8A8 对照的 BF16 服务。

**输入**：`${MODEL_PATH}`。

**操作**：性能评测配置如下：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export BF16_LOG="${LOG_DIR}/vllm_bf16_4096.log"

python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name qwen3-30b-a3b-bf16 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --enforce-eager \
    --gpu-memory-utilization 0.7 \
    --max-model-len 4096 \
    --enable-expert-parallel \
    --distributed-executor-backend mp \
    2>&1 | tee "${BF16_LOG}"
```

精度评测时同样使用 `--max-model-len 40960`。

**输出**：BF16 OpenAI 兼容推理服务。

**记录**：`${BF16_LOG}`。本案例中 BF16 需要 TP=4；MoE 服务需显式开启专家并行，否则曾出现服务启动阶段静默 hang。

### 4.6 步骤 6：执行 AISBench 精度评测

**目标**：比较 BF16 与 W8A8 在 GPQA diamond 和 AIME 2024 上的精度。

**输入**：已启动的 W8A8 或 BF16 服务、AISBench 模型配置与数据集配置。

**操作**：

```bash
export EVAL_TAG=w8a8  # 评测 BF16 时改为 bf16
export AISBENCH_WORK_DIR="${LOG_DIR}/aisbench_accuracy_${EVAL_TAG}"
mkdir -p "${AISBENCH_WORK_DIR}"

# GPQA
ais_bench \
    --models vllm_api_general_chat \
    --datasets gpqa_gen_0_shot_str.py \
    --mode all \
    --dump-eval-details \
    --merge-ds \
    --work-dir "${AISBENCH_WORK_DIR}"

# AIME 2024
ais_bench \
    --models vllm_api_general_chat \
    --datasets aime2024_gen_0_shot_str.py \
    --mode all \
    --dump-eval-details \
    --merge-ds \
    --work-dir "${AISBENCH_WORK_DIR}"
```

对 W8A8 与 BF16 分别启动对应服务后各运行一遍上述命令，并分别使用 `EVAL_TAG=w8a8` 与 `EVAL_TAG=bf16`，确保结果目录互不覆盖。

本案例实际排障中发现，服务端上下文长度和 AISBench 输出长度均会影响长推理题结果：

- 服务端精度评测使用 `max-model-len=40960`。
- GPQA / AIME 的 `max_out_len` 需给足；本案例曾因输出长度过小导致 `<think>` 内容被截断并产生异常低分，后续已调整后重新取得本文结果。

**输出**：GPQA、AIME 的评测 summary。

**记录**：将 BF16 与 W8A8 的 summary 分开保存，避免后续通配符误读历史目录。

### 4.7 步骤 7：执行 AISBench 性能评测

**目标**：在相同数据集、请求数和 `max-model-len` 下比较两种实际部署形态的吞吐表现。

**输入**：分别启动的 BF16 TP=4 与 W8A8 TP=1 服务。

**操作**：

```bash
export EVAL_TAG=w8a8  # 评测 BF16 时改为 bf16
export PERF_WORK_DIR="${LOG_DIR}/aisbench_perf_${EVAL_TAG}"
mkdir -p "${PERF_WORK_DIR}"

ais_bench \
    --models vllm_api_general_chat \
    --datasets gsm8k_gen_0_shot_cot_str_perf.py \
    --summarizer default_perf \
    --mode perf \
    --num-prompts 50 \
    --work-dir "${PERF_WORK_DIR}"
```

BF16 与 W8A8 分别运行一次；运行 W8A8 时使用 `EVAL_TAG=w8a8`，运行 BF16 时使用 `EVAL_TAG=bf16`，从而将两组性能结果写入独立目录。

**输出**：吞吐、生成 token 数、成功/失败请求数等性能 summary。

**记录**：本案例两组均 Success 50 / Failed 0；Total Generated Tokens 分别为 76155 与 77258，可作为两次独立运行的辅助核对信息。

## 5. 结果与经验

### 5.1 关键结果汇总

| 关键步骤 | 操作 | 指标 | 变化 | 备注 |
|---|---|---|---|---|
| W8A8 量化 | `msmodelslim quant` 26.1.0 | 权重文件总体积 | 约 62GB → 约 30GB |不等同运行时 HBM 占用 |
|  产物核对 | `safetensors.safe_open` | 层完整性 | 48 / 48 层 | 8 个 safetensors、57,220 张量 |
| GPQA diamond | AISBench 精度评测 | accuracy | 59.60 → 58.59（-1.01） | 500 题；BF16 与 W8A8 同评测口径 |
| AIME 2024 | AISBench 精度评测 | accuracy | 70.00 → 66.67（-3.33） | 30 题 |
| 轻量性能对比 | GSM8K，50 请求，`max-model-len=4096` | Total Token Throughput | 91.16 → 94.32 token/s（约 +3.47%） | BF16 为 4 卡 TP=4；W8A8 为单卡 TP=1，属于实际部署形态对比，不是同卡数纯算子加速比 |
| 部署资源 | BF16 TP=4 vs W8A8 TP=1 | 部署卡数 | 4 卡 → 1 卡 | 减少 3 张卡，部署卡数降至原来的 1/4 |

### 5.2 精度结果

精度评测服务端均使用 `max-model-len=40960`。

| 数据集 | 浮点 BF16 | 量化 W8A8 | 差值 | 说明 |
|---|---:|---:|---:|---|
| GSM8K | 未测试 | 96.06 | — | W8A8 冒烟与基础能力验证 |
| GPQA diamond | 59.60 | 58.59 | -1.01 | 500 题；本案例中量化前后差异较小 |
| AIME 2024 | 70.00 | 66.67 | -3.33 | 30 题；对应 1 题差异 |

本案例可以据实得出：GPQA diamond 上 W8A8 相比 BF16 下降 1.01 分；AIME 2024 的单次实测下降 3.33 分。

### 5.3 性能结果

测试条件：GSM8K、50 请求、`max-model-len=4096`，两组均 Success 50 / Failed 0。

| 指标 | BF16（四卡 TP=4） | W8A8（单卡 TP=1） | 结果 |
|---|---:|---:|---|
| Total Token Throughput | 91.16 token/s | 94.32 token/s | W8A8 部署形态约高 3.47% |
| Output Token Throughput | 87.28 token/s | 90.36 token/s | W8A8 更高 |
| Total Generated Tokens | 76155 | 77258 | 两次运行生成 token 数不同 |
| Success / Failed | 50 / 0 | 50 / 0 | 均成功 |
| 部署卡数 | 4 | 1 | W8A8 减少 3 张卡 |

> 口径说明：这是“实际部署形态对比”，不是同卡数、同并行策略下的纯量化算子加速比。BF16 因本案例环境中的部署约束使用 4 卡 TP=4，W8A8 使用单卡 TP=1；本测试也仅为 50 请求的轻量负载，不代表饱和压测峰值。

### 5.4 经验总结

1. **先确认模型处于官方支持矩阵。** 本案例的可复现主线建立在 Qwen3-30B-A3B W8A8 已受 msModelSlim 支持的前提上。该经验不应直接外推到未适配的 Qwen3.5 或其他混合架构。
2. **MoE 多卡服务需关注专家并行配置。** 本案例 BF16 TP=4 使用 `--enable-expert-parallel --distributed-executor-backend mp` 后服务成功启动；该结论适用于本案例 vLLM-Ascend/CANN 组合，不等同于所有版本的固定要求。
3. **精度评测必须避免输出截断。** 长推理题对 `max-model-len` 和 AISBench `max_out_len` 敏感。本案例曾因输出长度不足产生异常低分，调整后才得到最终记录结果。
4. **性能对比必须对齐关键配置。** 本案例最终将双方 `max-model-len` 对齐为 4096，并使用相同 50 请求；不同 TP 规模是实际部署形态差异，因此结论应表述为部署效率差异，而非纯算子加速倍数。
5. **结果目录应独立保存。** AISBench 输出使用独立 `--work-dir`，同时核对生成 token 数，可降低误读历史目录或缓存结果的风险。

### 5.5 适用边界

- **硬件**：4× Ascend 910B4 实测环境；
- **软件**：CANN 9.0.0（V100R001C10SPC001B250）；量化 transformers 4.57.6；评测 transformers 5.5.4；AISBench 3.1.20260630。其他版本可能存在行为变化。
- **数据**：GPQA diamond、AIME 2024、GSM8K
- **性能负载**：GSM8K 50 请求，`max-model-len=4096`，属于轻量负载，不代表饱和吞吐上限。
- **对比口径**：W8A8 TP=1 vs BF16 TP=4，反映本案例中的实际可部署方案，不是同卡数 benchmark。

## 6. 异常处理

以下问题均来自本案例实际排障过程。

### 6.1 vLLM 图捕获阶段显存申请失败

**现象**：服务在图捕获阶段报显存申请失败。

**处理**：本案例使用：

性能评测使用：

```text
--enforce-eager --gpu-memory-utilization 0.9 --max-model-len 4096
```

精度评测使用：

```text
--enforce-eager --gpu-memory-utilization 0.9 --max-model-len 40960
```

必要时先清理残留进程，再将 `--gpu-memory-utilization` 从 0.9 调低到 0.7。

### 6.2 MoE BF16 多卡服务静默 hang

**现象**：服务停在 `Breakable cudagraph` 附近，无明显 Python traceback。

**处理**：本案例 BF16 TP=4 补充：

```text
--enable-expert-parallel --distributed-executor-backend mp
```

### 6.3 AISBench 精度异常偏低或为 0

**现象**：GPQA/AIME 结果异常低，生成内容中的长推理过程未完成。

**排查结论**：本案例曾因 `max_out_len` 太小而截断 `<think>` 输出。

**处理**：提高服务端 `max-model-len` 和 AISBench `max_out_len`；本案例精度评测服务端最终使用 `max-model-len=40960`。

### 6.4 残留 EngineCore 占卡

**现象**：新服务启动后等待或无法正常占用设备。

**处理**：确认不再需要旧任务后清理相关残留进程，例如：

```bash
pkill -9 -f 'vllm|EngineCore|api_server'
```

> 执行前应确认当前用户没有仍需保留的同名任务。

### 6.5 输出目录所在存储卷空间不足

**现象**：出现 `No space`、`Errno 5`、`ENOTCONN` 等 I/O 错误，但主盘仍有空间。

**处理**：使用 `df -h` 分挂载点检查容量，将 AISBench `--work-dir` 和日志目录切换到空间充足的挂载点。

### 6.6 量化目录缺少 tokenizer 文件

**现象**：量化权重已生成，但 vLLM 无法从量化目录完整加载 tokenizer。

**处理**：从原 BF16 模型目录复制本模型实际使用的 tokenizer 配置文件到量化目录，并核对文件与原模型版本一致。

## 7. 附录

### 7.1 结果数据速查

- **W8A8**：SUCCESS；8 个 safetensors，约 30GB，57,220 张量，48 层完整。
- **GPQA diamond**：BF16 59.60，W8A8 58.59。
- **AIME 2024**：BF16 70.00，W8A8 66.67。
- **W8A8 GSM8K**：96.06。
- **性能**：BF16 TP=4 总吞吐 91.16 token/s；W8A8 TP=1 总吞吐 94.32 token/s。

### 7.2 关联 Issue / PR

- 任务 Issue：[#206](https://gitcode.com/Ascend/msmodelslim/issues/206)
- 交付跟踪 Issue：[#414](https://gitcode.com/Ascend/msmodelslim/issues/414)
- 当前文档 PR：[#933](https://gitcode.com/Ascend/msmodelslim/pull/933)
