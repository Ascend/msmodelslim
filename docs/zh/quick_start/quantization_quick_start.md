# 量化快速入门

## 1. 概述

msModelSlim 提供了两种量化方式：**一键量化（V1）**和**传统量化（V0）**。

- **一键量化（V1）**：面向零基础用户，通过命令行方式快速完成量化，具备“开箱即用”的特性。系统会自动匹配最佳实践配置，用户只需指定必要参数即可；此外也支持自定义精细化混合量化策略，灵活性高。
- **传统量化（V0）**：通过 Python 脚本方式执行量化，在泛化性、可读性等方面均低于一键量化，已停止演进，通常用于一键量化尚未支持的模型。

下面将以 Qwen2.5-7B-Instruct 为例完成量化并基于vllm-ascend完成一次推理。

## 2. 环境准备

### 2.1 镜像准备

vllm-ascend提供用于部署的Docker镜像，可以从镜像仓库[ascend/vllm-ascend](https://quay.io/repository/ascend/vllm-ascend?tab=tags)拉取预构建镜像，具体参考[vllm-ascend快速入门](https://docs.vllm.ai/projects/ascend/en/latest/)。

### 2.2 镜像内安装 msModelSlim

安装命令具体参考《[msModelSlim安装指导](../install_guide/install_guide.md)》

### 2.3 下载大模型原始浮点权重

以 Qwen2.5-7B-Instruct 为例，可前往 [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) 获取原始模型权重。

### 2.4 安装其他依赖（与模型相关，参考huggingface Model card）

```shell
pip install transformers==4.43.1
```

### 2.5 准备校准数据

传统量化方式需要准备校准数据文件（`.jsonl` 格式），用于量化过程中的校准。示例数据文件位于 [`example/common/`](../../../example/common/) 目录下，如 [`boolq.jsonl`](../../../example/common/boolq.jsonl)、[`teacher_qualification.jsonl`](../../../example/common/teacher_qualification.jsonl) 等。

## 3. 量化方式选择

### 3.1 方式一：一键量化（推荐）

一键量化通过命令行方式启动，系统会自动匹配最佳实践配置。

#### 3.1.1 命令格式

```bash
msmodelslim quant [ARGS]
```

#### 3.1.2 参数说明

参数说明可以参考《[一键量化参数说明](../user_guide/feature_guide/quick_quantization_v1/usage.md#42-参数说明)》

**说明：**

- 最佳实践库中的配置文件放在 `msmodelslim/lab_practice` 中。
- 若最佳实践库中未匹配到符合条件的最优配置，系统将依据预设规则为你推荐其他可用配置，并向你确认是否使用该推荐配置继续量化。
- 如果需要打印量化运行日志，可通过环境变量 `MSMODELSLIM_LOG_LEVEL` 进行设置，可选值为 `INFO`（默认）或 `DEBUG`。

#### 3.1.3 使用示例

使用一键量化功能量化 Qwen2.5-7B-Instruct 模型，量化方式采用 w8a8：

```bash
msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen2.5-7B-Instruct --quant_type w8a8 --trust_remote_code True
```

其中：

- `${MODEL_PATH}` 为 Qwen2.5-7B-Instruct 原始浮点权重路径
- `${SAVE_PATH}` 为用户自定义的量化权重保存路径

### 3.2 方式二：传统量化（Python 脚本方式）

传统量化通过 Python 脚本执行。

#### 3.2.1 命令格式

不同模型有对应的量化脚本，以 Qwen 模型为例：

```bash
python3 example/Qwen/quant_qwen.py [ARGS]
```

#### 3.2.2 主要参数说明

| 参数名称 | 解释 | 是否可选 | 说明 |
|---------|------|---------|------|
| `model_path` | 模型路径 | 必选 | 原始浮点模型权重路径 |
| `save_directory` | 量化权重保存路径 | 必选 | 量化后权重的保存目录 |
| `calib_file` | 校准数据文件 | 可选 | 校准数据文件路径（`.jsonl` 格式），默认使用 `example/common/teacher_qualification.jsonl` |
| `w_bit` | 权重量化位数 | 可选 | 默认值：8。大模型量化场景下，可配置为 8 或 16；稀疏量化场景下，需配置为 4 |
| `a_bit` | 激活值量化位数 | 可选 | 默认值：8。大模型量化场景下，可配置为 8 或 16；稀疏量化场景下，需配置为 8 |
| `device_type` | 设备类型 | 可选 | 默认值：`cpu`。可选值：`cpu`、`npu` |
| `act_method` | 激活值量化方法 | 可选 | 默认值：1。1 代表 min-max 量化方式；2 代表 histogram 量化方式；3 代表自动混合量化方式（推荐） |
| `anti_method` | 离群值抑制方法 | 可选 | 可选值：`m1`（SmoothQuant）、`m2`（SmoothQuant 加强版）、`m3`（AWQ）、`m4`（smooth 优化）、`m5`（CBQ）、`m6`（Flex smooth） |
| `model_type` | 模型类型 | 可选 | 对于 Qwen 模型，可选值：`qwen1`、`qwen1.5`、`qwen2`、`qwen2.5`、`qwen3`，默认值：`qwen2` |
| `trust_remote_code` | 是否信任自定义代码 | 可选 | 默认值：`False` |

**更多参数：** 传统量化支持更多高级参数，如 `disable_names`（手动回退的量化层）、`fraction`（稀疏量化异常值占比）、`use_kvcache_quant`（KV Cache 量化）等。详细参数说明请参考各模型目录下的 README.md 文件。

#### 3.2.3 使用示例

**示例 1：Qwen2.5-7B-Instruct W8A8 量化**

> **注意：** 传统量化方式需要**通过源码安装 msModelSlim**。执行以下命令克隆代码并进入源码目录：
>
> ```bash
> git clone https://gitcode.com/Ascend/msmodelslim.git
> cd msmodelslim
> ```

进入源码目录后，执行量化脚本：

```bash
python3 example/Qwen/quant_qwen.py \
    --model_path ${MODEL_PATH} \
    --save_directory ${SAVE_PATH} \
    --calib_file example/common/boolq.jsonl \
    --w_bit 8 \
    --a_bit 8 \
    --device_type npu \
    --trust_remote_code True
```

**说明：**

- 不同模型的量化脚本位于 [`example/`](../../../example/) 目录下对应的模型子目录中，如 [`example/Qwen/quant_qwen.py`](../../../example/Qwen/quant_qwen.py)、[`example/Llama/quant_llama.py`](../../../example/Llama/quant_llama.py) 等。
- 各模型的具体量化参数和最佳实践请参考对应模型目录下的 README.md 文件。
- 如果需要使用 NPU 多卡量化，请先配置环境变量：

  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

## 4. 量化结果输出

量化完成后，在保存路径目录下会生成量化权重及相关配置文件。不同量化方式的输出文件略有差异，分别说明如下：

### 4.1 一键量化（V1）输出目录

```yaml
├── config.json                          # 原始模型配置文件
├── generation_config.json               # 原始生成配置文件
├── quant_model_description.json         # 量化权重描述文件
├── quant_model_weight_w8a8.safetensors  # 量化权重文件
├── tokenizer_config.json                # 原始分词器配置文件
├── tokenizer.json                        # 原始分词器词汇表
├── {model_type}_best_practice.yaml      # 量化配置协议文件
└── vocab.json                            # 原始词汇映射文件（部分模型）
```

### 4.2 传统量化（V0）输出目录

```yaml
├── config.json                          # 原始模型配置文件
├── generation_config.json               # 原始生成配置文件
├── quant_model_description.json         # 量化权重描述文件
├── quant_model_weight_w8a8.safetensors  # 量化权重文件
├── tokenizer_config.json                # 原始分词器配置文件
├── tokenizer.json                        # 原始分词器词汇表
└── vocab.json                            # 原始词汇映射文件（部分模型）
```

>[!NOTE]
>
>传统量化方式不会生成 `best_practice.yaml` 文件，其他文件内容与一键量化基本一致。

### 4.3 文件说明

- `quant_model_description.json`（或 `quant_model_description_{quant_type}.json`）- 包含量化参数和配置信息，描述了每个权重的量化类型（W8A8、FLOAT 等）
- `quant_model_weight_{quant_type}.safetensors` - 实际的量化模型权重文件
- `{model_type}_best_practice.yaml`（仅一键量化生成）- 记录本次量化所使用的完整配置信息，可用于复现该量化权重
- 其他文件为模型推理所需的配置和词汇表文件，来自原始浮点目录

## 5. 量化后权重的使用

量化完成后，您可以使用生成的量化权重进行推理。根据不同的推理框架，使用方法如下：

### 5.1 在 vllm-ascend 中使用

可参考 vllm-ascend 官方文档 [Qwen3-32B-W4A4 教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-32B-W4A4.html)运行Docker容器。

#### 5.1.1 环境准备与模型目录结构

假设您已经参考前文使用 msModelSlim 完成了 Qwen2.5-7B-Instruct 的 W8A8 量化，量化后权重保存路径为：

```bash
SAVE_PATH=/home/models/Qwen2.5-7B-w8a8
```

#### 5.1.2 单卡在线服务部署

在 Ascend 设备上使用 vllm-ascend 提供在线服务时，可执行：

```bash
vllm serve /home/models/Qwen2.5-7B-w8a8 \
  --served-model-name "Qwen2.5-7B-w8a8" \
  --max-model-len 4096 \
  --quantization ascend
```

说明：

- `model` 路径 `/home/models/Qwen2.5-7B-w8a8` 即为 msModelSlim 输出的量化模型目录。
- `--quantization ascend`：指定使用适配 Ascend 的量化推理后端，加载由 msModelSlim 生成的权重。
- 其余参数（如 `--max-model-len`）可根据实际业务场景调整。

服务启动后，可以通过 HTTP 接口发起推理请求，例如：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen2.5-7B-w8a8",
        "prompt": "what is large language model?",
        "max_tokens": 128,
        "top_p": 0.95,
        "top_k": 40,
        "temperature": 0.0
      }'
```

其中：

- `model` 字段需要与 `--served-model-name` 保持一致。
- 推理时仅需指定量化后模型目录，无需再传入原始浮点权重。

#### 5.1.3 单卡离线推理（Python API）

如果希望在 Python 脚本中直接加载量化后模型进行离线推理，可以使用 vllm-ascend 的 `LLM` 接口：

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=40,
)

llm = LLM(
    model="/home/models/Qwen2.5-7B-w8a8",  # msModelSlim 量化输出目录
    max_model_len=4096,
    quantization="ascend",                # 启用 Ascend 量化推理
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

要点说明：

- `model` 依然指向量化后权重所在目录。
- `quantization="ascend"` 必须显式设置，以启用 Ascend 量化推理路径。
- 其余采样参数可根据业务调整。

## 6. 其他说明

### 6.1 支持的模型和量化类型

可通过《[大模型支持矩阵](../user_guide/model_support/foundation_model_support_matrix.md)》查看不同模型的支持情况：

- 标记了`一键量化`的模型支持一键量化方式。
- 所有在 [`example/`](../../../example/) 目录下有量化脚本的模型都支持传统量化方式。

### 6.2 大模型量化建议

对于过大的模型（7B 及以上），如果遇到显存不足的问题，可以尝试：

1. **使用逐层量化**：在一键量化中默认生效《[逐层量化](../user_guide/feature_guide/quick_quantization_v1/usage.md#51-逐层量化及分布式逐层量化)》，传统量化中不支持。
2. **使用 CPU 量化**：设置 `--device cpu`（一键量化）或 `--device_type cpu`（传统量化），速度较慢但显存占用低。

### 6.3 支持的量化算法

对于一键量化支持的多种算法，可以参考《[一键量化 V1 架构支持的算法](../user_guide/quantization_algorithms/README.md)》。

### 6.4 常见问题

**Q: 量化过程中出现显存不足怎么办？**

A: 可以尝试以下方法：

1. 使用逐层量化，在一键量化中默认生效，传统量化中不支持。
2. 使用 CPU 进行量化（`--device cpu` 或 `--device_type cpu`，速度较慢但显存占用低）。

**Q: 量化后的模型精度下降明显怎么办？**

A: 可以尝试：

1. 使用更高精度的量化类型（如从 w4a8 改为 w8a8）。
2. 参考 [`msmodelslim/lab_practice`](../../../lab_practice/) 路径下模型对应的最佳实践配置。
3. 检查离群值抑制算法、量化策略、校准数据集等是否合适，参考《[量化精度调优指南](../best_practices/quantization_precision_tuning_guide.md)》。

**Q: 如何验证量化效果？**

A: 可以使用推理框架在线/离线推理，在相同输入下对比量化前后的输出差异，以及比较量化前后的[数据集评分](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/mode.html#id2)差值。

**Q: 如何选择使用哪种量化方式？**

A: 量化方式选择建议：

- 如果模型支持一键量化，建议使用一键量化。
- 如果模型不支持一键量化，建议使用传统量化（已停止演进）。

**Q: 一键量化和传统量化生成的权重文件有什么区别？**

A: 两种方式生成的权重文件格式相同，都可以用于推理。主要区别在于：

- 一键量化使用最佳实践配置，可能包含一些优化。
- 传统量化支持生成MindIE推理框架独占格式，可用于老版本兼容；一键量化支持AscendV1格式（关于该格式的更多信息，请参考 [AscendV1Config](../../../msmodelslim/core/quant_service/modelslim_v1/save/ascendv1.py) 中的说明），可用于多框架（MindIE、vllm-ascend、SGLang）使用。
