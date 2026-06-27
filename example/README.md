# msModelSlim 推荐实践集

msModelSlim 推荐实践集提供了各种大语言模型、多模态理解模型和多模态生成模型的量化实践案例，帮助用户快速上手模型量化功能。

## 目录结构

### 大语言模型量化说明

- **[DeepSeek](./DeepSeek/)** - DeepSeek 系列模型量化说明
- **[GLM](./GLM/)** - GLM 系列模型量化说明
- **[GPT-NeoX](./GPT-NeoX/)** - GPT-NeoX 系列模型量化说明
- **[HunYuan](./HunYuan/)** - HunYuan 系列模型量化说明
- **[InternLM2](./InternLM2/)** - InternLM2 系列模型量化说明
- **[Llama](./Llama/)** - LLaMA 系列模型量化说明
- **[LongCat-Flash](./longcat_flash/)** - LongCat-Flash 系列模型量化说明
- **[MiniMax-M2](./MiniMax-M2/)** - MiniMax-M2 系列模型量化说明
- **[Qwen](./Qwen/)** - Qwen 系列模型量化说明
- **[Qwen3_5](./Qwen3_5/)** - Qwen3.5 系列模型量化说明
- **[Qwen3-MOE](./Qwen3-MOE/)** - Qwen3-MOE 系列模型量化说明
- **[Qwen3-Next](./Qwen3-Next/)** - Qwen3-Next 系列模型量化说明

### 多模态理解模型量化说明

- **[multimodal_vlm](./multimodal_vlm/)** - 多模态理解模型量化说明
  - LLaVA 系列模型
  - Qwen-VL 系列模型
  - InternVL2 系列模型
  - Qwen2-VL 系列模型
  - Qwen2.5-VL 系列模型
  - Qwen2.5-Omni 系列模型
  - Qwen3-VL 系列模型
  - Qwen3-VL-MoE 系列模型
  - GLM-4.1V 系列模型
  - Qwen3-Omni 系列模型
  - GLM-4.6V 模型
  - Kimi-K2.5 模型
  - Kimi-K2.6 模型
  - InternVL3.5 系列模型
  - InternVL3.5-MoE 系列模型

### 多模态生成模型量化说明

- **[multimodal_sd](./multimodal_sd/)** - 多模态生成模型量化说明
  - Stable Diffusion 系列模型
  - Flux 系列模型
  - HunYuanVideo 系列模型
  - OpenSoraPlanV1_2 系列模型
  - Wan2.1 系列模型

### 其他功能

- **[common](./common/)** - 通用工具和校准数据
- **[osp1_2](./osp1_2/)** - OpenSora Plan 1.2 相关功能
- **[ms_to_vllm.py](./ms_to_vllm.py)** - msModelSlim 到 vLLM 格式转换工具

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../docs/zh/install_guide/install_guide.md)。
- 不同模型系列可能依赖特定的版本，请参考各模型目录下的具体说明。

### 使用多卡量化功能

**重要提醒：Atlas 300I Duo 卡仅支持单卡单芯片处理器量化。**

如需使用 NPU 多卡量化，请先配置环境变量：

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
```

### deq_scale 转 int64 后处理（deqscale2int64.py）

当模型默认 `torch_dtype` 为 bf16 时，AscendV1 保存器会将 `deq_scale` 以 float32 原样写入；若推理侧需要 int64 格式的 deq_scale（与算子适配），可使用本脚本对已保存的量化权重做后处理，将权重中的 deq_scale 从 float32/bf16 转为 int64（与 ascendv1 非 bf16 保存时的格式一致）。脚本不依赖 NPU，支持单文件与分片两种权重布局。

**使用场景：**

- 量化时使用 bf16 默认保存，希望得到 int64 格式的 deq_scale 以便在 MindIE / vLLM Ascend 等推理侧使用。
- 对已有 AscendV1 量化目录做一次性转换，无需重新量化。

**命令示例：**

```shell
# 进入示例目录（或在项目根目录下将下方路径改为 example/deqscale2int64.py）
cd example

# 原地转换（直接覆盖原权重目录中的 safetensors 文件）
python deqscale2int64.py --model_path {量化权重目录路径}

# 输出到新目录（保留原目录不变，并复制 config、描述文件等）
python deqscale2int64.py --model_path {量化权重目录路径} --output_dir {输出目录路径}

# 仅查看将转换的 key，不写文件
python deqscale2int64.py --model_path {量化权重目录路径} --dry_run
```

**参数说明：**

| 参数名 | 含义 | 默认值 | 说明 |
|--------|------|--------|------|
| model_path | 量化权重目录 | 必填 | 须包含 `quant_model_description.json` 以及 `quant_model_weights.safetensors`（单文件）或 `quant_model_weights.safetensors.index.json` + 分片文件。 |
| output_dir | 输出目录 | 与 model_path 相同 | 不指定则原地覆盖；指定则先拷贝完整目录再在输出目录内做转换。 |
| dry_run | 仅预览不写入 | False | 加 `--dry_run` 时只打印会被转换的 deq_scale key 及数量，不修改任何文件。 |

脚本会优先根据 `quant_model_description.json` 中类型为 W8A8/W8A8_MIX 的 `.deq_scale` 键识别待转换项；若无描述或未匹配到，则按 key 名（含 `.deq_scale`）及 dtype（float32/bf16）识别。已是 int64 的 deq_scale 会跳过。
