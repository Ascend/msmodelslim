# Qwen3.5 量化说明

## 模型介绍

**Qwen3.5** 是 Qwen 系列最新的旗舰**多模态**模型，采用 **MoE (Mixture of Experts)** 架构，在保持极强模型能力的同时显著降低推理成本。核心架构特点包括：原生多模态能力（Vision Encoder + 图文融合）、混合注意力机制（常规 Attention 与 Linear-Attention 交替）、MTP 多 Token 预测分支、以及高性能 MoE 专家路由与共享专家机制。

## 校准模态支持

Qwen3.5 适配器按 **VLM（文本 + 可选图像）** 路径处理校准数据（`multimodal_vlm_modelslim_v1`）。推荐使用 `index.jsonl`（见[校准数据路径配置（参考）](../../docs/zh/knowledge_base/ptq/vlm/usage_vision_transformer_quantization.md#dataset---校准数据路径配置)）。

| 有效模态组合 | index.jsonl 字段示意 | 是否支持 |
|-------------|----------------------|----------|
| 纯文本 | `{"text":"..."}` | ✅ |
| 文本 + 图像 | `{"text":"...","image":"xxx.jpg"}` | ✅ |
| 文本 + 音频 / 视频 | 含 `audio` / `video` | ❌ 当前适配器不消费音频/视频字段 |

约束说明：

- **每条样本必须包含非空 `text`**。
- **同一量化任务内样本须同质**：不可在同一校准集中混用纯文本与图文样本。
- 依赖：`transformers==5.2.0`（见 `config.ini` `[ModelAdapterDependencies] qwen3_5_moe`）。纯文本校准的 chat template `content` 须为 list-of-parts 形式（适配器已处理）。
- **visit（data-free）** 仍会遍历 visual + decoder；**forward（calibration）** 在无 `pixel_values` 时跳过 visual，仅走文本 decoder。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见《[msModelSlim 安装指南](../../docs/zh/install_guide/install_guide.md)》。

- transformers 版本需要配置安装 5.2.0 版本。
  - pip install transformers==5.2.0

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接                                                 | W8A8 | W8A16 | W4A8 | W4A16 | W4A4  | 稀疏量化 | KV Cache | Attention | 量化命令                                          |
|---------|---------|---------------------------------------------------------------|-----|-----|-----|--------|------|---------|----------|-----------|-----------------------------------------------|
| **Qwen3.5-MoE** | Qwen3.5-397B-A17B | [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)   | ✅ |  | ✅ |      |   |  |   |   | [W8A8](#Qwen3.5-397B-A17B-w8a8)/[W4A8](#Qwen3.5-397B-A17B-w4a8)|
| **Qwen3.5-MoE** | Qwen3.5-122B-A10B | [Qwen3.5-122B-A10B](https://modelscope.cn/models/Qwen/Qwen3.5-122B-A10B)   | ✅ |  |    |        |   |  |   |   | [W8A8](#Qwen3.5-122B-A10B-w8a8)|
| **Qwen3.5-MoE** | Qwen3.5-35B-A3B | [Qwen3.5-35B-A3B](https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B)   | ✅ |  |    |        |   |  |   |   | [W8A8](#Qwen3.5-35B-A3B-w8a8)|
| **Qwen3.5-Dense** | Qwen3.5-27B | [Qwen3.5-27B](https://modelscope.cn/models/Qwen/Qwen3.5-27B)   | ✅ |  |    |        |   |  ✅|   |   | [W8A8](#Qwen3.5-27B-w8a8)/[W8A8SC](#Qwen3.5-27B-w8a8s)|
| **Qwen3.5-Dense** | Qwen3.5-4B | [Qwen3.5-4B](https://modelscope.cn/models/Qwen/Qwen3.5-4B)   | ✅ |  |    |        |   |  |   |   | [W8A8](#Qwen3.5-4B-w8a8)|

**说明：**

- ✅ 表示该量化策略已通过 msModelSlim 官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过 msModelSlim 官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令。

## 量化权重生成

### 使用示例

- 请将{MODEL_PATH}替换为用户实际浮点权重路径，{SAVE_PATH}替换为量化权重保存路径。

#### 1. Qwen3.5-397B-A17B

##### <span id="Qwen3.5-397B-A17B-w8a8">Qwen3.5-397B-A17B W8A8量化</span>

该模型的量化已集成至《[一键量化](../../docs/zh/user_guide/usage_one_click_quantization.md)》。

  ```shell
  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-397B-A17B --quant_type w8a8 --trust_remote_code true
  ```

##### <span id="Qwen3.5-397B-A17B-w4a8">Qwen3.5-397B-A17B W4A8量化</span>

该模型的量化已集成至《[一键量化](../../docs/zh/user_guide/usage_one_click_quantization.md)》。

  ```shell
  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-397B-A17B --quant_type w4a8 --trust_remote_code true
  ```

#### 2. Qwen3.5-122B-A10B

##### <span id="Qwen3.5-122B-A10B-w8a8">Qwen3.5-122B-A10B W8A8量化</span>

该模型的量化已集成至《[一键量化](../../docs/zh/user_guide/usage_one_click_quantization.md)》。

  ```shell
  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-122B-A10B --quant_type w8a8 --trust_remote_code true
  ```

#### 3. Qwen3.5-35B-A3B

##### <span id="Qwen3.5-35B-A3B-w8a8">Qwen3.5-35B-A3B W8A8量化</span>

该模型的量化已集成至《[一键量化](../../docs/zh/user_guide/usage_one_click_quantization.md)》。

  ```shell
  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-35B-A3B --quant_type w8a8 --trust_remote_code true
  ```

#### 4. Qwen3.5-27B

##### <span id="Qwen3.5-27B-w8a8">Qwen3.5-27B W8A8量化</span>

该模型的量化已集成至《[一键量化](../../docs/zh/user_guide/usage_one_click_quantization.md)》。

  ```shell
  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-27B --quant_type w8a8 --trust_remote_code true
  ```

##### <span id="Qwen3.5-27B-w8a8s">Qwen3.5-27B W8A8SC（W8A8S → 压缩）</span>

用于 Atlas 推理产品（如 **300I DUO**）部署前的稀疏量化链路：先生成 `W8A8S` 中间权重，再压缩为对外交付的 **`W8A8SC`**。

参考配置：[`lab_practice/qwen3_5_moe/qwen3_5_27b_w8a8s.yaml`](../../lab_practice/qwen3_5_moe/qwen3_5_27b_w8a8s.yaml)。校准图放在 `lab_calib/calibImages`（或 `example/multimodal_vlm/calibImages`），建议不少于 30 张。视觉塔 Linear 通过 YAML `mix_cfg` 保持浮点（OCR/TextVQA）。

**步骤 1：一键生成 W8A8S（推荐）**

  ```shell
  # 确保 lab_calib/calibImages 下有校准图（短名 calibImages）
  msmodelslim quant \
    --model_path ${MODEL_PATH} \
    --save_path ${SAVE_PATH_W8A8S} \
    --device npu \
    --model_type Qwen3.5-27B \
    --quant_type w8a8s \
    --trust_remote_code true
  ```

生成结果中应包含 `quant_model_description.json`，且 `model_quant_type` 为 `W8A8S`。浮点层策略仅在实践配置 `mix_cfg` 中声明，勿另开脚本维护名单。

> 说明：W8A8S 已接入 `msmodelslim quant --quant_type w8a8s`（`modelslim_v0` + `Calibrator`/`is_lowbit`）。对外交付形态为 **W8A8SC**；压缩与推理依赖 vLLM-Ascend 对应 scheme。

#### 5. Qwen3.5-4B

##### <span id="Qwen3.5-4B-w8a8">Qwen3.5-4B W8A8量化</span>

该模型的量化已集成至《[一键量化](../../docs/zh/user_guide/usage_one_click_quantization.md)》。

  ```shell
  msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-4B --quant_type w8a8 --trust_remote_code true
  ```
