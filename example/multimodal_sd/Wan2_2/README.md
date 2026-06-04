# Wan2.2 量化使用说明

## Wan2.2 模型介绍

Wan2.2 是阿里巴巴在 Wan 系列上的新一代开源视频基础模型，面向更高质量、更可控的影视级视频生成；在 Wan2.1 的基础上进一步扩充训练数据与能力，并引入面向视频扩散的混合专家（MoE）等设计，在保持开放生态的同时提升生成效率与观感。支持文本到视频（T2V）、图像到视频（I2V）以及文本+图像到视频（TI2V） 等多种模式。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/getting_started/install_guide/)。
- 环境安装参考魔乐社区[Wan2.2](https://modelers.cn/models/MindIE/Wan2.2)

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | 模型仓库链接 | W8A8 | W8A16 | W4A16 | W4A4 | 时间步量化 | FA3量化 | 异常值抑制量化 | 量化命令 |
|---------|---------|-------------|-----|-------|-------|------|-----------|---------|-------------|----------|
| **Wan2.2** | Wan2.2-T2V-A14B | [Wan2.2-T2V-A14B](https://modelers.cn/models/MindIE/Wan2.2) | ✅ |   |   |   |   | ✅ |   | [FA3+W8A8动态量化](#wan22-t2v-fa3w8a8动态量化) |
| | Wan2.2-I2V-A14B | [Wan2.2-I2V-A14B](https://modelers.cn/models/MindIE/Wan2.2) | ✅ |   |   |   |   | ✅  |  | [FA3+W8A8动态量化](#wan22-i2v-fa3w8a8动态量化) |
| | Wan2.2-TI2V-5B | [Wan2.2-TI2V-5B](https://modelers.cn/models/MindIE/Wan2.2) | ✅ |   |   |   |   | ✅  |  | [FA3+W8A8动态量化](#wan22-ti2v-fa3w8a8动态量化) |

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令
- 注意执行量化需要在模型文件路径下

## Wan2.2 量化支持

Wan2.2 的 DiT 采用 **双专家**（低噪声 / 高噪声）结构，msModelSlim 对两个专家分别进行逐层量化，输出目录下通常包含 `low_noise_model/`、`high_noise_model/` 子目录。

### 量化特性

- **逐层量化**: 支持逐层处理，大幅降低内存占用
- **单卡量化**: 结合逐层量化特性，可实现在Atlas 800I/800T A2(64G)设备上的单卡量化

## 量化命令

当前适配 Wan2.2 FA3+W8A8（MXFP8）动态量化，请通过如下命令切换魔乐社区 MindIE Wan2.2 推理仓版本：

```bash
git checkout 521cee68abc4d1b8bde30b6a26855e34f23a0073
```

### <span id="wan22-t2v-fa3w8a8动态量化">Wan2.2-T2V-A14B FA3+W8A8动态量化</span>

#### 使用quant_type参数进行一键量化

W8A8(MXFP8)+FA3(FP8动态)

```bash
msmodelslim quant \
    --model_path /path/to/wan2_2_t2v_float_weights \
    --save_path /path/to/wan2_2_t2v_quantized_weights \
    --device npu \
    --model_type Wan2.2-T2V-A14B \
    --quant_type w8a8f8 \
    --trust_remote_code True
```

### <span id="wan22-i2v-fa3w8a8动态量化">Wan2.2-I2V-A14B FA3+W8A8动态量化</span>

#### 使用quant_type参数进行一键量化

W8A8(MXFP8)+FA3(FP8动态)

```bash
msmodelslim quant \
    --model_path /path/to/wan2_2_i2v_float_weights \
    --save_path /path/to/wan2_2_i2v_quantized_weights \
    --device npu \
    --model_type Wan2.2-I2V-A14B \
    --quant_type w8a8f8 \
    --trust_remote_code True
```

### <span id="wan22-ti2v-fa3w8a8动态量化">Wan2.2-TI2V-5B FA3+W8A8动态量化</span>

#### 使用quant_type参数进行一键量化

W8A8(MXFP8)+FA3(FP8动态)

```bash
msmodelslim quant \
    --model_path /path/to/wan2_2_ti2v_float_weights \
    --save_path /path/to/wan2_2_ti2v_quantized_weights \
    --device npu \
    --model_type Wan2.2-TI2V-5B \
    --quant_type w8a8f8 \
    --trust_remote_code True
```

## 配置文件说明

### 基础配置结构

```yaml
apiversion: multimodal_sd_modelslim_v1

spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: True
          method: "minmax"
        weight:
          scope: "per_block"
          dtype: "mxfp8"
          symmetric: True
          method: "mse_round"
      include:
        - "*"
    - type: "online_quarot"
      include:
        - "*.self_attn.*"
      exclude:
        - "*blocks.0.self_attn*"
    - type: "fa3_quant"
      qconfig:
        dtype: "fp8_e4m3"
        scope: "per_token"
        symmetric: True
        method: "minmax"
      include:
        - "*self_attn"
      exclude:
        - "*blocks.0.self_attn*"

  dataset: wan2_2_t2v   # I2V: wan2_2_i2v；TI2V: wan2_2_ti2v

  save:
    - type: "mindie_format_saver"
      part_file_size: 0

  multimodal_sd_config:
    dump_config:
      enable_dump: False    # 全动态量化示例；静态/离群值抑制请改为 True
      capture_mode: "args"
      dump_data_dir: ""     # 空则使用 save_path；pth 见下文命名规则
    inference_config:       # 推荐；勿与已废弃的 model_config 同时配置
      size: "1280*720"
      frame_num: 81
      sample_steps: 40
      convert_model_dtype: True
      task: "t2v-A14B"      # 须与 --model_type 场景一致
```

### 关键配置参数

#### 量化配置 (process)

- **linear_quant**：DiT 线性层 W8A8（MXFP8 per-block）。
- **online_quarot**：注意力 Q/K 在线旋转；示例中排除首层 `blocks.0`。
- **fa3_quant**：注意力 FA3 动态 FP8 量化。

#### 校准数据集 (dataset)

- **作用**：指定 `index.json` / `index.jsonl` 或目录路径；短名称在 [`lab_calib`](../../../lab_calib) 下解析。
- **T2V**：每条须含非空 `text`，**不得**含 `image`。
- **I2V**：须含 `text` 与可访问的 `image`。
- **TI2V**：须含 `text`；`image` 可选。

#### 多模态配置 (multimodal_sd_config)

- **dump_config**
  - `enable_dump`：是否 load/dump 校准 pth；纯动态量化可设 `False`（仍须为每个专家保留 `calib_data` 的 key）。
  - `capture_mode`：当前仅支持 `"args"`。
  - `dump_data_dir`：pth 根目录；为空时使用 `--save_path`。
  - **pth 命名**（双专家）：`calib_data_<task>_low_noise_model.pth`、`calib_data_<task>_high_noise_model.pth`（例如 `calib_data_t2v-A14B_low_noise_model.pth`）。目录内文件齐全则加载，任一缺失则触发浮点推理 dump。
- **inference_config**（推荐）：推理参数，字段须与原 Wan2.2 推理仓 CLI 一致，由适配器 Pydantic 校验后桥接到 `model_args`。合法字段以各场景 `*InferenceConfig` 为准。
- **model_config**（Legacy）：仅 `--model_type Wan2_2` / `Wan2.2` 单体入口使用，**将废弃**；与 `inference_config` 不可同配。

| inference_config 常见字段 | 作用 | 说明 |
|---------------------------|------|------|
| size | 生成尺寸 | 如 `"1280*720"`，须在原仓 `SUPPORTED_SIZES` 内 |
| frame_num | 帧数 | 正整数 |
| sample_steps | 扩散步数 | 正整数 |
| task | 任务标识 | 若填写须与当前 `model_type` 绑定（如 T2V 为 `t2v-A14B`） |
| convert_model_dtype | 权重 dtype 转换 | 布尔 |
| base_seed | 随机种子 | 可选 |

## FAQ

**如何自定义量化配置？**
修改 YAML 中 `spec.process` 的处理器链与 `include`/`exclude`；场景相关推理参数放在 `multimodal_sd_config.inference_config`。

**能否只量化 low_noise_model？**
不能。双专家须全部完成量化，且 `calib_data` 中须包含 `low_noise_model`、`high_noise_model` 两个 key。

**量化报错 calib data missing for expert？**
检查 `dump_data_dir` / `save_path` 下 pth 是否按专家命名齐全，或设 `enable_dump: True` 重新 dump。

## 附录

### 相关资源

- [Wan2.2模型仓库](https://modelers.cn/models/MindIE/Wan2.2)
- [《多模态生成模型接入指南（开发者）》](../../../docs/zh/developer_guide/integrating_multimodal_generation_model.md)
- [一键量化配置协议说明](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/feature_guide/quick_quantization_v1/usage/#%E9%87%8F%E5%8C%96%E9%85%8D%E7%BD%AE%E5%8D%8F%E8%AE%AE%E8%AF%A6%E8%A7%A3)
- [逐层量化特性说明](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/feature_guide/quick_quantization_v1/usage/#%E9%80%90%E5%B1%82%E9%87%8F%E5%8C%96%E5%8F%8A%E5%88%86%E5%B8%83%E5%BC%8F%E9%80%90%E5%B1%82%E9%87%8F%E5%8C%96)
