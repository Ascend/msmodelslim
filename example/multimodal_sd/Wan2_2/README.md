# Wan2.2 量化使用说明

## Wan2.2 模型介绍

Wan2.2 是阿里巴巴在 Wan 系列上的新一代开源视频基础模型，面向更高质量、更可控的影视级视频生成；在 Wan2.1 的基础上进一步扩充训练数据与能力，并引入面向视频扩散的混合专家（MoE）等设计，在保持开放生态的同时提升生成效率与观感。支持文本到视频（T2V）、图像到视频（I2V）以及文本+图像到视频（TI2V） 等多种模式。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/getting_started/install_guide.md)。
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

Wan2.2模型基于Transformer架构，msmodelslim支持对其中Transformer部分进行量化，并支持逐层量化，能够显著降低量化过程中的内存占用。

### 量化特性

- **逐层量化**: 支持逐层处理，大幅降低内存占用
- **单卡量化**: 结合逐层量化特性，可实现在Atlas 800I/800T A2(64G)设备上的单卡量化

## 量化命令

### <span id="wan22-t2v-fa3w8a8动态量化">Wan2.2-T2V-A14B FA3+W8A8动态量化</span>

#### 使用config_path参数指定配置文件进行一键量化

```bash
msmodelslim quant \
    --model_path /path/to/wan2_2_t2v_float_weights \
    --save_path /path/to/wan2_2_t2v_quantized_weights \
    --device npu \
    --model_type Wan2_2 \
    --config_path /lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_t2v.yaml \
    --trust_remote_code True
```

### <span id="wan22-i2v-fa3w8a8动态量化">Wan2.2-I2V-A14B FA3+W8A8动态量化</span>

#### 使用config_path参数指定配置文件进行一键量化

```bash
msmodelslim quant \
    --model_path /path/to/wan2_2_i2v_float_weights \
    --save_path /path/to/wan2_2_i2v_quantized_weights \
    --device npu \
    --model_type Wan2_2 \
    --config_path /lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_i2v.yaml \
    --trust_remote_code True
```

### <span id="wan22-ti2v-fa3w8a8动态量化">Wan2.2-TI2V-5B FA3+W8A8动态量化</span>

#### 使用config_path参数指定配置文件进行一键量化

```bash
msmodelslim quant \
    --model_path /path/to/wan2_2_ti2v_float_weights \
    --save_path /path/to/wan2_2_ti2v_quantized_weights \
    --device npu \
    --model_type Wan2_2 \
    --config_path /lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_ti2v.yaml \
    --trust_remote_code True
```

### 一键量化命令参数说明

一键量化参数基本说明可参考：[一键量化参数说明](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/feature_guide/quick_quantization_v1/usage.md#参数说明)

针对Wan2.2模型，有不同的限制：

|参数名称|解释|是否可选| 范围                                                                                    |
|--------|--------|--------|---------------------------------------------------------------------------------------|
|model_path|Wan2.2浮点权重目录|必选| 类型：Str                                                                                |
|save_path|Wan2.2量化权重保存路径|必选| 类型：Str                                                                                |
|device|量化设备|必选| 1. 类型：Str <br>2. 仅支持"npu"                                     |
|model_type|模型名称|必选| 1. 类型：Str <br>2. 大小写敏感，需要配置为"Wan2_2"                                         |
|config_path|指定配置路径|与"quant_type"二选一| 1. 类型：Str <br>2. 配置文件格式为yaml <br>3. 当前只支持最佳实践库中已验证的配置[wan2_2_w8a8f8_mxfp_t2v.yaml](../../../lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_t2v.yaml)、[wan2_2_w8a8f8_mxfp_i2v.yaml](../../../lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_i2v.yaml)、[wan2_2_w8a8f8_mxfp_ti2v.yaml](../../../lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_ti2v.yaml)，若自定义配置，msmodelslim不为量化结果负责 <br> |
|quant_type|量化类型|与"config_path"二选一| 1. 类型：Str <br>2. 当前Wan2.2模型仅支持config_path
|trust_remote_code|是否信任自定义代码|可选| 1. 类型：Bool，默认值：False <br>2. 指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载(请确保所加载的自定义代码文件来源可靠，避免潜在的安全风险)。                           |

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
          method: "minmax"
      include:
        - "*"
    - type: "online_quarot"
      include: 
        - "*.self_attn.*"
    - type: "fa3_quant"
      qconfig:
        dtype: "fp8_e4m3"
        scope: "per_token"
        symmetric: True
        method: "minmax"
      include:
        - "*self_attn"
  save:
    - type: "mindie_format_saver"
      part_file_size: 0

  # 基础配置
  multimodal_sd_config:
    dump_config:
      capture_mode: "args"
      dump_data_dir: ""  # default is save_path
    model_config:
      prompt: "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
      # 模型加载参数
      convert_model_dtype: True
      task: "t2v-A14B"
```

### 关键配置参数

#### 量化配置 (process)

- **type**: 处理器类型，固定为"linear_quant"
- **qconfig.act**: 激活值量化配置
  - `scope`: 量化范围，mxfp8配合使用"per_block"
  - `dtype`: 数据类型
  - `symmetric`: 是否对称量化，推荐True
  - `method`: 量化方法，推荐"minmax"
- **qconfig.weight**: 权重量化配置
  - `scope`: 量化范围，mxfp8配合使用"per_block"
  - `dtype`: 数据类型
  - `symmetric`: 是否对称量化，推荐True
  - `method`: 量化方法，推荐"minmax"

#### 保存配置 (save)

- **type**: 保存器类型，使用"mindie_format_saver"
- **part_file_size**: 分片文件大小，0表示不分片

#### 多模态配置 (multimodal_sd_config)

- **dump_config**: 校准数据捕获配置
  - `capture_mode`: 捕获模式，当前仅支持配置为"args"
  - `dump_data_dir`: 校准数据保存目录，配置为空字符串时会自动转换为使用量化权重保存路径
- **model_config**: 模型加载与推理配置，具体可配置的参数需要参考原始推理工程仓[Wan2.2模型仓库](https://modelers.cn/models/MindIE/Wan2.2)

  | 字段名 | 作用 | 说明 | 可选值 |
  |--------|------|------|--------|
  | prompt | 校准提示词 | 用于生成校准数据的文本描述 | 字符串 |
  | offload_model | 模型卸载 | 是否在推理后卸载模型到CPU，值为True时开启 | True/False |
  | frame_num | 生成帧数 | 视频生成的帧数 | 大于0的整数 |
  | task | 任务类型 | 指定模型任务类型，"t2v-A14B"表示文本生成视频任务、"i2v-A14B"表示图像生成视频任务，"ti2v-5B"表示文本+图像生成视频任务 | "t2v-A14B", "i2v-A14B", "ti2v-5B" |
  | size | 生成尺寸 | 视频或图像的尺寸规格 | 参考[Wan2.2模型仓库](https://modelers.cn/models/MindIE/Wan2.2)配置|
  | sample_steps | 采样步数 | 扩散模型的采样步数 | 大于0的整数 |

## FAQ

现象：如何自定义量化配置？  
解决方案：可以修改配置文件中的process部分，调整量化参数和层选择策略。

## 附录

### 相关资源

- [Wan2.2模型仓库](https://modelers.cn/models/MindIE/Wan2.2)
- [一键量化配置协议说明](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/feature_guide/quick_quantization_v1/usage.md#%E9%87%8F%E5%8C%96%E9%85%8D%E7%BD%AE%E5%8D%8F%E8%AE%AE%E8%AF%A6%E8%A7%A3)
- [逐层量化特性说明](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/feature_guide/quick_quantization_v1/usage.md#%E9%80%90%E5%B1%82%E9%87%8F%E5%8C%96%E5%8F%8A%E5%88%86%E5%B8%83%E5%BC%8F%E9%80%90%E5%B1%82%E9%87%8F%E5%8C%96)
