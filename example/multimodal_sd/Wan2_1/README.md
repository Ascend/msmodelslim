# Wan2.1 量化使用说明

## Wan2.1 模型介绍

Wan2.1是阿里巴巴发布的一套全面且开放的视频基础模型，它突破了视频生成的界限。支持多种生成任务：

- **文本到视频 (T2V)**: 根据文本描述生成视频
- **图像到视频 (I2V)**: 根据输入图像生成视频
- **文本到图像 (T2I)**: 根据文本描述生成图像

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/getting_started/install_guide.md)。

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | 模型仓库链接 | W8A8 | W8A16 | W4A16 | W4A4 | 时间步量化 | FA3量化 | 异常值抑制量化 | 量化命令 |
|---------|---------|-------------|-----|-------|-------|------|-----------|---------|-------------|----------|
| **Wan2.1** | Wan2.1-14B | [Wan2.1-14B](https://modelers.cn/models/MindIE/Wan2.1) | ✅ |   |   |   |   |   |   | [W8A8动态量化](#wan21-14b-w8a8动态量化) |
| | Wan2.1-1.3B | [Wan2.1-1.3B](https://modelers.cn/models/MindIE/Wan2.1) | ✅ |   |   |   |   |   |   | [W8A8动态量化](#wan21-13b-w8a8动态量化) |

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令

## Wan2.1 量化支持

Wan2.1模型基于Transformer架构，msmodelslim支持对其中Transformer部分进行量化，并支持逐层量化，能够显著降低量化过程中的内存占用。

### 已验证的量化类型

| 量化类型 | 描述 | 适用场景 | 配置示例 |
|----------|------|----------|----------|
| w8a8_dynamic | 权重8bit per-channel对称量化，激活值8bit动态量化 | 推荐使用，适应不同输入 | [wan2_1_w8a8_dynamic.yaml](../../../lab_practice/wan2_1/wan2_1_w8a8_dynamic.yaml) |

### 量化特性

- **逐层量化**: 支持逐层处理，大幅降低内存占用
- **单卡量化**: 结合逐层量化特性，可实现在Atlas 800I/800T A2(64G)设备上的单卡量化
- **动态激活值量化**: 激活值使用per_token量化范围，提高精度

## 量化命令

### <span id="wan21-14b-w8a8动态量化">Wan2.1-14B W8A8动态量化</span>

#### 方式一：使用quant_type参数进行一键量化

```bash
msmodelslim quant \
    --model_path /path/to/wan2_1_14b_float_weights \
    --save_path /path/to/wan2_1_14b_quantized_weights \
    --device npu \
    --model_type Wan2_1 \
    --quant_type w8a8 \
    --trust_remote_code True
```

#### 方式二：使用config_path参数指定配置文件进行一键量化

```bash
msmodelslim quant \
    --model_path /path/to/wan2_1_14b_float_weights \
    --save_path /path/to/wan2_1_14b_quantized_weights \
    --device npu \
    --model_type Wan2_1 \
    --config_path /path/to/wan2_1_w8a8_dynamic.yaml \
    --trust_remote_code True
```

### <span id="wan21-13b-w8a8动态量化">Wan2.1-1.3B W8A8动态量化</span>

#### 方式一：使用quant_type参数进行一键量化

```bash
msmodelslim quant \
    --model_path /path/to/wan2_1_float_weights \
    --save_path /path/to/wan2_1_quantized_weights \
    --device npu \
    --model_type Wan2_1 \
    --quant_type w8a8 \
    --trust_remote_code True
```

#### 方式二：使用config_path参数指定配置文件进行一键量化

```bash
msmodelslim quant \
    --model_path /path/to/wan2_1_float_weights \
    --save_path /path/to/wan2_1_quantized_weights \
    --device npu \
    --model_type Wan2_1 \
    --config_path /path/to/wan2_1_w8a8_dynamic.yaml \
    --trust_remote_code True
```

### 一键量化命令参数说明

一键量化参数基本说明可参考：[一键量化参数说明](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/feature_guide/quick_quantization_v1/usage.md#%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E)

针对Wan2.1模型，有不同的限制：

|参数名称|解释|是否可选| 范围                                                                                    |
|--------|--------|--------|---------------------------------------------------------------------------------------|
|model_path|Wan2.1浮点权重目录|必选| 类型：Str                                                                                |
|save_path|Wan2.1量化权重保存路径|必选| 类型：Str                                                                                |
|device|量化设备|必选| 1. 类型：Str <br>2. 仅支持"npu"                                     |
|model_type|模型名称|必选| 1. 类型：Str <br>2. 大小写敏感，需要配置为"Wan2_1"                                         |
|config_path|指定配置路径|与"quant_type"二选一| 1. 类型：Str <br>2. 配置文件格式为yaml <br>3. 当前只支持最佳实践库中已验证的配置[wan2_1_w8a8_dynamic.yaml](../../../lab_practice/wan2_1/wan2_1_w8a8_dynamic.yaml)，若自定义配置，msmodelslim不为量化结果负责 <br> |
|quant_type|量化类型|与"config_path"二选一| 1. 类型：Str <br>2. 当前仅支持配置为"w8a8"
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
          scope: "per_token"   # 激活值量化范围
          dtype: "int8"        # 激活值量化数据类型
          symmetric: True      # 是否对称量化
          method: "minmax"      # 量化方法
        weight:
          scope: "per_channel"   # 权重量化范围
          dtype: "int8"        # 权重量化数据类型
          symmetric: True       # 是否对称量化
          method: "minmax"      # 量化方法
      include: ["*"]           # 包含的层模式
      exclude: ["*ffn.2*"]     # 排除的层模式

  save:
    - type: "mindie_format_saver"
      part_file_size: 0

  multimodal_sd_config:
    dump_config:
      capture_mode: "args"
      dump_data_dir: ""
    model_config:
      prompt: "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
      offload_model: True
      frame_num: 121
```

### 关键配置参数

#### 量化配置 (process)

- **type**: 处理器类型，固定为"linear_quant"
- **qconfig.act**: 激活值量化配置
  - `scope`: 量化范围，推荐使用"per_token"
  - `dtype`: 数据类型，固定为"int8"
  - `symmetric`: 是否对称量化，推荐True，"per_token"量化下仅支持设置为True
  - `method`: 量化方法，推荐"minmax"
- **qconfig.weight**: 权重量化配置
  - `scope`: 量化范围，固定为"per_channel"
  - `dtype`: 数据类型，固定为"int8"
  - `symmetric`: 是否对称量化，推荐True
  - `method`: 量化方法，推荐"minmax"

#### 保存配置 (save)

- **type**: 保存器类型，使用"mindie_format_saver"
- **part_file_size**: 分片文件大小，0表示不分片

#### 多模态配置 (multimodal_sd_config)

- **dump_config**: 校准数据捕获配置
  - `capture_mode`: 捕获模式，当前仅支持配置为"args"
  - `dump_data_dir`: 校准数据保存目录，配置为空字符串时会自动转换为使用量化权重保存路径
- **model_config**: 模型加载与推理配置，具体可配置的参数需要参考原始推理工程仓[Wan2.1模型仓库](https://modelers.cn/models/MindIE/Wan2.1)

  | 字段名 | 作用 | 说明 | 可选值 |
  |--------|------|------|--------|
  | prompt | 校准提示词 | 用于生成校准数据的文本描述 | 字符串 |
  | offload_model | 模型卸载 | 是否在推理后卸载模型到CPU，值为True时开启 | True/False |
  | frame_num | 生成帧数 | 视频生成的帧数 | 大于0的整数 |
  | task | 任务类型 | 指定模型任务类型，"t2v-14B"表示14B模型的文本生成视频任务、"t2v-1.3B"表示1.3B模型的文本生成视频任务 | "t2v-14B", "t2v-1.3B" |
  | size | 生成尺寸 | 视频或图像的尺寸规格 | "1280\*720", "832\*480" |
  | sample_steps | 采样步数 | 扩散模型的采样步数 | 大于0的整数 |

## FAQ

问题：是否支持w8a8静态量化？  
解决方案: 可以修改配置文件中的process部分，调整qconfig.act.scope为"per_tensor"启动静态量化，但精度损失严重，不推荐。

现象：如何自定义量化配置？  
解决方案：可以修改配置文件中的process部分，调整量化参数和层选择策略。

## 附录

### 相关资源

- [Wan2.1模型仓库](https://modelers.cn/models/MindIE/Wan2.1)
- [一键量化配置协议说明](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/feature_guide/quick_quantization_v1/usage.md#%E9%87%8F%E5%8C%96%E9%85%8D%E7%BD%AE%E5%8D%8F%E8%AE%AE%E8%AF%A6%E8%A7%A3)
- [逐层量化特性说明](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/feature_guide/quick_quantization_v1/usage.md#%E9%80%90%E5%B1%82%E9%87%8F%E5%8C%96%E5%8F%8A%E5%88%86%E5%B8%83%E5%BC%8F%E9%80%90%E5%B1%82%E9%87%8F%E5%8C%96)
