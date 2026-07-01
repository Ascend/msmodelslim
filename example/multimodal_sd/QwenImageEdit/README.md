# Qwen Image Edit 量化使用说明

## Qwen Image Edit 模型介绍

[Qwen-Image-Edit](https://github.com/QwenLM/Qwen-Image) 是阿里巴巴通义千问团队基于 Qwen-Image 图像基础模型推出的开源图像编辑模型，兼顾语义级改动（如风格、构图、物体增删与替换）与外观级细节控制，支持中英文画面内文字的精准修改。当前 msModelSlim 一键量化面向 **Qwen-Image-Edit-2509** 权重与 [MindIE/Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509) 推理工程对接。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/getting_started/install_guide.md)。
- 浮点推理环境与依赖请参考[魔乐 Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509) 及 [README](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509/blob/main/README.md)，确保在量化前可正常完成浮点推理（需从推理工程仓加载 `qwenimage_edit` 等模块）。

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | 模型仓库链接 | W8A8 | W8A16 | W4A16 | W4A4 | 时间步量化 | FA3量化 | 异常值抑制量化 | 量化命令 |
|---------|---------|-------------|-----|-------|-------|------|-----------|---------|-------------|----------|
| **Qwen-Image-Edit** | Qwen-Image-Edit-2509 | [Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509) | ✅ |   |   |   |   | ✅ |   | [FA3+W8A8动态量化](#qwen-image-edit-2509-fa3w8a8动态量化) |

**说明：**

- ✅ 表示该量化策略已通过 msModelSlim 官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过 msModelSlim 官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令。

## Qwen Image Edit 量化支持

Qwen-Image-Edit-2509 的 Transformer 部分基于扩散与 Transformer 结构，msModelSlim 支持对其线性层等进行量化，并配合 **online_quarot** 与 **FA3** 等流程；支持逐层量化，有利于降低量化过程中的内存占用。

### 量化特性

- **逐层量化**: 支持逐层处理，大幅降低内存占用
- **单卡量化**: 结合逐层量化特性，可实现在Atlas 800I/800T A2(64G)设备上的单卡量化

## 量化命令

### <span id="qwen-image-edit-2509-fa3w8a8动态量化">Qwen-Image-Edit-2509 FA3+W8A8 动态量化</span>

#### 使用 config_path 参数指定配置文件进行一键量化

```bash
msmodelslim quant \
    --model_path /path/to/Qwen-Image-Edit-2509 \
    --save_path /path/to/qwen_image_edit_quantized_weights \
    --device npu \
    --model_type Qwen-Image-Edit-2509 \
    --config_path lab_practice/qwen_image_edit/qwen-image-edit-w8a8f8-mxfp.yaml \
    --trust_remote_code True
```

### 一键量化命令参数说明

一键量化参数基本说明可参考：[一键量化参数说明](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/feature_guide/quick_quantization_v1/usage.md#%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E)。

针对 Qwen-Image-Edit-2509，补充说明如下：

| 参数名称 | 解释 | 是否可选 | 范围 |
|----------|------|----------|------|
| model_path | 浮点权重目录（需包含 `transformer` 子目录及 Pipeline 所需文件，与 MindIE 推理仓一致） | 必选 | 类型：Str |
| save_path | 量化权重保存路径 | 必选 | 类型：Str |
| device | 量化设备 | 必选 | 1. 类型：Str <br>2. 仅支持 `npu` |
| model_type | 模型名称 | 必选 | 1. 类型：Str <br>2. 大小写敏感，需配置为 `qwen_image_edit` |
| config_path | 指定配置路径 | 与 `quant_type` 二选一 | 1. 类型：Str <br>2. 配置文件格式为 yaml <br>3. 建议优先使用最佳实践库中已验证配置 [qwen-image-edit-w8a8f8-mxfp.yaml](../../../lab_practice/qwen_image_edit/qwen-image-edit-w8a8f8-mxfp.yaml)；若自定义配置，msModelSlim 不为量化结果负责 |
| quant_type | 量化类型 | 与 `config_path` 二选一 | 1. 类型：Str <br>2. 当前Qwen-Image-Edit模型仅支持config_path |
| trust_remote_code | 是否信任自定义代码 | 可选 | 类型：Bool，默认 False；加载自定义代码时建议设为 `True`（请确保代码来源可靠） |

## 配置文件说明

### 基础配置结构

以下结构与仓库内 [qwen-image-edit-w8a8f8-mxfp.yaml](../../../lab_practice/qwen_image_edit/qwen-image-edit-w8a8f8-mxfp.yaml) 一致，便于理解各段含义：

```yaml
apiversion: multimodal_sd_modelslim_v1

metadata:
  config_id: qwen-image-edit-mxw8a8
  verified_model_types:
    - Qwen-Image-Edit-2509

default_w8a8_dynamic: &default_w8a8_dynamic
  act:
    scope: "per_block"
    dtype: "mxfp8"
    symmetric: True
    method: "minmax"
    ext:
      axes: -1
  weight:
    scope: "per_block"
    dtype: "mxfp8"
    symmetric: True
    method: "minmax"
    ext:
      axes: -1

spec:
  process:
    - type: "linear_quant"
      qconfig: *default_w8a8_dynamic
      exclude: ['*txt_mlp.net.2*', '*img_mod.1*', '*txt_mod.1*']
    - type: "online_quarot"
      include:
        - "*"
    - type: "fa3_quant"
      qconfig:
        dtype: "fp8_e4m3"
        scope: "per_token"
        symmetric: True
        method: "minmax"
      include:
        - "*"
  save:
    - type: "mindie_format_saver"
      part_file_size: 0

  multimodal_sd_config:
    dump_config:
      enable_dump: False
    model_config:
      img_paths: ""
      prompt_file: ""
```

### 关键配置参数

#### 量化配置 (process)

- **linear_quant**：对线性层等进行 W8A8（mxfp8）动态量化；`exclude` 中模式用于排除部分子模块，以稳定精度。
- **online_quarot**：在线旋转相关配置，与注意力等模块配合。
- **fa3_quant**：Flash Attention 3 路径上的 FP8 量化配置（如 `fp8_e4m3`、`per_token`）。

#### 保存配置 (save)

- **type**：保存器类型，使用 `mindie_format_saver`。
- **part_file_size**：分片大小，`0` 表示不分片。

#### 多模态配置 (multimodal_sd_config)

- **dump_config**：校准数据导出相关；当前默认示例中 `enable_dump` 为 `False`。若后续开启校准 dump，需按一键量化协议与适配器约定配置。
- **model_config**：可与推理参数对齐的占位字段，例如：
  - `img_paths`：输入图像路径（多图可用逗号分隔等约定，以推理仓为准）。
  - `prompt_file`：提示词文件路径。

更细的协议说明见：[一键量化配置协议说明](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/feature_guide/quick_quantization_v1/usage.md#%E9%87%8F%E5%8C%96%E9%85%8D%E7%BD%AE%E5%8D%8F%E8%AE%AE%E8%AF%A6%E8%A7%A3)。

## FAQ

**现象：量化时报错无法导入 `qwenimage_edit`。**  
**解决方案：** 请按 [MindIE/Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509) 将推理工程置于 Python 路径或按说明安装，使 `qwenimage_edit.transformer_qwenimage`、`qwenimage_edit.pipeline_qwenimage_edit_plus` 可被正常导入。

**现象：如何自定义量化配置？**  
**解决方案：** 可在 `process` 中调整 `exclude`/`include`、量化 dtype 与范围等；自定义配置需自行验证精度与兼容性，官方仅对最佳实践库中已验证配置提供保证。

## 附录

### 相关资源

- [Qwen-Image-Edit-2509（Hugging Face）](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- [Qwen-Image-Edit-2509模型仓库](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509)
- [一键量化配置协议说明](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/feature_guide/quick_quantization_v1/usage.md#%E9%87%8F%E5%8C%96%E9%85%8D%E7%BD%AE%E5%8D%8F%E8%AE%AE%E8%AF%A6%E8%A7%A3)
- [逐层量化特性说明](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/zh/feature_guide/quick_quantization_v1/usage.md#%E9%80%90%E5%B1%82%E9%87%8F%E5%8C%96%E5%8F%8A%E5%88%86%E5%B8%83%E5%BC%8F%E9%80%90%E5%B1%82%E9%87%8F%E5%8C%96)
