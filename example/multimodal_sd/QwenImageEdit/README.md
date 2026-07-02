# Qwen Image Edit 量化使用说明

## Qwen Image Edit 模型介绍

[Qwen-Image-Edit](https://github.com/QwenLM/Qwen-Image) 是阿里巴巴通义千问团队基于 Qwen-Image 图像基础模型推出的开源图像编辑模型，兼顾语义级改动（如风格、构图、物体增删与替换）与外观级细节控制，支持中英文画面内文字的精准修改。当前 msModelSlim 一键量化面向 **Qwen-Image-Edit-2509** 权重与 [MindIE/Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509) 推理工程对接。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/install_guide/install_guide/)。
- 浮点推理环境与依赖请参考[魔乐 Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509) 及 [README](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509/blob/main/README.md)，确保在量化前可正常完成浮点推理（需从推理工程仓加载 `qwenimage_edit` 等模块）。

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | 模型仓库链接 | W8A8 | W8A16 | W4A16 | W4A4 | 时间步量化 | FA3量化 | 异常值抑制量化 | 量化命令 |
|---------|---------|-------------|-----|-------|-------|------|-----------|---------|-------------|----------|
| **Qwen-Image-Edit** | Qwen-Image-Edit-2509 | [Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509) | ✅ |   |   | ✅ |   | ✅ |   | [W8A8F8动态量化](#qwen-image-edit-2509-w8a8f8动态量化) / [W4A4F4动态量化](#qwen-image-edit-2509-w4a4f4动态量化) |

**说明：**

- ✅ 表示该量化策略已通过 msModelSlim 官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过 msModelSlim 官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令。
- 推荐使用MindIE-SD [3.0.0版本](https://gitcode.com/Ascend/MindIE-SD)

## Qwen Image Edit 量化支持

Qwen-Image-Edit-2509 的 Transformer 部分基于扩散与 Transformer 结构，msModelSlim 支持对其线性层等进行量化，并配合 **online_quarot** 与 **FA3** 等流程；当前适配 **W8A8（MXFP8）F8（FP8）** 与 **W4A4（MXFP4 DualScale）F4（MXFP4）** 两种一键量化方案，支持逐层量化，有利于降低量化过程中的内存占用。

### 量化特性

- **逐层量化**: 支持逐层处理，大幅降低内存占用
- **单卡量化**: 结合逐层量化特性，可实现在Atlas 800I/800T A2(64G)设备上的单卡量化

## 量化命令

### <span id="qwen-image-edit-2509-w8a8f8动态量化">Qwen-Image-Edit-2509 W8A8F8 动态量化</span>

#### 使用 quant_type 参数进行一键量化

W8A8(MXFP8)+FA3(FP8动态)

```bash
msmodelslim quant \
    --model_path /path/to/Qwen-Image-Edit-2509 \
    --save_path /path/to/qwen_image_edit_quantized_weights \
    --device npu \
    --model_type Qwen-Image-Edit-2509 \
    --quant_type w8a8f8 \
    --trust_remote_code True
```

### <span id="qwen-image-edit-2509-w4a4f4动态量化">Qwen-Image-Edit-2509 W4A4F4 动态量化</span>

#### 使用 quant_type 参数进行一键量化

W4A4(MXFP4 DualScale)+FA3(MXFP4)

```bash
msmodelslim quant \
    --model_path /path/to/Qwen-Image-Edit-2509 \
    --save_path /path/to/qwen_image_edit_quantized_weights \
    --device npu \
    --model_type Qwen-Image-Edit-2509 \
    --quant_type w4a4f4 \
    --trust_remote_code True
```

## 配置文件说明

### W8A8F8 基础配置结构

可以参考仓库内 [qwen-image-edit-w8a8f8-mxfp.yaml](../../../lab_practice/qwen_image_edit/qwen-image-edit-w8a8f8-mxfp.yaml)。

### W4A4F4 基础配置结构

可以参考仓库内 [qwen-image-edit-w4a4f4-mxfp.yaml](../../../lab_practice/qwen_image_edit/qwen-image-edit-w4a4f4-mxfp.yaml)。

与 W8A8 配置的主要差异：

- **linear_quant**：采用 **DualScale** 双尺度量化（`scope: dual_scale`、`method: dualscale`），权重与激活均为 `mxfp4`，`dual_block_size` 为 512；详见 [DualScale 量化方案说明](../../../docs/zh/user_guide/quantization_algorithms/quantization_algorithms/dual_scale.md)。
- **fa3_quant**：FA3 路径同样使用 `mxfp4`（`per_block` + `minmax`），与 W8A8 方案中的 `fp8_e4m3` 不同。

### 关键配置参数

#### 元数据 (metadata)

- **config_id**：配置标识，与 YAML 文件名对应（如 `qwen-image-edit-w8a8f8-mxfp`、`qwen-image-edit-w4a4f4-mxfp`）。
- **score**：官方验证评分，数值越高表示该配置在验证场景下表现越稳定。
- **verified_tags**：已验证的模型类型及对接环境标签；当前 `Qwen-Image-Edit-2509` 对应 MindIE-SD 推理与 Atlas_350 设备。
- **label**：量化能力标签，便于检索与对照：
  - `w_bit` / `a_bit`：权重与激活位宽（W8A8 为 8，W4A4 为 4）。
  - `is_sparse`：是否稀疏量化（当前为 `False`）。
  - `fa_quant`：是否启用 FA 量化（当前为 `True`，与下方 `fa3_quant` 流程一致）。

#### 默认 W8A8 动态量化锚点 (default_w8a8_dynamic)

- **act** / **weight**：线性层 W8A8 动态量化，`per_block` + `mxfp8` + `minmax`，供 `linear_quant` 通过 YAML 锚点 `*default_w8a8_dynamic` 引用。

#### 默认 W4A4 动态量化锚点 (default_w4a4_dynamic)

- **act** / **weight**：线性层 W4A4 动态量化，`dual_scale` + `mxfp4` + `dualscale`，`dual_block_size` 为 512，供 `linear_quant` 通过 YAML 锚点 `*default_w4a4_dynamic` 引用。

#### 量化配置 (process)

- **linear_quant**：对线性层进行 W8A8（mxfp8）或 W4A4（mxfp4 DualScale）动态量化；`exclude` 中模式用于排除部分子模块，以稳定精度。
- **online_quarot**：在线旋转相关配置，与注意力等模块配合。
- **fa3_quant**：Flash Attention 3 路径上的量化配置；W8A8F8 方案为 `fp8_e4m3` + `per_token`，W4A4F4 方案为 `mxfp4` + `per_block`。

#### 保存配置 (save)

- **type**：保存器类型，使用 `mindie_format_saver`。
- **part_file_size**：分片大小，`0` 表示不分片。

#### 多模态配置 (multimodal_sd_config)

- **dump_config**：校准数据导出相关；当前默认示例中 `enable_dump` 为 `False`。若后续开启校准 dump，需按一键量化协议与适配器约定配置。
- **model_config**：可与推理参数对齐的占位字段，例如：
  - `img_paths`：输入图像路径（多图可用逗号分隔等约定，以推理仓为准）。
  - `prompt_file`：提示词文件路径。

更细的协议说明见：[一键量化配置协议说明](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/user_guide/feature_guide/quick_quantization_v1/usage/#6-%E9%87%8F%E5%8C%96%E9%85%8D%E7%BD%AE%E5%8D%8F%E8%AE%AE%E8%AF%A6%E8%A7%A3)。

## FAQ

**现象：量化时报错无法导入 `qwenimage_edit`。**
**解决方案：** 请按 [MindIE/Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509) 将推理工程置于 Python 路径或按说明安装，使 `qwenimage_edit.transformer_qwenimage`、`qwenimage_edit.pipeline_qwenimage_edit_plus` 可被正常导入。

**现象：如何自定义量化配置？**
**解决方案：** 可在 `process` 中调整 `exclude`/`include`、量化 dtype 与范围等；自定义配置需自行验证精度与兼容性，官方仅对最佳实践库中已验证配置提供保证。

## 附录

### 相关资源

- [Qwen-Image-Edit-2509（Hugging Face）](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- [Qwen-Image-Edit-2509模型仓库](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509)
- [一键量化配置协议说明](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/user_guide/feature_guide/quick_quantization_v1/usage/#6-%E9%87%8F%E5%8C%96%E9%85%8D%E7%BD%AE%E5%8D%8F%E8%AE%AE%E8%AF%A6%E8%A7%A3)
- [逐层量化特性说明](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/user_guide/feature_guide/quick_quantization_v1/usage/#51-%E9%80%90%E5%B1%82%E9%87%8F%E5%8C%96%E5%8F%8A%E5%88%86%E5%B8%83%E5%BC%8F%E9%80%90%E5%B1%82%E9%87%8F%E5%8C%96)
