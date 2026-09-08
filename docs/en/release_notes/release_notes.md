# Release Notes

## Version Mapping

### Product Versions

| Product       | Version          | Version Type|
|-------------|----------------|------|
| msModelSlim | 26.0.0.alpha02 | Internal test version|
| msModelSlim | 26.0.0.alpha01 | Internal test version|
| msModelSlim | 8.3.0          | Official version|

### Related Product Versions

| msModelSlim Version| CANN Version| PyTorch Version         | TorchNPU Version   | Python Version        | Transformers Version|
|---------------|--------|-------------------|----------------|------------------|----------------|
| 26.0.0.alpha02         | No specific version requirement| Depends on the specific model. See the corresponding model documentation.| Depends on the specific model. See the corresponding model documentation.| Python 3.10 and 3.11| Depends on the specific model. See the corresponding case description in the [example](https://gitcode.com/Ascend/msmodelslim/tree/master/example) directory.|
| 26.0.0.alpha01         | No specific version requirement| Depends on the specific model. See the corresponding model documentation.| Depends on the specific model. See the corresponding model documentation.| Python 3.10 and 3.11| Depends on the specific model. See the corresponding case description in the [example](https://gitcode.com/Ascend/msmodelslim/tree/master/example) directory.|
| 8.3.0         | 8.2.RC1 or later| Depends on the specific model. See the corresponding model documentation.| Depends on the specific model. See the corresponding model documentation.| Python 3.10 and 3.11| Depends on the specific model. See the corresponding case description in the [example](https://gitcode.com/Ascend/msmodelslim/tree/master/example) directory.|

### Wheel Package Downloads

|       Version       |     Download Link          |        Checksum                |
|:--------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------:|
| 26.0.0-alpha.2 | [msmodelslim-26.0.0a2-py3-none-any.whl](https://gitcode.com/Ascend/msmodelslim/releases/download/tag_mindstudio_26.0.0.alpha02/msmodelslim-26.0.0a2-py3-none-any.whl) | 4711edb30c4354fcb99fb69a2e0351561b013bb1298d6f54a0ee409bf979a264 |
| 26.0.0-alpha.1 | [msmodelslim-26.0.0a1-py3-none-any.whl](https://gitcode.com/Ascend/msmodelslim/releases/download/tag_MindStudio_26.0.0-alpha.1/msmodelslim-26.0.0a1-py3-none-any.whl) | 60383c42bf103cf2f78304b3b974e2dac0190f0f20706a5ef347e55855048f42 |

For more details, see [release](https://gitcode.com/Ascend/msmodelslim/releases?presetConfig={%22tags%22:32,%22release%22:2}).

## Version Compatibility

Refer to the preceding table for the compatibility information of each version.

## Feature Updates

### 26.0.0.alpha02

1. Supports custom `practice` directories through an entry point, laying the groundwork for the plugin-based `model_adapter` capability.
2. Improves automatic tuning.
3. Supports W4A8 quantization for the Qwen3-Coder-480B model and W8A8 quantization for the Qwen3.5 MoE model.
4. Supports W8A8 quantization for the GLM-4.7 model and W4A8 quantization for the GLM-5 model.
5. Supports W8A8 quantization for the Qwen2.5-Omni-7B model and the Qwen3-Omni-30B-A3B model.

### 26.0.0.alpha01

1. Supports W8A8 quantization for Qwen3-VL-32B-Instruct.
2. Supports automatic tuning based on quantization-accuracy feedback and can automatically search for the optimal quantization configuration based on accuracy requirements.
3. Supports self-managed quantization for multimodal understanding models and supports quantization integration for those models.
4. Quick quantization supports multi-card quantization and distributed layer-by-layer quantization, improving the efficiency of large-model quantization.
5. Supports W8A8 quantization for DeepSeek-V3.2. You can run it on a single card with 64 GB of accelerator memory and 100 GB of system memory.
6. Supports W4A8 quantization for DeepSeek-V3.2-Exp. You can run it on a single card with 64 GB of accelerator memory and 100 GB of system memory.
7. Supports W8A8 quantization for Qwen3-VL-235B-A22B.

### 8.3.0

1. Supports W8A8 quantization for DeepSeek-V3.2-Exp.
2. Supports W8A8C8 quantization for DeepSeek-V3.1.
3. Supports W8A8C8 quantization for Qwen3-32B.
4. Supports W8A8 quantization for Qwen3-Next-80B.
5. Supports W4A8C8 quantization for DeepSeek-R1-0528.
