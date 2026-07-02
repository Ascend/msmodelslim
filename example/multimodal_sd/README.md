# 多模态生成模型量化说明

## 模型介绍

[SD3](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/stable_diffusion/stable_diffusion_3.md) Stable Diffusion 3, 由stability.ai发布的强大的文本到图像模型，在多主题提示、图像质量和拼写功能方面的性能得到了大幅提升。

[Open-Sora-Plan v1.2](https://github.com/PKU-YuanGroup/Open-Sora-Plan) 是一个开源的多模态视频生成模型，由北大-兔展AIGC联合实验室共同发起，专注于高效视频生成任务。

[Flux.1](https://github.com/black-forest-labs/flux)是由 Black Forest Labs 开发的一款开源的 120 亿参数的图像生成模型，它能够根据文本描述生成高质量的图像。

[HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) 是腾讯发布的一种新颖的开源视频基础模型，它在视频生成方面的性能可与领先的闭源模型相媲美，甚至优于领先的闭源模型。

[Wan2.1](https://github.com/Wan-Video/Wan2.1) 是阿里巴巴发布的一套全面且开放的视频基础模型，它突破了视频生成的界限。支持文本到视频(T2V)、图像到视频(I2V)、文本到图像(T2I)等多种生成任务。

[Wan2.2](https://github.com/Wan-Video/Wan2.2) 是阿里巴巴在 Wan 系列上的新一代开源视频基础模型，面向更高质量、更可控的影视级视频生成；在 Wan2.1 的基础上进一步扩充训练数据与能力，并引入面向视频扩散的 混合专家（MoE） 等设计，在保持开放生态的同时提升生成效率与观感。支持 文本到视频（T2V）、图像到视频（I2V） 以及 文本+图像到视频（TI2V） 等多种模式。

[Qwen-Image-Edit](https://github.com/QwenLM/Qwen-Image) 是阿里巴巴通义千问团队基于 Qwen-Image 图像基础模型推出的开源图像编辑模型，兼顾语义级改动（如风格、构图、物体增删与替换）与外观级细节控制。支持中英文画面内文字的精准修改。

## 使用前准备

- 配套CANN版本请选择8.2.RC1及之后的版本
- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/install_guide/install_guide/)。
- 当前多模态生成模型统一接口依赖于pydantic库
  - pip install pydantic
- SD3-Medium依赖于diffusers库
  - pip install -U diffusers
- Open-Sora-Plan v1.2相关环境配置参考[MindIE/open_sora_planv1_2](https://modelers.cn/models/MindIE/open_sora_planv1_2)
  - 参考 [open_sora_planv1_2 readme](https://modelers.cn/models/MindIE/open_sora_planv1_2) 安装浮点模型的环境依赖，并确保浮点推理能正常运行
  - pip install huggingface_hub==0.25.2
- Flux.1-dev相关环境配置参考[MindIE/FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev)
  - 参考 [Flux readme](https://modelers.cn/models/MindIE/FLUX.1-dev) 安装浮点模型的环境依赖，并确保浮点推理能正常运行
- HunyuanVideo相关环境配置参考[MindIE/hunyuan_video](https://modelers.cn/models/MindIE/hunyuan_video)
  - 参考 [HunyuanVideo readme](https://modelers.cn/models/MindIE/hunyuan_video) 安装浮点模型的环境依赖，并确保浮点推理能正常运行
- Wan2.1相关环境配置参考[MindIE/Wan2.1](https://modelers.cn/models/MindIE/Wan2.1)
  - 参考 [Wan2.1 readme](https://modelers.cn/models/MindIE/Wan2.1/blob/main/README.md) 安装浮点模型的环境依赖，并确保浮点推理能正常运行
- Wan2.2相关环境配置参考[MindIE/Wan2.2](https://modelers.cn/models/MindIE/Wan2.2)
  - 参考 [Wan2.2 readme](https://modelers.cn/models/MindIE/Wan2.2/blob/main/README.md) 安装浮点模型的环境依赖，并确保浮点推理能正常运行
- Qwen-Image-Edit-2509相关环境配置参考[MindIE/Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509)
  - 参考 [Qwen-Image-Edit-2509 readme](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509/blob/main/README.md) 安装浮点模型的环境依赖，并确保浮点推理能正常运行

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接 | W8A8 | W8A16 | W4A16 | W4A4 | 稀疏量化 | KV Cache | Attention | 时间步量化 | FA3量化 | 量化命令 |
|---------|---------|---------------------------------------------------------------|-----|-------|-------|------|---------|----------|-----------|----------|----------|----------|
| **SD3** | SD3-Medium | [SD3-Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium) | ✅ | | | | | | | | | [W8A8静态量化](#sd3-medium-w8a8静态量化) |
| **Open-Sora-Plan** | Open-Sora-Plan v1.2 | [Open-Sora-Plan v1.2](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0) | ✅ | | | | | | | | | [W8A8静态量化](#open-sora-plan-v12-w8a8静态量化) |
| **FLUX** | FLUX.1-dev | [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main) | ✅ | | | | | | ✅ | ✅ | ✅ | [W8A8静态量化](#flux1-dev-w8a8静态量化) / [W8A8分时间步量化](#flux1-dev-w8a8分时间步量化) / [FA3+W8A8动态量化](#flux1-dev-fa3w8a8动态量化) / [异常值抑制+W8A8动态量化](#flux1-dev-异常值抑制w8a8动态量化) |
| **HunyuanVideo** | HunyuanVideo | [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo) | ✅ | | | | | | ✅ | ✅ | ✅ | [W8A8静态量化](#hunyuanvideo-w8a8静态量化) / [W8A8分时间步量化](#hunyuanvideo-w8a8分时间步量化) / [FA3+W8A8动态量化](#hunyuanvideo-fa3w8a8动态量化) / [异常值抑制+W8A8动态量化](#hunyuanvideo-异常值抑制w8a8动态量化) |
| **Wan2.1** | Wan2.1-T2V-14B | [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | ✅ | | | | | | | | | [W8A8动态量化](#wan21-w8a8动态量化) |
| **Wan2.2** | Wan2.2-T2V-A14B | [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | ✅ | | | | | |✅ | | | [FA3+W8A8动态量化](#wan22-fa3w8a8动态量化)|
| **Wan2.2** | Wan2.2-I2V-A14B | [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) | ✅ | | | | | |✅ | | |  [FA3+W8A8动态量化](#wan22-fa3w8a8动态量化)|
| **Wan2.2** | Wan2.2-TI2V-5B | [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | ✅ | | | | | |✅ | | | [FA3+W8A8动态量化](#wan22-fa3w8a8动态量化)|
| **Wan2.2** | Wan2.2-T2V-A14B | [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | | | | ✅ | | | ✅ | | | [FA3+W4A4动态量化](#wan22-fa3w4a4动态量化) |
| **Qwen-Image-Edit** | Qwen-Image-Edit-2509 | [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) | ✅ | | | ✅ | | |✅ | | | [W8A8F8动态量化](#qwen-image-edit-2509-w8a8f8动态量化) / [W4A4F4动态量化](#qwen-image-edit-2509-w4a4f4动态量化) |

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令。
- 其中FLUX.1-dev、HunyuanVideo、Wan2.2、Qwen-Image-Edit-2509支持在昇腾950代际产品上运行的mxfp8量化，详情点击查看具体量化命令。

## 使用示例

使用量化前，需要加载模型和校准数据，其中加载模型依赖于diffusers库（如SD3-Medium）或多模态生成模型[魔乐社区](https://modelers.cn/models/)推理工程仓（如Open-Sora-Plan v1.2、Flux.1-dev、HunyuanVideo、Wan2.1），请先确保依据推理工程仓可以正常进行浮点推理。

- Open-Sora-Plan v1.2推理工程仓：[MindIE/open_sora_planv1_2](https://modelers.cn/models/MindIE/open_sora_planv1_2)
- Flux.1-dev推理工程仓：[MindIE/FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev)
- HunyuanVideo推理工程仓[MindIE/hunyuan_video](https://modelers.cn/models/MindIE/hunyuan_video)
- Wan2.1推理工程仓[MindIE/Wan2.1](https://modelers.cn/models/MindIE/Wan2.1)
- Wan2.2推理工程仓[MindIE/Wan2.2](https://modelers.cn/models/MindIE/Wan2.2)
- Qwen-Image-Edit-2509推理工程仓[MindIE/Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509)

### <span id="sd3-medium-w8a8静态量化">SD3-Medium W8A8静态量化</span>

请参考《[SD3-Medium 量化使用说明](./SD3/README.md)》

### <span id="open-sora-plan-v12-w8a8静态量化">Open-Sora-Plan v1.2 W8A8静态量化</span>

请参考《[Open-Sora-Plan V1.2 量化使用说明](./OpenSoraPlanV1_2/README.md)》

### <span id="flux1-dev-w8a8静态量化">FLUX.1-dev W8A8静态量化</span>

请参考《[FLUX.1-dev 量化使用说明](./Flux/README.md)》

### <span id="flux1-dev-w8a8分时间步量化">FLUX.1-dev W8A8分时间步量化</span>

请参考《[FLUX.1-dev 量化使用说明](./Flux/README.md)》

### <span id="flux1-dev-fa3w8a8动态量化">FLUX.1-dev FA3+W8A8动态量化</span>

请参考《[FLUX.1-dev 量化使用说明](./Flux/README.md)》

### <span id="flux1-dev-异常值抑制w8a8动态量化">FLUX.1-dev 异常值抑制+W8A8动态量化</span>

请参考《[FLUX.1-dev 量化使用说明](./Flux/README.md)》

### <span id="hunyuanvideo-w8a8静态量化">HunyuanVideo W8A8静态量化</span>

请参考《[HunyuanVideo 量化使用说明](./HunYuanVideo/README.md)》

### <span id="hunyuanvideo-w8a8分时间步量化">HunyuanVideo W8A8分时间步量化</span>

请参考《[HunyuanVideo 量化使用说明](./HunYuanVideo/README.md)》

### <span id="hunyuanvideo-fa3w8a8动态量化">HunyuanVideo FA3+W8A8动态量化</span>

请参考《[HunyuanVideo 量化使用说明](./HunYuanVideo/README.md)》

### <span id="hunyuanvideo-异常值抑制w8a8动态量化">HunyuanVideo 异常值抑制+W8A8动态量化</span>

请参考《[HunyuanVideo 量化使用说明](./HunYuanVideo/README.md)》

### <span id="wan21-w8a8动态量化">Wan2.1 W8A8动态量化</span>

请参考《[Wan2.1 量化使用说明](./Wan2_1/README.md)》

### <span id="wan22-fa3w8a8动态量化">Wan2.2 FA3+W8A8动态量化</span>

请参考《[Wan2.2 量化使用说明](./Wan2_2/README.md)》

### <span id="wan22-fa3w4a4动态量化">Wan2.2 FA3+W4A4动态量化</span>

请参考《[Wan2.2 量化使用说明](./Wan2_2/README.md)》

### <span id="qwen-image-edit-2509-w8a8f8动态量化">Qwen-Image-Edit-2509 W8A8F8 动态量化</span>

请参考《[Qwen Image Edit 量化使用说明](./QwenImageEdit/README.md)》

### <span id="qwen-image-edit-2509-w4a4f4动态量化">Qwen-Image-Edit-2509 W4A4F4 动态量化</span>

请参考《[Qwen Image Edit 量化使用说明](./QwenImageEdit/README.md)》
