# Multimodal Generation Model Quantification Description

## Model Introduction

[SD3](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/stable_diffusion/stable_diffusion_3.md)    Stable Diffusion 3, a powerful text-to-image model released by stability.ai, offers a huge performance boost in multi-topic prompting, image quality, and spelling capabilities.

[Open-Sora-Plan v1.2](https://github.com/PKU-YuanGroup/Open-Sora-Plan)    It is an open-source multi-modal video generation model, co-sponsored by the AIGC Joint Laboratory of Peking University-rabbit exhibition, focusing on efficient video generation tasks.

[Flux.1](https://github.com/black-forest-labs/flux)    is an open source 12 billion parameter image generation model developed by Black Forest Labs that generates high-quality images based on text descriptions.

[HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo)    It is a novel open source video basic model released by Tencent. Its performance in video generation is comparable to or even superior to the leading closed-source model.

[Wan2.1](https://github.com/Wan-Video/Wan2.1)    Alibaba is a comprehensive and open video basic model released by Alibaba, which breaks through the boundaries of video generation. Supports multiple generation tasks, such as text-to-video (T2V), image-to-video (I2V), and text-to-image (T2I).

[Wan2.2](https://github.com/Wan-Video/Wan2.2)    Alibaba is a new generation of open source video basic model on the Wan series, oriented to higher quality, more controllable film-level video generation; Based on Wan2.1, further expand training data and capabilities, and introduce design such as Hybrid Expert (MoE) for video diffusion to improve generation efficiency and visual perception while maintaining an open ecosystem. Support for multiple modes, including text-to-video (T2V), image-to-video (I2V), and text+image-to-video (TI2V).

[Qwen-Image-Edit](https://github.com/QwenLM/Qwen-Image)    It is an open source image editing model launched by Alibaba Tongyi Qianwen team based on the Qwen-Image image basic model, taking into account semantic changes. (e.g. style, composition, object addition, deletion and replacement) With appearance level detail control. Precisely modify the text in Chinese and English images.

## Preparation Before Use

 * Select 8.2.RC1 or a later version.
 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/en/getting_started/install_guide.md)    .
 * Currently, the unified interface for generating multi-modal models depends on the pydantic library.
    
     * pip install pydantic
 * SD3-Medium depends on the diffusers library.
    
     * pip install -U diffusers
 * Open-Sora-Plan v1.2 Environment Configuration Reference[MindIE/open_sora_planv1_2](https://modelers.cn/models/MindIE/open_sora_planv1_2)    
    
     * Reference[open_sora_planv1_2 readme](https://modelers.cn/models/MindIE/open_sora_planv1_2)    Install the environmental dependencies for the floating-point model and ensure that floating-point inference works properly.
     * pip install huggingface_hub==0.25.2
 * Flux.1-dev Environment Configuration Reference[MindIE/FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev)    
    
     * Reference[Flux readme](https://modelers.cn/models/MindIE/FLUX.1-dev)    Install the environmental dependencies for the floating-point model and ensure that floating-point inference works properly.
 * HunyuanVideo Environment Configuration Reference[MindIE/hunyuan_video](https://modelers.cn/models/MindIE/hunyuan_video)    
    
     * Reference[HunyuanVideo readme](https://modelers.cn/models/MindIE/hunyuan_video)    Install the environmental dependencies for the floating-point model and ensure that floating-point inference works properly.
 * Wan2.1 Environment Configuration Reference[MindIE/Wan2.1](https://modelers.cn/models/MindIE/Wan2.1)    
    
     * Reference[Wan2.1 readme](https://modelers.cn/models/MindIE/Wan2.1/blob/main/README.md)    Install the environmental dependencies for the floating-point model and ensure that floating-point inference works properly.
 * Wan2.2 Environment Configuration Reference[MindIE/Wan2.2](https://modelers.cn/models/MindIE/Wan2.2)    
    
     * Reference[Wan2.2 readme](https://modelers.cn/models/MindIE/Wan2.2/blob/main/README.md)    Install the environmental dependencies for the floating-point model and ensure that floating-point inference works properly.
 * Qwen-Image-Edit-2509 Environment Configuration Reference[MindIE/Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509)    
    
     * Reference[Qwen-Image-Edit-2509 readme](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509/blob/main/README.md)    Install the environmental dependencies for the floating-point model and ensure that floating-point inference works properly.

## Supported Model Versions and Quantification Policies

| Model Series        | Model version        | HuggingFace Link                                                                  | W8A8 | W8A16 | W4A16 | W4A4 | sparse quantization | KV Cache | Attention | time step quantization | FA3 Quantification | Quantization command                                                                                                                                                                                                                                            |
| ------------------- | -------------------- | --------------------------------------------------------------------------------- | ---- | ----- | ----- | ---- | ------------------- | -------- | --------- | ---------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SD3**             | SD3-Medium           | [SD3-Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium)           | ✅    |       |       |      |                     |          |           |                        |                    | [W8A8 static quantization](#sd3-medium-w8a8-static-quantization)                                                                                                                                                                                                                   |
| **Open-Sora-Plan**  | Open-Sora-Plan v1.2  | [Open-Sora-Plan v1.2](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0)     | ✅    |       |       |      |                     |          |           |                        |                    | [W8A8 static quantization](#open-sora-plan-v12-w8a8-static-quantization)                                                                                                                                                                                                           |
| **FLUX**            | FLUX.1-dev           | [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main)          | ✅    |       |       |      |                     |          | ✅         | ✅                      | ✅                  | [W8A8 static quantization](#flux1-dev-w8a8-static-quantization)    /[W8A8 time step quantization](#flux1-dev-w8a8-time-step-quantization)    /[FA3+W8A8 dynamic quantization](#flux1-dev-fa3w8a8-dynamic-quantization)    /[Abnormal value suppression + W8A8 dynamic quantization](#flux1-dev-abnormal-value-suppression--w8a8-dynamic-quantization)                 |
| **HunyuanVideo**    | HunyuanVideo         | [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo)                          | ✅    |       |       |      |                     |          | ✅         | ✅                      | ✅                  | [W8A8 static quantization](#hunyuanvideo-w8a8-static-quantization)    /[W8A8 time step quantization](#hunyuanvideo-w8a8-minute-time-step-quantization)    /[FA3+W8A8 dynamic quantization](#hunyuanvideo-fa3w8a8-dynamic-quantization)    /[Abnormal value suppression + W8A8 dynamic quantization](#hunyuanvideo-abnormal-value-suppression--w8a8-dynamic-quantization)     |
| **Wan2.1**          | Wan2.1-T2V-14B       | [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)                       | ✅    |       |       |      |                     |          |           |                        |                    | [W8A8 Dynamic Quantization](#wan21-w8a8-dynamic-quantization)                                                                                                                                                                                                                       |
| **Wan2.2**          | Wan2.2-T2V-A14B      | [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)                     | ✅    |       |       |      |                     |          | ✅         |                        |                    | [FA3+W8A8 dynamic quantization](#wan22-fa3w8a8-dynamic-quantization)                                                                                                                                                                                                                |
| **Wan2.2**          | Wan2.2-I2V-A14B      | [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)                     | ✅    |       |       |      |                     |          | ✅         |                        |                    | [FA3+W8A8 dynamic quantization](#wan22-fa3w8a8-dynamic-quantization)                                                                                                                                                                                                                |
| **Wan2.2**          | Wan2.2-TI2V-5B       | [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)                       | ✅    |       |       |      |                     |          | ✅         |                        |                    | [FA3+W8A8 dynamic quantization](#wan22-fa3w8a8-dynamic-quantization)                                                                                                                                                                                                                |
| **Qwen-Image-Edit** | Qwen-Image-Edit-2509 | [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)             | ✅    |       |       |      |                     |          | ✅         |                        |                    | [FA3+W8A8 dynamic quantization](#qwen-image-edit-2509-fa3w8a8-dynamic-quantization)                                                                                                                                                                                                 |

**Description:**

 * ' indicates that the quantification policy has been officially verified by msModelSlim and has complete functions and stable performance. It is recommended that the quantification policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements. However, the quantification effect and function stability cannot be officially guaranteed.
 * You can click the link in the Quantization Command column to go to the corresponding quantization command.
 * FluX.1-dev, HunyuanVideo, Wan2.2, and Qwen-Image-Edit-2509 support mxfp8 quantification running on Ascend 950. For details, click to view the specific quantification command.

## Use Example

Before using quantization, you need to load the model and calibration data. Loading the model depends on the diffusers library (such as SD3-Medium) or multimodal generation model.[Magic Music Community](https://modelers.cn/models/)    Inference engineering repository (such as Open-Sora-Plan v1.2, Flux.1-dev, HunyuanVideo, and Wan2.1) Ensure that floating-point inference can be properly performed based on the inference project repository.

 * Open-Sora-Plan v1.2 inference engineering repository:[MindIE/open_sora_planv1_2](https://modelers.cn/models/MindIE/open_sora_planv1_2)    
 * Flux.1-dev inference project repository:[MindIE/FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev)    
 * HunyuanVideo inference engineering warehouse[MindIE/hunyuan_video](https://modelers.cn/models/MindIE/hunyuan_video)    
 * Wan2.1 Inference Engineering Warehouse[MindIE/Wan2.1](https://modelers.cn/models/MindIE/Wan2.1)    
 * Wan2.2 Inference Engineering Warehouse[MindIE/Wan2.2](https://modelers.cn/models/MindIE/Wan2.2)    
 * Qwen-Image-Edit-2509 Inference Engineering Repository[MindIE/Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509)    

### SD3-Medium W8A8 static quantization

For details, see.[SD3-Medium Quantization Instructions](./SD3/README.md)    

### Open-Sora-Plan v1.2 W8A8 Static Quantization

For details, see.[Open-Sora-Plan V1.2 Quantification Instructions](./OpenSoraPlanV1_2/README.md)    

### FLUX.1-dev W8A8 Static Quantization

For details, see.[FLUX.1-dev Quantification Instructions](./Flux/README.md)    

### FLUX.1-dev W8A8 time step quantization

For details, see.[FLUX.1-dev Quantification Instructions](./Flux/README.md)    

### FLUX.1-dev FA3+W8A8 Dynamic Quantization

For details, see.[FLUX.1-dev Quantification Instructions](./Flux/README.md)    

### FLUX.1-dev Abnormal Value Suppression + W8A8 Dynamic Quantization

For details, see.[FLUX.1-dev Quantification Instructions](./Flux/README.md)    

### HunyuanVideo W8A8 Static Quantization

For details, see.[HunyuanVideo Quantization Instructions](./HunYuanVideo/README.md)    

### HunyuanVideo W8A8-Minute Time Step Quantization

For details, see.[HunyuanVideo Quantization Instructions](./HunYuanVideo/README.md)    

### HunyuanVideo FA3+W8A8 Dynamic Quantization

For details, see.[HunyuanVideo Quantization Instructions](./HunYuanVideo/README.md)    

### HunyuanVideo Abnormal Value Suppression + W8A8 Dynamic Quantization

For details, see.[HunyuanVideo Quantization Instructions](./HunYuanVideo/README.md)    

### Wan2.1 W8A8 Dynamic Quantization

For details, see.[Wan2.1 Quantification Instructions](./Wan2_1/README.md)    

### Wan2.2 FA3+W8A8 dynamic quantization

For details, see.[Wan2.2 Quantification Instructions](./Wan2_2/README.md)    

### Qwen-Image-Edit-2509 FA3+W8A8 dynamic quantization

For details, see.[Qwen Image Edit Quantification Instructions](./QwenImageEdit/README.md)    
