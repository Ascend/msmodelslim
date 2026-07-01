# Qwen2.5-Omni Quantification Description

## Model Introduction

[Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni)    is an end-to-end multimodal model that simultaneously perceives text, images, audio, and video, and streams text and natural speech. The main features are as follows:

 * **Omni and Thinker-Talker architecture: End-to-end multi-modal model, supporting joint perception of text, image, audio, and video, and simultaneously generating text and natural speech in streaming mode; Time-aligned Multimodal RoPE (TMRoPE) position encoding is used to align the video and audio timestamps.**
 * **Real-time voice and video dialogue: supports full real-time interaction between block input and instant output.**
 * **Natural and robust speech generation: Excellent performance in streaming and non-streaming scenarios, with outstanding naturalness and robustness of speech.**
 * **Multi-modal capability balance: Excellent performance on the same scale single-modal benchmark, superior audio capability to the same scale Qwen2-Audio, and comparable visual capability to the Qwen2.5-VL-7B.**
 * **End-to-end voice command compliance: In the MMLU, GSM8K, and other benchmarks, the voice command compliance effect is equivalent to the text input effect.**

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/zh/getting_started/install_guide.md)    .
 * For Qwen2.5-Omni, the transformers version must be 4.57. 3:
    
    ```bash
    pip install transformers==4.57.3
    ```

 * The qwen_omni_utils dependency must be installed to preprocess model data.
    
    ```bash
    pip install qwen_omni_utils
    ```

 * You need to install the FFmpeg (for audio and video preprocessing) in the environment.
    
    ```bash
    #Ubuntu
      sudo apt-get update && sudo apt install -y ffmpeg
    
      #CentOS
      sudo yum install -y ffmpeg
    
      #Verifying the FFmpeg Installation
      ffmpeg -version
    ```

## Qwen2.5-Omni Model Currently Validated Quantification Methods

| model           | Raw floating-point weight                                       | Quantization mode         | Inference Framework Support                                | Quantization command                                  |
| --------------- | --------------------------------------------------------------- | ------------------------- | ---------------------------------------------------------- | ----------------------------------------------------- |
| Qwen2.5-Omni-7B | [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)     | W8A8 Dynamic Quantization | MindIE to be supported<br/>vLLM Ascend is being supported. | [W8A8 Dynamic Quantization](#qwen25-omni-7b-w8a8-dynamic-quantization)     |

**Note: Click the link in the Quantization Command column to go to the specific quantization command.**

## Calibration Data Description

Calibration data supported modes. For details, see.[Dataset Configuration Description](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#dataset---校准数据路径配置)    \:

For Qwen2.5-Omni, index.json or index.jsonl is recommended. (file path or a directory that contains only one index.json or index.jsonl) Added support for multi-modal fields in. Provide each sample during calibration`text`and multimodal combinations consistent with the inference scenario (`image`,`audio`,`video`), the current missing sample will be skipped.

`dataset`Can be configured as a short name (on the`lab_calib`), absolute path, or relative path. For example, see.[qwen2_5_omni_thinker_w8a8.yaml](../../../lab_practice/qwen2_5_omni_thinker/qwen2_5_omni_thinker_w8a8.yaml)    \:`dataset`Specify the calibration dataset,`default_text`It can be set to multi-modal description prompt, for example, "What are the elements can you see and hear in these medias.".

## Generate quantified weights

### Qwen2.5-Omni-7B W8A8 Dynamic Quantization

Quantification of the model has been integrated into[One-click quantization](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#参数说明)    .

```shell
msmodelslim quant \
    --model_path /path/to/qwen2_5_omni_float_weights \
    --save_path /path/to/qwen2_5_omni_quantized_weights \
    --device npu \
    --model_type Qwen2.5-Omni-7B \
    --quant_type w8a8 \
    --trust_remote_code True
```

## Appendixes

 * [Configuration Description of the multimodal_vlm_modelslim_v1 Quantification Service](../../../docs/zh/feature_guide/quick_quantization_v1/usage.md#multimodal_vlm_modelslim_v1-配置详解)    
