# Wan2.2 Quantification Instructions

## Introduction to the Wan2.2 Model

Wan2.2 is Alibaba's new-generation open-source video basic model on the Wan series. It is oriented to higher-quality and more controllable film-level video generation. Based on Wan2.1, further expand training data and capabilities, and introduce design such as Hybrid Expert (MoE) for video diffusion to improve generation efficiency and visual perception while maintaining an open ecosystem. Support for multiple modes, including text-to-video (T2V), image-to-video (I2V), and text+image-to-video (TI2V).

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/en/getting_started/install_guide.md)    .
 * Environment installation reference Magic Music Community[Wan2.2](https://modelers.cn/models/MindIE/Wan2.2)    

## Supported Model Versions and Quantification Policies

| Model Series | Model version   | Model Repository Link                                        | W8A8 | W8A16 | W4A16 | W4A4 | time step quantization | Quantification of FA3 | outlier suppression quantization | Quantization command                                      |
| ------------ | --------------- | ------------------------------------------------------------ | ---- | ----- | ----- | ---- | ---------------------- | --------------------- | -------------------------------- | --------------------------------------------------------- |
| **Wan2.2**   | Wan2.2-T2V-A14B | [Wan2.2-T2V-A14B](https://modelers.cn/models/MindIE/Wan2.2)     | ✅    |       |       |      |                        | ✅                     |                                  | [FA3+W8A8 dynamic quantization](#wan22-t2v-a14b-fa3w8a8-dynamic-quantization)      |
|              | Wan2.2-I2V-A14B | [Wan2.2-I2V-A14B](https://modelers.cn/models/MindIE/Wan2.2)     | ✅    |       |       |      |                        | ✅                     |                                  | [FA3+W8A8 dynamic quantization](#wan22-i2v-a14b-fa3w8a8-dynamic-quantization)      |
|              | Wan2.2-TI2V-5B  | [Wan2.2-TI2V-5B](https://modelers.cn/models/MindIE/Wan2.2)      | ✅    |       |       |      |                        | ✅                     |                                  | [FA3+W8A8 dynamic quantization](#wan22-ti2v-5b-fa3w8a8-dynamic-quantization)     |

**Description:**

 * " indicates that the quantification policy has been officially verified by msModelSlim and has complete functions and stable performance. It is recommended that the quantification policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the quantization command column to go to the specific quantization command.
 * Note that the quantification must be performed in the model file path.

## Wan2.2 Quantification Support

The Wan2.2 model is based on the Transformer architecture. The msmodelslim supports quantization of the Transformer part and layer by layer, which significantly reduces the memory usage during quantization.

### quantization characteristics

 * **Layer-by-layer quantization: supports layer-by-layer processing, greatly reducing memory usage.**
 * **Single-card quantization: With the layer-by-layer quantization feature, single-card quantization can be implemented on the Atlas 800I/800T A2 (64G) device.**

## Quantization command

### Wan2.2-T2V-A14B FA3+W8A8 dynamic quantization

#### Use the config_path parameter to specify the configuration file for one-click quantization

```bash
msmodelslim quant \
    --model_path /path/to/wan2_2_t2v_float_weights \
    --save_path /path/to/wan2_2_t2v_quantized_weights \
    --device npu \
    --model_type Wan2_2 \
    --config_path /lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_t2v.yaml \
    --trust_remote_code True
```

### Wan2.2-I2V-A14B FA3+W8A8 dynamic quantization

#### Use the config_path parameter to specify the configuration file for one-click quantization

```bash
msmodelslim quant \
    --model_path /path/to/wan2_2_i2v_float_weights \
    --save_path /path/to/wan2_2_i2v_quantized_weights \
    --device npu \
    --model_type Wan2_2 \
    --config_path /lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_i2v.yaml \
    --trust_remote_code True
```

### Wan2.2-TI2V-5B FA3+W8A8 dynamic quantization

#### Use the config_path parameter to specify the configuration file for one-click quantization

```bash
msmodelslim quant \
    --model_path /path/to/wan2_2_ti2v_float_weights \
    --save_path /path/to/wan2_2_ti2v_quantized_weights \
    --device npu \
    --model_type Wan2_2 \
    --config_path /lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_ti2v.yaml \
    --trust_remote_code True
```

### One-click quantization command parameters

For the basic description of one-click quantization parameters, see the following:[One-click Quantization Parameters](../../../docs/en/feature_guide/quick_quantization_v1/usage.md#parameters)    

For the WAN 2.2 model, the restrictions are as follows:

| Parameter Name | Explained                                                | Optional or not           | Scope                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| -------------- | -------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path     | Wan2.2 Floating Point Weight Directory                   | Mandatory.                | Type: Str                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| save_path      | Path for storing the weights of the Wan2.2 quantization. | Mandatory.                | Type: Str                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| device         | quantization equipment                                   | Mandatory.                | 1. Type: Str. 2. Only npu is supported.                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| model_type     | Model name                                               | Mandatory.                | 1. Type: Str. 2. Case-sensitive. Set this parameter to Wan2_2.                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| config_path    | Specify the configuration path.                          | Select either Quant_type. | 1. Type: Str. 2. Configuration file format: YAML 3. Only verified configurations in the best practice library are supported.[wan2_2_w8a8f8_mxfp_t2v.yaml](../../../lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_t2v.yaml)    ,[wan2_2_w8a8f8_mxfp_i2v.yaml](../../../lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_i2v.yaml)    ,[wan2_2_w8a8f8_mxfp_ti2v.yaml](../../../lab_practice/wan2_2/wan2_2_w8a8f8_mxfp_ti2v.yaml)    If the configuration is customized, msmodelslim is not responsible for the quantization result. |

\|quant_type\|Quantization type\|Choose either "config_path" \| 1. Type: Str
2. Currently, the WAN2.2 model supports only config_path.

\|trust_remote_code\|Whether to trust the custom code \| Optional \| 1. Type: Bool; Default value: False 2. Specified`trust_remote_code=True`Enable the modified custom code file to be loaded correctly (Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.\|

## Configuration File Description

### Basic Configuration Structure

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

  #Basic Configuration
  multimodal_sd_config:
    dump_config:
      capture_mode: "args"
      dump_data_dir: ""  #default is save_path
    model_config:
      prompt: "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
      #Parameters for loading models
      convert_model_dtype: True
      task: "t2v-A14B"
```

### Key Configuration Parameters

#### Quantified configuration (process)

 * **type: processor type, which has a fixed value of linear_quant.**
 * **qconfig.act: activation value quantization configuration**
    
     * `scope`\: quantization range. Use "per_block" with mxfp8.
     * `dtype`\: Data type
     * `symmetric`\: indicates whether symmetric quantization is enabled. True is recommended.
     * `method`\: quantization method. Minmax is recommended.
 * **qconfig.weight: weight quantization configuration**
    
     * `scope`\: quantization range. Use "per_block" with mxfp8.
     * `dtype`\: Data type
     * `symmetric`\: indicates whether symmetric quantization is enabled. The recommended value is True.
     * `method`\: quantization method. Minmax is recommended.

#### Saving Configurations (save)

 * **type: saver type, using "mindie_format_saver"**
 * **part_file_size: size of a segment file. The value 0 indicates that the segment file is not segmented.**

#### Multimodal Configuration (multimodal_sd_config)

 * **dump_config: Calibrate Data Capture Configuration**
    
     * `capture_mode`\: capture mode. Currently, only args is supported.
     * `dump_data_dir`\: directory for storing calibration data. If this parameter is set to an empty character string, the directory is automatically converted to the path for storing calibration data using quantization weights.
 * **model_config: model loading and inference configuration. For details about configurable parameters, see the original inference project repository.**[Wan2.2 model warehouse](https://modelers.cn/models/MindIE/Wan2.2)    

| Field name    | action                  | Description                                                                                                                                                                                                                                                                          | Optional Value                                                                            |
| ------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| prompt        | Calibration prompt word | Text description used to generate calibration data                                                                                                                                                                                                                                   | String                                                                                    |
| offload_model | Model Uninstallation    | Indicates whether to unload the model to the CPU after inference. The value True indicates that this function is enabled.                                                                                                                                                            | True/False                                                                                |
| frame_num     | Generated Frames        | Number of frames generated by the video                                                                                                                                                                                                                                              | An integer greater than 0.                                                                |
| task          | Task Type               | Specifies the model task type. The options are as follows: t2v-A14B indicates the task of generating video files using text, i2v-A14B indicates the task of generating video files using image, and ti2v-5B indicates the task of generating video files using text and image files. | "t2v-A14B", "i2v-A14B", "ti2v-5B"                                                         |
| size          | Generate Dimension      | Video or image size specifications                                                                                                                                                                                                                                                   | Reference[Wan2.2 model warehouse](https://modelers.cn/models/MindIE/Wan2.2)    Configuration |
| sample_steps  | Sampling Steps          | Sampling Steps for Diffusion Model                                                                                                                                                                                                                                                   | An integer greater than 0.                                                                |

## FAQ

Symptom: How do I customize the quantization configuration?
Solution: Modify the process part in the configuration file to adjust the quantization parameters and layer selection policy.

## Appendixes

### Related Resources

 * [Wan2.2 model warehouse](https://modelers.cn/models/MindIE/Wan2.2)    
 * [Protocol Description for One-Click Quantization Configuration](../../../docs/en/feature_guide/quick_quantization_v1/usage.md#configuration-protocol-overview)    
 * [Layer-by-Layer Quantization Feature Description](../../../docs/en/feature_guide/quick_quantization_v1/usage.md#layer-wise-and-dp-layer-wise-quantization)    
