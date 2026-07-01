# Wan2.1 Quantification Instructions

## Wan2.1 Model Introduction

Wan2.1 is a comprehensive and open video basic model released by Alibaba, which breaks through the boundaries of video generation. Multiple generation tasks are supported:

 * **Text-to-Video (T2V): Generate video from text description**
 * **Image to Video (I2V): Generates video from the input image**
 * **Text-to-Image (T2I): Generates an image from a text description**

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/en/getting_started/install_guide.md)    .

## Supported Model Versions and Quantification Policies

| Model Series | Model version | Model Repository Link                                    | W8A8 | W8A16 | W4A16 | W4A4 | time step quantization | Quantification of FA3 | outlier suppression quantization | Quantization command                              |
| ------------ | ------------- | -------------------------------------------------------- | ---- | ----- | ----- | ---- | ---------------------- | --------------------- | -------------------------------- | ------------------------------------------------- |
| **Wan2.1**   | Wan2.1-14B    | [Wan2.1-14B](https://modelers.cn/models/MindIE/Wan2.1)      | ✅    |       |       |      |                        |                       |                                  | [W8A8 Dynamic Quantization](#wan21-14b-w8a8-dynamic-quantization)     |
|              | Wan2.1-1.3B   | [Wan2.1-1.3B](https://modelers.cn/models/MindIE/Wan2.1)     | ✅    |       |       |      |                        |                       |                                  | [W8A8 Dynamic Quantization](#wan21-13b-w8a8-dynamic-quantization)     |

**Description:**

 * ' indicates that the quantization policy has passed the official verification of msModelSlim. The functions are complete and the performance is stable. It is recommended that the quantization policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy as required, but the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the quantization command column to go to the specific quantization command.

## Wan2.1 Quantification Support

The Wan2.1 model is based on the Transformer architecture. The msmodelslim supports quantization of the Transformer part and layer by layer, significantly reducing the memory usage during quantization.

### Validated Quantization Type

| Quantization type | Description                                                                                | Application Scenario             | Configuration Example                                                              |
| ----------------- | ------------------------------------------------------------------------------------------ | -------------------------------- | ---------------------------------------------------------------------------------- |
| w8a8_dynamic      | Weight: 8-bit per-channel symmetric quantization; active value: 8-bit dynamic quantization | Recommended for different inputs | [wan2_1_w8a8_dynamic.yaml](../../../lab_practice/wan2_1/wan2_1_w8a8_dynamic.yaml)     |

### quantization characteristics

 * **Layer-by-layer quantization: supports layer-by-layer processing, greatly reducing memory usage.**
 * **Single-card quantization: With the layer-by-layer quantization feature, single-card quantization can be implemented on the Atlas 800I/800T A2 (64G) device.**
 * **Dynamic activation value quantization: The activation value uses the per_token quantization range to improve the precision.**

## Quantization command

### Wan2.1-14B W8A8 Dynamic Quantization

#### Method 1: Use the quant_type parameter to perform one-click quantization

```bash
msmodelslim quant \
    --model_path /path/to/wan2_1_14b_float_weights \
    --save_path /path/to/wan2_1_14b_quantized_weights \
    --device npu \
    --model_type Wan2_1 \
    --quant_type w8a8 \
    --trust_remote_code True
```

#### Method 2: Use the config_path parameter to specify the configuration file for one-click quantization

```bash
msmodelslim quant \
    --model_path /path/to/wan2_1_14b_float_weights \
    --save_path /path/to/wan2_1_14b_quantized_weights \
    --device npu \
    --model_type Wan2_1 \
    --config_path /path/to/wan2_1_w8a8_dynamic.yaml \
    --trust_remote_code True
```

### Wan2.1-1.3B W8A8 Dynamic Quantization

#### Method 1: Use the quant_type parameter to perform one-click quantization

```bash
msmodelslim quant \
    --model_path /path/to/wan2_1_float_weights \
    --save_path /path/to/wan2_1_quantized_weights \
    --device npu \
    --model_type Wan2_1 \
    --quant_type w8a8 \
    --trust_remote_code True
```

#### Method 2: Use the config_path parameter to specify the configuration file for one-click quantization

```bash
msmodelslim quant \
    --model_path /path/to/wan2_1_float_weights \
    --save_path /path/to/wan2_1_quantized_weights \
    --device npu \
    --model_type Wan2_1 \
    --config_path /path/to/wan2_1_w8a8_dynamic.yaml \
    --trust_remote_code True
```

### One-click quantization command parameters

For the basic description of one-click quantization parameters, see the following:[One-click Quantization Parameters](../../../docs/en/feature_guide/quick_quantization_v1/usage.md#parameters)    

For the Wan2.1 model, the restrictions are as follows:

| Parameter name | Explained                                                | Optional or not           | Scope                                                                                                                                                                                                                                                                                                                |
| -------------- | -------------------------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path     | Wan2.1 Floating Point Weight Directory                   | Mandatory.                | Type: Str                                                                                                                                                                                                                                                                                                            |
| save_path      | Path for storing the weights in the Wan2.1 quantization. | Mandatory.                | Type: Str                                                                                                                                                                                                                                                                                                            |
| device         | quantization equipment                                   | Mandatory.                | 1. Type: Str 2. Only npu is supported.                                                                                                                                                                                                                                                                               |
| model_type     | Model name                                               | Mandatory.                | 1. Type: Str. 2. Case-sensitive. Set this parameter to Wan2_1.                                                                                                                                                                                                                                                       |
| config_path    | Specify the configuration path.                          | Select either Quant_type. | 1. Type: Str. 2. The configuration file format is YAML. 3. Only verified configurations in the best practice library are supported.[wan2_1_w8a8_dynamic.yaml](../../../lab_practice/wan2_1/wan2_1_w8a8_dynamic.yaml)    If the configuration is customized, msmodelslim is not responsible for the quantization result. |

\|quant_type\|Quantization type\|Choose either config_path\|1. Type: Str
2. Currently, the value can only be w8a8.

\|trust_remote_code\|Whether to trust the custom code \| Optional \| 1. Type: Bool; Default: False 2. Specify`trust_remote_code=True`Enable the modified custom code file to be loaded correctly (Ensure that the source of the loaded custom code file is reliable to avoid potential security risks.)\|

## Configuration File Description

### Basic Configuration Structure

```yaml
apiversion: multimodal_sd_modelslim_v1

spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_token"   #Activated Value Quantization Range
          dtype: "int8"        #Active Value Quantization Data Type
          symmetric: True      #Symmetric Quantization
          method: "minmax"      #Quantification method
        weight:
          scope: "per_channel"   #Weight Quantization Range
          dtype: "int8"        #Weight Quantization Data Type
          symmetric: True       #Symmetric Quantization
          method: "minmax"      #Quantification method
      include: ["*"]           #Included Layer Patterns
      exclude: ["*ffn.2*"]     #Excluded Layer Pattern

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

### Key Configuration Parameters

#### Quantified configuration (process)

 * **type: processor type. The value is fixed to linear_quant.**
 * **qconfig.act: activation value quantization configuration**
    
     * `scope`\: quantization range. The value per_token is recommended.
     * `dtype`\: data type. The fixed value is int8.
     * `symmetric`\: indicates whether symmetric quantization is enabled. The recommended value is True. In per_token quantization, this parameter can only be set to True.
     * `method`\: quantization method. Minmax is recommended.
 * **qconfig.weight: weight quantization configuration**
    
     * `scope`\: quantization range, which has a fixed value of per_channel.
     * `dtype`\: data type. The fixed value is int8.
     * `symmetric`\: indicates whether symmetric quantization is enabled. True is recommended.
     * `method`\: quantization method. Minmax is recommended.

#### Saving Configurations (save)

 * **type: saver type, using "mindie_format_saver"**
 * **part_file_size: size of a segment file. The value 0 indicates that the segment file is not segmented.**

#### Multimodal Configuration (multimodal_sd_config)

 * **dump_config: Calibrate data capture configuration**
    
     * `capture_mode`\: capture mode. Currently, only args is supported.
     * `dump_data_dir`\: directory for storing calibration data. If this parameter is set to an empty character string, it is automatically converted to the path for storing the quantization weight.
 * **model_config: model loading and inference configuration. For details about configurable parameters, see the original inference project repository.**[Wan2.1 model repository](https://modelers.cn/models/MindIE/Wan2.1)    

| Field name    | action                  | Description                                                                                                                                                                                    | Optional Value              |
| ------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| prompt        | Calibration prompt word | Text description used to generate calibration data                                                                                                                                             | String                      |
| offload_model | Model Uninstallation    | Indicates whether to unload the model to the CPU after inference. The value True indicates that this function is enabled.                                                                      | True/False                  |
| frame_num     | Generated Frames        | Number of frames generated by the video                                                                                                                                                        | An integer greater than 0.  |
| task          | Task Type               | Specifies the model task type. The value t2v-14B indicates the text generation video task of the 14B model, and the value t2v-1.3B indicates the text generation video task of the 1.3B model. | "t2v-14B", "t2v-1.3B"       |
| size          | Generate Dimension      | Video or image size specifications                                                                                                                                                             | "1280 \* 720", "832 \* 480" |
| sample_steps  | Sampling Steps          | Sampling Steps for Diffusion Model                                                                                                                                                             | An integer greater than 0.  |

## FAQ

Question: Does W8a8 Static Quantization Support?
Solution: You can modify the process part in the configuration file and set qconfig.act.scope to per_tensor to enable static quantization. However, the precision loss is severe. Therefore, it is not recommended.

Symptom: How do I customize the quantization configuration?
Solution: You can modify the process part in the configuration file to adjust the quantization parameters and layer selection policy.

## Appendixes

### Related Resources

 * [Wan2.1 model repository](https://modelers.cn/models/MindIE/Wan2.1)    
 * [One-click Quantization Configuration Protocol Description](../../../docs/en/feature_guide/quick_quantization_v1/usage.md#configuration-protocol-overview)    
 * [Layer-by-Layer Quantization Feature Description](../../../docs/en/feature_guide/quick_quantization_v1/usage.md#layer-wise-and-dp-layer-wise-quantization)    
