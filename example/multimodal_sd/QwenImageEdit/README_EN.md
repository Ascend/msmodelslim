# Qwen Image Edit Quantification Instructions

## Qwen Image Edit Model

[Qwen-Image-Edit](https://github.com/QwenLM/Qwen-Image)    It is an open source image editing model launched by Alibaba Tongyi Qianwen team based on the Qwen-Image image basic model, taking into account semantic changes. (e.g. style, composition, object addition, deletion and replacement) Controls appearance details and supports precise modification of text in Chinese and English images. Current msModelSlim One-Click Quantization for Qwen-Image-Edit-2509 Weight vs.[MindIE/Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509)    Inference engineering interconnection.

## Preparation Before Use

 * Install the msModelSlim tool. For details, see.[msModelSlim Installation Guide](../../../docs/en/getting_started/install_guide.md)    .
 * For details about the floating-point inference environment and dependency, see.[Magic Music Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509)    and the[README](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509/blob/main/README.md)    Ensure that floating-point inference can be completed normally before quantization (loaded from the inference engineering repository).`qwenimage_edit`modules, etc.).

## Supported Model Versions and Quantification Policies

| Model Series        | Model version        | Model Repository Link                                                           | W8A8 | W8A16 | W4A16 | W4A4 | time step quantization | FA3 Quantification | outlier suppression quantization | Quantization command                                                |
| ------------------- | -------------------- | ------------------------------------------------------------------------------- | ---- | ----- | ----- | ---- | ---------------------- | ------------------ | -------------------------------- | ------------------------------------------------------------------- |
| **Qwen-Image-Edit** | Qwen-Image-Edit-2509 | [Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509)     | ✅    |       |       |      |                        | ✅                  |                                  | [FA3+W8A8 dynamic quantization](#qwen-image-edit-2509-fa3w8a8-dynamic-quantization)     |

**Description:**

 * ' indicates that the quantification policy has been officially verified by msModelSlim and has complete functions and stable performance. It is recommended that the quantification policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * You can click the link in the Quantization Command column to go to the corresponding quantization command.

## Qwen Image Edit Quantification Support

The transformer part of Qwen-Image-Edit-2509 is based on the diffusion and transformer structure. The msModelSlim supports the quantization of the linear layer and works with the online_quarot and FA3 processes. Supports layer-by-layer quantization, which helps reduce memory usage during quantization.

### quantization characteristics

 * **Layer-by-layer quantization: supports layer-by-layer processing, greatly reducing memory usage.**
 * **Single-card quantization: With the layer-by-layer quantization feature, single-card quantization can be implemented on the Atlas 800I/800T A2 (64G) device.**

## Quantization command

### Qwen-Image-Edit-2509 FA3+W8A8 dynamic quantization

#### Use the config_path parameter to specify the configuration file for one-click quantization

```bash
msmodelslim quant \
    --model_path /path/to/Qwen-Image-Edit-2509 \
    --save_path /path/to/qwen_image_edit_quantized_weights \
    --device npu \
    --model_type Qwen-Image-Edit-2509 \
    --config_path lab_practice/qwen_image_edit/qwen-image-edit-w8a8f8-mxfp.yaml \
    --trust_remote_code True
```

### One-click quantization command parameters

For the basic description of one-click quantization parameters, see the following:[One-click Quantization Parameters](../../../docs/en/feature_guide/quick_quantization_v1/usage.md#parameters)    .

Added the following information about Qwen-Image-Edit-2509:

| Parameter Name    | To explain                                                                                                                                          | Optional or not                                             | Scope                                                                                                                                                                                                                                                                                                                                                |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_path        | Floating Point Weight Directory (Required`transformer`Subdirectories and pipeline files, which are consistent with the MindIE inference repository) | Mandatory.                                                  | Type: Str                                                                                                                                                                                                                                                                                                                                            |
| save_path         | Path for storing quantization weights.                                                                                                              | Mandatory.                                                  | Type: Str                                                                                                                                                                                                                                                                                                                                            |
| device            | quantization equipment                                                                                                                              | Mandatory.                                                  | 1. Type: Str 2. Only supported`npu`                                                                                                                                                                                                                                                                                                                  |
| model_type        | Model name                                                                                                                                          | Mandatory.                                                  | 1. Type: Str. 2. Case-sensitive. Set this parameter to`qwen_image_edit`                                                                                                                                                                                                                                                                              |
| config_path       | Specify the configuration path.                                                                                                                     | To be associated with the`quant_type`Choose one of the two  | 1. Type: Str. 2. The configuration file format is YAML. 3. You are advised to use the verified configuration in the best practice library.[qwen-image-edit-w8a8f8-mxfp.yaml](../../../lab_practice/qwen_image_edit/qwen-image-edit-w8a8f8-mxfp.yaml)    If the configuration is customized, msModelSlim is not responsible for the quantization result. |
| quant_type        | Quantization type                                                                                                                                   | To be associated with the`config_path`Choose one of the two | 1. Type: Str. 2. Currently, the Qwen-Image-Edit model supports only config_path.                                                                                                                                                                                                                                                                     |
| trust_remote_code | Trust Custom Code                                                                                                                                   | Optional.                                                   | Type: Bool. The default value is False. When loading custom code, it is recommended that you set this parameter to`True`(Ensure that the code source is reliable.)                                                                                                                                                                                   |

## Configuration File Description

### Basic Configuration Structure

The following structures and warehouses[qwen-image-edit-w8a8f8-mxfp.yaml](../../../lab_practice/qwen_image_edit/qwen-image-edit-w8a8f8-mxfp.yaml)    Consistent and easy to understand the meaning of each paragraph:

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

### Key Configuration Parameters

#### Quantified configuration (process)

 * **linear_quant: performs W8A8 (mxfp8) dynamic quantization on the linear layer.**`exclude`Medium mode is used to exclude some submodules to stabilize precision.
 * **online_quarot: configuration related to online rotation, which works with the attention module.**
 * **fa3_quant: FP8 quantization configuration on the Flash Attention 3 path (e.g.**`fp8_e4m3`,`per_token`).

#### Saving Configurations (save)

 * **type: saver type.**`mindie_format_saver`.
 * **part_file_size: fragment size,**`0`Indicates that the data is not fragmented.

#### Multimodal Configuration (multimodal_sd_config)

 * **dump_config: calibration data export; In the current default example**`enable_dump`To the`False`. If the calibration dump function is enabled later, the one-click quantization protocol and adapter must be used to configure the calibration dump function.
 * **model_config: placeholder field that can be aligned with the inference parameter, for example:**
    
     * `img_paths`\: Input image path (Multi-maps can be separated by commas, and the reasoning warehouse prevails.)
     * `prompt_file`\: indicates the path of the prompt word file.

For details about the protocol, see:[One-Click Quantization Configuration Protocol Description](../../../docs/en/feature_guide/quick_quantization_v1/usage.md#configuration-protocol-overview)    .

## FAQ

**Symptom: An error is reported during quantification and cannot be imported.`qwenimage_edit`.**
Solution: Please press[MindIE/Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509)    Place the inference project in the Python path or install it as described so that`qwenimage_edit.transformer_qwenimage`,`qwenimage_edit.pipeline_qwenimage_edit_plus`The data can be imported.

**Symptom: How do I customize the quantization configuration?**
Solution: Available in`process`Medium adjustment`exclude`/`include`1. Quantify the dtype and scope. The precision and compatibility of customized configurations need to be verified. The official system only provides assurance for verified configurations in the best practice library.

## Appendixes

### Related Resources

 * [Qwen-Image-Edit-2509 (Hugging Face)](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)    
 * [Qwen-Image-Edit-2509 Model Repository](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509)    
 * [One-click Quantization Configuration Protocol Description](../../../docs/en/feature_guide/quick_quantization_v1/usage.md#configuration-protocol-overview)    
 * [Layer-by-Layer Quantization Feature Description](../../../docs/en/feature_guide/quick_quantization_v1/usage.md#layer-wise-and-dp-layer-wise-quantization)    
