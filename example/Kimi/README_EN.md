# Kimi Quantification Case

## Model Introduction

Kimi-K2-Instruct-0905 is an efficient command-following language model developed by Moonshot AI. It is optimized based on the Kimi Chat 1.0 architecture and focuses on accurately understanding and executing complex human instructions. The model has outstanding performance in code generation, multi-round dialogue, logic reasoning and Chinese processing, and has strong context processing ability. Its design focuses on response quality and practicability, and is suitable for serving as the core AI engine in a variety of scenarios, such as intelligent assistants, content creation, and developer tools.

## Environment Configuration

 * For details about the environment configuration, see.[Installation guide](../../docs/zh/getting_started/install_guide.md)    .
 * For the Kimi series models, because the models are large, please complete the "Check Before Running" ([Check Before Running the Kimi](#check-before-running)    ).
 * Model Quantization has high requirements on the video memory. Therefore, ensure that the video memory of a single SIM card is greater than or equal to 64 GB.

## Supported Model Versions and Quantification Policies

| Model Series              | Model version         | HuggingFace Link                                                                  | W8A8 | W8A16 | W4A8 | W8A8C8 | W4A8C8 | sparse quantization | KV Cache | Attention | FA3 Quantification | MTP quantization | Quantization command                   |
| ------------------------- | --------------------- | --------------------------------------------------------------------------------- | ---- | ----- | ---- | ------ | ------ | ------------------- | -------- | --------- | ------------------ | ---------------- | -------------------------------------- |
| **Kimi K2-Instruct-0905** | Kimi K2-Instruct-0905 | [Kimi-K2-Instruct-0905](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905)     | ✅    |       |      |        |        |                     |          |           |                    |                  | [W8A8](#kimi-k2-instruct-0905-w8a8-quantification)     |

**Description:**

 * " indicates that the quantification policy has been officially verified by msModelSlim and has complete functions and stable performance. It is recommended that the quantification policy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements. However, the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the Quantization Command column to go to the corresponding quantization command.

## Quantized weight generation

### Use Cases

Please put the`${model_path}`And to the`${save_path}`Replace it with the actual path of the user.

#### Kimi-K2-Instruct Series

##### Check Before Running

The Kimi-K2-Instruct-0905 model is large and needs to be manually adapted. To avoid wasting time, modify the related content according to the following mandatory check items before running the script:

1. The Ascend (Ascend) does not support the flash_attn library. Some code in the modeling_deepseek.py file in the weight folder needs to be commented out during running.

   ![img.png](img.png)

2. Transformers 4.48.2 must be installed.

3. Currently, transformers do not support loading in FP8 quantization format. You need to delete the following fields from the config.json file in the weight folder:

![img_1.png](img_1.png)    

##### Kimi-K2-Instruct-0905 W8A8 Quantification

 * Generate the W8A8 quantization weight of the Kimi-K2-Instruct-0905 model, use the activation value quantization mode of the automatic mixing min-max and histogram, SmoothQuant enhanced algorithm, and perform the operation on the NPU.
    
    ```shell
    msmodelslim quant --model_path ${model_path} --save_path ${save_path} --device npu --model_type Kimi-K2-Instruct-0905 --quant_type w8a8 --trust_remote_code True
    ```
