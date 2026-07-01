# Quantification Description of DeepSeek R1 Distill

## Preparation Before Use

MindIE 1.0 is used.[Official image](https://gitcode.com/Ascend/ascend-docker-image/tree/dev/mindie#%E5%90%AF%E5%8A%A8%E5%AE%B9%E5%99%A8), e.g. 1.0.0-800I-A2-py311-openeuler24.03-lts

## Supported Model Versions and Quantification Policies

| Model Series                  | Model version                 | HuggingFace Link                                                                                             | W8A8 | W8A16 | W4A8 | W8A8C8 | sparse quantization | KV Cache | Attention | Quantization command                                                                            |
| ----------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------ | ---- | ----- | ---- | ------ | ------------------- | -------- | --------- | ----------------------------------------------------------------------------------------------- |
| **DeepSeek-R1-Distill-Qwen** | DeepSeek-R1-Distill-Qwen-1.5B | [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/tree/main)| ✅| ||| ✅||| [W8A8](#deepseek-r1-distill-qwen-15b-w8a8-quantization)/[sparseness](#deepseek-r1-distill-qwen-15b-sparse-quantization)|
|| DeepSeek-R1-Distill-Qwen-7B|[DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/tree/main)| ✅|||| ✅||| [W8A8](#deepseek-r1-distill-qwen-7b-w8a8-quantization)/[sparseness](#deepseek-r1-distill-qwen-7b-sparse-quantization)|
|| DeepSeek-R1-Distill-Qwen-14B  | [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/tree/main)| ✅    |       |      |        | ✅                   |          |           | [W8A8](#deepseek-r1-distill-qwen-14b-w8a8-quantization)/[sparseness](#deepseek-r1-distill-qwen-14b-sparse-quantization)|
|                               | DeepSeek-R1-Distill-Qwen-32B| [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/tree/main)| ✅    |       |      |        | ✅                   |          |           | [W8A8](#deepseek-r1-distill-qwen-32b-w8a8-quantization)/[sparseness](#deepseek-r1-distill-qwen-32b-sparse-quantization)|
| **DeepSeek-R1-Distill-Llama** | DeepSeek-R1-Distill-Llama-8B  | [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/tree/main)| ✅    |       |      |        | ✅                   |          |           | [W8A8](#deepseek-r1-distill-llama-8b-w8a8-quantization)/[sparseness](#deepseek-r1-distill-llama-8b-sparse-quantization)|
|                               | DeepSeek-R1-Distill-Llama-70B | [DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/tree/main)| ✅    |       |      |        |                     |          |           | [W8A8](#deepseek-r1-distill-llama-70b-w8a8-quantization)|

**Description:**

 * " indicates that the quantitative strategy has passed the official verification of msModelSlim. The functions are complete and the performance is stable. It is recommended that the quantitative strategy be used preferentially.
 * A space indicates that the quantification policy has not been officially verified by msModelSlim. You can configure the policy based on actual requirements, but the quantification effect and function stability cannot be officially guaranteed.
 * Click the link in the quantization command column to go to the specific quantization command.
 * **Note: Currently, the Atlas 300I DUO supports only monolithic quantization. Ensure that the size of the model to be quantized can adapt to the single-chip capacity to ensure that the quantization process is normal.**

## Quantized weight generation

 * To use NPU multi-card quantization, configure environment variables first to support multi-card quantization.
    
    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    ```

 * To load a custom model, call`from_pretrained`To be specified when the function`trust_remote_code=True`The modified custom code file can be loaded correctly. (Ensure the security of the loaded custom code file.)

## Use Example

### DeepSeek-R1-Distill-Llama Quantification

#### DeepSeek-R1-Distill-Llama-8B W8A8 Quantization

Atlas 800I A2 W8A8 Quantification

```shell
cd msmodelslim/example/Llama
python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --anti_method m1 --trust_remote_code True
```

#### DeepSeek-R1-Distill-Llama-8B sparse quantization

The Atlas 300I DUO uses the following sparse quantization method

 * sparse quantization

```shell
#Specify the available logical NPU cores on the current host. Modify the value of export ASCEND_RT_VISIBLE_DEVICES in the convert_quant_weight.sh file to specify the card number and quantity. 
  cd msmodelslim/example/Llama
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type npu --use_sigma True --is_lowbit True --trust_remote_code True
```

 * Weight Compression

```shell
#The number of TPs is the number of tensor parallel parallel devices.
  export IGNORE_INFER_ERROR=1
  torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
```

#### DeepSeek-R1-Distill-Llama-70B W8A8 Quantization

Atlas 800I A2 W8A8 Quantification

```shell
cd msmodelslim/example/Llama
python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L5 --anti_method m4 --act_method 3 --trust_remote_code True
```

### DeepSeek-R1-Distill-Qwen Quantification

#### DeepSeek-R1-Distill-Qwen-1.5B W8A8 Quantization

Atlas 800I A2 W8A8 Quantification

```shell
cd msmodelslim/example/Qwen
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
```

OrangePi

 * To use the OrangePi inference, another Atlas 800I A2 or Atlas 300I DUO needs to be prepared for W8a8 quantization. After the quantization, the weight is transferred to the OrangePi.

```shell
#W8a8 quantization instruction
cd msmodelslim/example/Qwen
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_names "lm_head" --anti_method m4 --trust_remote_code True
```

#### DeepSeek-R1-Distill-Qwen-1.5B sparse quantization

The Atlas 300I DUO uses the following sparse quantization method

 * sparse quantization

```shell
cd msmodelslim/example/Qwen
#Specify the available logical NPU cores on the current host. Modify the value of export ASCEND_RT_VISIBLE_DEVICES in the convert_quant_weight.sh file to specify the card number and quantity. 
export ASCEND_RT_VISIBLE_DEVICES=0
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type npu --use_sigma True --is_lowbit True --trust_remote_code True
```

 * Weight Compression

```shell
#The number of TPs is the number of tensor parallel parallels.
  export IGNORE_INFER_ERROR=1
  torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
```

#### DeepSeek-R1-Distill-Qwen-7B W8A8 Quantization

Atlas 800I A2 W8A8 Quantification

```shell
cd msmodelslim/example/Qwen
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
```

#### DeepSeek-R1-Distill-Qwen-7B Sparse Quantization

The Atlas 300I DUO uses the following sparse quantization method

 * sparse quantization

```shell
cd msmodelslim/example/Qwen
#Specify the available logical NPU cores on the current host. Modify the value of export ASCEND_RT_VISIBLE_DEVICES in the convert_quant_weight.sh file to specify the card number and number. 
export ASCEND_RT_VISIBLE_DEVICES=0
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type npu --use_sigma True --is_lowbit True --trust_remote_code True
```

 * Weighted Compression

```shell
#The number of TPs is the number of tensor parallel parallel devices.
  export IGNORE_INFER_ERROR=1
  torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
```

#### DeepSeek-R1-Distill-Qwen-14B W8A8 Quantization

Atlas 800I A2 W8A8 Quantification

```shell
cd msmodelslim/example/Qwen
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
```

#### DeepSeek-R1-Distill-Qwen-14B Sparse Quantization

 * sparse quantization

The Atlas 300I DUO uses the following sparse quantization method

```shell
cd msmodelslim/example/Qwen
#Specify the available logical NPU cores on the current host. Modify the value of export ASCEND_RT_VISIBLE_DEVICES in the convert_quant_weight.sh file to specify the card number and quantity. 
export ASCEND_RT_VISIBLE_DEVICES=0
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/cn_en.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type npu --use_sigma True --is_lowbit True --sigma_factor 4.0 --anti_method m4 --trust_remote_code True
```

 * Weight Compression

```shell
#The number of TPs is the number of tensor parallel parallel devices.
  export IGNORE_INFER_ERROR=1
  torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --multiprocess_num 4 --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
```

#### DeepSeek-R1-Distill-Qwen-32B W8A8 Quantization

Atlas 800I A2 W8A8 Quantification

```shell
cd msmodelslim/example/Qwen
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
```

#### DeepSeek-R1-Distill-Qwen-32B sparse quantization

 * sparse quantization

The Atlas 300I DUO uses the following sparse quantization method

```shell
cd msmodelslim/example/Qwen
#Specify the available logical NPU cores on the current host. Modify the value of export ASCEND_RT_VISIBLE_DEVICES in the convert_quant_weight.sh file to specify the card number and quantity. 
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/cn_en.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type npu --use_sigma True --is_lowbit True --sigma_factor 4.0 --anti_method m4 --trust_remote_code True
```

 * Weight Compression

```shell
#The number of TPs is the number of tensor parallel parallel devices.
export IGNORE_INFER_ERROR=1
torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --multiprocess_num 4 --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
```
