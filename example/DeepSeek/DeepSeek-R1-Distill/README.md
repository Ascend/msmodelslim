# DeepSeek R1 Distill 量化说明

## 使用前准备

- 使用 MindIE1.0版本 [官方镜像](https://gitcode.com/Ascend/ascend-docker-image/tree/dev/mindie#%E5%90%AF%E5%8A%A8%E5%AE%B9%E5%99%A8)，如1.0.0-800I-A2-py311-openeuler24.03-lts

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接 | W8A8 | W8A16 | W4A8 | W8A8C8 | 稀疏量化 | KV Cache | Attention | 量化命令 |
|---------|---------|---------------------------------------------------------------|-----|-------|-------|------|---------|----------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **DeepSeek-R1-Distill-Qwen** | DeepSeek-R1-Distill-Qwen-1.5B | [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/tree/main) | ✅ |   |   |   | ✅ |   |   | [W8A8](#deepseek-r1-distill-qwen-15b-w8a8量化) / [稀疏](#deepseek-r1-distill-qwen-15b-稀疏量化) |
| | DeepSeek-R1-Distill-Qwen-7B | [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/tree/main) | ✅ |   |   |   | ✅ |   |   | [W8A8](#deepseek-r1-distill-qwen-7b-w8a8量化) / [稀疏](#deepseek-r1-distill-qwen-7b-稀疏量化) |
| | DeepSeek-R1-Distill-Qwen-14B | [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/tree/main) | ✅ |   |   |   | ✅ |   |   | [W8A8](#deepseek-r1-distill-qwen-14b-w8a8量化) / [稀疏](#deepseek-r1-distill-qwen-14b-稀疏量化) |
| | DeepSeek-R1-Distill-Qwen-32B | [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/tree/main) | ✅ |   |   |   | ✅ |   |   | [W8A8](#deepseek-r1-distill-qwen-32b-w8a8量化) / [稀疏](#deepseek-r1-distill-qwen-32b-稀疏量化) |
| **DeepSeek-R1-Distill-Llama** | DeepSeek-R1-Distill-Llama-8B | [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/tree/main) | ✅ |   |   |   | ✅ |   |   | [W8A8](#deepseek-r1-distill-llama-8b-w8a8量化) / [稀疏](#deepseek-r1-distill-llama-8b-稀疏量化) |
| | DeepSeek-R1-Distill-Llama-70B | [DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/tree/main) | ✅ |   |   |   |   |   |   | [W8A8](#deepseek-r1-distill-llama-70b-w8a8量化) |

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令
- **注意：** Atlas 300I DUO目前仅支持单片量化，请先确保待量化模型的大小可适配单片容量，以保证量化过程正常进行。

## 量化权重生成

- 如果需要使用NPU多卡量化，请先配置环境变量，支持多卡量化：

  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

- 若加载自定义模型，调用`from_pretrained`函数时要指定`trust_remote_code=True`让修改后的自定义代码文件能够正确的被加载。(请确保加载的自定义代码文件的安全性)

## 使用示例

### DeepSeek-R1-Distill-Llama 量化

#### <span id="deepseek-r1-distill-llama-8b-w8a8量化">DeepSeek-R1-Distill-Llama-8B W8A8量化</span>

Atlas 800I A2 w8a8量化

  ```shell
  cd msmodelslim/example/Llama
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --anti_method m1 --trust_remote_code True
  ```

#### <span id="deepseek-r1-distill-llama-8b-稀疏量化">DeepSeek-R1-Distill-Llama-8B 稀疏量化</span>

Atlas 300I DUO  使用以下方法稀疏量化

- 稀疏量化

```shell
  # 指定当前机器上可用的逻辑NPU核心 通过修改convert_quant_weight.sh文件中export ASCEND_RT_VISIBLE_DEVICES值 指定使用卡号及数量 
  cd msmodelslim/example/Llama
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type npu --use_sigma True --is_lowbit True --trust_remote_code True
```

- 权重压缩

```shell
  # TP数为tensor parallel并行个数
  export IGNORE_INFER_ERROR=1
  torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
```

#### <span id="deepseek-r1-distill-llama-70b-w8a8量化">DeepSeek-R1-Distill-Llama-70B W8A8量化</span>

Atlas 800I A2 w8a8量化

  ```shell
  cd msmodelslim/example/Llama
  python3 quant_llama.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_level L5 --anti_method m4 --act_method 3 --trust_remote_code True
  ```

### DeepSeek-R1-Distill-Qwen 量化

#### <span id="deepseek-r1-distill-qwen-15b-w8a8量化">DeepSeek-R1-Distill-Qwen-1.5B W8A8量化</span>

Atlas 800I A2 w8a8量化

  ```shell
  cd msmodelslim/example/Qwen
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
  ```

OrangePi

- 使用 OrangePi 推理，需要准备另外一台Atlas 800I A2 或 Atlas 300I DUO 进行w8a8量化，量化后把权重转移至香橙派上

```shell
# w8a8 量化指令
cd msmodelslim/example/Qwen
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl  --device_type npu --disable_names "lm_head" --anti_method m4 --trust_remote_code True
```

#### <span id="deepseek-r1-distill-qwen-15b-稀疏量化">DeepSeek-R1-Distill-Qwen-1.5B 稀疏量化</span>

Atlas 300I DUO 使用以下方法稀疏量化

- 稀疏量化

```shell
cd msmodelslim/example/Qwen
# 指定当前机器上可用的逻辑NPU核心 通过修改convert_quant_weight.sh文件中export ASCEND_RT_VISIBLE_DEVICES值 指定使用卡号及数量 
export ASCEND_RT_VISIBLE_DEVICES=0
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type npu --use_sigma True --is_lowbit True --trust_remote_code True
```

- 权重压缩

```shell
  # TP数为tensor parallel并行个数
  export IGNORE_INFER_ERROR=1
  torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
```

#### <span id="deepseek-r1-distill-qwen-7b-w8a8量化">DeepSeek-R1-Distill-Qwen-7B W8A8量化</span>

Atlas 800I A2 w8a8量化

  ```shell
  cd msmodelslim/example/Qwen
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
  ```

#### <span id="deepseek-r1-distill-qwen-7b-稀疏量化">DeepSeek-R1-Distill-Qwen-7B 稀疏量化</span>

Atlas 300I DUO 使用以下方法稀疏量化

- 稀疏量化

```shell
cd msmodelslim/example/Qwen
# 指定当前机器上可用的逻辑NPU核心 通过修改convert_quant_weight.sh文件中export ASCEND_RT_VISIBLE_DEVICES值 指定使用卡号及数量 
export ASCEND_RT_VISIBLE_DEVICES=0
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type npu --use_sigma True --is_lowbit True --trust_remote_code True
```

- 权重压缩

```shell
  # TP数为tensor parallel并行个数
  export IGNORE_INFER_ERROR=1
  torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
```

#### <span id="deepseek-r1-distill-qwen-14b-w8a8量化">DeepSeek-R1-Distill-Qwen-14B W8A8量化</span>

Atlas 800I A2 w8a8量化

  ```shell
  cd msmodelslim/example/Qwen
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
  ```

#### <span id="deepseek-r1-distill-qwen-14b-稀疏量化">DeepSeek-R1-Distill-Qwen-14B 稀疏量化</span>

- 稀疏量化

Atlas 300I DUO 使用以下方法稀疏量化

```shell
cd msmodelslim/example/Qwen
# 指定当前机器上可用的逻辑NPU核心 通过修改convert_quant_weight.sh文件中export ASCEND_RT_VISIBLE_DEVICES值 指定使用卡号及数量 
export ASCEND_RT_VISIBLE_DEVICES=0
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/cn_en.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type npu --use_sigma True --is_lowbit True --sigma_factor 4.0 --anti_method m4 --trust_remote_code True
```

- 权重压缩

```shell
  # TP数为tensor parallel并行个数
  export IGNORE_INFER_ERROR=1
  torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --multiprocess_num 4 --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
```

#### <span id="deepseek-r1-distill-qwen-32b-w8a8量化">DeepSeek-R1-Distill-Qwen-32B W8A8量化</span>

Atlas 800I A2 w8a8量化

  ```shell
  cd msmodelslim/example/Qwen
  python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --trust_remote_code True
  ```

#### <span id="deepseek-r1-distill-qwen-32b-稀疏量化">DeepSeek-R1-Distill-Qwen-32B 稀疏量化</span>

- 稀疏量化

Atlas 300I DUO 使用以下方法稀疏量化

```shell
cd msmodelslim/example/Qwen
# 指定当前机器上可用的逻辑NPU核心 通过修改convert_quant_weight.sh文件中export ASCEND_RT_VISIBLE_DEVICES值 指定使用卡号及数量 
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --calib_file ../common/cn_en.jsonl --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --device_type npu --use_sigma True --is_lowbit True --sigma_factor 4.0 --anti_method m4 --trust_remote_code True
```

- 权重压缩

```shell
# TP数为tensor parallel并行个数
export IGNORE_INFER_ERROR=1
torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --multiprocess_num 4 --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
```
