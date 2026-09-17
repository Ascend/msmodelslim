# Qwen3-MOE 量化说明

## 模型介绍

- [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)、[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) Qwen3 是 Qwen 系列中最新一代大型语言模型，提供全面的密集模型和混合专家 (MoE) 模型。Qwen3 基于丰富的训练经验，在推理、指令遵循、代理能力和多语言支持方面取得了突破性进展。其中Qwen3-MoE结构模型典型代表模型有Qwen3-235B-A22B和Qwen3-30B-A3B。

## 使用前准备

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../docs/zh/install_guide/install_guide.md)。
- transformers版本需要配置安装4.57.6版本
    - pip install transformers==4.57.6

## 支持的模型版本与量化策略

| 模型系列 | 模型版本 | HuggingFace链接                                                 | W8A8 | W8A16 | W4A8 | W4A16 | W4A4  | 稀疏量化 | KV Cache | Attention | 量化命令                                          |
|---------|---------|---------------------------------------------------------------|-----|-----|-----|--------|------|---------|----------|-----------|-----------------------------------------------|
| **Qwen3-MOE** | Qwen3-30B-A3B | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)   | ✅ |  |  ✅   |        |   |  |   |   | [W8A8](#qwen3-30b-a3b-w8a8-混合量化) /[W4A8](#qwen3-30b-a3b-w4a8-混合量化)               |
| | Qwen3-235B-A22B | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) | ✅ |   |  ✅ |   |   |  |   |   | [W8A8](#qwen3-235b-a22b-w8a8-混合量化) /[W4A8](#qwen3-235b-a22b-w4a8-混合量化) |
| | Qwen3-Coder-480B-A35B | [Qwen3-Coder-480B-A35B](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) |  |   |  ✅ |   |   |  |   |   | [W4A8](#qwen3-coder-480b-a35b-w4a8-混合量化) |

**说明：**

- ✅ 表示该量化策略已通过msModelSlim官方验证，功能完整、性能稳定，建议优先采用。
- 空格表示该量化策略暂未通过msModelSlim官方验证，用户可根据实际需求进行配置尝试，但量化效果和功能稳定性无法得到官方保证。
- 点击量化命令列中的链接可跳转到对应的具体量化命令

## 量化权重生成

- 量化权重可使用[quant_qwen_moe_w8a8.py](./quant_qwen_moe_w8a8.py)脚本生成。

## 使用示例

- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用NPU多卡量化，请先配置环境变量，支持多卡量化：

  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```

### Qwen3-30B-A3B

#### <span id="qwen3-30b-a3b-w8a8-混合量化">Qwen3-30B-A3B W8A8混合量化</span>

生成Qwen3-30B-A3B模型W8A8混合量化权重（Attention:w8a8量化，MoE:w8a8 dynamic量化）

  ```shell
  python3 quant_qwen_moe_w8a8.py --model_path {浮点权重路径} \
  --save_path {W8A8量化权重路径} \
  --anti_dataset ../common/qwen3-moe_anti_prompt_50.json \
  --calib_dataset ../common/qwen3-moe_calib_prompt_50.json \
  --trust_remote_code True
  ```

#### <span id="qwen3-30b-a3b-w4a8-混合量化">Qwen3-30B-A3B W4A8混合量化</span>

生成Qwen3-30B-A3B模型W4A8混合量化权重（Attention:w8a8 dynamic量化，MoE:w4a8 dynamic量化）

  ```shell
  msmodelslim quant --model_type Qwen3-30B --model_path {浮点权重路径} --save_path {W4A8量化权重路径} --quant_type w4a8 --trust_remote_code true
  ```

### Qwen3-235B-A22B

#### <span id="qwen3-235b-a22b-w8a8-混合量化">Qwen3-235B-A22B W8A8混合量化</span>

生成Qwen3-235B-A22B模型W8A8混合量化权重（Attention:w8a8量化，MoE:w8a8 dynamic量化）

  ```shell
  python3 quant_qwen_moe_w8a8.py --model_path {浮点权重路径} \
  --save_path {W8A8量化权重路径} \
  --anti_dataset ../common/qwen3-moe_anti_prompt_50.json \
  --calib_dataset ../common/qwen3-moe_calib_prompt_50.json \
  --trust_remote_code True \
  --rot
  ```

#### <span id="qwen3-235b-a22b-w4a8-混合量化">Qwen3-235B-A22B W4A8混合量化</span>

生成Qwen3-235B-A22B模型W4A8混合量化权重（Attention:w8a8 dynamic量化，MoE:w4a8 dynamic量化）

  ```shell
  msmodelslim quant --model_type Qwen3-235B --model_path {浮点权重路径} --save_path {W4A8量化权重路径} --quant_type w4a8 --trust_remote_code true
  ```

### Qwen3-Coder-480B-A35B

#### <span id="qwen3-coder-480b-a35b-w4a8-混合量化">Qwen3-Coder-480B-A35B W4A8混合量化</span>

生成Qwen3-Coder-480B-A35B模型W4A8混合量化权重（Attention:w8a8 量化，MoE:w4a8 dynamic量化）

  ```shell
  msmodelslim quant --model_type  Qwen3-Coder-480B-A35B --model_path {浮点权重路径} --save_path {W4A8量化权重路径} --quant_type w4a8 --trust_remote_code true
  ```

## 附录

### quant_qwen_moe_w8a8.py 量化参数说明

| 参数名           | 含义           | 默认值  | 使用方法                          |
|---------------|--------------|------|-------------------------------|
| model_path    | 浮点权重路径       | 无默认值 | 必选参数；<br>输入Qwen3-MOE权重目录路径。    |
| save_path     | 量化权重路径       | 无默认值 | 必选参数；<br>输出量化结果目录路径。          |
| layer_count   | 模型层数       | 0 | 可选参数；<br>用于调试，实际量化的层数。0表示使用所有层。        |
| anti_dataset  | 反异常值校准数据集路径 | ../common/qwen3-moe_anti_prompt_50.json | 可选参数；<br>用于反异常值处理的校准数据集路径。 |
| calib_dataset | 量化校准数据集路径   | ../common/qwen3-moe_calib_prompt_50.json | 可选参数；<br>量化校准集路径。             |
| batch_size     | 输入batch size | 4  | 可选参数；<br>生成量化校准数据时使用的batch size。batch size越大，校准速度越快，但也要求更多的显存和内存，如资源受限，请降低batch size。  |
| mindie_format | 是否开启旧的权重配置文件保存格式 | False | 开启`mindie_format`时保存的量化权重格式能够兼容MindIE 2.1.RC1及之前的版本。 |
| trust_remote_code | 是否信任自定义代码 | False | 指定`trust_remote_code=True`让修改后的自定义代码文件能够正确地被加载(请确保所加载的自定义代码文件来源可靠，避免潜在的安全风险)。 |
| rot | 开启基于旋转矩阵的预处理 | 不开启 | 可选参数；开启即指定。 |

注：在量化脚本里面通过transformers库对模型进行加载时，调用`from_pretrained`函数时会指定`trust_remote_code=True`让修改后的modeling文件能够正确地被加载(请确保所加载的modeling文件来源可靠，避免潜在的安全风险)。

更多参数配置要求，请参考量化过程中配置的参数 [QuantConfig](../../docs/zh/api_reference/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_QuantConfig.md)
以及量化参数配置类 [Calibrator](../../docs/zh/api_reference/python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/pytorch_Calibrator.md)
