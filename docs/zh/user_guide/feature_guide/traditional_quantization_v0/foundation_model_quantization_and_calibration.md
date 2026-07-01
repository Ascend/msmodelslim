# 大模型量化与校准

本文聚焦大模型场景，包含低显存量化、混合校准数据集与 FA3 量化。

## 1. 低显存量化

### 1.1 简介

在量化大模型的时候，显存受限或模型参数过多（如千亿级）时模型无法完整加载到显存中，量化报显存不足错误，可以启用低显存量化模式。
该模式将模型大部分模块存放于内存中，仅计算时使用NPU，可以限制显存使用。

### 1.2 注意

本文中的**显存**实际含义为**NPU片上内存**，为方便用户理解，借用**显存**的表述。

### 1.3 使用前准备

依赖版本：accelerate >= 0.28.0

### 1.4 功能介绍

<font color="red">注意：开启后量化耗时更久！！！</font>

使用transformers库的from_pretrained方法加载模型时，通过调整<font color="orange">device_map</font>和<font color="orange">
max_memory</font>参数控制模型加载时的显存和内存约束。

* <font color="orange">device_map</font>：模块设备映射，设置为auto
* <font color="orange">max_memory</font>：显存和内存限制
  * 每张NPU卡显存最大值分别设置为容量的80%，卡号使用**整数**
  * cpu内存最大值配置为总内存容量

示例如下：

```python
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    local_files_only=True,
    torch_dtype='auto',
    device_map="auto",
    max_memory={
        0: "25GiB",  # NPU0最多使用25GB显存
        1: "25GiB",  # NPU1最多使用25GB显存
        2: "25GiB",  # NPU2最多使用25GB显存
        3: "25GiB",  # NPU3最多使用25GB显存
        "cpu": "500GiB",  # 加载模型时，最多使用500GB的Host侧内存
    }
)
```

### 1.5 使用样例

[Deepseek w8a8量化示例](https://gitcode.com/Ascend/msmodelslim/blob/master/example/DeepSeek/README.md)

## 2. 混合校准数据集使用方法说明

### 2.1 简介

混合校准集接口，通过CalibrationData类混合指定的数据集，支持用户自定义数据集

### 2.2 使用前准备

前提条件：config文件，用于配置基础数据集的路径，名称包括 boolq、ceval_5_shot、gsm8k、mmlu

数据集下载链接：

- [ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam)
- [boolq](https://huggingface.co/datasets/google/boolq)
- [mmlu](https://huggingface.co/datasets/cais/mmlu)
- [gsm8k](https://huggingface.co/datasets/openai/gsm8k)

### 2.3 功能介绍

### 2.4 接口说明

请参考 [CalibrationData](../../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/CalibrationData.md)

操作步骤：

1. 如需自定义数据集，创建自定义数据集处理类，继承自DatasetProcessorBase类，并重写process_data()和verify_positive_prompt()方法
2. 实例化CalibrationData，如需正样本混合校准集，需要实例化tokenizer和model，并作为参数传入CalibrationData；否则设置为None。如需保存需要设置保存路径
3. 如有自定义数据集，通过add_custormized_dataset_processor()接口传入自定义数据集名称和处理类的实例
4. 设置样本数量，通过set_sample_size()接口
5. 设置batch_size，通过set_batch_size()接口
6. 设置随机种子，通过set_shuffle_seed()接口
7. 调用process接口运行，生成混合校准集

### 2.5 config文件示例

- 第一层为dict，key为"configurations"，value为一个list，包含多个数据集信息

- 每个数据集为一个dict，key为"dataset_name"和"dataset_path"，用来配置数据集的名称和路径

```json
{"configurations":
    [
        {
          "dataset_name": "boolq",
          "dataset_path": "./boolq/dev.jsonl"
        },
        {
          "dataset_name": "ceval_5_shot",
          "dataset_path": "./ceval_5_shot/"
        },
        {
          "dataset_name": "gsm8k",
          "dataset_path": "./gsm8k/GSM8K.jsonl"
        },
        {
          "dataset_name": "mmlu",
          "dataset_path": "./mmlu/"
        }
    ]
}
```

### 2.6 调用示例

请注意`trust_remote_code`为`True`时可能执行浮点模型权重中的代码文件，请确保浮点模型来源安全可靠。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from msmodelslim.pytorch.llm_ptq.mix_calibration.calib_select import CalibrationData
from msmodelslim.pytorch.llm_ptq.mix_calibration.dataset_processor_base import DatasetProcessorBase # 用户自定义数据集时需要引入

# 继承自DatasetProcessorBase，并重写抽象方法
class CustomizedProcessor(DatasetProcessorBase):
    def __init__(self, dataset_path, tokenizer=None, model=None):
        super().__init__(dataset_path, tokenizer, model)
        self.ori_prompts = []
        self.ori_answers = []

    def process_data(self, indexs):
        """用于获取一组样本，输出为[{"prompt": prompt1, "ans": ans1},{"prompt": prompt2, "ans": ans2}]"""
        prmpts_anses = []
        for idx in indexs:
            prmpts_anses.append({"prompt": self.ori_prompts[idx], "ans": self.ori_answers[idx]})
        return prmpts_anses

    def verify_positive_prompt(self, prompts, labels):
        """用于验证一组prompts中的正样本，labels为对应标签，输出为[{"prompt": prompt1, "ans": ans1},{"prompt": prompt2, "ans": ans2}]"""
        prpt_ans = []
        with torch.no_grad():
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=20)

            answers = []
            for idx in range(len(outputs)):
                output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                response = self.tokenizer.decode(output)
                answers.append(response)
            answers = [answer.lstrip()[0] if answer.lstrip() else "-1" for answer in answers]

            for ans, label, prmpt in zip(answers, labels, prompts):
                if ans == label:
                    prpt_ans.append({"prompt": prmpt, "ans": ans})

        return prpt_ans

MODEL_PATH = "./model"
CONFIG_PATH = "./mix_config.json"
SAVE_PATH = "./mix_dataset.json"

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_PATH,
                                          trust_remote_code=True,
                                          local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL_PATH,
                                             trust_remote_code=True,
                                             config=config,
                                             torch_dtype='auto',
                                             device_map='auto',
                                             local_files_only=True)

# 基础支持的校准集包括boolq、ceval_5_shot、gsm8k、mmlu。 customized_dataset_name为用户自定义数据集名称
# 当sample_size中设置了非config.json中配置的数据集且非用户自定义数据集名称时，会报错“Dataset {dataset_name} has no handler”
# 当sample_size为空时，返回空结果
sample_size = {"boolq": 4, "ceval_5_shot": 3, "gsm8k": 3, "mmlu": 2, "customized_dataset_name": 3}

# 用户自定义数据集
customized_dataset_path = "./customized_dataset"
customized_processor = CustomizedProcessor(customized_dataset_path, tokenizer=tokenizer, model=model)

calib_select = CalibrationData(config_path=CONFIG_PATH, save_path=SAVE_PATH, tokenizer=tokenizer, model=model)  # 若不需要正样本，tokenizer和model均置为None
calib_select.add_custormized_dataset_processor("customized_dataset_name", customized_processor)     # 该调用需在设置采样数量之前
calib_select.set_sample_size(sample_size)
calib_select.set_batch_size(4)  # 该调用仅用于设置获取正样本时的batch，不对输出产生影响， 输入为int类型
calib_select.set_shuffle_seed(1)

mixed_dataset = calib_select.process()
print(mixed_dataset)
```

### 2.7 混合校准集解析使用示例

通过get_anti_dataset()方法，以混合校准集生成的mixed_dataset为输入，输出可以应用于离群值抑制模块``AntiOutlier(model, calib_data=mixed_dataset, cfg=anti_config)``中``calib_data``的输入 <br>

通过get_calib_dataset()方法，以混合校准集生成的mixed_dataset为输入，输出可以应用于量化模块``Calibrator(model, quant_config, calib_data=mixed_dataset, disable_level='L0')``中``calib_data``的输入

```python
import torch
import torch.nn.functional as F

def get_anti_dataset(tokenizer, mixed_dataset, device="npu"):
    """用于离群值抑制的校准集"""
    anti_data = []
    for prpt_ans in mixed_dataset:
        calib_dataset = []
        calib_list = [prpt_ans["prompt"]]
        max_len = 0
        for calib_data in calib_list:
            inputs = tokenizer(calib_data, return_tensors='pt')
            calib_dataset.append(inputs.data['input_ids'].to(device))
            max_len = max(max_len, inputs.data['input_ids'].size(1))
        for i in range(len(calib_dataset)):
            calib_dataset[i] = F.pad(calib_dataset[i], (0, max_len - calib_dataset[i].size(1)), value=0)
        anti_data.append(torch.cat(calib_dataset))

    anti_dataset = []
    for data in anti_data:
        anti_dataset.append([data])

    return anti_dataset

def get_calib_dataset(tokenizer, mixed_dataset, device='npu'):
    """用于量化的校准集"""
    dataset_calib = []
    for prpt_ans in mixed_dataset:
        calib_list = [prpt_ans["prompt"]]
        calib_dataset = []
        for calib_data in calib_list:
            inputs = tokenizer(calib_data, return_tensors='pt').to(device)
            calib_dataset.append([inputs.data['input_ids']])
        dataset_calib += calib_dataset

    return dataset_calib
```

## 3. FA3量化

**Flash Attention 3（FA3）**，在KV-Cache的基础上增强了在硬件设备上的利用率，提升了整体在推理场景中的计算效率，以低精度的数据格式完成更快的处理和更少的内存占用。

### 3.1 使用前准备

前提条件参考大模型量化的[使用前准备](foundation_model_compression.md#11-使用前准备)。

说明：仅Atlas 800I A2推理产品支持FA3量化功能。当前 FA3 量化功能已完成对大语言模型 Llama3.1-70B、Qwen2.5-72B 和多模态模型 Flux.1-dev、HunyuanVideo 的验证。

### 3.2 功能介绍

### 3.3 大语言模型FA3量化关键步骤说如下

#### 3.3.1 修改modeling文件

（1）找到对应版本的modeling文件：

- 方式一：从transformers库中找到modeling文件，并复制到模型权重路径下，以便后续修改：
<br>`cp {transformers库路径}/models/{模型名称}/modeling_{model_type}.py {权重路径}/modeling_{model_type}_fa3.py`

- 方式二：通过`pip show transformers`查询transformers的`Version`，假设为`4.43.1`，则可以去transformer库里找到对应版本的`modeling_{model_type}.py`，以Qwen2.5-72B为例，4.43.1版本的modeling文件地址为[modeling_qwen2.py](https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/qwen2/modeling_qwen2.py)

- 注意：Llama3.1-70B模型需使用4.43.0及以上的modeling文件。

- `model_type`可从config文件中查询，下附Qwen2.5-72B权重config。

```python
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 8192,
  "initializer_range": 0.02,
  "intermediate_size": 29568,
  "max_position_embeddings": 32768,
  "max_window_layers": 70,
  "model_type": "qwen2",
  "num_attention_heads": 64,
  "num_hidden_layers": 80,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 131072,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 150000
}
```

（2）确定Attention：
需确定模型基于哪一个Attention进行实现，以Qwen2.5模型为例，有Qwen2Attention、Qwen2FlashAttention2和Qwen2SdpaAttention三种Attention。如未特殊指定，默认为Qwen2Attention。

（3）修改modeling文件：

- 添加引用依赖：

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer
from msmodelslim import logger
```

- 在使用的Attention处调用工具：

在init初始化处添加：

```python
self.fa_quantizer = FAQuantizer(self.config, logger)
```

在forward部分添加:

```python
query_states = self.fa_quantizer.quant(query_states, qkv="q")
key_states = self.fa_quantizer.quant(key_states, qkv="k")
value_states = self.fa_quantizer.quant(value_states, qkv="v")

```

注意：新增的query_states、key_states和value_states的量化代码需放置在`if past_key_value is not None:`代码块之后，`key_states = repeat_kv(key_states, self.num_key_value_groups)`代码块之前。如果某些attention结构（如mha）没有`key_states = repeat_kv(key_states, self.num_key_value_groups)`代码块，则将量化代码放置在`if past_key_value is not None:`代码块之后即可。

- 整体修改如下：

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer
from msmodelslim import logger

class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        ...
        # 其他未修改的代码部分
        ...

     # 新增的代码部分
        # --------------------------------------------------
     self.fa_quantizer = FAQuantizer(self.config, logger)
        # --------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        ...
        # 其他未修改的代码部分
        ...

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

      # 新增的代码部分
        # --------------------------------------------------
        query_states = self.fa_quantizer.quant(query_states, qkv="q")
        key_states = self.fa_quantizer.quant(key_states, qkv="k")
        value_states = self.fa_quantizer.quant(value_states, qkv="v")
        # --------------------------------------------------

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        ...
        # 其他未修改的代码部分
        ...

```

**注意**：部分模型在transformers的库中对其组件的依赖是采用的相对路径，在改写了modeling文件之后需要将这部分相对路径的导入依赖改成绝对路径，例如：

```python
"""
# 修改前的导入方式
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import _flash_attention_forward
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
# 以qwen模型为例
from .configuration_qwen2 import Qwen2Config
"""
# 修改后的导入方式
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
# 以qwen模型为例
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
```

（4）对config文件进行修改来指定模型加载时所使用的modeling文件。其一般形式为：

```json
"auto_map": {
"AutoModelForCausalLM": "{文件名}.{architectures[0]}"}

```

假设修改后的modeling文件名为`modeling_qwen2_fa3.py`，`architectures[0]`从config中可知为`Qwen2ForCausalLM`，则对config文件做如下修改：

```json
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
    // 新增配置
    // --------------------------------------------------
    "auto_map": {
    "AutoModelForCausalLM": "modeling_qwen2_fa3.Qwen2ForCausalLM"
  },
    // --------------------------------------------------
    ...
    // 其他未修改的代码部分
    ...
}

```

**注意**：在量化脚本里面通过transformers库对模型进行加载时，调用`from_pretrained`函数时一定要指定`trust_remote_code=True`让修改后的modeling文件能够正确的被加载。(请确保加载的modeling文件的安全性)

#### 3.3.2 配置config

`config = QuantConfig().fa_quant()`

在QuantConfig初始化中完成核心参数`(w_bit, a_bit, disable_names, disable_last_linear, dev_type, dev_id)`的配置后，如果需要使用FA量化的新特性，通过调用QuantConfig的`fa_quant` 函数完成配置。

具体的参数说明如下：

| **量化类型**                          | **需要配置的参数列表**                                       | **调用示例**                                                 |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| fa_quant(fa_amp=5) | fa_amp用于配置自动精度回退，根据想要回退的layer的数量来设置。<br>数据类型为int，默认值为0。数据取值范围是大于等于0，并且小于等于模型layer数量，如果超出模型的layer数量将会取模型的最大layer数量为回退层数。 | quant_config=QuantConfig(w_bit=8,  a_bit=8, disable_names=disable_names,dev_type='npu',dev_id=0).fa_quant(fa_amp=5)|

### 3.4 量化步骤（以Qwen2.5-7B为例）

1. 用户自行准备模型、权重文件和校准数据，将修改好的modeling文件和config放入权重目录下，本样例以Qwen2.5-7B为例，目录示例如下：

```bash

├── config.json
├── modeling_qwen2_fa3.py
├── generation_config.json
├── merges.txt
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
├── README.md
├── tokenizer_config.json
├── tokenizer.json
├── vocab.json

```

2.新建模型的量化脚本quant.py，
此处可以参考：[量化脚本（NPU）](#361-量化脚本npu)

3.启动模型量化任务，并在指定的输出目录获取模型量化参数，量化后权重文件的介绍请参见[量化后权重文件](#35-量化后权重文件)，若使用MindIE进行后续的推理部署任务，请保存为safetensors格式，具体请参见[大语言模型列表](https://www.hiascend.com/software/mindie/modellist)章节中已适配量化的模型。

### 3.5 量化后权重文件

- **npy格式**

当save_type设置为['numpy']或不设置时，量化权重会保存为npy文件，npy储存格式为字典，其中key值为各层Linear的名字，例如Qwen2.5-7B模型的transformer.encoder.layers.0.self_attention.query_key_value，value值为第0层query_key_value的Linear权重。

```bash

├── anti_fp_norm.npy  #Qwen模型已启用离群抑制功能，具体操作请参见使用离群值抑制功能，将会生成此文件。antioutlier算法生成浮点权重中的norm层权重文件，用于量化层的input和post norm的权重适配
├── deq_scale.npy    #W8A8量化的量化参数权重文件，Tensor数据类型为int64，deq_scale已针对量化算子进行数据类型转换，可直接适配算子。在量化BF16模型情况下，数据类型不会转换为int64，仍然为float32
├── fa_quant_offset.npy    #FA3量化的激活值量化偏移值参数文件，Tensor数据类型为bfloat16或float16
├── fa_quant_scale.npy   #FA3量化的激活值量化缩放因子参数文件，Tensor数据类型为bfloat16或float16
├── input_offset.npy  #W8A8量化的激活值量化偏移值权重文件，Tensor数据类型为float32
├── input_scale.npy   #W8A8量化的激活值量化缩放因子权重文件，Tensor数据类型为float32
├── quant_bias.npy   #W8A8量化的量化参数权重文件，Tensor数据类型为int32，quant_bias已考虑原始浮点模型linear层的bias值
├── quant_weight.npy  #量化权重文件，Tensor数据类型为int8

```

推理部署时读取上述文件的示例代码：quant_param_dict = np.load("xxx.npy", allow_pickle=True).item()。

- **safetensors格式**

当save_type设置为['safe_tensor']时，量化权重会保存为safetensors文件和json描述文件。

- safetensors中储存格式为字典，包含量化权重和量化不修改的浮点权重。其中量化权重的key值为各层Linear的名字加上对应权重的名字，module.weight和module.bias对应anti_fp_norm.npy，weight对应quant_weight.npy，quant_bias对应quant_bias.npy等以此类推。例如Qwen2.5-7B模型的model.layers.0.self_attn.q_proj.deq_scale对应npy格式权重中deq_scale.npy中的model.layers.0.self_attn.q_proj;

```python
# qwen模型量化生成的权重文件部分内容
{
  "model.embed_tokens.weight": tensor([...]),
  "model.layers.0.self_attn.q_proj.weight": tensor([...]),
  "model.layers.0.self_attn.q_proj.input_scale": tensor([...]),
  "model.layers.0.self_attn.q_proj.input_offset": tensor([...]),
  "model.layers.0.self_attn.q_proj.quant_bias": tensor([...]),
  "model.layers.0.self_attn.q_proj.deq_scale": tensor([...]),
  "model.layers.0.self_attn.k_proj.weight": tensor([...]),
   ...
   "model.layers.0.self_attn.fa_q.scale": tensor([...]),
   "model.layers.0.self_attn.fa_q.offset": tensor([...]),
   "model.layers.0.self_attn.fa_k.scale": tensor([...]),
   "model.layers.0.self_attn.fa_k.offset": tensor([...]),
   "model.layers.0.self_attn.fa_v.scale": tensor([...]),
   "model.layers.0.self_attn.fa_v.offset": tensor([...]),
   ...
}
```

- json描述文件中储存的量化权重的总体类型model_quant_type，是否启用FA3量化fa_quant_type，和其中各个权重的类型，来自原始浮点权重则为FLOAT，来自W8A8量化则为W8A8。

```python
{
  "model_quant_type": "W8A8",                                # 整体量化类型为W8A8量化
  "fa_quant_type": "FAQuant",                         # 量化过程开启了FA3量化
  "model.embed_tokens.weight": "FLOAT",                      # 来自原始浮点模型的embed_tokens权重
  "model.layers.0.self_attn.q_proj.weight": "W8A8",          # 量化新增的第0层self_attn.q_proj的quant_weight
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8",     # 量化新增的第0层self_attn.q_proj的input_scale
  "model.layers.0.self_attn.q_proj.input_offset": "W8A8",    # 量化新增的第0层self_attn.q_proj的input_offset
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8",      # 量化新增的第0层self_attn.q_proj的quant_bias
  "model.layers.0.self_attn.q_proj.deq_scale": "W8A8",       # 量化新增的第0层self_attn.q_proj的deq_scale
  "model.layers.0.self_attn.k_proj.weight": "W8A8",          # 量化新增的第0层self_attn.k_proj的quant_weight
   ...
   "model.layers.0.self_attn.fa_q.scale": "FAQuant",         # 量化新增的第0层self_attn的query_states的scale
   "model.layers.0.self_attn.fa_q.offset": "FAQuant",        # 量化新增的第0层self_attn的query_states的offset
   "model.layers.0.self_attn.fa_k.scale": "FAQuant",         # 量化新增的第0层self_attn的key_states的scale
   "model.layers.0.self_attn.fa_k.offset": "FAQuant",        # 量化新增的第0层self_attn的key_states的offset
   "model.layers.0.self_attn.fa_v.scale": "FAQuant",         # 量化新增的第0层self_attn的key_states的scale
   "model.layers.0.self_attn.fa_v.offset": "FAQuant",        # 量化新增的第0层self_attn的key_states的offset
   ...
}
```

### 3.6 FA3精度调优

#### 3.6.1 量化脚本（NPU）

当前 FA 量化脚本和命令可以参考 example 的相关内容。跳转链接见下表：

| 脚本文件                                          | 参考资料                                                     |
| ------------------------------------------------- | ------------------------------------------------------------ |
| [quant_qwen.py](https://gitcode.com/Ascend/msmodelslim/blob/master/example/Qwen/quant_qwen.py)    | [Qwen2.5-72B 支持Attention量化](https://gitcode.com/Ascend/msmodelslim/blob/master/example/Qwen/README.md#qwen25-72b-支持attention量化) |
| [quant_llama.py](https://gitcode.com/Ascend/msmodelslim/blob/master/example/Llama/quant_llama.py) | [Llama3.1-70B W8A8量化搭配Attention量化](https://gitcode.com/Ascend/msmodelslim/blob/master/example/Llama/README.md#llama31-70b-w8a8量化搭配attention量化) |

#### 本文仅给出FA3场景下Llama3.1-70B和Qwen2.5-72B的量化推荐配置，可按实际情况进行参数调整，详见《[精度调优策略](../../../best_practices/w8a8_accuracy_tuning_policy.md)》

#### 3.6.2 Llama3.1-70B 量化参数设置

- 离群值抑制(AntiOutlier) ：anti_method = "m3"

```python
anti_config = AntiOutlierConfig(anti_method="m3", dev_type="npu", dev_id=model.device.index)
```

- 量化参数(QuantConfig)

激活值量化方法：act_method = 3

```python
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=device_id,
    act_method=3,
    pr=1.0,
    w_sym=True,
    mm_tensor=False
).fa_quant(fa_amp=0)

calibrator = Calibrator(
    model,
    quant_config,
    calib_data=dataset_calib,
    disable_level='L5'
)
```

- 校准数据(calib_set)
使用50条左右boolq数据集校准

- 量化回退(disable_names)
（1）disable_level='L5': 自动回退5个layer
<br>

（2）回退所有down层：

```python
disable_names = []
num_layers = 80
disable_idx_lst = list(range(num_layers))
for layer_index in disable_idx_lst:
    down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
    disable_names.append(down_proj_name)

```

（3）（可选）调用fa_quant时设置回退层数。本模型无需设置该参数精度即可达标。

```python
fa_quant(fa_amp=5)
```

#### 3.6.3 Qwen2.5-72B 量化参数设置

- （可选）离群值抑制(AntiOutlier) ：无需离群值抑制精度即可达标

- 量化参数(QuantConfig)

激活值量化方法：act_method = 1

```python
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=disable_names,
    dev_type='npu',
    dev_id=device_id,
    act_method=1,
    pr=1.0,
    w_sym=True,
    mm_tensor=False
).fa_quant(fa_amp=0)

calibrator = Calibrator(
    model,
    quant_config,
    calib_data=dataset_calib,
    disable_level='L0'
)
```

- 校准数据(calib_set)
使用50条左右boolq数据集校准

- 量化回退(disable_names)
（1）（可选）disable_level='L0': 本模型设置L0精度即可达标。

<br>（2）回退所有down层：

```python
disable_names = []
num_layers = 80
disable_idx_lst = list(range(num_layers))
for layer_index in disable_idx_lst:
    down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
    disable_names.append(down_proj_name)

```

（3）（可选）调用fa_quant时设置回退层数。本模型无需设置该参数精度即可达标。

```python
fa_quant(fa_amp=5)
```

### 3.7 多模态FA3量化关键步骤说明如下

请参考[Flux FA3量化](../../../../../example/multimodal_sd/Flux/README.md#flux-fa3-quantization)与[HunyuanVideo FA3](../../../../../example/multimodal_sd/HunYuanVideo/README.md#hunyuanvideo-fa3-量化)的详细使用说明。
