# Foundation Model Quantization and Calibration

This document focuses on foundation model scenarios, including low-memory quantization, mixed calibration datasets, and FA3 quantization.

## 1. Low-Memory Quantization

### 1.1 Overview

During the quantization of a foundation model, if the model cannot be fully loaded due to constrained hardware resources or excessive parameters (such as hundreds of billions), the quantization process throws an out-of-memory error. To resolve this, you can enable low-memory quantization mode.
This mode keeps most model modules stored in the host memory and offloads execution to the NPU only during computation, minimizing NPU memory consumption.

### 1.2 Note

In this document, **NPU memory** specifically refers to **NPU on-chip memory**. The **NPU memory** is used throughout this text to ensure general readability.

### 1.3 Preparations

Dependency version: `accelerate >= 0.28.0`

### 1.4 Function

<font color="red">Note: Enabling this feature significantly increases quantization processing time.</font>

When loading a model by using the `from_pretrained` method of the `transformers` library, you can adjust the <font color="orange">device_map</font> and <font color="orange">
max_memory</font> parameters to regulate NPU memory and host memory constraints.

* <font color="orange">device_map</font>: module device mapping. Set it to `auto`.
* <font color="orange">max_memory</font>: NPU memory and Host memory constraints.
  * Set the maximum allocation limit for each individual NPU to 80% of its total capacity. Specify card IDs using **integers**.
  * Configure the maximum CPU limit to match your total available host memory capacity.

Example:

```python
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    local_files_only=True,
    torch_dtype='auto',
    device_map="auto",
    max_memory={
        0: "25GiB",  # NPU0 can use a maximum of 25 GiB NPU memory.
        1: "25GiB",  # NPU1 can use a maximum of 25 GiB NPU memory.
        2: "25GiB",  # NPU2 can use a maximum of 25 GiB NPU memory.
        3: "25GiB",  # NPU3 can use a maximum of 25 GiB NPU memory.
        "cpu": "500GiB",  # The system uses a maximum of 500 GiB host memory when loading the model.
    }
)
```

### 1.5 Usage Examples

[Deepseek w8a8 Quantization Examples](https://gitcode.com/Ascend/msmodelslim/blob/master/example/DeepSeek/README.md)

## 2. Usage of Mixed Calibration Datasets

### 2.1 Overview

The mixed calibration dataset interface mixes specified datasets using the `CalibrationData` class and supports user-defined custom datasets.

### 2.2 Preparations

Prerequisites: A configuration file has been provided to specify the paths of the basic datasets. Supported dataset names include `boolq`, `ceval_5_shot`, `gsm8k`, and `mmlu`.

Dataset download links:

- [ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam)
- [boolq](https://huggingface.co/datasets/google/boolq)
- [mmlu](https://huggingface.co/datasets/cais/mmlu)
- [gsm8k](https://huggingface.co/datasets/openai/gsm8k)

### 2.3 Function

### 2.4 API Description

For details, see [CalibrationData](../../../python_api_v0/foundation_model_compression_apis/foundation_model_quantization_apis/CalibrationData.md).

Procedure:

1. To use a custom dataset, create a custom dataset processing class that inherits from the `DatasetProcessorBase` class, then override the `process_data()` and `verify_positive_prompt()` methods.
2. Instantiate `CalibrationData`. If a positive-sample mixed calibration dataset is required, instantiate the `tokenizer` and `model` to pass them as arguments to `CalibrationData`. Otherwise, set these parameters to `None`. To save the output, specify the directory to save the file.
3. When using a custom dataset, call the `add_custormized_dataset_processor()` API to pass the custom dataset name and the instance of the processing class.
4. Set the sample size by calling the `set_sample_size()` API.
5. Set the batch size by calling the `set_batch_size()` API.
6. Set the random seed by calling the `set_shuffle_seed()` API.
7. Call the `process` API to execute the pipeline and generate the mixed calibration dataset.

### 2.5 `config` File Example

- The top-level structure is a `dict`, where the key is `"configurations"` and the value is a `list` containing metadata for multiple datasets.

- Each dataset entry is represented as a `dict`, using the keys `"dataset_name"` and `"dataset_path"` to configure the respective dataset identity and path.

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

### 2.6 Example

If `trust_remote_code` is set to `True`, code files in the weights directory of the floating-point model may be executed. Ensure that the source of the floating-point model is secure and reliable.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from msmodelslim.pytorch.llm_ptq.mix_calibration.calib_select import CalibrationData
from msmodelslim.pytorch.llm_ptq.mix_calibration.dataset_processor_base import DatasetProcessorBase # Required only for custom dataset processing.

# Inherit from DatasetProcessorBase and override abstract methods.
class CustomizedProcessor(DatasetProcessorBase):
    def __init__(self, dataset_path, tokenizer=None, model=None):
        super().__init__(dataset_path, tokenizer, model)
        self.ori_prompts = []
        self.ori_answers = []

    def process_data(self, indexs):
        """Retrieves a group of samples. Output format: [{"prompt": prompt1, "ans": ans1},{"prompt": prompt2, "ans": ans2}]"""
        prompts_anses = []
        for idx in indexs:
            prompts_anses.append({"prompt": self.ori_prompts[idx], "ans": self.ori_answers[idx]})
        return prompts_anses

    def verify_positive_prompt(self, prompts, labels):
        """Validates positive samples within a group of prompts. Output format: [{"prompt": prompt1, "ans": ans1},{"prompt": prompt2, "ans": ans2}]"""
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

            for ans, label, prompt in zip(answers, labels, prompts):
                if ans == label:
                    prpt_ans.append({"prompt": prompt, "ans": ans})

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

# Basic supported calibration datasets: boolq, ceval_5_shot, gsm8k, and mmlu. The key "customized_dataset_name" maps to user-defined datasets.
# Specifying a dataset name in sample_size that is missing from mix_config.json and not defined as a custom dataset triggers a "Dataset {dataset_name} has no handler" error.
# If sample_size is empty, an empty result is returned.
sample_size = {"boolq": 4, "ceval_5_shot": 3, "gsm8k": 3, "mmlu": 2, "customized_dataset_name": 3}

# User-defined dataset
customized_dataset_path = "./customized_dataset"
customized_processor = CustomizedProcessor(customized_dataset_path, tokenizer=tokenizer, model=model)

calib_select = CalibrationData(config_path=CONFIG_PATH, save_path=SAVE_PATH, tokenizer=tokenizer, model=model)  # Set tokenizer and model to None if positive sample filtering is not required.
calib_select.add_custormized_dataset_processor("customized_dataset_name", customized_processor)     # Call this API before setting the sample size.
calib_select.set_sample_size(sample_size)
calib_select.set_batch_size(4)  # Specifies the batch size for retrieving positive samples. This setting does not affect the output. The input must be of type int.
calib_select.set_shuffle_seed(1)

mixed_dataset = calib_select.process()
print(mixed_dataset)
```

### 2.7 Usage Examples for Parsing Mixed Calibration Datasets

The `get_anti_dataset()` method accepts the generated `mixed_dataset` as an input and returns a parsed dataset compatible with the `calib_data` parameter of the `AntiOutlier(model, calib_data=mixed_dataset, cfg=anti_config)` API.<br>

The `get_calib_dataset()` method accepts the generated `mixed_dataset` as an input and returns a parsed dataset compatible with the `calib_data` parameter of the `Calibrator(model, quant_config, calib_data=mixed_dataset, disable_level='L0')` API.

```python
import torch
import torch.nn.functional as F

def get_anti_dataset(tokenizer, mixed_dataset, device="npu"):
    """Calibration dataset for outlier suppression"""
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
    """Calibration dataset for quantization"""
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

## 3. FA3 Quantization

**Flash Attention 3 (FA3)** enhances hardware utilization on top of KV-Cache, boosting overall computing efficiency in inference scenarios. It utilizes low-precision data formats to achieve faster execution speed and a reduced memory footprint.

### 3.1 Preparations

For details about the prerequisites, see [Preparations](foundation_model_compression.md#11-preparations) in "Foundation Model Quantization".

Note: Only Atlas 800I A2 inference products support FA3 quantization. Currently, the FA3 quantization capability has been verified for the large language models `Llama3.1-70B` and `Qwen2.5-72B`, and the multimodal models `Flux.1-dev` and `HunyuanVideo`.

### 3.2 Function

### 3.3 Key Steps for Large Language Model FA3 Quantization

#### 3.3.1 Modifying the Modeling File

(1) Locate the modeling file matching your specific framework version.

- Method 1: Locate the modeling file within your installed `transformers` library, then copy it to your model weight directory for subsequent modification.
<br>`cp {transformers library path}/models/{model_name}/modeling_{model_type}.py {weight_directory}/modeling_{model_type}_fa3.py`

- Method 2: Execute `pip show transformers` to check your `transformers` framework version. Assume that the version is `4.43.1`, locate the matching `modeling_{model_type}.py` file in the official Transformers repository. For example, the modeling file path for `Qwen2.5-72B` version 4.43.1 is available at [modeling_qwen2.py](https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/qwen2/modeling_qwen2.py).

- Note: The `Llama3.1-70B` model requires a modeling file corresponding to version 4.43.0 or later.

- The `model_type` can be extracted from the model configuration file. The configuration dictionary for `Qwen2.5-72B` weights is provided as follows:

```json
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

(2) Identify the target attention.
Determine the exact attention implementation utilized by the target architecture. For example, the `Qwen2.5` model contains three distinct attention modules: `Qwen2Attention`, `Qwen2FlashAttention2`, and `Qwen2SdpaAttention`. If no explicit implementation is designated during loading, the system applies `Qwen2Attention` by default.

(3) Modify the modeling file.

- Import the required dependencies:

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer
from msmodelslim import logger
```

- Call the tool within the attention mechanism being used:

Add the following content to the `__init__` initialization block:

```python
self.fa_quantizer = FAQuantizer(self.config, logger)
```

Add the following content to the `forward` block:

```python
query_states = self.fa_quantizer.quant(query_states, qkv="q")
key_states = self.fa_quantizer.quant(key_states, qkv="k")
value_states = self.fa_quantizer.quant(value_states, qkv="v")

```

Note: Place the quantization hooks for `query_states`, `key_states`, and `value_states` after the `if past_key_value is not None:` block, and before the `key_states = repeat_kv(key_states, self.num_key_value_groups)` block. If an attention variant (such as standard `MHA`) lacks the `key_states = repeat_kv(key_states, self.num_key_value_groups)` block, position the quantization code directly after the `if past_key_value is not None:` block.

- The overall modification is as follows:

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
        # Other unchanged code
        ...

     # New code
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
        # Other unchanged code
        ...

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

      # New code
        # --------------------------------------------------
        query_states = self.fa_quantizer.quant(query_states, qkv="q")
        key_states = self.fa_quantizer.quant(key_states, qkv="k")
        value_states = self.fa_quantizer.quant(value_states, qkv="v")
        # --------------------------------------------------

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        ...
        # Other unchanged code
        ...

```

**Note**: Some component dependencies within the `transformers` library rely on relative path imports. After modifying the target modeling file, you must convert these relative imports into absolute paths. For example:

```python
"""
# Import method before modification
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
# Qwen model example
from .configuration_qwen2 import Qwen2Config
"""
# Import method after modification
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
# Qwen model example
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
```

(4) Modify the configuration file to map the custom modeling file used during model loading. The general format is as follows:

```json
"auto_map": {
"AutoModelForCausalLM": "{file_name}.{architectures[0]}"}

```

If the modified modeling file is named `modeling_qwen2_fa3.py` and the `architectures[0]` entry within the configuration file maps to `Qwen2ForCausalLM`, apply the following configuration changes:

```json
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
    // New configuration
    // --------------------------------------------------
    "auto_map": {
    "AutoModelForCausalLM": "modeling_qwen2_fa3.Qwen2ForCausalLM"
  },
    // --------------------------------------------------
    ...
    // Other unchanged code
    ...
}

```

**Note**: When loading the model through the `transformers` library in your quantization script, you must set `trust_remote_code=True` when calling the `from_pretrained` function to load the modified modeling file correctly. (Ensure the security of the loaded modeling file.)

#### 3.3.2 Configuring the `config` File

`config = QuantConfig().fa_quant()`

Initialize the core parameters `(w_bit, a_bit, disable_names, disable_last_linear, dev_type, dev_id)` within `QuantConfig`. To enable FlashAttention quantization extensions, call the `fa_quant()` method of the `QuantConfig` instance to complete the configuration.

The following table describes the tables.

| **Quantization Type**                         | **Parameters to Be Configured**                                      | **Example**                                                |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| fa_quant(fa_amp=5) | The `fa_amp` parameter configures automatic accuracy fallback and accepts an integer specifying the exact number of layers to roll back.<br>Data type: `int`. Default value: `0`. The value must be greater than or equal to 0 and less than or equal to the total layer count in the model. If the specified value exceeds the total layer count, the configuration automatically applies the maximum layer count as the number of fallback layers.| quant_config=QuantConfig(w_bit=8,  a_bit=8, disable_names=disable_names,dev_type='npu',dev_id=0).fa_quant(fa_amp=5)|

### Quantization Procedure (Using Qwen2.5-7B as an Example)

1. Prepare the model, weight files, and calibration data, and place the modified modeling file and configuration file into the model weight directory. Target directory structure example (using Qwen2.5-7B):

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

2. Create a quantization script named `quant.py`.
For details, see [Quantization Scripts (NPU)](#361-quantization-scripts-npu).

3. Start the model quantization task and obtain model quantization parameters from the specified output directory. For details about the quantized weight files, see [Quantized Weight Files](#35-quantized-weight-files). To use MindIE for subsequent inference and deployment tasks, save the file in safetensors format. For details about models supported for quantization, see [Large Language Models](https://www.hiascend.com/software/mindie/modellist).

### 3.5 Quantized Weight Files

- **Files in `.npy` format**

When `save_type` is set to `['numpy']` or left unspecified, the system saves quantized weights as `.npy` files. Each `.npy` file stores a Python dictionary, where the keys are full names of linear layers (such as `transformer.encoder.layers.0.self_attention.query_key_value` in the Qwen2.5-7B model) and the values are the corresponding linear layer weights (such as the weight of the layer 0 `query_key_value` module).

```bash

├── anti_fp_norm.npy   # This file is generated for the Qwen model when outlier suppression is enabled. For details, see "Using the Outlier Suppression Function". The outlier suppression algorithm generates the norm layer weight file in the floating-point weights, which is used for weight adaptation of the input and post-norm of the quantization layer.
├── deq_scale.npy      # File of quantization parameter weights for W8A8 quantization. The tensor data type is int64. The deq_scale file has undergone data type conversion for the quantization operators and can directly adapt to the operators. When quantizing a BF16 model, the data type is not converted to int64 and remains float32.
├── fa_quant_offset.npy    # Parameter file of activation quantization offsets for FA3 quantization. The tensor data type is bfloat16 or float16.
├── fa_quant_scale.npy   # Parameter file of activation quantization scale factors for FA3 quantization. The tensor data type is bfloat16 or float16.
├── input_offset.npy # File of activation quantization offset weights for W8A8 quantization. The tensor data type is float32.
├── input_scale.npy # File of activation quantization scale factor weights for W8A8 quantization. The tensor data type is float32.
├── quant_bias.npy # File of quantization parameter weights for W8A8 quantization. The tensor data type is int32. The quant_bias already incorporates the bias value of the linear layer from the original floating-point model.
├── quant_weight.npy   # Quantization weight file. The tensor data type is int8.

```

Code sample for reading the preceding files during inference and deployment: `quant_param_dict = np.load("xxx.npy", allow_pickle=True).item()`

- **Files in safetensors format**

When `save_type` is set to `['safe_tensor']`, the quantized weights are saved as a `.safetensors` file and a `.json` description file.

- The storage format inside a `.safetensors` file is a dictionary containing both quantized weights and unmodified floating-point weights. The key for a quantized weight consists of the linear layer name at each layer and its corresponding weight suffix: `module.weight` and `module.bias` correspond to `anti_fp_norm.npy`, `weight` corresponds to `quant_weight.npy`, and `quant_bias` corresponds to `quant_bias.npy`. For example, `model.layers.0.self_attn.q_proj.deq_scale` in the Qwen2.5-7B model corresponds to `model.layers.0.self_attn.q_proj` within the `deq_scale.npy` file of the `.npy` format weights.

```json
# Partial content of a weight file generated during quantization of a Qwen model
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

- The JSON description file stores the overall quantization type (`model_quant_type`), the FA3 quantization enabling status (`fa_quant_type`), and the type of each individual weight. The weight types are defined as follows: `FLOAT` (from the original floating-point weights) and `W8A8` (from W8A8 quantization).

```json
{
  "model_quant_type": "W8A8",                               # The overall quantization type is W8A8.
  "fa_quant_type": "FAQuant",                         # FA3 quantization is enabled during the quantization process.
  "model.embed_tokens.weight": "FLOAT",                      # The embed_tokens weight from the original floating-point model.
  "model.layers.0.self_attn.q_proj.weight": "W8A8",          # The quant_weight of self_attn.q_proj at layer 0 added during quantization.
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8",     # The input_scale of self_attn.q_proj at layer 0 added during quantization.
  "model.layers.0.self_attn.q_proj.input_offset": "W8A8",    # The input_offset of self_attn.q_proj at layer 0 added during quantization.
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8",      # The quant_bias of self_attn.q_proj at layer 0 added during quantization.
  "model.layers.0.self_attn.q_proj.deq_scale": "W8A8",       # The deq_scale of self_attn.q_proj at layer 0 added during quantization.
  "model.layers.0.self_attn.k_proj.weight": "W8A8",          # The quant_weight of self_attn.k_proj at layer 0 added during quantization.
   ...
   "model.layers.0.self_attn.fa_q.scale": "FAQuant",         # The scale of query_states of self_attn at layer 0 added during quantization.
   "model.layers.0.self_attn.fa_q.offset": "FAQuant",        # The offset of query_states of self_attn at layer 0 added during quantization.
   "model.layers.0.self_attn.fa_k.scale": "FAQuant",         # The scale of key_states of self_attn at layer 0 added during quantization.
   "model.layers.0.self_attn.fa_k.offset": "FAQuant",        # The offset of key_states of self_attn at layer 0 added during quantization.
   "model.layers.0.self_attn.fa_v.scale": "FAQuant",         # The scale of key_states of self_attn at layer 0 added during quantization.
   "model.layers.0.self_attn.fa_v.offset": "FAQuant",        # The offset of value_states of self_attn at layer 0 added during quantization.
   ...
}
```

### 3.6 FA3 Accuracy Tuning

#### 3.6.1 Quantization Scripts (NPU)

For details about the current FA quantization scripts and commands, see the code samples. The following table lists the links.

| Script File                                         | Reference                                                    |
| ------------------------------------------------- | ------------------------------------------------------------ |
| [quant_qwen.py](../../../../../example/Qwen/quant_qwen.py)    | [Attention Quantization Supported by Qwen2.5-72B](../../../../../example/Qwen/README.md#qwen25-72b-支持attention量化)|
| [quant_llama.py](../../../../../example/Llama/quant_llama.py) | [Llama3.1-70B W8A8 Quantization Combined with Attention Quantization](../../../../../example/Llama/README.md#llama31-70b-w8a8量化搭配attention量化)|

#### This document provides only the recommended quantization configurations for Llama3.1-70B and Qwen2.5-72B in FA3 scenarios. You can adjust the parameters based on the actual situation. For details, see [Accuracy Tuning Strategies](../../../best_practices/w8a8_accuracy_tuning_policy.md)

#### 3.6.2 Quantization Parameter Settings for Llama3.1-70B

- Outlier suppression (AntiOutlier): `anti_method = "m3"`

```python
anti_config = AntiOutlierConfig(anti_method="m3", dev_type="npu", dev_id=model.device.index)
```

- Quantization parameters (QuantConfig)

Activation quantization method: `act_method = 3`

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

- Calibration data (`calib_set`)
Use about 50 BoolQ datasets for calibration.

- Quantization fallback (`disable_names`)
(1) `disable_level='L5'`: automatically rolls back 5 fallback layers.
<br>

(2) Roll back all `down` layers:

```python
disable_names = []
num_layers = 80
disable_idx_lst = list(range(num_layers))
for layer_index in disable_idx_lst:
    down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
    disable_names.append(down_proj_name)

```

(3) (Optional) Set the number of fallback layers when calling the `fa_quant` method. This model already meets the accuracy requirement. This parameter does not need to be set.

```python
fa_quant(fa_amp=5)
```

#### 3.6.3 Quantization Parameter Settings for Qwen2.5-72B

- (Optional) Outlier suppression (`AntiOutlier`): The accuracy meets the requirement without outlier suppression.

- Quantization parameters (QuantConfig)

Activation quantization method: `act_method = 1`

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

- Calibration data (`calib_set`)
Use about 50 BoolQ datasets for calibration.

- Quantization fallback (`disable_names`)
(1) (Optional) `disable_level='L0'`: The model can meet the accuracy requirement with the L0 configuration.

<br>(2) Roll back all `down` layers:

```python
disable_names = []
num_layers = 80
disable_idx_lst = list(range(num_layers))
for layer_index in disable_idx_lst:
    down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
    disable_names.append(down_proj_name)

```

(3) (Optional) Set the number of fallback layers when calling the `fa_quant` method. This model already meets the accuracy requirement. This parameter does not need to be set.

```python
fa_quant(fa_amp=5)
```

### 3.7 Key Steps for Multimodal FA3 Quantization

For details, see usage instructions in [Flux FA3 Quantization](../../../../../example/multimodal_sd/Flux/README.md#flux-fa3-quantization) and [HunyuanVideo FA3 Quantization](../../../../../example/multimodal_sd/HunYuanVideo/README.md#hunyuanvideo-fa3-量化).
