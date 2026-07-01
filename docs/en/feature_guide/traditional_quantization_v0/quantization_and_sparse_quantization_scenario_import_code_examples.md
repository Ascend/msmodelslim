# Quantization Code Samples

## Code Samples for Quantization and Sparse Quantization Scenarios

### W8A8 KVCache Quantization Scenario

The following is a code sample for the W8A8 KVCache quantization scenario:

```python
# Importing dependencies
import torch  
import torch_npu   # To perform quantization on the CPU, skip this step.
from transformers import AutoTokenizer, AutoModel

# For local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
).npu()  # To perform multi-rank quantization on the NPU, refer to the prerequisites for configuration and set device_map='auto'. When creating a model, remove .npu(). To perform quantization on the CPU, set torch_dtype=torch.float32 and remove .npu().
# Prepare calibration data and modify the data as needed.
calib_list = ["Where is the capital of China?",
              "Please write a poem:",
              "How can I learn Python?",
              "Please help me write a report on the optimization of foundation model inference:",
              "List the most worth-seeing tourist attractions in China."
]
# Define the function for obtaining calibration data.
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)    
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  # Obtain calibration data.

# Quantization configuration (modify it as needed)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # Import quantization configuration interfaces.
# Specify quantization parameters and return a quantization configuration instance by using QuantConfig.
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,       
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'], 
    dev_type='npu', # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    dev_id=model.device.index,
    act_method=3,
    pr=1.0, 
    mm_tensor=False,
    use_kvcache_quant=True
  )  
# Define calibration by using the Calibrator interface with the loaded original model, quantization configuration, and calibration data.
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  
calibrator.run()     # Use run() to perform quantization.
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])     # Save model quantization parameters by using save(). Modify the path and format as needed.
print('Save quant weight success!')
```

### W8A8 Low-bit Quantization Scenario

The following is a code sample for the W8A8 low-bit quantization scenario:

```python
# Importing dependencies
import torch 
import torch_npu   # To perform quantization on the CPU, skip this step.
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
  ).npu()   # To perform multi-rank quantization on the NPU, refer to the prerequisites for configuration and set device_map='auto'. When creating a model, remove .npu(). To perform quantization on the CPU, set torch_dtype=torch.float32 and remove .npu().
# Prepare calibration data and modify the data as needed.
calib_list = ["Where is the capital of China?",
              "Please write a poem:",
              "How can I learn Python?",
              "Please help me write a report on the optimization of foundation model inference:",
              "List the most worth-seeing tourist attractions in China."
]
# Define the function for obtaining calibration data.
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)   
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])     
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  # Obtain calibration data.

# Quantization configuration (modify it as needed)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # Import quantization configuration interfaces.
# Specify quantization parameters and return a quantization configuration instance by using QuantConfig.
quant_config = QuantConfig(
    a_bit=8, 
    w_bit=8,       
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'], 
    dev_id=model.device.index, 
    dev_type='npu', # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    act_method=2,
    sigma_factor=3.0,
    do_smooth=False,                          
    is_lowbit=True,                          
    use_sigma=False,
  )  
# Define calibration by using the Calibrator interface with the loaded original model, quantization configuration, and calibration data.
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0') 
calibrator.run()     # Use run() to perform quantization.
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])     # Save model quantization parameters by using save(). Modify the path and format as needed.
print('Save quant weight success!')
```

### W8A8 `per_token` Quantization Scenario

Note:
W8A8 `per_token` quantization does not support quantization of MoE model weights on Atlas inference products.
The following is a code sample for the W8A8 `per_token` quantization scenario:

```python
# Importing dependencies
import torch 
import torch_npu   # To perform quantization on the CPU, skip this step.
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
  ).npu()
# To perform multi-rank quantization on the NPU, refer to the prerequisites for configuration and set device_map='auto'. When creating a model, remove .npu(). To perform quantization on the CPU, set torch_dtype=torch.float32 and remove .npu().
# Prepare calibration data and modify the data as needed.
calib_list = ["Where is the capital of China?",
              "Please write a poem:",
              "How can I learn Python?",
              "Please help me write a report on the optimization of foundation model inference:",
              "List the most worth-seeing tourist attractions in China."
]
# Define the function for obtaining calibration data.
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)   
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])     
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  # Obtain calibration data.

# Quantization configuration (modify it as needed)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # Import quantization configuration interfaces.
# Specify quantization parameters and return a quantization configuration instance by using QuantConfig.
quant_config = QuantConfig(
    a_bit=8, 
    w_bit=8,       
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'], 
    dev_id=model.device.index, 
    dev_type='npu', # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    act_method=1,
    w_sym=True, 
    mm_tensor=False,   
    is_dynamic=True
  )  
# Define calibration by using the Calibrator interface with the loaded original model, quantization configuration, and calibration data.
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0') 
calibrator.run()     # Use run() to perform quantization.
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])     # Save model quantization parameters by using save(). Modify the path and format as needed.
print('Save quant weight success!')
```

### W8A16 `per_channel` MinMax Quantization Scenario

The following is a code sample for the W8A16 `per_channel` MinMax quantization scenario:

```python
# Importing dependencies
import torch
import torch_npu   # To perform quantization on the CPU, skip this step.
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True) 
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
    ).npu() # To perform multi-rank quantization on the NPU, refer to the prerequisites for configuration and set device_map='auto'. When creating a model, remove .npu(). To perform quantization on the CPU, set torch_dtype=torch.float32 and remove .npu().

# Prepare calibration data and modify the data as needed. In W8A16 data-free mode, skip this step.
calib_list = ["Where is the capital of China?",
              "Please write a poem:",
              "How can I learn Python?",
              "Please help me write a report on the optimization of foundation model inference:",
              "List the most worth-seeing tourist attractions in China."
]
# Define the function for obtaining calibration data.
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  # Obtain calibration data.

# Quantization configuration (modify it as needed)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # Import quantization configuration interfaces.
# Specify quantization parameters and return a quantization configuration instance by using QuantConfig.
quant_config = QuantConfig(
    w_bit=8,    
    a_bit=16,         
    disable_names=[], 
    dev_id=model.device.index, 
    dev_type='npu', # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    w_sym=True,
    mm_tensor=False, 
    w_method='MinMax'
  )  
# Define calibration by using the Calibrator interface with the loaded original model, quantization configuration, and calibration data.
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  # Specify calib_data=[] in the data-free scenario.
calibrator.run()     # Use run() to perform quantization.
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])     # Save model quantization parameters by using save(). Modify the path and format as needed.
print('Save quant weight success!')
```

### W8A16 `per_channel` HQQ Quantization Scenario

The following is a code sample for the W8A16 `per_channel` HQQ quantization scenario:

```python
# Importing dependencies
import torch
import torch_npu   # To perform quantization on the CPU, skip this step.
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True) 
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
    ).npu() # To perform multi-rank quantization on the NPU, refer to the prerequisites for configuration and set device_map='auto'. When creating a model, remove .npu(). To perform quantization on the CPU, set torch_dtype=torch.float32 and remove .npu().
# Prepare calibration data and modify the data as needed. In W8A16 data-free mode, skip this step.
calib_list = ["Where is the capital of China?",
              "Please write a poem:",
              "How can I learn Python?",
              "Please help me write a report on the optimization of foundation model inference:",
              "List the most worth-seeing tourist attractions in China."
]
# Define the function for obtaining calibration data.
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  # Obtain calibration data.

# Quantization configuration (modify it as needed)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # Import quantization configuration interfaces.
# Specify quantization parameters and return a quantization configuration instance by using QuantConfig.
quant_config = QuantConfig(
    w_bit=8,    
    a_bit=16,         
    disable_names=[], 
    dev_id=model.device.index, 
    dev_type='npu', # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    w_sym=True, 
    mm_tensor=False, 
    w_method='HQQ'
  )  
# Define calibration by using the Calibrator interface with the loaded original model, quantization configuration, and calibration data.
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  # Specify calib_data=[] in the data-free scenario.
calibrator.run()     # Use run() to perform quantization.
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])     # Save model quantization parameters by using save(). Modify the path and format as needed.
print('Save quant weight success!')
```

### W8A16 `per_channel` GPTQ Quantization Scenario

Note:
When you use GPTQ to process an MoE model, the tool uses the MinMax algorithm by default to quantize linear layers not reached by the calibration dataset.
When you use GPTQ to process an MoE model, the low-bit algorithm is not supported.
The following is a code sample for the W8A16 `per_channel` GPTQ quantization scenario:

```python
# Importing dependencies
import torch
import torch_npu   # To perform quantization on the CPU, skip this step.
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True) 
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
    ).npu()    # To perform multi-rank quantization on the NPU, refer to the prerequisites for configuration and set device_map='auto'. When creating a model, remove .npu(). To perform quantization on the CPU, set torch_dtype=torch.float32 and remove .npu().
# Prepare calibration data and modify the data as needed.
calib_list = ["Where is the capital of China?",
              "Please write a poem:",
              "How can I learn Python?",
              "Please help me write a report on the optimization of foundation model inference:",
              "List the most worth-seeing tourist attractions in China."
]
# Define the function for obtaining calibration data.
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  # Obtain calibration data.

# Quantization configuration (modify it as needed)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # Import quantization configuration interfaces.
# Specify quantization parameters and return a quantization configuration instance by using QuantConfig.
quant_config = QuantConfig(
    w_bit=8,    
    a_bit=16,         
    disable_names=[], 
    dev_id=model.device.index, 
    dev_type='npu', # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    w_sym=True, 
    mm_tensor=False, 
    w_method='GPTQ'
  )  
# Define calibration by using the Calibrator interface with the loaded original model, quantization configuration, and calibration data.
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  
calibrator.run()     # Use run() to perform quantization.
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])   # Save model quantization parameters by using save(). Modify the path and format as needed.
print('Save quant weight success!')
```

### W8A16 `per_channel` AWQ Quantization Scenario

Note:
When using AWQ to process an MoE model, the tool does not perform any processing on the expert structure.
The following is a code sample for the W8A16 `per_channel` AWQ quantization scenario:

```python
# Importing dependencies
import torch
import torch_npu   # To perform quantization on the CPU, skip this step.
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True) 
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
    ).npu() # To perform multi-rank quantization on the NPU, refer to the prerequisites for configuration and set device_map='auto'. When creating a model, remove .npu(). To perform quantization on the CPU, set torch_dtype=torch.float32 and remove .npu().

# Prepare calibration data and modify the data as needed.
calib_list = ["Where is the capital of China?",
              "Please write a poem:",
              "How can I learn Python?",
              "Please help me write a report on the optimization of foundation model inference:",
              "List the most worth-seeing tourist attractions in China."
]
# Define the function for obtaining calibration data.
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  # Obtain calibration data.

# Perform outlier suppression.
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
w_sym = False
anti_config = AntiOutlierConfig(
    a_bit=16, 
    w_bit=8,
    anti_method='m3', 
    dev_id=model.device.index,
    dev_type='npu', # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    w_sym=w_sym
)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()

# Quantization configuration (modify it as needed)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # Import quantization configuration interfaces.
# Specify quantization parameters and return a quantization configuration instance by using QuantConfig.
quant_config = QuantConfig(
    w_bit=8,    
    a_bit=16,         
    disable_names=[],
    dev_id=model.device.index, 
    dev_type='npu',  # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    w_sym=w_sym,
    mm_tensor=False, 
    w_method='MinMax'
  )  
# Define calibration by using the Calibrator interface with the loaded original model, quantization configuration, and calibration data.
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  
calibrator.run()     # Use run() to perform quantization.
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])   # Save model quantization parameters by using save(). Modify the path and format as needed.
print('Save quant weight success!')
```

### W8A16 KVCache Quantization Scenario

The following is a code sample for the W8A16 KVCache quantization scenario:

```python
# Importing dependencies
import torch  
import torch_npu   # To perform quantization on the CPU, skip this step.
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
).npu()  # To perform multi-rank quantization on the NPU, refer to the prerequisites for configuration and set device_map='auto'. When creating a model, remove .npu(). To perform quantization on the CPU, set torch_dtype=torch.float32 and remove .npu().
# Prepare calibration data and modify the data as needed.
calib_list = ["Where is the capital of China?",
              "Please write a poem:",
              "How can I learn Python?",
              "Please help me write a report on the optimization of foundation model inference:",
              "List the most worth-seeing tourist attractions in China."
]
# Define the function for obtaining calibration data.
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)    
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  # Obtain calibration data.

# Quantization configuration (modify it as needed)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # Import quantization configuration interfaces.
# Specify quantization parameters and return a quantization configuration instance by using QuantConfig.
quant_config = QuantConfig(
    a_bit=16,
    w_bit=8,       
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'], 
    dev_type='npu', # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    dev_id=model.device.index,
    act_method=3,
    pr=1.0, 
    mm_tensor=False,
    use_kvcache_quant=True
  ).kv_quant(kv_sym=True)
# Define calibration by using the Calibrator interface with the loaded original model, quantization configuration, and calibration data.
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  
calibrator.run()     # Use run() to perform quantization.
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])     # Save model quantization parameters by using save(). Modify the path and format as needed.
print('Save quant weight success!')
```

### W8A16 `per_group` Quantization Scenario

Note:
W8A16 supports `per_group` quantization by using the MinMax, HQQ, GPTQ, or AWQ algorithm.
The following is a code sample for the W8A16 `per_group` quantization scenario:

```python
# Importing dependencies
import torch
import torch_npu   # To perform quantization on the CPU, skip this step.
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True) 
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
).npu()  # To perform multi-rank quantization on the NPU, refer to the prerequisites for configuration and set device_map='auto'. When creating a model, remove .npu(). To perform quantization on the CPU, set torch_dtype=torch.float32 and remove .npu().

# Prepare calibration data and modify the data as needed. In W8A16 data-free mode for HQQ and MinMax, skip this step.
calib_list = ["Where is the capital of China?",
              "Please write a poem:",
              "How can I learn Python?",
              "Please help me write a report on the optimization of foundation model inference:",
              "List the most worth-seeing tourist attractions in China."
]
# Define the function for obtaining calibration data.
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)  
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  # Obtain calibration data.

"""
# This step is required in the AWQ quantization scenario.
# Perform outlier suppression.
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
anti_config = AntiOutlierConfig(
    w_bit=8, 
    a_bit=16, 
    anti_method='m3', 
    dev_id=model.device.index,
    dev_type='npu', # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    w_sym=True
)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()
# End of optional configurations.
"""   

# Quantization configuration (modify it as needed)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # Import quantization configuration interfaces.
# Specify quantization parameters and return a quantization configuration instance by using QuantConfig.
quant_config = QuantConfig(
    w_bit=8,     
    a_bit=16,         
    disable_names=[], 
    dev_id=model.device.index, 
    dev_type='npu',  # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    w_sym=True,
    mm_tensor=False,     
    w_method='MinMax',   # Use the default value 'MinMax' for MinMax and AWQ. Set this parameter to 'GPTQ' for GPTQ, or 'HQQ' for HQQ.     
    is_lowbit=True,
    open_outlier=False,
    group_size=64
)  
# Define calibration by using the Calibrator interface with the loaded original model, quantization configuration, and calibration data.
calibrator = Calibrator(model, quant_config, disable_level='L0')
calibrator.run()     # Use run() to perform quantization.
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])     # Save model quantization parameters by using save(). Modify the path and format as needed.
print('Save quant weight success!')
```

### Low-bit Sparse Quantization Scenario

The following is a code sample for the low-bit sparse quantization scenario:

```python
# Importing dependencies
import torch
import torch_npu   # To perform quantization on the CPU, skip this step.
from transformers import AutoTokenizer, AutoModel
# for local path
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
) 
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2',
    torch_dtype=torch.float16, 
    local_files_only=True
  ).npu() # To perform multi-rank quantization on the NPU, refer to the prerequisites for configuration, set device_map='auto', and set torch_dtype to the default data type of the model. To perform quantization on the NPU, move the model to the NPU for single-rank calibration (model = model.npu()). This is not required for multi-rank calibration.
# Prepare calibration data and modify the data as needed.
calib_list = ["Where is the capital of China?",
              "Please write a poem:",
              "How can I learn Python?",
              "Please help me write a report on the optimization of foundation model inference:",
              "List the most worth-seeing tourist attractions in China."
]
# Define the function for obtaining calibration data.
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)  
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset
dataset_calib = get_calib_dataset(tokenizer, calib_list)  # Obtain calibration data.

# Sparse quantization configuration (modify it as needed)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # Import sparse quantization configuration interfaces.
# Specify sparse quantization parameters and return a configuration instance by using QuantConfig.
quant_config = QuantConfig(
    w_bit=4, 
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'], 
    dev_type='npu',  # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    dev_id=model.device.index,
    act_method=2,
    mm_tensor=False, 
    sigma_factor=3.0,
    do_smooth=False,
    is_lowbit=True,
    use_sigma=True
 )  
# Define calibration by using Calibrator with the loaded original model, sparse quantization configuration, and calibration data.
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()     # Use run() to perform quantization.
calibrator.save('./quant_weight')      # Save model quantization parameters by using save(). Modify the path as needed.
print('Save quant weight success!')
```

### W4A8 Dynamic Quantization Scenario

 The following is a code sample for the W4A8 dynamic quantization scenario:

```python
# Importing dependencies
import torch
import torch_npu   # To perform quantization on the CPU, skip this step.
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='./Llama3.1-8B-Instruct', local_files_only=True
    ) 
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./Llama3.1-8B-Instruct', local_files_only=True
    ).npu() # To perform multi-rank quantization on the NPU, refer to the prerequisites for configuration and set device_map='auto'. When creating a model, remove .npu(). To perform quantization on the CPU, set torch_dtype=torch.float32 and remove .npu().

# Quantization configuration (modify it as needed)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # Import quantization configuration interfaces.
# Specify quantization parameters and return a quantization configuration instance by using QuantConfig.
quant_config = QuantConfig(
    a_bit=8, 
    w_bit=4,
    dev_id=model.device.index,
    dev_type='npu', # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    w_sym=True,
    is_lowbit=True,
    mm_tensor=False,
    is_dynamic=True,
    group_size=32,
    open_outlier=False,
  )  
# Define calibration by using the Calibrator interface with the loaded original model, quantization configuration, and calibration data.
calibrator = Calibrator(model, quant_config, disable_level='L0')  
calibrator.run()     # Use run() to perform quantization.
calibrator.save('./quant_weight', save_type=['safe_tensor'])   # Save model quantization parameters by using save(). Modify the path and format as needed.
print('Save quant weight success!')
```

### W4A4 Dynamic Quantization Scenario

 The following is a code sample for the W4A4 dynamic quantization scenario:

```python
# Importing dependencies
import torch
import torch_npu   # To perform quantization on the CPU, skip this step.
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='./Qwen3-32B', local_files_only=True
    ) 
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./Qwen3-32B', local_files_only=True
    ).npu() # To perform multi-rank quantization on the NPU, refer to the prerequisites for configuration and set device_map='auto'. When creating a model, remove .npu(). To perform quantization on the CPU, set torch_dtype=torch.float32 and remove .npu().

# Quantization configuration (modify it as needed)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # Import quantization configuration interfaces.
# Specify quantization parameters and return a quantization configuration instance by using QuantConfig.
quant_config = QuantConfig(
    a_bit=4, 
    w_bit=4,
    dev_id=model.device.index,
    dev_type='npu', # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    w_sym=True,
    is_dynamic=True,
  )  
# Define calibration by using the Calibrator interface with the loaded original model, quantization configuration, and calibration data.
calibrator = Calibrator(model, quant_config, disable_level='L0')  
calibrator.run()     # Use run() to perform quantization.
calibrator.save('./quant_weight', save_type=['safe_tensor'])   # Save model quantization parameters by using save(). Modify the path and format as needed.
print('Save quant weight success!')
```

### NF4 Quantization Scenario

The following is a code sample for the NF4 quantization scenario:

```python
# Importing dependencies
import torch
import torch_npu   # To perform quantization on the CPU, skip this step.
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
# To perform multi-rank quantization on the NPU, refer to the prerequisites for configuration and set device_map='auto'. When creating a model, remove .npu(). To perform quantization on the CPU, set torch_dtype=torch.float32 and remove .npu().
model = AutoModel.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True).npu() 

# NF4 quantization is commonly used in training scenarios such as QLoRA. Activation operations, such as AntiOutlier, are not recommended in this scenario.

# Quantization configuration (modify it as needed)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # Import quantization configuration interfaces.
# Specify quantization parameters and return a quantization configuration instance by using QuantConfig.
quant_config = QuantConfig(
    w_bit=4,    
    a_bit=16,         
    dev_id=model.device.index,
    dev_type='npu', # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
  ).weight_quant(w_method='NF', block_size=64)
# Define calibration by using the Calibrator interface with the loaded original model, quantization configuration, and calibration data.
calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')  # Specify calib_data=[] in the data-free scenario.
calibrator.run()     # Use run() to perform quantization.
calibrator.save('./quant_weight', save_type=['safe_tensor'])      # Save model quantization parameters by using save(). Modify the path and format as needed.
print('Save quant weight success!')
```

### Simulated Multi-rank Quantization Scenario

Note:
Simulated multi-rank quantization is applicable only to the TensorParallel multi-rank inference and deployment scenario. Other inference and deployment modes are not supported.
The following is a code sample for the simulated multi-rank quantization scenario:

```python
# Importing dependencies
import torch 
import torch_npu   # To perform quantization on the CPU, skip this step.
from transformers import AutoTokenizer, AutoModel
# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
 ).npu()    # To perform multi-rank quantization on the NPU, refer to the prerequisites for configuration and set device_map='auto'. When creating a model, remove .npu(). To perform quantization on the CPU, set torch_dtype=torch.float32 and remove .npu().
# Prepare calibration data and modify the data as needed.
calib_list = ["Where is the capital of China?",
              "Please write a poem:",
              "How can I learn Python?",
              "Please help me write a report on the optimization of foundation model inference:",
              "List the most worth-seeing tourist attractions in China."
]
# Define the function for obtaining calibration data.
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)   
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])     
    return calib_dataset
dataset_calib = get_calib_dataset(tokenizer, calib_list)  # Obtain calibration data.
# Quantization configuration (modify it as needed)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # Import quantization configuration interfaces.
# Specify quantization parameters and return a quantization configuration instance by using QuantConfig.
quant_config = QuantConfig(
    a_bit=8, 
    w_bit=8,       
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'], 
    dev_id=model.device.index, 
    dev_type='npu',   # To perform quantization on the CPU, set dev_type='cpu' and remove the dev_id=model.device.index configuration.
    act_method=3,
    pr=0.5, 
    mm_tensor=False
  ).simulate_tp(tp_size=4, enable_communication_quant=True, enable_per_device_quant=True)
# Define calibration by using the Calibrator interface with the loaded original model, quantization configuration, and calibration data.
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  
calibrator.run()     # Use run() to perform quantization.
calibrator.save('./quant_weight', save_type=['numpy', 'safe_tensor'])      # Save model quantization parameters by using save(). Modify the path as needed.
print('Save quant weight success!')
```
