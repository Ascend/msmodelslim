# 量化代码示例

量化及稀疏量化场景导入代码样例如下：

## 1. W8A8_kvcache量化场景

W8A8_kvcache量化场景导入的样例代码如下：

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在CPU上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
).npu()  #若在NPU上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在CPU上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()
# 准备校准数据，请根据实际情况修改
calib_list = ["中国的首都在哪里？",
              "请作一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的述职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'],
    dev_type='npu', # 在CPU上进行量化时，需要配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    dev_id=model.device.index,
    act_method=3,
    pr=1.0,
    mm_tensor=False,
    use_kvcache_quant=True
  )
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=['numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径及保存的格式
print('Save quant weight success!')
```

## 2. W8A8_lowbit算法量化场景

W8A8_lowbit算法量化场景导入的样例代码如下：

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在CPU上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
  ).npu()   # 若在NPU上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在CPU上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()
# 准备校准数据，请根据实际情况修改
calib_list = ["中国的首都在哪里？",
              "请作一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的述职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'],
    dev_id=model.device.index,
    dev_type='npu',   # 在CPU上进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    act_method=2,
    sigma_factor=3.0,
    do_smooth=False,
    is_lowbit=True,
    use_sigma=False,
  )
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=['numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径及保存的格式
print('Save quant weight success!')
```

## 3. W8A8_per_token算法量化场景

说明
W8A8_per_token不支持在Atlas 推理系列产品中对MoE模型权重进行量化。
W8A8_per_token算法量化场景导入的样例代码如下：

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在CPU上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
  ).npu()
# 若在NPU上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在CPU上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()
# 准备校准数据，请根据实际情况修改
calib_list = ["中国的首都在哪里？",
              "请作一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的述职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'],
    dev_id=model.device.index,
    dev_type='npu',   # 在CPU上进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    act_method=1,
    w_sym=True,
    mm_tensor=False,
    is_dynamic=True
  )
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=['numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径及保存的格式
print('Save quant weight success!')
```

## 4. W8A16 per-channel_MinMax算法量化场景

W8A16 per-channel_MinMax算法量化场景导入的样例代码如下：

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在CPU上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
    ).npu() # 若在NPU上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在CPU上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()

# 准备校准数据，请根据实际情况修改，W8A16 Data-Free模式下请忽略此步骤
calib_list = ["中国的首都在哪里？",
              "请作一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的述职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    w_bit=8,
    a_bit=16,
    disable_names=[],
    dev_id=model.device.index,
    dev_type='npu',   # 在CPU上进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    w_sym=True,
    mm_tensor=False,
    w_method='MinMax'
  )
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  # Data Free场景下calib_data=[]
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=['numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径及保存的格式
print('Save quant weight success!')
```

## 5. W8A16 per-channel_HQQ算法量化场景

W8A16 per-channel_HQQ算法量化场景导入的样例代码如下：

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在CPU上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
    ).npu() # 若在NPU上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在CPU上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()
# 准备校准数据，请根据实际情况修改，W8A16 Data-Free模式下请忽略此步骤
calib_list = ["中国的首都在哪里？",
              "请作一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的述职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    w_bit=8,
    a_bit=16,
    disable_names=[],
    dev_id=model.device.index,
    dev_type='npu',   # 在CPU上进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    w_sym=True,
    mm_tensor=False,
    w_method='HQQ'
  )
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  # Data Free场景下calib_data=[]
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=['numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径及保存的格式
print('Save quant weight success!')
```

## 6. W8A16 per-channel_GPTQ算法量化场景

说明
GPTQ方式处理MOE模型时，对校准集没运行到的线性层，会默认使用MinMax进行量化。
GPTQ方式处理MOE模型时，不支持lowbit算法量化场景。
W8A16 per-channel_GPTQ算法量化示例

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在CPU上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
    ).npu()    # 若在NPU上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在CPU上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()
# 准备校准数据，请根据实际情况修改
calib_list = ["中国的首都在哪里？",
              "请作一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的述职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    w_bit=8,
    a_bit=16,
    disable_names=[],
    dev_id=model.device.index,
    dev_type='npu',   # 在CPU上进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    w_sym=True,
    mm_tensor=False,
    w_method='GPTQ'
  )
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=['numpy', 'safe_tensor'])   #使用save()保存模型量化参数，请根据实际情况修改路径及保存的格式
print('Save quant weight success!')
```

## 7. W8A16 per-channel_AWQ算法量化场景

说明
AWQ方式处理MOE模型时，不对专家结构做任何处理。
W8A16 per-channel_AWQ算法量化导入的样例代码：

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在CPU上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
    ).npu() # 若在NPU上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在CPU上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()

# 准备校准数据，请根据实际情况修改
calib_list = ["中国的首都在哪里？",
              "请作一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的述职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 执行离群值抑制的操作
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
w_sym = False
anti_config = AntiOutlierConfig(
    a_bit=16,
    w_bit=8,
    anti_method='m3',
    dev_id=model.device.index,
    dev_type='npu', # 在CPU上进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    w_sym=w_sym
)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    w_bit=8,
    a_bit=16,
    disable_names=[],
    dev_id=model.device.index,
    dev_type='npu',  # 在CPU上进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    w_sym=w_sym,
    mm_tensor=False,
    w_method='MinMax'
  )
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=['numpy', 'safe_tensor'])   #使用save()保存模型量化参数，请根据实际情况修改路径及保存的格式
print('Save quant weight success!')
```

## 8. W8A16_kvcache量化场景

W8A16_kvcache量化场景导入的样例代码如下：

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在CPU上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
).npu()  #若在NPU上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在CPU上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()
# 准备校准数据，请根据实际情况修改
calib_list = ["中国的首都在哪里？",
              "请作一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的述职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    a_bit=16,
    w_bit=8,
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'],
    dev_type='npu', # 在CPU上进行量化时，需要配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    dev_id=model.device.index,
    act_method=3,
    pr=1.0,
    mm_tensor=False,
    use_kvcache_quant=True
  ).kv_quant(kv_sym=True)
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=['numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径及保存的格式
print('Save quant weight success!')
```

## 9. W8A16 per-group量化场景

说明
W8A16支持使用MinMax、HQQ、GPTQ或AWQ算法进行per-group量化。
W8A16 per-group场景导入的样例代码如下：

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在CPU上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
).npu()  # 若在NPU上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在CPU上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()

# 准备校准数据，请根据实际情况修改，HQQ和MinMax的W8A16 Data-Free模式下请忽略此步骤
calib_list = ["中国的首都在哪里？",
              "请作一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的述职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset

dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

"""
# AWQ算法量化场景下，需要配置此步骤
# 执行离群值抑制的操作
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
anti_config = AntiOutlierConfig(
    w_bit=8,
    a_bit=16,
    anti_method='m3',
    dev_id=model.device.index,
    dev_type='npu',   # 在CPU上进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    w_sym=True
)
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()
# 可选配置结束
"""

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    w_bit=8,
    a_bit=16,
    disable_names=[],
    dev_id=model.device.index,
    dev_type='npu',  # 在CPU上进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    w_sym=True,
    mm_tensor=False,
    w_method='MinMax',   # MinMax和AWQ算法量化场景使用默认值'MinMax'，GPTQ算法量化场景需配置为'GPTQ',HQQ算法量化场景需配置为'HQQ'
    is_lowbit=True,
    open_outlier=False,
    group_size=64
)
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, disable_level='L0')
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=['numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径及保存的格式
print('Save quant weight success!')
```

## 10. lowbit算法稀疏量化场景

lowbit算法稀疏量化场景导入代码样例：

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在CPU上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel
# for local path
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2',
    torch_dtype=torch.float16,
    local_files_only=True
  ).npu()    # 如果需要在NPU上进行多卡量化，需要先参考前提条件进行配置，并配置以下参数device_map='auto', torch_dtype为当前使用模型的默认数据类型；在NPU上进行量化时，单卡校准需将模型移到npu上model = model.npu()，多卡校准时不需要
# 准备校准数据，请根据实际情况修改
calib_list = ["中国的首都在哪里？",
              "请作一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的述职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset
dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取

# 稀疏量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入稀疏量化配置接口
# 使用QuantConfig接口，配置稀疏量化参数，并返回配置实例
quant_config = QuantConfig(
    w_bit=4,
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'],
    dev_type='npu',  # 在CPU上进行量化时，需要配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    dev_id=model.device.index,
    act_method=2,
    mm_tensor=False,
    sigma_factor=3.0,
    do_smooth=False,
    is_lowbit=True,
    use_sigma=True
 )
#使用Calibrator接口，输入加载的原模型、稀疏量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight')      #使用save()保存模型量化参数，请根据实际情况修改路径
print('Save quant weight success!')
```

## 11. W4A8 dynamic 量化场景

 W4A8 dynamic 量化场景导入代码样例：

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在CPU上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='./Llama3.1-8B-Instruct', local_files_only=True
    )
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./Llama3.1-8B-Instruct', local_files_only=True
    ).npu() # 若在NPU上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在CPU上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    a_bit=8,
    w_bit=4,
    dev_id=model.device.index,
    dev_type='npu', # 在CPU上进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    w_sym=True,
    is_lowbit=True,
    mm_tensor=False,
    is_dynamic=True,
    group_size=32,
    open_outlier=False,
  )
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, disable_level='L0')
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=['safe_tensor'])   #使用save()保存模型量化参数，请根据实际情况修改路径及保存的格式
print('Save quant weight success!')
```

## 12. W4A4 dynamic 量化场景

 W4A4 dynamic 量化场景导入代码样例：

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在CPU上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='./Qwen3-32B', local_files_only=True
    )
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./Qwen3-32B', local_files_only=True
    ).npu() # 若在NPU上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在CPU上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    a_bit=4,
    w_bit=4,
    dev_id=model.device.index,
    dev_type='npu', # 在CPU上进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
    w_sym=True,
    is_dynamic=True,
  )
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, disable_level='L0')
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=['safe_tensor'])   #使用save()保存模型量化参数，请根据实际情况修改路径及保存的格式
print('Save quant weight success!')
```

## 13. NF4算法量化场景

NF4算法量化场景导入的样例代码如下：

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在CPU上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
# 若在NPU上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto'，创建model时需去掉.npu()；若在CPU上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()
model = AutoModel.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True).npu()

# NF4量化常用于QLoRA等训练场景，AntiOutlier离群值抑制等对激活的操作不推荐在该场景使用

# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    w_bit=4,
    a_bit=16,
    dev_id=model.device.index,
    dev_type='npu',   # 在CPU上进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
  ).weight_quant(w_method='NF', block_size=64)
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')  # Data Free场景下calib_data=[]
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=['safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径及保存的格式
print('Save quant weight success!')
```

## 14. 模拟多卡量化场景

说明
模拟多卡量化仅适用于TensorParallel多卡推理部署场景，暂不支持其他推理部署方式。
模拟多卡量化场景导入代码样例：

```python
# 导入相关依赖
import torch
import torch_npu   # 若需要在CPU上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel
# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
 ).npu()    # 若在NPU上进行多卡量化时，需要先参考前提条件进行配置，并配置device_map='auto',创建model时需去掉.npu()；若在CPU上进行量化时，需要配置torch_dtype=torch.float32，创建model时需去掉.npu()
# 准备校准数据，请根据实际情况修改
calib_list = ["中国的首都在哪里？",
              "请作一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的述职报告：",
              "中国最值得去的几个景点"]
#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return calib_dataset
dataset_calib = get_calib_dataset(tokenizer, calib_list)  #校准数据获取
# 量化配置，请根据实际情况修改
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口
# 使用QuantConfig接口，配置量化参数，并返回量化配置实例
quant_config = QuantConfig(
    a_bit=8,
    w_bit=8,
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value','transformer.encoder.layers.0.self_attention.dense', 'transformer.encoder.layers.0.mlp.dense_h_to_4h'],
    dev_id=model.device.index,
    dev_type='npu',   # 在CPU上进行量化时，需配置参数dev_type='cpu'，并取消dev_id=model.device.index参数的配置
    act_method=3,
    pr=0.5,
    mm_tensor=False
  ).simulate_tp(tp_size=4, enable_communication_quant=True, enable_per_device_quant=True)
#使用Calibrator接口，输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()     #使用run()执行量化
calibrator.save('./quant_weight', save_type=['numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径
print('Save quant weight success!')
```
