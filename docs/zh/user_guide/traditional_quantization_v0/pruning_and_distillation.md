# 训练加速与模型改造

## 1. 基于重要性评估的剪枝调优

### 1.1 简介

msModelSlim工具提供了基于重要性评估的模型剪枝调优API，用户只需要提供模型实例，即可调用剪枝API完成模型的剪枝。剪枝后的模型提升了一定的性能，减少了模型的大小，提升推理过程中的效率。

### 1.2 使用前准备

目前支持PyTorch框架下的模型剪枝调优。
安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../install_guide/install_guide.md)。

- 注意：该功能仅支持 PyTorch 2.0.0 以上版本。

### 1.3 功能介绍

### 1.4 操作步骤

1.用户自行准备待剪枝模型实例以及训练脚本。本样例以torchvision中的vgg16为例。

2.打开待剪枝模型的训练脚本vision/references/classification/train.py。编辑train.py文件，导入剪枝接口。剪枝API接口说明请参考PruneTorch。

```python
from msmodelslim.pytorch.prune.prune_torch import PruneTorch
```

3.（可选）调整日志输出等级，启动调优任务后，将打屏显示设置级别的日志信息。[日志级别说明](../../api_reference/python_api_v0/common_apis.md#参数说明)

```python
from msmodelslim import set_logger_level
set_logger_level("info")        #根据实际情况配置
```

4.在原脚本初始化网络，并已经加载权重后，使用PruneTorch接口自定义配置剪枝的重要性评估函数、算子节点保留的参数比例、剪枝率等。

```python
desc = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32)).prune(0.8)
```

5.启动模型剪枝调优任务。建议使用原始训练过程最终学习率，训练10epoch即可。

```bash
python3 train.py --model vgg16 --lr 1e-5 --epochs 10 --pretrained --batch-size 256 -j 48
```

将得到一个剪枝后的模型，可进行后续的训练任务。

6.在后续的评估过程中，参考如下示例配置，加载步骤4返回模型剪枝信息。

```python
PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32)).prune_by_desc(desc)
```

## 2. Transformer类模型权重剪枝调优

### 2.1 简介

msModelSlim工具提供了API方式的Transformer类模型权重剪枝调优，可将模型权重进行裁剪，并加载到同一模型结构下的小模型中。用户只需提供同一模型结构下小模型（通过配置较小初始化参数得到的模型实例，例如Bert模型中缩小intermediate_size和num_hidden_layers参数）和原始模型权重文件，即可调用剪枝API完成模型权重的剪枝。

### 2.2 使用前准备

目前支持MindSpore和PyTorch框架下Transformer类模型的剪枝调优。
安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../install_guide/install_guide.md)。

- 注意：该功能仅支持 PyTorch 2.0.0 以上版本。

模型剪枝期间，用户可手动配置参数对预训练模型的权重进行裁剪，并将裁剪后的权重加载至小模型中，获取一个权重加载完毕的Transformer模型。剪枝后模型不保证精度，需要进行一定的训练来提升精度，例如通过模型蒸馏进行训练。

### 2.3 功能介绍

#### 2.3.1 操作步骤

以下步骤以PyTorch框架的Transformer类模型为例，MindSpore框架的模型仅在调用部分接口时，入参配置有所差异，使用时请参照具体的API接口说明。

1. 用户自行准备同一种模型结构下的原始模型实例（待剪枝模型）和原始模型权重文件。本样例以Bert为例，在ModelZoo搜索下载Bert代码和原始模型权重文件。

2. 新建待剪枝模型的Python脚本，例如test_prune_model.py。编辑test_prune_model.py文件，导入如下接口。剪枝API接口说明请参考剪枝接口。

```python
from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig
from msmodelslim.common.prune.transformer_prune.prune_model import prune_model_weight
```

3.（可选）调整日志输出等级，启动调优任务后，将打屏显示设置级别的日志信息。[日志级别说明](../../api_reference/python_api_v0/common_apis.md#参数说明)

```python
from msmodelslim import set_logger_level
set_logger_level("info")        #根据实际情况配置
```

4.使用PruneConfig接口自定义配置剪枝的步骤和block，请参考PruneConfig进行配置。

```python
prune_config = PruneConfig()
prune_config.set_steps(['prune_blocks', 'prune_bert_intra_block']). \
    add_blocks_params(pattern="bert.encoder.layer.(\d+).",layer_id_map={0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 11})
```

- 说明：若set_steps方法中配置的剪枝步骤包含“prune_blocks”，则必须调用“add_blocks_params”方法进行配置。

5.使用prune_model_weight接口调用剪枝配置项修改预训练的模型权重，并将剪枝后的权重加载小模型中，小模型通过较小的初始化参数生成。
以Bert为例，初始化较小模型时，需提前修改bert_config下的json配置，例如intermediate_size参数改小为1536，num_hidden_layers 参数改小为7。修改后在Python脚本中导入如下内容进行配置。

```python
import modeling # 导入bert模型
bert_config = modeling.BertConfig.from_json_file(bert_config_file) # 载入bert配置，初始化较小的模型。
bert_model = modeling.BertForQuestionAnswering(bert_config) # 实例化bert模型
prune_model_weight(bert_model, prune_config, weight_file_path = "/home/xxx/xxx.pt")   #model根据实际情况配置待剪枝模型实例，weight_file_path根据实际情况配置原模型的权重文件
```

MindSpore模型的权重文件需为ckpt格式，PyTorch框架的权重文件需为pt/pth/pkl/bin格式，具体请参考prune_model_weight进行配置。

6.启动模型剪枝调优任务，将原始权重进行裁剪并载入小模型中。

```python
python3 test_prune_model.py
```

## 3. Sparse tool

### 3.1 简介

稀疏算法是一种优化深度神经网络的技术，可以将linear网络层中不必要的参数置为0，部署阶段借助昇腾芯片unzip单元在线解码能力，可获得更加轻量化的模型，以提高模型的推理速度和泛化能力。

### 3.2 功能介绍

用户需自行准备模型，模型是基于pytorch网络结构，本样例以线性层为例。

1.使用SparseConfig接口，配置稀疏参数、稀疏方式，生成稀疏化算法配置

```python
sparse_config = SparseConfig(method = "magnitude", sparse_ratio = 0.5, progressive = False, uniform = True)
```

- method: 稀疏方式，可选值为：'magnitude','hessian','par','par_v2'，默认'magnitude'
- sparse_ratio: 0~1,用户可以自行设置稀疏率, 默认0.5
- progressive: 渐进式稀疏，默认False
- uniform: 均匀稀疏，默认True

2.用户自行准备一个batch的数据集作为稀疏算法的校准数据

```python
   test_dataset = [torch.randn(64, 100)]
```

3.模型稀疏调优任务

```python
import torch
from msmodelslim.pytorch.sparse.sparse_tools import SparseConfig, Compressor

# 定义一个简单的模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = torch.nn.Linear(100, 50)
        self.linear2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

generate_model = SimpleModel()
test_dataset = [torch.randn(64, 100)]
sparse_config = SparseConfig(method="magnitude", sparse_ratio=0.5)
prune_compressor = Compressor(generate_model, sparse_config)
prune_compressor.compress(dataset=test_dataset)
```

### 3.3 示例

```python
import torch
import torch_npu
from msmodelslim.pytorch.sparse.sparse_tools import SparseConfig, Compressor

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H, bias=True)
        self.linear2 = torch.nn.Linear(H, D_out, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        y_pred = self.linear2(x)
        return y_pred

D_in, H, D_out = 100, 10, 1
model = TwoLayerNet(100, 10, 1)
test_dataset = [torch.randn(64, 100)]
sparse_config = SparseConfig(method='magnitude')
prune_compressor = Compressor(model, sparse_config)
prune_compressor.compress(dataset=test_dataset)
```

## 4. 模型蒸馏

### 4.1 简介

msModelSlim工具支持API方式的蒸馏调优。蒸馏调优时，用户只需要提供teacher模型、student模型和数据集，调用API接口完成模型的蒸馏调优过程。

模型蒸馏期间，用户可将原始Transformer模型、配置较小参数的Transformer模型分别作为teacher和student进行知识蒸馏。通过手动配置参数，返回一个待蒸馏的DistillDualModels模型实例，用户对其进行训练。训练完毕后，从DistillDualModels模型实例获取训练后的student模型，即通过蒸馏训练后的模型。

### 4.2 使用前准备

目前支持MindSpore和PyTorch框架下Transformer类模型的蒸馏调优。
安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../install_guide/install_guide.md)。

### 4.3 功能介绍

以下步骤以PyTorch框架的模型为例，MindSpore框架的模型仅在调用部分接口时，入参配置有所差异，使用时请参照具体的API接口说明。

1.用户自行准备原始Transformer模型、配置较小参数的Transformer模型，分别作为模型蒸馏调优的teacher模型和student模型。本样例以Bert为例，在ModelZoo搜索下载Bert代码和原模型权重文件。

2.新建待蒸馏模型的Python脚本，例如distill_model.py。编辑distill_model.py文件，导入如下接口。蒸馏API接口说明请参考蒸馏接口。

```python
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig, get_distill_model
```

3.（可选）调整日志输出等级，启动调优任务后，将打屏显示蒸馏调优的日志信息。

```python
from msmodelslim import set_logger_level
set_logger_level("info")        #根据实际情况配置
```

4.使用KnowledgeDistillConfig接口自定义配置模型蒸馏的参数，请参考KnowledgeDistillConfig进行配置。

```python
distill_config = KnowledgeDistillConfig()
distill_config.add_output_soft_label({
                "t_output_idx": 1,
                "s_output_idx": 1,
                "loss_func": [{"func_name": "KDCrossEntropy",
                               "func_weight": 1,
                               "temperature": 1}]})
```

5.使用get_distill_model接口调用蒸馏配置项并返回一个待蒸馏的DistillDualModels模型实例，请参考get_distill_model进行配置。teacher_model、student_model为Bert的实例，通过修改bert_configs下的json配置，初始化不同大小的Bert模型。

```python
distill_model = get_distill_model(teacher_model, student_model, distill_config)   #请传入teacher模型、student模型的实例
```

6.用户自行对待蒸馏的DistillDualModels模型实例进行训练，可参考teacher、student模型的训练脚本、MindSpore官网或PyTorch官网进行训练。以Bert为例，参考原始训练代码run_squad.py进行如下重点信息修改，并执行命令进行训练。

- 将原始代码中model = modeling.BertForQuestionAnswering(config)改为model = distill_model.student_model，从而为student模型设置optimizer。
- 将原始代码中start_logits, end_logits = model(input_ids, segment_ids, input_mask)改为loss, student_outputs, teacher_outputs = distill_model (input_ids, segment_ids, input_mask)，并注释原始的loss计算部分，从而对 DistillDualModels模型实例进行训练。
训练完成后，可以使用get_student_model方法，获取训练后的student模型（MindSpore框架的模型使用get_student_model方法后，无法再次对DistillDualModels模型实例进行训练）。

7.训练完成后，可以使用get_student_model方法，获取训练后的student模型（MindSpore框架的模型使用get_student_model方法后，无法再次对DistillDualModels模型实例进行训练）。

```python
student_model = distill_model.get_student_model()
```

8.启动模型蒸馏调优任务，将获取一个训练后的student模型。

```bash
python3 distill_model.py
```
