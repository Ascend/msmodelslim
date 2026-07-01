# 稀疏加速训练

## 1. 简介

深度学习的一次训练往往涉及到几万甚至上百万的迭代次数，其中存在着大量的计算冗余，本算法基于网络扩增训练的思想，结合参数继承方法，分别实现了宽度、深度两个层面的网络扩增算法应对不同场景。

## 2. 功能介绍

### 2.1 宽度扩增模型稀疏训练加速

#### 2.1.1 基本使用流程

  - 在模型以及优化器 optimizer 初始化后，使用 `sparse_model_width` 将模型及优化器包装为稀疏化训练的调用

  ```py
  from msmodelslim.pytorch.sparse import sparse_model_width

  model = sparse_model_width(model, optimizer, steps_per_epoch=100, epochs_each_stage=[10, 20, -1])
  ```

#### 2.1.2 接口说明

  - **sparse_model_width** 提供对外接口
  - 参数 **model** 初始化后的 PyTorch 模型
  - 参数 **optimizer** 初始化后的 PyTorch 优化器 optimizer
  - 参数 **steps_per_epoch** 数据集单个 epoch 需要的迭代数，int 值，一般为数据集按照 batch 划分之后的长度 `len(train_loader)`
  - 参数 **epochs_each_stage** 稀疏化每个阶段的 epoch 数量，列表值，如 `[10, 20, -1]` 表示分 3 个阶段
    - 第 1 个阶段，从原模型裁剪为 `1/4` 的初始模型开始训练 10 个 epoch
    - 第 2 个阶段将初始模型扩增 2 倍，训练 20 个 epoch
    - 第 3 个阶段 epoch 数量 `-1` 表示训练直到总的 epoch 结束，初始模型扩增为 4 倍，恢复为原模型大小训练

#### 2.1.3 样例

  ```py
  import os
import torch
import torch_npu
import apex
from torch import nn
from apex import amp

from ascend_utils.common.utils import count_parameters
from msmodelslim.pytorch import sparse

device = torch.device("npu:{}".format(os.getenv('DEVICE_ID', 0)))
torch.npu.set_device(device)

model = nn.Sequential(
    nn.Conv2d(3, 32, 1, 1, bias=False),
    nn.Sequential(nn.Conv2d(32, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 32, 1, 1, bias=False)),
    nn.Sequential(nn.Conv2d(32, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 32, 1, 1, bias=False)),
    nn.Sequential(nn.Conv2d(32, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 32, 1, 1, bias=False)),
    nn.Sequential(nn.Conv2d(32, 64, 1, 1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 32, 1, 1, bias=False)),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 10, bias=False),
).to(device)

optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), lr=0.1)

steps_per_epoch, epochs_each_stage = 10, [2, 3, 1]
original_model_params = count_parameters(model)  # 10826
model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O2", combine_grad=False)

# 添加宽度稀疏化训练方式
model = sparse.sparse_model_width(
    model, optimizer, steps_per_epoch=steps_per_epoch, epochs_each_stage=epochs_each_stage
)

# 模型训练
for _ in range(steps_per_epoch * sum(epochs_each_stage)):
    optimizer.zero_grad()
    output = model(torch.ones([1, 3, 32, 32]).npu())
    loss = torch.mean(output)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
  ```

***

### 2.2 深度扩增模型稀疏训练加速

#### 2.2.1 基本使用流程

  - 在模型以及优化器 optimizer 初始化后，使用 `sparse_model_depth` 将模型及优化器包装为稀疏化训练的调用

  ```py
  from msmodelslim.pytorch.sparse import sparse_model_depth

  model = sparse_model_depth(model, optimizer, steps_per_epoch=100, epochs_each_stage=[10, 20, -1])
  ```

#### 2.2.2 接口说明

  - **sparse_model_depth** 提供对外接口
  - 参数 **model** 初始化后的 PyTorch 模型
  - 参数 **optimizer** 初始化后的 PyTorch 优化器 optimizer
  - 参数 **steps_per_epoch** 数据集单个 epoch 需要的迭代数，int 值，一般为数据集按照 batch 划分之后的长度 `len(train_loader)`
  - 参数 **epochs_each_stage** 稀疏化每个阶段的 epoch 数量，列表值，如 `[10, 20, -1]` 表示分 3 个阶段
    - 第 1 个阶段，从原模型裁剪为 `1/4` 的初始模型开始训练 10 个 epoch
    - 第 2 个阶段将初始模型扩增 2 倍，训练 20 个 epoch
    - 第 3 个阶段 epoch 数量 `-1` 表示训练直到总的 epoch 结束，初始模型扩增为 4 倍，恢复为原模型大小训练

#### 2.2.3 样例

  - `宽度扩增模型稀疏训练加速` - `样例` 中使用 `sparse_model_depth` 替换 `sparse_model_width`
