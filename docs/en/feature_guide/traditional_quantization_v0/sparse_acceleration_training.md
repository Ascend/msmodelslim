# Sparse Training Acceleration

## Overview

Deep learning training often involves tens of thousands or millions of iterations, introducing significant computational redundancy. Based on network augmentation principles and combined with parameter inheritance methods, this algorithm provides width-level and depth-level network augmentation capabilities to handle different deployment scenarios.

## Function

### Width-Augmented Model Sparse Training Acceleration

#### Basic Workflow

  - After initializing the model and the optimizer, use `sparse_model_width` to wrap both components for sparse training execution:

  ```py
  from msmodelslim.pytorch.sparse import sparse_model_width

  model = sparse_model_width(model, optimizer, steps_per_epoch=100, epochs_each_stage=[10, 20, -1])
  ```

#### API Description

  - `sparse_model_width` provides the following external interface parameters:
  - `model`: initialized PyTorch model instance.
  - `optimizer`: initialized PyTorch optimizer instance.
  - `steps_per_epoch`: iterations required for a single epoch. The value is an `int`, matching the dataset batch length (typically `len(train_loader)`).
  - `epochs_each_stage`: epoch count for each sparsification phase. The value is a list (such as `[10, 20, -1]` for a three-phase workflow).
    - Phase 1: The original model is pruned to an initial `1/4` scale and trained for 10 epochs.
    - Phase 2: The initial model is expanded by a factor of 2 and trained for 20 epochs.
    - Phase 3: An epoch count of `-1` specifies that training continues until total execution completes. The initial model is expanded by a factor of 4, restoring it to the original model size.

#### Sample

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

# Add width-level sparse training wrapper
model = sparse.sparse_model_width(
    model, optimizer, steps_per_epoch=steps_per_epoch, epochs_each_stage=epochs_each_stage
)

# Execute model training
for _ in range(steps_per_epoch * sum(epochs_each_stage)):
    optimizer.zero_grad()
    output = model(torch.ones([1, 3, 32, 32]).npu())
    loss = torch.mean(output)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
  ```

***

### Depth-Augmented Model Sparse Training Acceleration

#### Basic Workflow

  - After initializing the model and the optimizer, use `sparse_model_depth` to wrap both components for sparse training execution:

  ```py
  from msmodelslim.pytorch.sparse import sparse_model_depth

  model = sparse_model_depth(model, optimizer, steps_per_epoch=100, epochs_each_stage=[10, 20, -1])
  ```

#### API Description

  - `sparse_model_depth` provides the following external interface parameters:
  - `model`: initialized PyTorch model instance.
  - `optimizer`: initialized PyTorch optimizer instance.
  - `steps_per_epoch`: iterations required for a single epoch. The value is an `int`, matching the dataset batch length (typically `len(train_loader)`).
  - `epochs_each_stage`: epoch count for each sparsification phase. The value is a list (such as `[10, 20, -1]` for a three-phase workflow).
    - Phase 1: The original model is pruned to an initial `1/4` scale and trained for 10 epochs.
    - Phase 2: The initial model is expanded by a factor of 2 and trained for 20 epochs.
    - Phase 3: An epoch count of `-1` specifies that training continues until total execution completes. The initial model is expanded by a factor of 4, restoring it to the original model size.

#### Sample

  - To execute depth-augmented sparse training, substitute the `sparse_model_width` API call inside the **Width-Augmented Model Code Sample** block with the `sparse_model_depth`.
