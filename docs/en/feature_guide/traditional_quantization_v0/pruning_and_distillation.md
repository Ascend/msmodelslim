# Training Acceleration and Model Reconstruction

## Pruning Based on Importance Evaluation

### Overview

msModelSlim provides model pruning APIs based on importance evaluation. You only need to provide a model instance and call the pruning APIs to prune a model. The pruned model achieves improved performance and reduced size, leading to higher inference efficiency.

### Preparations

Currently, model pruning is supported under the PyTorch framework. 
Install msModelSlim. For details, see [msModelSlim Installation Guide](../../getting_started/install_guide.md).

- Note: This feature supports only PyTorch 2.0.0 or later.

### Function

### Procedure

1. Prepare the model instance to be pruned and the corresponding training script. The following example shows how to perform configuration by using VGG16 from `torchvision`.

2. Open the training script `vision/references/classification/train.py` of the model to be pruned. Edit the `train.py` file and import the pruning APIs. For details about the pruning APIs, see the `PruneTorch` documentation.

    ```python
    from msmodelslim.pytorch.prune.prune_torch import PruneTorch
    ```

3. (Optional) Adjust the log output level. After starting the tuning task, the system displays log information at the specified level. For details, see [Log Level Description](../../python_api_v0/common_apis.md#parameters).

    ```python
    from msmodelslim import set_logger_level
    set_logger_level("info")        # Set this parameter as needed.
    ```

4. After initializing the network and loading the weights in the original script, use the `PruneTorch` APIs to configure the importance evaluation function, the parameter retention ratio for operator nodes, and the pruning rate.

    ```python
    desc = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32)).prune(0.8)
    ```

5. Start the model pruning task. You are advised to use the final learning rate from the original training process and execute training for 10 epochs.

    ```bash
    python3 train.py --model vgg16 --lr 1e-5 --epochs 10 --pretrained --batch-size 256 -j 48
    ```

    This process generates a pruned model for subsequent training tasks.

6. During subsequent evaluation steps, load the model pruning information returned in step 4 by referring to the following configuration example.

    ```python
    PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32)).prune_by_desc(desc)
    ```

## Weight Pruning for Transformer Models

### Overview

msModelSlim provides API-based weight pruning for transformer models. This feature prunes model weights and loads them into a smaller model instance sharing the same architecture. You only need to provide the smaller model instance (obtained by specifying smaller initialization parameters, such as reducing the `intermediate_size` and `num_hidden_layers` parameters in a BERT model) along with the original model weight file, and then call the pruning APIs to prune the weights.

### Preparations

Currently, weight pruning for transformer models is supported under the MindSpore and PyTorch frameworks. 
Install msModelSlim. For details, see [msModelSlim Installation Guide](../../getting_started/install_guide.md).

- Note: This feature supports only PyTorch 2.0.0 or later.

During model pruning, you can manually configure parameters to prune the weights of a pre-trained model and load the pruned weights into the smaller model to obtain a transformer model with fully loaded weights. The accuracy of the pruned model is not guaranteed immediately after pruning. You must perform subsequent training, such as training through model distillation, to improve the accuracy.

### Function

### Procedure

The following procedure uses a transformer model under the PyTorch framework as an example. The input parameter configurations for MindSpore models vary only for certain API calls. For details, refer to the corresponding API specifications.

1. Prepare the original model instance (the model to be pruned) and its weight file sharing the same architecture. This example uses BERT as an example. Search for and download the BERT code and the original model weight file from ModelZoo.

2. Create a Python script for the model to be pruned, for example, `test_prune_model.py`. Edit the `test_prune_model.py` file and import the following APIs. For details about the pruning APIs, see the pruning API descriptions.

    ```python
    from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig
    from msmodelslim.common.prune.transformer_prune.prune_model import prune_model_weight
    ```

3. (Optional) Adjust the log output level. After starting the tuning task, the system displays log information at the specified level. For details, see [Log Level Description](../../python_api_v0/common_apis.md#parameters).

    ```python
    from msmodelslim import set_logger_level
    set_logger_level("info")        # Set this parameter as needed.
    ```

4. Use the `PruneConfig` APIs to configure parameters for the pruning steps and blocks. For details, see the `PruneConfig` documentation.

    ```python
    prune_config = PruneConfig()
    prune_config.set_steps(['prune_blocks', 'prune_bert_intra_block']). \
        add_blocks_params(pattern="bert.encoder.layer.(\d+).",layer_id_map={0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 11})
    ```

    - Note: If the pruning steps configured in the `set_steps` method contain `prune_blocks`, you must call the `add_blocks_params` method for configuration.

5. Call the `prune_model_weight` API to invoke pruning configuration items to modify the pre-trained model weights and load the pruned weights into the smaller model, which is generated using smaller initialization parameters.
    The following example shows how to perform configuration using BERT. When initializing a smaller model, modify the JSON configuration under `bert_config` in advance. For example, set the value of the `intermediate_size` parameter to `1536` and `num_hidden_layers` parameter to `7`. After the modification, import the following content into the Python script:

    ```python
    import modeling # Import the BERT model.
    bert_config = modeling.BertConfig.from_json_file(bert_config_file) # Load the BERT configuration and initialize a smaller model.
    bert_model = modeling.BertForQuestionAnswering(bert_config) # Instantiate the BERT model.
    prune_model_weight(bert_model, prune_config, weight_file_path = "/home/xxx/xxx.pt")   # Configure the target model to be pruned based on the actual model instance, and specify the original model weight file path based on the actual file path.
    ```

    The weight file for a MindSpore model must be in CKPT format. Weight files for the PyTorch framework must be in PT, PTH, PKL, or BIN format. For details, see the `prune_model_weight` documentation.

6. Start the model pruning task to prune the original weights and load them into the smaller model.

    ```python
    python3 test_prune_model.py
    ```

## Sparse Tool

### Overview

Sparsification algorithms optimize deep neural networks by setting unnecessary parameters in linear layers to 0. During deployment, the on-chip unzip unit of Ascend processors enables online weight decoding, which yields a more lightweight model and improves both inference speed and generalization performance.

### Function

You must prepare a model based on the PyTorch framework architecture. The following example shows how to perform configuration using a linear layer.

1. Use the `SparseConfig` APIs to configure sparsity parameters and methods, which generates the sparsity algorithm configuration.

    ```python
    sparse_config = SparseConfig(method = "magnitude", sparse_ratio = 0.5, progressive = False, uniform = True)
    ```

    - `method`: specifies the sparsification method. Valid values: `"magnitude"`, `"hessian"`, `"par"`, or `"par_v2"`. Default value: `"magnitude"`.
    - `sparse_ratio`: specifies the sparsification ratio, ranging from 0 to 1. Set this parameter as needed. Default value: `0.5`.
    - `progressive`: specifies whether to enable progressive sparsification. Default value: `False`.
    - `uniform`: specifies whether to enable uniform sparsification. Default value: `True`.

2. Prepare a single batch dataset to serve as the calibration data for the sparsification algorithm.

    ```python
    test_dataset = [torch.randn(64, 100)]
    ```

3. Execute the model sparsification tuning task.

    ```python
    import torch
    from msmodelslim.pytorch.sparse.sparse_tools import SparseConfig, Compressor

    # Define a simple model.
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

### Example

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

## Model Distillation

### Overview

msModelSlim provides API-based knowledge distillation for model tuning. You only need to provide a teacher model, a student model, and a dataset, and then call the distillation APIs to execute the distillation tuning process.

During model distillation, you can use the original transformer model and a transformer model configured with smaller parameters as the teacher and student models, respectively. Manually configuring the parameters returns a `DistillDualModels` model instance to be distilled, which can be used for training. After training is complete, you can obtain the trained student model from the `DistillDualModels` model instance.

### Preparations

Currently, distillation tuning for transformer models is supported under the MindSpore and PyTorch frameworks.
Install msModelSlim. For details, see [msModelSlim Installation Guide](../../getting_started/install_guide.md).

### Function

The following procedure uses a transformer model under the PyTorch framework as an example. The input parameter configurations for MindSpore models vary only for certain API calls. For details, refer to the corresponding API specifications.

1. Prepare the original transformer model and a transformer model configured with smaller parameters to serve as the teacher model and student model for the distillation tuning process. The following example shows how to perform configuration using BERT. Search for and download the BERT code and the original model weight file from ModelZoo.

2. Create a Python script for the model to be distilled, such as `distill_model.py`. Edit the `distill_model.py` file and import the following APIs. For details about the distillation API specifications, see the distillation API descriptions.

    ```python
    from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig, get_distill_model
    ```

3. (Optional) Adjust the log output level. After starting the tuning task, the system displays the distillation tuning log information on the screen.

    ```python
    from msmodelslim import set_logger_level
    set_logger_level("info")        # Set this parameter as needed.
    ```

4. Use the `KnowledgeDistillConfig` APIs to configure parameters for model distillation. For details, see the `KnowledgeDistillConfig` documentation.

    ```python
    distill_config = KnowledgeDistillConfig()
    distill_config.add_output_soft_label({
                    "t_output_idx": 1,
                    "s_output_idx": 1,
                    "loss_func": [{"func_name": "KDCrossEntropy",
                                "func_weight": 1,
                                "temperature": 1}]})
    ```

5. Use the `get_distill_model` API to invoke distillation configuration items and return a `DistillDualModels` model instance to be distilled. For details, see the `get_distill_model` documentation. The `teacher_model` and `student_model` arguments represent BERT instances. You can modify the JSON configuration under `bert_configs` to initialize BERT models of different sizes.

    ```python
    distill_model = get_distill_model(teacher_model, student_model, distill_config) # Pass the instances of the teacher and student models.
    ```

6. Train the `DistillDualModels` model instance. For details, refer to the training scripts of the teacher and student models, or visit the official MindSpore or PyTorch websites. The following example shows how to perform configuration using BERT. Modify the key information by referring to the original training code `run_squad.py`, and then execute the command to perform training:

    - Change `model = modeling.BertForQuestionAnswering(config)` in the original code to `model = distill_model.student_model` to configure the optimizer for the student model.
    - Change `start_logits, end_logits = model(input_ids, segment_ids, input_mask)` in the original code to `loss, student_outputs, teacher_outputs = distill_model(input_ids, segment_ids, input_mask)` and comment out the original loss calculation section to train the `DistillDualModels` model instance.
 
7. Use the `get_student_model` method to obtain the trained student model after training is complete. (For models under the MindSpore framework, you cannot train the `DistillDualModels` model instance again after executing the `get_student_model` method.)

    ```python
    student_model = distill_model.get_student_model()
    ```

8. Execute the model distillation tuning task to obtain the trained student model.

```bash
python3 distill_model.py
```
