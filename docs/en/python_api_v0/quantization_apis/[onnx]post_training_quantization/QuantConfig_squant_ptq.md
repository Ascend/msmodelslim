# QuantConfig

## Function

Quantization parameter configuration class, which stores the parameters configured during the quantization process.

## Prototype

```python
QuantConfig(w_bit=8, a_bit=8, w_signed=True, a_signed=False, w_sym=True, a_sym=False, input_shape=None, act_quant=True, act_method=0, quant_mode=0, disable_names=None, amp_num=0, squant_mode='squant', keep_acc=None, sigma=25, is_fp=False, disable_first_layer=True, disable_last_layer=True, is_optimize_graph=True, is_dynamic_shape=False, use_onnx=True, num_input=0, quant_param_ops=None, atc_input_shape=None, graph_optimize_level=0, shut_down_structures=None, device_id=0, om_method='aoe')
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| w_bit | Input | Weight quantization bit. | Optional.<br>Data Type: int.<br>Default value is 8. Other bit quantization is currently not supported and cannot be modified. |
| a_bit | Input | Activation layer quantization bit. | Optional.<br>Data Type: int.<br>Default value is 8. Other bit quantization is currently not supported and cannot be modified. |
| w_signed | Input | Whether to perform signed quantization on weights. | Optional.<br>Data Type: bool.<br>Default value: True. |
| a_signed | Input | Whether to perform signed quantization on activations. | Optional.<br>Data Type: bool.<br>Default value: False.<br>It is recommended to set to False for CV models using relu, and True for other models. |
| w_sym | Input | Whether weights use symmetric quantization. | Optional.<br>Data Type: bool.<br>Default value: True. |
| a_sym | Input | Whether activations use symmetric quantization. | Optional.<br>Data Type: bool.<br>Default value: False. |
| input_shape | Input | When the input model supports dynamic shapes, the user must specify the input_shape parameter to generate calibration data for quantization. | Optional, but must be specified when the model supports dynamic shapes.<br>Data Type: list [list]<br>Default Value: [] <br>When the model has multiple inputs, specify input_shape in order, for example: \[[1, 3,224, 224], [1, 3, 640, 640]]. |
| act_quant | Input | Whether to quantize activations. | Optional.<br>Data Type: bool.<br>Default value: True.<br>Modification is currently not supported. |
| act_method | Input | Activation quantization method. | Optional.<br>Data Type: int.<br>Optional values [0,1,2], default value is 0.<br>(1) 0 represents the quantization method for Data-Free scenarios (specifically determined by the sigma parameter).<br>(2) 1 represents the min-max observer method for Label-Free scenarios. Option 1 is recommended for Label-Free scenarios.<br>(3) 2 represents the histogram observer method for Label-Free scenarios. |
| quant_mode | Input | Quantization mode. | Optional.<br>Data Type: int.<br>Optional values are [0,1], default value is 0.<br>(1) 0 represents the Data-Free quantization mode.<br>(2) 1 represents the Label-Free quantization mode. |
| disable_names | Input | Names of nodes to be excluded from quantization, i.e., the names of quantization layers to be manually rolled back.<br>If accuracy is poor, it is recommended to roll back quantization-sensitive layers, such as classification layers, input layers, and detection head layers. | Optional.<br>Data Type: list[str].<br>Default value []. |
| amp_num | Input | Number of layers rolled back for mixed-precision quantization.<br>When accuracy drops significantly, the number of rolled-back layers can be increased. It is recommended to prioritize rolling back 3~7 layers. If accuracy recovery is not obvious, increase the number of rolled-back layers further. | Optional.<br>Data Type: int.<br>Default value: 0. |
| squant_mode | Input | Quantization method. | Optional.<br>Data Type: String.<br>Supports configuration with the default value 'squant' (Data-Free quantization algorithm). Modification is currently not supported. |
| keep_acc | Input | Accuracy preservation strategy.<br>(1) admm and round_opt are used to improve weight quantization and reduce weight quantization errors. They are recommended for use in Label-Free mode to moderately improve quantization effects.<br>(2) easy_quant is used to improve activation quantization and reduce activation quantization errors. It is recommended for use in Label-Free mode and usually achieves good improvement effects. | Optional.<br>Data Type: dict.<br>Contains the following three accuracy preservation strategies:<br>(1) admm strategy: Data Type [bool, int], bool configures whether to enable, int configures the number of optimization iterations. (2) easy_quant: (Recommended) Data Type [bool, int], bool configures whether to enable, int configures the number of optimization iterations. (3) round_opt: Data Type [bool], bool configures whether to enable.<br>Input template is: keep_acc={'admm': [False, 1000], 'easy_quant': [False, 1000], 'round_opt': False}. |
| sigma | Input | Quantization statistics method for Label-Free.<br>Recommended input values are 0 or 25. Convolutional models tend to perform better with sigma statistics, while transformer models tend to perform better with min-max statistics. | Optional.<br>Data Type: int.<br>Default value: 25.<br>(1) When sigma=25, the sigma statistical method is used.<br>(2) When sigma=0, the Min-Max statistical method is used. |
| is_fp | Input | Whether to enable layer-wise quantization calibration. | Optional.<br>Data Type: bool.<br>Default value: False. Modification is currently not supported. |
| disable_first_layer | Input | Whether to automatically roll back the first quantization layer. | Optional.<br>Data Type: bool.<br>Default value: True. |
| disable_last_layer | Input | Whether to automatically roll back the last quantization layer. | Optional.<br>Data Type: bool.<br>Default value: True. |
| is_optimize_graph | Input | Whether to perform graph optimization. | Optional.<br>Data Type: bool.<br>Default value: True. |
| is_dynamic_shape | Input | Specifies whether the input model supports dynamic shapes. | Optional. When the input model supports dynamic shapes, another configuration parameter input_shape must also be specified.<br>Data Type: bool.<br>Default value: False.<br>True: The input model supports dynamic shapes. False: The input model has static shapes. |
| use_onnx | Input | Indicates whether to use onnx_runtime for quantization calibration (onnx_runtime only supports model quantization for models < 2GB). If the model is larger than 2GB, it is recommended to disable this parameter and use ACL for calibration. | Optional.<br>Data Type: bool.<br>Default value: True.<br>True: onnx_runtime quantization calibration. False: ACL quantization calibration. |
| num_input | Input | The number of network input data. | Optional.<br>Data Type: int.<br>Default value: 0. If use_onnx is configured to False, the number of model input data must be manually entered. |
| quant_param_ops | Input | Selects the network layers to be quantized. | Optional.<br>Data Type: list.<br>Default value: ['Conv', 'Gemm', 'MatMul'].<br>If using ACL quantization calibration to assist quantization (i.e., use_onnx is configured to False), this parameter must be configured as ['Conv']. |
| atc_input_shape | Input | The input data shape for converting the model to an om model using the ATC tool. | Optional.<br>Data Type: String.<br>Default value: None.<br>If use_onnx is configured to False, the input shape of the model must be manually entered. The input format requirements are as follows:<br>(1) If the model has a single input, the shape information is "input_name:n,c,h,w"; the specified node must be enclosed in double quotes.<br>(2) If the model has multiple inputs, the shape information is --input_shape="input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2"; different inputs are separated by English semicolons. input_name must be the node name in the network model before conversion. |
| graph_optimize_level | Input | Graph optimization level. | Optional.<br>Data Type: int.<br>Values are as follows:<br>(1) 0: Default value is 0. No graph optimization is performed on either the floating-point model or the quantized model.<br>(2) 1: Graph optimization is performed only on the floating-point model.<br>(3) 2: Graph optimization is performed on both the floating-point model and the quantized model. |
| shut_down_structures | Input | List of graph optimization structures to be disabled. | Optional.<br>Data Type: list.<br>Default value: None, meaning all optimizable structures are quantized.<br>Value range: ['ChangeGAPCONVOptimization', 'ChangeResizeOptimization', 'CombineMatmulOptimization', 'DeleteConcatOptimization', 'DoubleFuseBatchNormOptimization', 'DoubleReshapeOptimization', 'FastClipOptimization', 'FuseBatchNormOptimization', 'FuseDivMatmulOptimization', 'GeluErf2FastGeluOptimization', 'GeluErf2SigmoidOptimization', 'GeluErf2TanhOptimization', 'GeluTanh2SigmoidOptimization', 'LayerNormOptimization', 'Matmul2GemmOptimization', 'PatchMerging2ConvOptimizationV0', 'PatchMerging2ConvOptimizationV1', 'PatchMerging2ConvOptimizationV2', 'PatchMerging2ConvOptimizationV3', 'RemoveDoubleResizeOptimization', 'ReplaceAscendQuantOptimizationV1', 'ReplaceAscendQuantOptimizationV2', 'ReplaceConcatQuantOptimizationV1', 'ReplaceConcatQuantOptimizationV2', 'ReplaceConcatQuantOptimizationV3', 'ReplaceConcatQuantOptimizationV4', 'ReplaceConcatQuantOptimizationV5', 'ReplaceConcatQuantOptimizationV6', 'ReplaceConcatQuantOptimizationV7', 'ReplaceConcatQuantOptimizationV8', 'ReplaceConcatQuantOptimizationV9', 'ReplaceHardSigmoidOptimization', 'ReplaceLeakyReluOptimization', 'ReplaceMaxPoolBlockOptimizationV1', 'ReplaceMaxPoolBlockOptimizationV2', 'ReplaceRelu6Optimization', 'ReplaceReluOptimization', 'ReplaceReshapeTransposeOptimizationV1', 'ReplaceReshapeTransposeOptimizationV2', 'ReplaceReshapeTransposeOptimizationV3', 'ReplaceResizeQuantOptimization', 'ReplaceSigmoidOptimizationV1', 'ReplaceSigmoidOptimizationV2', 'ReplaceSoftmaxOptimizationV1', 'ReplaceSoftmaxOptimizationV2', 'Resize2ConvTransposeOptimization', 'SimplifyShapeOptimization', 'SimplifyShapeOptimizationV2']. |
| device_id | Input | The DEVICE ID of the Ascend AI processor. | Optional.<br>Data Type: int.<br>Value range [0,7], default value is 0. |
| om_method | Input | The method for converting an onnx model to an om model. | Optional.<br>Data Type: String.<br>Supports configuration as 'aoe' and 'atc', default value is 'aoe', meaning conversion is performed using the aoe tool. |

## Sample

```python
from msmodelslim.onnx.squant_ptq import QuantConfig
config = QuantConfig(disable_names=[],
                      quant_mode=0,
                      amp_num=0,
                      a_sym=True,
                      keep_acc={'admm': [False, 1000], 'easy_quant': [True, 1000], 'round_opt': False},
                      disable_first_layer=True,
                      disable_last_layer=True
)
```
