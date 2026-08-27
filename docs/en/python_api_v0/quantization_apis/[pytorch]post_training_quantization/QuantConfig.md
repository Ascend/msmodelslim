# QuantConfig

## Function

Quantization parameter configuration class, which stores parameters configured during the quantization process.

## Prototype

```python
QuantConfig(w_bit=8, a_bit=8, w_signed=True, a_signed=False, w_sym=True, a_sym=False, input_shape=None, act_quant=True, act_method=0, quant_mode=0, disable_names=None, amp_num=0, keep_acc=None, sigma=25, device='cpu')
```

## Parameters

| Parameter| Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| w_bit | Input | Weight quantization bit. | Optional.<br>Data Type: int.<br>Defaults to 8. Other bit quantization is currently not supported and cannot be modified. |
| a_bit | Input | Activation layer quantization bit. | Optional.<br>Data Type: int.<br>Defaults to 8. Other bit quantization is currently not supported and cannot be modified. |
| w_signed | Input | Whether to apply signed quantization to weights. | Optional.<br>Data Type: bool.<br>Defaults to True. |
| a_signed | Input | Whether to apply signed quantization to activations. | Optional.<br>Data Type: bool.<br>Defaults to False.<br>It is recommended to set to False for CV models using ReLU, and True for other models. |
| w_sym | Input | Whether to apply symmetric quantization to weights. | Optional.<br>Data Type: bool.<br>Defaults to True. |
| a_sym | Input | Whether to apply symmetric quantization to activations. | Optional.<br>Data Type: bool.<br>Defaults to False. |
| input_shape | Input | The shape of the model input, used to construct virtual data for Label-Free quantization.<br>(1) Currently only supports models with a single input and float input data format.<br>(2) For models with multiple inputs or requiring custom input formats, if Label-Free quantization is needed, users can construct virtual input data themselves without specifying input_shape. | Optional, but must be specified when the model supports dynamic shapes.<br>Data Type: list [list [int]] <br>Default: `None` |
| act_quant | Input | Whether to quantize activations. | Optional.<br>Data Type: bool.<br>Defaults to True.<br>Modification is currently not supported. |
| act_method | Input | Activation quantization method. | Optional.<br>Data Type: int.<br>Optional values: [0, 1, 2]. Defaults to 0.<br>(1) 0 represents the Data-Free quantization method (specifically determined by the sigma parameter).<br>(2) 1 represents the min-max observer method for Label-Free scenarios. 1 is recommended for Label-Free scenarios.<br>(3) 2 represents the histogram observer method for Label-Free scenarios. |
| quant_mode | Input | Quantization mode. | Optional.<br>Data Type: int.<br>Optional values: [0, 1]. Defaults to 0.<br>(1) 0 represents Data-Free quantization mode.<br>(2) 1 represents Label-Free quantization mode. |
| disable_names | Input | Names of nodes to be excluded from quantization, i.e., names of quantization layers for manual fallback.<br>If accuracy is poor, it is recommended to fallback quantization-sensitive layers, such as classification layers, input layers, detection head layers, etc. | Optional.<br>Data Type: list[str].<br>Default: []. |
| amp_num | Input | Number of mixed precision quantization fallback layers.<br>When accuracy drops significantly, the number of fallback layers can be increased. It is recommended to prioritize falling back 3~7 layers. If accuracy recovery is not significant, further increase the number of fallback layers. | Optional.<br>Data Type: int.<br>Defaults to 0. |
| keep_acc | Input | Accuracy preservation strategy.<br>(1) admm and round_opt are used to improve weight quantization and reduce weight quantization error. They are recommended for use in Label-Free mode to moderately improve quantization effects.<br>(2) easy_quant is used to improve activation quantization and reduce activation quantization error. It is recommended for use in Label-Free mode and usually achieves good improvement effects. | Optional.<br>Data Type: dict.<br>Contains the following three accuracy preservation strategies:<br>(1) admm strategy: Data Type [bool, int], bool configures whether to enable, int configures the number of optimization iterations. (2) easy_quant: Data Type [bool, int], bool configures whether to enable, int configures the number of optimization iterations. (3) round_opt: Data Type [bool], bool configures whether to enable.<br>Input template: keep_acc={'admm': [False, 1000], 'easy_quant': [False, 1000], 'round_opt': False}. |
| sigma | Input | Label-Free quantization statistics method.<br>Recommended input values are 0 or 25. Convolutional models generally perform better with sigma statistics, while transformer models perform better with min-max statistics. | Optional.<br>Data Type: int.<br>Defaults to 25.<br>(1) When sigma=25, the sigma statistics method is used.<br>(2) When sigma=0, the min-max statistics method is used. |
| device | Input | Selects the device for model execution. | Optional.<br>Optional values: ["cpu", "npu"].<br>Data Type: str.<br>Defaults to "cpu". Note: Currently, only multimodal quantization scenarios support "npu", and multimodal quantization scenarios only support "npu". |

## Sample

```python
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig
disable_names = []
input_shape = [1, 3, 224, 224]
keep_acc={'admm': [False, 1000], 'easy_quant': [False, 1000], 'round_opt': False}
quant_config = QuantConfig(disable_names=disable_names, amp_num=0, input_shape=input_shape, keep_acc=keep_acc)
```
