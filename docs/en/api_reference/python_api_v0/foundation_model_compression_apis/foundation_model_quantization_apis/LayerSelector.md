# LayerSelector

## Function

Constructs a class for layer selection, which helps users select layers to skip quantization by analyzing the quantization difficulty of each layer in the model.

## Prototype

```python
LayerSelector(model, layer_names=None, range_method="quantile")
```

## Parameters

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| model | Input | The model for which layer selection analysis is to be performed. | Required.<br>Data type: torch.nn.Module |
| layer_names | Input | A list of layer names to be analyzed. | Optional.<br>Data type: list.<br>Default value: None, indicating that all linear and convolutional layers are analyzed. |
| range_method | Input | The method used to calculate quantization difficulty. | Optional.<br>Data type: str.<br>Default value: "quantile".<br>Optional values: "quantile" or "std". |

## API Description

### run

```python
run(calib_data)
```

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| calib_data | Input | Calibration data used to analyze the quantization difficulty of layers. | Required.<br>Data type: list.<br>Input template: \[[input1],[input2],[input3]]. |

### select_layers_by_threshold

```python
select_layers_by_threshold(threshold)
```

| Parameter | Input/Return Value | Description | Usage Constraints |
| ------ | ------ | ------ | ------ |
| threshold | Input | Quantization Difficulty threshold. | Required.<br>Data type: float.<br>Value Range: >0. |
| Return value | Return | A list of layer names whose Quantization Difficulty exceeds the threshold. | Data type: list. |

### select_layers_by_disable_level

```python
select_layers_by_disable_level(disable_level)
```

| Parameter | Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| disable_level | Input | The number of layer levels to skip quantization. | Required.<br>Data type: int.<br>Value range: ≥0. |
| Return value | Return | A list of layer names in the selected levels. | Data type: list. |

## Sample

Use LayerSelector to analyze the Quantization Difficulty of each model layer and select the layers to skip quantization based on actual requirements.

```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.layer_select import LayerSelector

# Initialize LayerSelector.
layer_selector = LayerSelector(model, range_method="quantile")

# Run analysis.
layer_selector.run(calib_data)

# Select layers by threshold.
disable_names = layer_selector.select_layers_by_threshold(threshold=1.0)

# Or select layers by level.
disable_names = layer_selector.select_layers_by_disable_level(disable_level=10)

# Apply selected layers to quantization config.
quant_config = QuantConfig(disable_names=disable_names)
calibrator = Calibrator(model, quant_config, calib_data=calib_data)
calibrator.run() 
```
