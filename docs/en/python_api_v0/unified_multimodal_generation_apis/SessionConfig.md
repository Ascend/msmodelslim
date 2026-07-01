# Quantization Session Configuration

## SessionConfig

### Function

The quantization session configuration class is used to configure quantization-related parameters, calibration data, and the runtime device.

### Class Prototype

```python
class SessionConfig(BaseModel):
    processor_cfg_map: Dict[str, BaseModel] = {}
    calib_data: Optional[List[Any]] = None
    device: str = 'cpu'
```

### Parameters

| Parameter| Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| processor_cfg_map | Input | Quantization processor configuration map | Required.<br>Data type: Dictionary. Defaults to {}. When using quantization functionality, at least one quantization processor (such as W8A8ProcessorConfig) must be configured. The save processor (SaveProcessorConfig) cannot be configured alone.<br>Each key-value pair corresponds to a quantization processor name and a quantization processor configuration class. Currently available quantization processor names: ['m3', 'm4', 'm6', 'w8a8', 'w8a8_dynamic', 'w8a8_timestep', 'fa3', 'save'], which correspond one-to-one with the available quantization processor configurations [M3ProcessorConfig, M4ProcessorConfig, M6ProcessorConfig, W8A8ProcessorConfig, W8A8DynamicProcessorConfig, W8A8TimeStepProcessorConfig, FA3ProcessorConfig, SaveProcessorConfig]. Among them, 'fa3' must be used together with 'w8a8_dynamic'.|
| calib_data | Input | Outlier suppression and quantization calibration data | Optional.<br>Data type: List. Defaults to None, which is for the Data-Free scenario. Input is required for the Label-Free scenario. In multimodal generative model quantization scenarios, calibration data must be dumped in advance, loaded, and passed in as calib_data.|
| device | Input | Device on which the quantization process runs | Optional.<br>Data type: String. Defaults to 'cpu'. Optional values: ['cpu', 'npu'].|

### Sample

```python
import torch
from ascend_utils.common.security.pytorch import safe_torch_load
from msmodelslim.quant.session.session import W8A8ProcessorConfig, W8A8QuantConfig, SaveProcessorConfig
from msmodelslim.quant.session.session import SessionConfig

session_config = SessionConfig(
    processor_cfg_map={
        "w8a8": W8A8ProcessorConfig(
            cfg=W8A8QuantConfig(
                act_method='minmax'
            ),
            disable_names=[]
        ),
        "save": SaveProcessorConfig(
            output_path="./",
            safetensors_name=None,
            json_name=None,
            save_type=['safe_tensor'],
            part_file_size=None
        )
    },
    calib_data=safe_torch_load("calib_data.pth"),
    device="npu"
)
```

## W8A8ProcessorConfig

### Function

W8A8 quantization processor configuration class, used to configure parameters related to the W8A8 quantization processor.

### Class Prototype

```python
class W8A8ProcessorConfig(BaseModel):
    cfg: W8A8QuantConfig
    disable_names: list
```

### Parameters

| Parameter| Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| cfg | Input | W8A8 quantization configuration | Required.<br>Data type: W8A8QuantConfig, the W8A8 quantization configuration class.|
| disable_names | Input | Fallback layers | Required.<br>Data type: List. Each element in the list is a fallback layer name.|

### Sample

```python
from msmodelslim.quant.session.session import W8A8ProcessorConfig, W8A8QuantConfig

w8a8_processor_cfg = W8A8ProcessorConfig(
    cfg=W8A8QuantConfig(
        act_method='minmax'
    ),
    disable_names=[]
)
```

## W8A8QuantConfig

### Function

W8A8 quantization configuration class, used to configure parameters related to W8A8 quantization.

### Class Prototype

```python
class W8A8QuantConfig(BaseModel):
    act_method: str = 'minmax'
```

### Parameters

| Parameter| Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| act_method | Input | Activation Quantization Method | Optional.<br>Data type: String. Optional values: ['minmax', 'histogram', 'mix'], which correspond to MinMax Activation Quantization, Histogram-based Activation Quantization, and a mix of MinMax and Histogram-based Activation Quantization, respectively.|

### Sample

```python
from msmodelslim.quant.session.session import W8A8QuantConfig

w8a8_quant_cfg = W8A8QuantConfig(
    act_method='minmax'
)
```

## W8A8DynamicProcessorConfig

### Function

W8A8 dynamic quantization processor configuration class, used to configure parameters related to the W8A8 dynamic quantization processor.

### Class Prototype

```python
class W8A8DynamicProcessorConfig(BaseModel):
    cfg: W8A8DynamicQuantConfig
    disable_names: list
```

### Parameters

| Parameter| Input/Return Value | Description | Constraints |
| ------ | ------ | ------ | ------ |
| cfg | Input | W8A8 dynamic quantization configuration | Required.<br>Data type: W8A8DynamicQuantConfig, the W8A8 dynamic quantization configuration class.|
| disable_names | Input | Fallback layers | Required.<br>Data type: List. Each element in the list is a fallback layer name.|

### Sample

```python
from msmodelslim.quant.session.session import W8A8DynamicProcessorConfig, W8A8DynamicQuantConfig

w8a8dynamic_processor_cfg = W8A8DynamicProcessorConfig(
    cfg=W8A8DynamicQuantConfig(
        act_method='minmax'
    ),
    disable_names=[]
)
```

## W8A8DynamicQuantConfig

### Function

W8A8 dynamic quantization configuration class, used to configure parameters related to W8A8 dynamic quantization.

### Class Prototype

```python
class W8A8DynamicQuantConfig(BaseModel):
    act_method: str = 'minmax'
```

### Parameters

| Parameter| Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| act_method | Input | Activation Quantization Method | Optional.<br>Data Type: String. Optional Values: ['minmax', 'histogram', 'mix'], which corresponds to MinMax Activation Quantization, Histogram-based Activation Quantization, and Mixed Activation Quantization of MinMax and Histogram, respectively.|

### Sample

```python
from msmodelslim.quant.session.session import W8A8DynamicQuantConfig

w8a8dynamic_quant_cfg = W8A8DynamicQuantConfig(
    act_method='minmax'
)
```

## W8A8TimeStepProcessorConfig

### Function

The W8A8 timestep quantization processor configuration class, used to configure parameters related to the W8A8 timestep quantization processor.

### Class Prototype

```python
class W8A8TimeStepProcessorConfig(BaseModel):
    cfg: W8A8TimeStepQuantConfig
    disable_names: list
    timestep_sep: int
```

### Parameter Description

| Parameter| Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| cfg | Input | W8A8 timestep quantization configuration | Required.<br>Data type: W8A8TimeStepQuantConfig, the W8A8 timestep quantization configuration class.|
| disable_names | Input | Fallback layers | Required.<br>Data type: List. Each element in the list is a fallback layer name.|
| timestep_sep | Input | The dynamic/static quantization split threshold for timestep quantization | Required.<br>Data type: Integer. Typically set to half of the total inference timesteps in multimodal generative model view generation.|

### Sample

```python
from msmodelslim.quant.session.session import W8A8TimeStepProcessorConfig, W8A8TimeStepQuantConfig

w8a8timestep_processor_cfg = W8A8TimeStepProcessorConfig(
    cfg=W8A8TimeStepQuantConfig(
        act_method='minmax'
    ),
    disable_names=[],
    timestep_sep=25
)
```

## W8A8TimeStepQuantConfig

### Function

W8A8 timestep quantization configuration class, used to configure parameters related to W8A8 timestep quantization.

### Class Prototype

```python
class W8A8TimeStepQuantConfig(BaseModel):
    act_method: str = 'minmax'
```

### Parameters

| Parameter| Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| act_method | Input | Activation Quantization Method | Optional.<br>Data Type: String. Optional Values: ['minmax', 'histogram', 'mix'], which correspond to MinMax Activation Quantization, Histogram-based Activation Quantization, and a mix of MinMax and Histogram-based Activation Quantization, respectively.|

### Sample

```python
from msmodelslim.quant.session.session import W8A8TimeStepQuantConfig

w8a8timestep_quant_cfg = W8A8TimeStepQuantConfig(
    act_method='minmax'
)
```

## FA3ProcessorConfig

### Function

FA3 quantization processor configuration class, used to configure parameters related to the FA3 quantization processor.

### Class Prototype

```python
class FA3ProcessorConfig(BaseModel):
    pass
```

### Sample

```python
from msmodelslim.quant.session.session import FA3ProcessorConfig

fa3_processor_cfg = FA3ProcessorConfig()
```

## SaveProcessorConfig

### Function

Quantization saving processor configuration class, used to configure parameters related to the quantization saving processor.

### Class Prototype

```python
class SaveProcessorConfig(BaseModel):
    output_path: str
    safetensors_name: Optional[str] = None
    json_name: Optional[str] = None
    save_type: list = ['safe_tensor']
    part_file_size: Optional[int] = None
```

### Parameters

| Parameter| Input/Return| Description| Constraints|
| ------ | ------ | ------ | ------ |
| output_path | Input| Quantization saving path.| Required.<br>Data type: String. No default value.|
| safetensors_name | Input| Name of the quantization weight safetensors file.| Optional.<br>Data type: String. Default value is None, generated based on the quantization type, e.g., quant_model_weight_w8a8.safetensors.|
| json_name | Input| Name of the quantization weight description JSON file.| Optional.<br>Data type: String. Default value is None, generated based on the quantization type, e.g., quant_model_description_w8a8.json.|
| save_type | Input| Saving format for quantization weights.| Optional.<br>Data type: List, elements are strings. Default value is ['safe_tensor']. The safetensors format is used by default in multimodal generative model quantization scenarios.|
| part_file_size | Input| When saving as safetensors weight files, the size of each part during sharded saving, in GB.| Optional.<br>Data type: Integer. Default value is None, which disables the sharded saving feature. Otherwise, sharding will be performed according to the user-set value; the actual saved weights may be slightly larger than the set value.|

### Sample

```python
from msmodelslim.quant.session.session import SaveProcessorConfig

save_processor_cfg = SaveProcessorConfig(
    output_path="./",
    safetensors_name=None,
    json_name=None,
    save_type=['safe_tensor'],
    part_file_size=None
)
```

## M3ProcessorConfig

### Function

The outlier suppression M3 algorithm processor configuration class, used to configure parameters related to the M3 outlier suppression processor.

### Class Prototype

```python
class M3ProcessorConfig(BaseModel):
    pass
```

### Sample

```python
from msmodelslim.quant.session.session import M3ProcessorConfig

m3_processor_cfg = M3ProcessorConfig()
```

## M4ProcessorConfig

### Function

Outlier Suppression M4 algorithm processor configuration class, used to configure parameters related to the M4 Outlier Suppression processor.

### Class Prototype

```python
class M4ProcessorConfig(BaseModel):
    pass
```

### Sample

```python
from msmodelslim.quant.session.session import M4ProcessorConfig

m4_processor_cfg = M4ProcessorConfig()
```

## M6Config

### Function

Configuration class for the M6 outlier suppression algorithm parameters, used to configure parameters related to M6 outlier suppression.

### Class Prototype

```python
class M6Config(BaseModel):
    alpha: float = None
    beta: float = None
```

### Parameters

| Parameter| Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| alpha | Input | Controls the smoothing degree of the algorithm. | Optional.<br>Data type: float. Value range: [0, 1], default value: None. If both alpha and beta are specified with specific values, these values are used directly; if either value is not provided, the algorithm will automatically perform optimization to calculate the optimal alpha and beta values for outlier suppression. |
| beta | Input | Controls the smoothing degree of the algorithm. | Optional.<br>Data type: float. Value range: [0, 1], default value: None. If both alpha and beta are specified with specific values, these values are used directly; if either value is not provided, the algorithm will automatically perform optimization to calculate the optimal alpha and beta values for outlier suppression. |

### Sample

```python
from msmodelslim.quant.session.session import M6Config

m6_cfg = M6Config(alpha=0.8, beta=0.2)
```

## M6ProcessorConfig

### Function

Configuration class for the M6 outlier suppression algorithm processor, used to configure parameters related to the M6 quantization processor.

### Class Prototype

```python
class M6ProcessorConfig(BaseModel):
    cfg: M6Config
```

### Parameters

| Parameter| Input/Return | Description | Constraints |
| ------ | ------ | ------ | ------ |
| cfg | Input | M6 outlier suppression algorithm configuration | Required.<br>Data type: M6Config, the M6 outlier suppression algorithm configuration class.|

### Sample

```python
from msmodelslim.quant.session.session import M6ProcessorConfig, M6Config

m6_processor_cfg = M6ProcessorConfig(
    cfg=M6Config(
        alpha=0.8,
        beta=0.2
    )
)
```
