# Histogram: Activation Quantization Algorithm

## Overview

- **Source**: Adapted from the relevant PyTorch implementation.
- **Problem**: Traditional MinMax quantizers are sensitive to outliers, which leads to excessively large quantization ranges, low effective bit utilization, and significant loss of quantization accuracy.
- **Objective**: Improve quantization accuracy and model performance through automatic search for the optimal clipping range and outlier filtering by analyzing the histogram distribution of activation values.

## Preparations

Install msModelSlim. For details, see [msModelSlim Installation Guide](../../../install_guide/install_guide.md).

## Principle and Implementation

### Principle

The histogram activation quantization algorithm automatically searches for the optimal clipping range (`clip_min`, `clip_max`) by analyzing the distribution histogram of the input tensor. This approach prevents excessively large quantization ranges.

### Implementation

- The algorithm is implemented in [`msmodelslim/core/quantizer/impl/histogram.py`](../../../../../msmodelslim/core/quantizer/impl/histogram.py) and [`msmodelslim/core/observer/histogram.py`](../../../../../msmodelslim/core/observer/histogram.py). The processing flow consists of four steps.

1. **Histogram statistics**
   - Divide the value range of the input tensor into a fixed number of bins (default `2048`).
   - Count the frequency of values in each bin to construct a distribution histogram.
   - Support upsampling (`upsample_rate=16`) to reduce quantization error.

2. **Clipping range search**
   - Gradually adjust the clipping range by moving a fixed percentile (`stepsize=1e-5`) in each iteration.
   - Evaluate the quality of candidate ranges by computing the quantization error. Stop the search when the quantization error no longer decreases or the range exceeds boundaries.

3. **Quantization error measurement**
   - **L2 norm error**: default quantization error metric. The algorithm computes the L2 norm difference between the real and quantized distributions.
   - **KL divergence error**: The algorithm computes the KL divergence between the real and quantized distributions. Currently, the accuracy and performance of this method are lower than the L2 norm method. No configuration entry is available for this method when using the quick quantization YAML file.

4. **Quantization parameter calculation**
   - Calculate and save the `scale` and `zero_point` by using the upper and lower bounds of the optimal clipping range as the `max` and `min` values.
   - Perform the fake-quantization operation and return the quantized tensor.

### Core Components

#### HistogramObserver

```python
class HistogramObserver(TorchHistogramObserver):
    def __init__(self, config: HistogramObserverConfig):
        super().__init__(qscheme=torch.per_tensor_affine)
        self.config = config
        self.clip_min = None
        self.clip_max = None
        self.upsample_rate = 16  # Specifies the upsampling rate to reduce quantization error.
```

**Core Method Implementation**

1. **`forward` Method**

   ```python
   # This method is inherited from the forward method of TorchHistogramObserver.
   # It is used to update histogram statistics.
   # It is called in the update method.
   ```

2. **`update` Method**

   ```python
   def update(self, x: torch.Tensor, sync: bool = False, group: Optional[dist.ProcessGroup] = None):
       """
       Update the histogram, search for the clipping values, and save the optimal quantization clipping values.

       Main steps:
       1. Input verification: Check tensor validity and filter out NaN and infinite values.
       2. Histogram update: Call the forward method of the parent class to update histogram statistics.
       3. Parameter search: Perform non-linear parameter search to find the optimal clipping range.

       Args:
            x: Specifies the input tensor.
            sync: Specifies whether to synchronize.
            group: Specifies the process group.

       Returns:
            None
       """
   ```

3. **Internal Search Method Implementation**

   **L2 Norm Search**

   ```python
   def _compute_l2_error(self, start_bin: int, end_bin: int):
       """
       Compute the L2 norm error between the real and quantized distributions.

       Algorithm principles:
       1. Calculate the target bin width.
       2. Compute the mapping from the source bin to the target bin.
       3. Decompose the error into three parts: start, middle, and end.
       4. Use the _get_norm method (explicitly calculating the integral) to compute the L2 norm error of each part.
       """
   ```

   **KL Divergence Search**

   ```python
   def _compute_kl_error(self, start_bin: int, end_bin: int):
       """
       Compute the KL divergence between the real and quantized distributions.

       Algorithm principles:
       1. Calculate the real distribution p_i.
       2. Calculate the quantized distribution q_i.
       3. Calculate the KL divergence: KL = sum(p_i * log(p_i / q_i)).
       """
   ```

4. **Non-linear Parameter Search**

   ```python
   def _non_linear_param_search(self) -> tuple[torch.Tensor, torch.Tensor]:
       """
       Use a binary search policy to find the optimal clipping range.

       Search policy:
       1. Initialization: alpha=0.0 (lower bound), beta=1.0 (upper bound).
       2. Iterative search: Move a fixed percentile each time (stepsize=1e-5).
       3. Boundary adjustment: Determine whether to move the left or right boundary based on the step size. Select the side with a longer single move (a sparser distribution).
       4. Early stopping condition: Stop when the quantization error no longer improves or the boundary exceeds the limit.
       5. Return result: The clipping values corresponding to the optimal start_bin and end_bin.
       """
   ```

#### ActPerTensorHistogram

```python
class ActPerTensorHistogram(AutoActQuantizer):
    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        histogram_config = HistogramObserverConfig(symmetric=config.symmetric)
        self.histogram_observer = HistogramObserver(histogram_config)
        self.q_param: Optional[QParam] = None
```

**Core Method Implementation**

1. **`forward` Method**

   ```python
   def forward(self, x: torch.Tensor) -> torch.Tensor:
       """
       Forward propagation method for quantization computation
       """
       # Update the histogram observer to collect statistics on the distribution of the input tensor.
       self.histogram_observer.update(x)
       # Obtain the optimal clipping values based on histogram statistics.
       # clip_min: minimum clipping value. clip_max: maximum clipping value.
       clip_min, clip_max = self.histogram_observer.get_clip_bounds()
       # Calculate quantization parameters based on the clipping values.
       self.q_param = calculate_qparam(
           min_val=clip_min,      # Minimum clipping value
           max_val=clip_max,      # Maximum clipping value
           q_dtype=QDType(self.config.dtype),
           q_scope=QScope(self.config.scope),
           symmetric=self.config.symmetric,
       )
       # Perform the fake-quantization operation.
       return fake_quantize(QStorage(dtype=QDType.FLOAT, value=x), self.q_param).value.clamp(clip_min, clip_max)
   ```

2. **Quantization Parameter Management**

   ```python
   def get_q_param(self) -> QParam:
       """
       Obtain the calculated quantization parameters.
       """
       if self.q_param is None:
          raise SpecError(
                  "No q_param was set",
                  action="Please call forward first"
              )
       return self.q_param
   ```

## Function Description

### YAML Configuration Example

Histogram is used as the activation quantization method (`method: "histogram"`) in the [`linear_quant`](linear_quant.md) processor. The following is an example of the YAML configuration:

```yaml
spec:
  process:
  - type: "linear_quant"
    qconfig:
      act:
        scope: "per_tensor"  # Specifies the quantization scope. Currently only per_tensor is supported.
        dtype: "int8"        # Specifies the quantization data type. Currently only int8 is supported.
        symmetric: false     # Enables symmetric quantization (True) or asymmetric quantization (False).
        method: "histogram"  # Specifies the quantization algorithm: "histogram" activation quantization.
      weight:
        scope: "per_channel"
        dtype: "int8"
        symmetric: true
        method: "minmax"     # Histogram weight quantization is not supported. Do not set it to "histogram" here.
```

### YAML Configuration Fields

#### `qconfig.act` - Activation Quantization Configuration

The following fields are part of the `linear_quant` processor configuration. For a complete description of the `linear_quant` fields, see [Linear Quantization Configuration Fields](linear_quant.md#yaml-configuration-fields).

| Field| Purpose| Optional Value| Description|
|--------|-------|--------|------|
| scope | Specifies the quantization scope.| `"per_tensor"` | Currently, histogram quantization supports only `per_tensor`.|
| dtype | Specifies the quantization data type.| `"int8"` | Currently, histogram quantization supports only `int8`.|
| symmetric | Specifies whether to enable symmetric quantization.| `true`, `false` | `true` enables symmetric quantization, while `false` enables asymmetric quantization.|
| method | Specifies the quantization method.| `"histogram"` | This value is fixed to enable histogram activation quantization.|

### Algorithm Parameters

**HistogramObserverConfig**

These parameters are automatically configured by the quantizer. Manual adjustment is not required.

```python
class HistogramObserverConfig(BaseModel):
    symmetric: bool = False                             # Specifies whether to enable symmetric quantization.
    search_method: SearchMethod = SearchMethod.L2_NORM  # Specifies the search method.
    dtype: QDType = QDType.INT8                         # Specifies the quantization data type.
    scope: QScope = QScope.PER_TENSOR                   # Specifies the quantization scope.
```

**Search Method Enumeration**

```python
class SearchMethod(str, Enum):
    L2_NORM = "l2_norm"              # Specifies L2 norm search.
    KL_DIVERGENCE = "kl_divergence"  # Specifies KL divergence search.
```

## FAQ

### Configuration Error

**Symptom**: A `ValidationError` appears in the log prompt.

**Possible causes**:

- The `histogram` method is incorrectly configured for `weight` in a scenario where activation quantization is supported.
- A configuration not supported by the `histogram` method is selected (such as INT4 quantization) in a scenario where activation quantization is supported.
- The `histogram` method is configured in a scenario where activation quantization is not supported.

**Solution**:

- Check whether the YAML file is incorrectly configured.

```yaml
- type: "linear_quant"
  qconfig:
   act:
     scope: "per_tensor" # Specifies the quantization scope. Currently only per_tensor is supported.
     dtype: "int8" # Specifies the quantization data type. Currently only int8 is supported.
     symmetric: False # Specifies whether to enable symmetric quantization (True) or asymmetric quantization (False).
     method: "histogram" # Specifies the quantization method. "histogram" enables histogram activation quantization.
   weight:
     scope: "per_channel"
     dtype: "int8"
     symmetric: True
     method: "minmax" # Histogram weight quantization is not supported. Do not set it to "histogram" here.
```

- Check whether `AutoActQuantizer` exists during the initialization of the corresponding quantizer. You can find the quantizer name by using the `type` field in the `process` list in the YAML file and view the corresponding code in [`msmodelslim/core/quantizer`](../../../../../msmodelslim/core/quantizer).

```python
class LinearQuantizer(nn.Module):

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(self, config: LinearQConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.input_quantizer = AutoActQuantizer.from_config(config.act) # Activation quantization is supported.
        self.weight_quantizer = AutoWeightQuantizer.from_config(config.weight)
        self.weight: Optional[nn.Parameter] = None
        self.bias: Optional[nn.Parameter] = None
        self.q_weight: Optional[QStorage] = None
```
