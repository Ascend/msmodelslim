# DitCacheSearchConfig

## Function

DiT cache search parameter configuration class, which stores configuration parameters for cache search.

## Prototype

```python
class DitCacheSearchConfig(cache_ratio=1.3, dit_block_num=None, num_sampling_steps=None)
```

## Parameters

| Parameter | Input/Return | Description | Constraints | Instructions |
|---------------------|--------------|------------|------------------------------------------------------------------|------------------------------------------------------------|
| `cache_ratio` | Input | Speedup ratio | **Optional**.<br>Data type: `float`.<br>Default: 1.3 <br>Value range: (1.0, 2.0) | Controls the speedup ratio for cache application. Larger values indicate a more significant expected speedup effect. |
| `dit_block_num` | Input | Number of DiT blocks | **Optional**.<br>Data type: `int`.<br>Default: `None` | Usually set automatically by the system; manual specification is not required. |
| `num_sampling_steps` | Input | Number of sampling steps | **Required**.<br>Data type: `int`.<br>Must be a positive integer | Should match the number of sampling steps during actual inference. |

## Sample

```python
from msmodelslim.pytorch.multi_modal.dit_cache import DitCacheSearchConfig

# Set the search configuration.
config = DitCacheSearchConfig(
    cache_ratio=1.3,
    num_sampling_steps=100
)
```

## Notes

1. cache_ratio should be set within a reasonable range. Excessively large values may lead to quality degradation.

2. The time complexity of the search process is proportional to num_sampling_steps.
