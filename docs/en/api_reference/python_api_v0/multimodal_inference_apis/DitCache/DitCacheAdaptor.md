# DitCacheAdaptor: DiT Model Cache Adapter

## 1. Function

A DiT (Diffusion Transformer) cache adapter class for optimizing the inference performance of DiT models. It reduces computational load by caching and reusing intermediate computation results, improving inference efficiency while maintaining generation quality.

Key Features:

- Automatically searches for the optimal cache configuration.

- Supports incremental computation and result reuse.

- Built-in quality evaluation mechanism.

- Saves and reuses search configurations.

## 2. API Reference

### 2.1 Class Definition

```python
class DitCacheAdaptor:
    def __init__(self, 
                 pipeline,
                 config: Optional[DitCacheSearchConfig] = None,
                 dit_block_path: str = "transformer.transformer_blocks")
```

### 2.2 Parameters

| Parameter | Type | Required/Optional | Description | Default Value |
|--------|------|-----------|------|--------|
| pipeline | OpenSoraPipeline | Required | Model pipeline instance, must contain transformer blocks | - |
| config | DitCacheSearchConfig | Optional | Cache search configuration object | None |
| dit_block_path | str | Optional | Access path for transformer blocks within the pipeline | "transformer.transformer_blocks" |

### 2.3 Exception Handling

- `ValueError`: Raised when the input parameters are invalid, including:

  - config is not of type DitCacheSearchConfig

  - The `dit_block_path` format is invalid or empty.

  - Unable to access the transformer blocks at the specified path.

  - The transformer blocks are not of type `nn.ModuleList` or are empty.

### 2.4 Methods

#### 2.4.1 set_timestep_idx

```python
@classmethod
def set_timestep_idx(cls, t_idx: int) -> None
```

Sets the current timestep index. **Must** be called at the beginning of each timestep.

##### Parameters

- t_idx (int): Current timestep index

##### Exceptions

- `ValueError`: If this method is not called before the forward pass of the DiT block, an exception occurs. 

#### 2.4.2 search

```python
def search(self,
          run_pipeline_and_save_videos: Callable,
          prompts_num: int = 1) -> DitCacheConfig
```

Performs a cache configuration search to find the optimal caching strategy.

##### Parameters

| Parameter | Type | Description | Default Value | Remarks |
|------------------------|--------------|------------------------------------------------|--------|----------------------------------------------------------------------|
| `run_pipeline_and_save_videos` | `Callable` | Function that runs the pipeline and returns generated videos | None | Input parameter: `pipeline (OpenSoraPipeline)`<br>Return value: `List[np.ndarray]`, each video has a shape of `(num_frames, h, w, c)` |
| `prompts_num` | `int` | Number of videos to generate | `1` | Controls the number of generated videos |

##### Returns

The return value is an object `DitCacheConfig` used to configure the DiT caching mechanism, containing the following fields:

| Field               | Type   | Description                                           |
|--------------------|--------|------------------------------------------------|
| `cache_step_start`     | `int` | The timestep at which caching starts to be used                              |
| `cache_step_interval`  | `int` | The timestep interval for cache computation, that is, how many steps between cache recomputations              |
| `cache_block_start`    | `int` | The index of the block at which caching starts, where `0` indicates starting from the first block        |
| `cache_num_blocks`     | `int` | The number of blocks to cache                                 |

#### 2.4.3 `update_cache_config`

Used to manually update the current caching strategy configuration, including the starting block, number of blocks, and timestep-related parameters for caching. Calling this method allows you to directly apply a specified caching strategy without re-searching.

```python
def update_cache_config(self,
                        cache_block_start: int,
                        cache_num_blocks: int,
                        cache_step_start: int,
                        cache_step_interval: int)
```

##### Parameters

| Parameter | Input/Output | Type | Description |
|--------------------|------------|--------|------------------------------|
| `cache_block_start` | Input | `int` | The starting block index for caching (starting from 0). |
| `cache_num_blocks` | Input | `int` | The number of blocks to cache. |
| `cache_step_start` | Input | `int` | The timestep from which caching is enabled. |
| `cache_step_interval` | Input | `int` | The interval (in timesteps) at which the cache is recomputed. |

##### Sample

```python
from msmodelslim.pytorch.multi_modal.dit_cache import DitCacheAdaptor, DitCacheSearchConfig

# Create an adapter to add caching functionality to the DiT model.
adaptor = DitCacheAdaptor(pipeline)

# Set the cache configuration.
cache_config = dict(
    cache_block_start=2,
    cache_num_blocks=4,
    cache_step_start=10,
    cache_step_interval=5
)
adaptor.update_cache_config(**cache_config)

# Use the pipeline to run inference and generate the videos.
...
```

## 3. Usage Guide

### 3.1 Sample

```python
# 1. Import necessary classes.
from msmodelslim.pytorch.multi_modal.dit_cache import DitCacheAdaptor, DitCacheSearchConfig

# 2. Define the function to run pipeline.
def run_pipeline_and_save_videos(pipeline):
    """Run pipeline and return the generated video list."""
    positive_prompt = "(masterpiece), (best quality), (ultra-detailed), {}"
    
    videos = pipeline(
        positive_prompt.format("a dog running on the beach"),
        num_frames=29,
        height=480,
        width=640,
        num_inference_steps=100,
        guidance_scale=7.5
    ).images
    
    return videos

# 3. Configure and initialize cache adapter.
config = DitCacheSearchConfig(
    cache_ratio=1.3,  # Cache speedup ratio
    num_sampling_steps=100  # Number of sampling steps
)
cache_adaptor = DitCacheAdaptor(pipeline, config)

# 4. Execute cache configuration search.
searched_config = cache_adaptor.search(
    run_pipeline_and_save_videos=run_pipeline_and_save_videos,
    prompts_num=1
)
```

### 3.2 Usage Flow

1. Initialize a DitCacheAdaptor instance.

2. In the diffusion loop, call set_timestep_idx() at the beginning of each timestep.

3. Call the search() method to search for cache configurations.

4. Perform inference using the returned cache configuration.

### 3.3 Precautions

1. **Timestep must be set**

    Call `DitCacheAdaptor.set_timestep_idx(step_id)` at the beginning of each timestep. This is typically performed within the model's denoising loop, as shown in the following example:

    ```python
    for step_id, t in enumerate(timesteps):
        DitCacheAdaptor.set_timestep_idx(step_id)  # Must be called at the beginning of each timestep.
        model_output = pipeline(...)
    ```

2. **Search Configuration**

    It is recommended that you set cache_ratio to 1.3, which represents the desired speedup ratio. The search process, which includes calibration video generation and configuration evaluation, may take a considerable amount of time. It is recommended to run the search process on a device with better performance.

3. **Configuration Saving and Reuse**

    The searched configuration can be saved as a JSON file, allowing it to be directly loaded and used in the same scenario without re-searching.

4. **Usage Scenarios**

    Currently, 29\*480p and 93\*720p scenarios are supported, achieving approximately 1.3x speedup while maintaining generation quality. Different scenarios may require re-searching for the optimal configuration.

5. **Parameter Consistency**

Ensure that the same model parameter configuration is used during both searching and inference. When applying the optimized cache configuration to inference, ensure that the model and data processing pipeline remain consistent with those used during searching, including but not limited to the number of sampling steps, image size, etc.

## 4. Technical Principles

### 4.1 Theoretical Basis

#### 4.1.1 Basic Assumptions

DiT cache optimization is based on the following core ideas:

- During the diffusion process, the output changes of transformer blocks at adjacent timesteps are gradual.

- The computation results of certain blocks can be obtained incrementally without full recomputation.

#### 4.1.2 Mathematical Model

In the diffusion model, let $h_{t,i}$ denote the hidden state output of the $i$-th transformer block at timestep $t$:

$$ h_{t,i} = \mathcal{F}_i(h_{t,i-1}), \quad i \in [1,N] $$

where:

- $\mathcal{F}_i$ represents the transformation function of the $i$-th transformer block.

- $N$ is the total number of transformer blocks.

- $t \in [0,T]$ is the diffusion timestep, and $T$ is the total number of timesteps.

### 4.2 Incremental Computation

#### 4.2.1 Interval Difference Definition

For any block interval $[i,j]$, its output difference is defined as:

$$ \Delta_{t,[i:j]} = h_{t,j} - h_{t,i}, \quad 1 \leq i < j \leq N $$

This represents the cumulative transformation effect from the $i$-th block to the $j$-th block.

#### 4.2.2 Continuity Assumption

During the diffusion process, the outputs of transformer blocks at adjacent timesteps exhibit local continuity:

$$ \|\Delta_{t,[i:j]} - \Delta_{t-1,[i:j]}\| \leq \epsilon, \quad \forall t > 0 $$

where $\epsilon$ is a small positive number representing the acceptable error margin.

#### 4.2.3 Incremental Approximation

Based on the continuity assumption, the difference from the previous timestep can be used to approximate the current timestep:

$$ h_{t,j} = h_{t,i} + \Delta_{t,[i:j]} \approx h_{t,i} + \Delta_{t-1,[i:j]} $$

### 4.3 Caching Strategy

#### 4.3.1 Basic Computation

For the starting block $(i = block\_start)$, compute directly:

$$ h_{t,i} = \mathcal{F}_i(h_{t,i-1}) $$

#### 4.3.2 Incremental Update Mechanism

At the cache update time point $(t \bmod interval = 0)$, compute and store the interval difference:

$$ \Delta_{t,[i:j]} = \mathcal{F}_{[i:j]}(h_{t,i}) - h_{t,i} $$

where $\mathcal{F}_{[i:j]}$ denotes the composite transformation from block $i$ to $j$.

#### 4.3.3 Incremental Reconstruction Process

During cache reuse $(t \bmod interval \neq 0)$, reconstruct the output using the stored difference:

$$ h_{t,j} = h_{t,i} + \Delta_{t-\delta t,[i:j]}, \quad \delta t = t \bmod interval $$

### 4.4 Engineering Implementation

#### 4.4.1 Core Architecture

The caching mechanism is implemented by replacing the forward method of the DiT block:

```python
def _add_cache_to_dit_block(self, dit_blocks: nn.ModuleList):
    """Add caching logic for transformer blocks.
    
    Caching process:
    1. Basic condition checks:
       - t_idx < cache_step_start: use original forward.
       - cache disabled: use original forward.
    
    2. Handle based on the block position:
       - Base block (index = cache_block_start): compute and cache input
       - Intermediate blocks: return placeholder DitCacheDummy
       - Reuse block (index = cache_block_start + cache_num_blocks - 1): 
         reconstruct output using cached delta
       - Other blocks: use original forward
    """
```

#### 4.4.2 Key Implementation Details

##### Basic Block Processing

```python
# Process the base block.
if _block_idx == blk_start:
    self.cache[START_HIDDEN_KEY] = hidden_states
    
    if is_step_to_store_cache:
        return orig_forward(hidden_states, *args, **kwargs)
    else:
        return DitCacheDummy()
```

##### Reuse Block Processing

```python
# Process the reuse block.
elif _block_idx == blk_end:
    last_block_hidden = self.cache.pop(START_HIDDEN_KEY)
    
    if is_step_to_store_cache:
        hidden_states = orig_forward(hidden_states, *args, **kwargs)
        delta = hidden_states - last_block_hidden
        self.cache[DELTA_HIDDEN_KEY] = delta
        return hidden_states
    else:
        return self.cache[DELTA_HIDDEN_KEY] + last_block_hidden
```

#### 4.4.3 Parameters

Key control parameters of the caching mechanism:

- `cache_step_start`: The timestep at which caching begins.

- `cache_step_interval`: Cache update interval.

- `cache_block_start`: Starting cache block position.

- `cache_num_blocks`: Number of cache blocks.

#### 4.4.4 Implementation Notes

1. `set_timestep_idx()` must be called at the beginning of each timestep.

2. Ensure that the cache parameters are set reasonably to avoid affecting generation quality.

3. Pay attention to memory usage and clear unnecessary caches promptly.
