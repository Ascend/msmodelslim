# Optimization of OSP1_2 Inference

## Model Introduction

[Open-Sora-Plan v1.2](https://github.com/PKU-YuanGroup/Open-Sora-Plan)    It is an open-source multi-modal video generation model, co-sponsored by the AIGC Joint Laboratory of Peking University-rabbit exhibition, focusing on efficient video generation tasks.

## Preparation Before Use

For details, see.[Multimodal view generation inference optimization tool](../../docs/zh/feature_guide/traditional_quantization_v0/inference_optimization_for_multimodal_generative_model.md#使用前准备)    The environment configuration is complete.

## Supported Model Versions and Optimization Policies

| Model Series       | Model version | HuggingFace Link                                                        | Sampling Optimization | DiT cache optimization | Optimized command                                                   |
| ------------------ | ------------- | ----------------------------------------------------------------------- | --------------------- | ---------------------- | ------------------------------------------------------------------- |
| **Open-Sora-Plan** | v1.2          | [Open-Sora-Plan v1.2](https://github.com/PKU-YuanGroup/Open-Sora-Plan)     | ✅                     | ✅                      | [Sampling Optimization](#sampling-optimization)    /[DiT cache optimization](#dit-cache-optimization)     |

**Description:**

 * " indicates that the optimization policy has passed the official verification of msModelSlim. The functions are complete and the performance is stable. You are advised to use the optimization policy preferentially.
 * A space indicates that the optimization policy has not been officially verified by msModelSlim. You can configure the policy as required, but the optimization effect and function stability cannot be officially guaranteed.
 * Click the link in the Optimization Command column to go to the specific optimization command.

### Validated Optimization Methods

| Optimization Type      | Supported Scenario      | Acceleration effect | precision loss |
| ---------------------- | ----------------------- | ------------------- | -------------- |
| Sampling optimization  | 29 fps 480p             | 2x                  | < 1%           |
| DiT cache optimization | 29 fps 480p/93 fps 720p | 1.3x                | < 1%           |

## Instructions for use

### sampling optimization

#### 1. Generate calibration video

Use the original model to generate a batch of videos, which will be used as the quality evaluation benchmark for subsequent optimization steps.

```bash
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m example.osp1_2.sample_t2v_sp \
    --model_path /path/to/checkpoint-xxx/model_ema \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/path/to/causalvideovae" \
    --save_img_path "./sample_video_test" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --save_memory \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit"
```

Complete sample script:[`generate_baseline_t2v_sp.sh`](generate_baseline_t2v_sp.sh)    

 * **Parameter description:**

| 参数 | 说明 |
|------|------|
| `--model_path` | 预训练的 DiT 模型权重路径 |
| `--num_frames` | 生成视频的帧数 |
| `--height`, `--width` | 生成视频的高度和宽度 |
| `--cache_dir` | Hugging Face 模型缓存目录 |
| `--text_encoder_name` | 文本编码器的名称或路径 |
| `--text_prompt` | 包含文本提示的 txt 文件路径 |
| `--ae` | 使用的自动编码器模型名称 |
| `--ae_path` | VAE 模型权重路径 |
| `--save_img_path` | 生成视频的保存路径 |
| `--fps` | 生成视频的帧率 |
| `--guidance_scale` | 文本引导的权重比例 |
| `--num_sampling_steps` | 原始采样步数 |
| `--enable_tiling` | 是否启用分块推理 |
| `--tile_overlap_factor` | 分块推理时的重叠因子 |
| `--save_memory` | 是否启用节省内存模式 |
| `--max_sequence_length` | 文本编码器的最大序列长度 |
| `--sample_method` | 使用的采样器方法 |
| `--model_type` | 模型类型，默认值："dit" |

#### 2. Search and optimize sampling steps

According to the generated calibration video, the optimal sampling time step combination is searched to ensure the quality and reduce the number of sampling steps.

```bash
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m example.osp1_2.search_t2v_sp \
    --model_path /path/to/checkpoint-xxx/model_ema \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/path/to/causalvideovae" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 50 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --save_memory \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit" \
    --save_dir "/path/to/save/schedule/timestep/file" \
    --videos_path "/path/of/calibration/videos" \
    --neighbour_type "uniform" \
    --monte_carlo_iters 5
```

Complete sample script:[`search_t2v_sp.sh`](search_t2v_sp.sh)    

 * **Parameter description:**

Other parameters are the same as the procedure for generating a calibration video. The following parameters are added:

| 参数 | 说明 |
|------|------|
| `--num_sampling_steps` | 目标优化的采样步数（例如，从100步优化到50步） |
| `--save_dir` | 保存搜索到的优化时间步配置文件的目录 |
| `--videos_path` | 第1步生成的校准视频所在的路径 |
| `--neighbour_type` | 采样过程中使用的邻域搜索类型，可选值为 "uniform" 或 "random" |
| `--monte_carlo_iters` | Monte Carlo 采样的迭代次数 |

#### 3. Use optimized configurations for inference

Use the optimized time step profile found in step 2 for inference to verify the acceleration and build quality.

```bash
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m example.osp1_2.sample_t2v_sp \
    --model_path /path/to/checkpoint-xxx/model_ema \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/path/to/causalvideovae" \
    --save_img_path "./sample_video_test" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --save_memory \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit" \
    --schedule_timestep "/path/of/schedule/timestep/file.txt"
```

Complete sample script:[`sample_t2v_sp.sh`](sample_t2v_sp.sh)    

 * **Parameter description:**

Other parameters are the same as the procedure for generating a calibration video. The following parameters are added:

| 参数 | 说明 |
|------|------|
| `--schedule_timestep` | 第2步搜索到的优化时间步配置文件的路径 |

### DiT cache optimization

#### 1. Search for cache configuration

Searches for the optimal DiT cache configuration, including the start layer, number of cache layers, start time step, and time step interval.

```bash
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m example.osp1_2.search_t2v_sp \
    --model_path /path/to/checkpoint-xxx/model_ema \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/path/to/causalvideovae" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --save_memory \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit" \
    --search_type "dit_cache" \
    --cache_ratio 1.3 \
    --cache_save_path /path/to/save/the/searched/config
```

Complete sample script:[`dit_cache_search_t2v_sp.sh`](dit_cache_search_t2v_sp.sh)    

 * **Parameter description:**

Other parameters are the same as the procedure for generating a calibration video. The following parameters are added:

| 参数 | 说明 |
|------|------|
| `--search_type` | 指定搜索类型为 "dit_cache" |
| `--cache_ratio` | 缓存搜索的加速比目标（例如1.3x） |
| `--cache_save_path` | 保存搜索到的缓存配置文件的路径 |

#### 2. Use the cache configuration for inference

Use the cached profile discovered in step 1 for inference to verify the acceleration and build quality.

```bash
torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m example.osp1_2.sample_t2v_sp \
    --model_path /path/to/checkpoint-xxx/model_ema \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/path/to/causalvideovae" \
    --save_img_path "./sample_video_test" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --save_memory \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit" \
    --dit_cache_config "/path/of/cache/config/file"
```

Complete sample script:[`dit_cache_sample_t2v_sp.sh`](dit_cache_sample_t2v_sp.sh)    

 * **Parameter description:**

Other parameters are the same as the procedure for generating a calibration video. The following parameters are added:

| 参数 | 说明 |
|------|------|
| `--dit_cache_config` | 第1步搜索到的缓存配置文件的路径 |
