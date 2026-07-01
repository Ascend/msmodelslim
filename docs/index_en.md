# Welcome to msModelSlim

![ModelSlim Slogan](assets/modelslim_slogan.png)

MindStudio ModelSlim (msModelSlim), an affinity compression tool that aims at acceleration, compression, and Ascend. It includes a series of inference optimization techniques, such as quantization and compression, designed to accelerate large language dense models, MoE models, multi-modal understanding models, multi-modal generation models, etc.

Ascend AI developers can invoke rich Python APIs provided by msModelSlim to flexibly implement algorithm adaptation and model compression, export weights in multiple formats, and optimize precision and performance throughout the process. Optimized models can seamlessly connect to mainstream inference frameworks, such as MindIE and vLLM Ascend, and implement efficient deployment on Ascend AI processors. :zap:

## \:star:Core Benefits

 * **Efficient compression:package:-Supports multiple quantization algorithms, significantly reducing memory usage.**
 * **Ascend Affinity:gear:-Deeply adapts to Ascend hardware to ensure optimal inference performance.**
 * **Easy to use:magic_wand: - Rich model best practice library, quickly implement model optimization.**

## \:loudspeaker: Latest news

### December 2025 events

 * The msModelSlim supports automatic optimization of quantization precision feedback and can automatically search for the optimal quantization configuration based on precision requirements.
 * The msModelSlim supports self-quantized multi-modal understanding models and supports quantitative access of multi-modal understanding models.
 * One-click quantification by msModelSlim supports multi-card quantification and distributed layer-by-layer quantification, improving the quantification efficiency of large models.
 * msModelSlim supports DeepSeek-V3.2 W8A8 quantization. A single card can be executed with 64 GB video memory and 100 GB memory.
 * msModelSlim supports DeepSeek-V3.2-Exp W4A8 quantization, which can be executed by a single card with 64 GB video memory and 100 GB memory.
 * msModelSlim supports Qwen3-VL-235B-A22B W8A8 quantization.

### November 2025 events

 * The msModelSlim model adaptation supports plug-in, configuration registration, and dependency pre-check.

### October 2025 events

 * The msModelSlim supports Qwen3-235B-A22B W4A8 and Qwen3-30B-A3B W4A8 quantization. The vLLM Ascend supports quantitative model inference deployment.

### September 2025 events

 * msModelSlim supports DeepSeek-V3.2-Exp W8A8 quantization. A single card has 64 GB video memory and 100 GB memory can be executed.
 * MsModelSlim has solved the problem that Qwen3-235B-A22B frequently displays abnormal tokens such as "game copy" under W8A8 quantization.
 * msModelSlim supports DeepSeek R1 W4A8 per-channel quantization \[Prototype\]
 * The msModelSlim supports large-scale model quantization and sensitive layer analysis.

### August 2025 events

 * msModelSlim supports one-click quantization of Wan2.1 models.
 * The msModelSlim supports layer-by-layer quantification of large models, significantly reducing the memory usage of large models.
 * The msModelSlim supports the SSZ weight quantization algorithm for large models and improves the quantization precision by iteratively searching for the optimal scaling factor and offset.

> **Note: The feature marked with Prototype has not been fully verified and may be unstable or defective. The features marked beta are non-commercial features.**

## msModelSlim Document

Welcome to the msModelSlim documentation. msModelSlim is a comprehensive model compression and optimization tool for Ascend AI processors.
