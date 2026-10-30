# Post-Training Quantization (PTQ) Knowledge Base

## What Is Post-Training Quantization (PTQ)

Post-training quantization (Post-Training Quantization, PTQ) is a quantization approach that, after a model has **finished training**, uses only a small amount of calibration data to gather statistics on the distributions of model weights and activations and maps floating-point parameters to low-bit integer or low-precision floating-point formats, thereby compressing model size and reducing the memory and compute overhead of inference. Unlike quantization-aware training (QAT), PTQ does not require retraining the model, so it is low-cost and fast to ship; it is the mainstream approach for large model inference acceleration.

## Basic Concept Entries

| Category | Description | Entry |
|------|------|------|
| Basic Concept | Basic concepts, technical characteristics, and applicable scope of post-training quantization (PTQ) | [Post-Training Quantization (PTQ) Entry](term_ptq.md) |

## Quantization Entries

The quantization focus and implementation path differ considerably across model types, so msModelSlim divides the related entries into the following four categories by quantization target and tool capability:

| Category | Description | Entry |
|------|------|------|
| LLM Quantization | Generally a Decoder-only autoregressive structure; weight and activation quantization are mature | [LLM Quantization Entry](llm/term_large_language_model_quantization.md) |
| VLM Quantization | Vision encoder + language model; requires multimodal calibration data | [VLM Quantization Entry](vlm/term_vision_transformer_quantization.md) |
| DiT Quantization | Text-to-image / text-to-video with a Diffusion Transformer backbone | [DiT Quantization Entry](dit/term_diffusion_transformer_quantization.md) |
| Weight Conversion | Offline, data-free format / precision transformation | [Weight Conversion Entry](convert/term_weight_conversion.md) |

## Usage Guides

| Category | Description | Entry |
|------|------|------|
| LLM Quantization | One-click quantization workflow for large language models | [Large Language Model (LLM) Quantization Usage Guide](llm/usage_large_language_model_quantization.md) |
| VLM Quantization | Quantization workflow for multimodal understanding models | [Multimodal Understanding Model (VLM) Quantization Usage Guide](vlm/usage_vision_transformer_quantization.md) |
| DiT Quantization | Quantization workflow for multimodal generation models | [Multimodal Generation Model (DiT) Quantization Usage Guide](dit/usage_diffusion_transformer_quantization.md) |
| Weight Conversion | Format / precision conversion workflow for existing weights | [Weight Conversion Usage Guide](convert/usage_weight_conversion.md) |

## Model Integration Guides

| Category | Description | Entry |
|------|------|------|
| LLM Integration | Integrating a new large language model (LLM) into the msModelSlim quantization workflow | [LLM Model Integration Quantization Workflow Guide](llm/integration_guide_large_language_model_quantization.md) |
| VLM Integration | Integrating a new multimodal vision-language model (VLM) into the msModelSlim quantization workflow | [VLM Model Integration Quantization Workflow Guide](vlm/integration_guide_vision_transformer_quantization.md) |
| DiT Integration | Integrating a new multimodal generation model (DiT) into the msModelSlim quantization workflow | [DiT Model Integration Quantization Workflow Guide](dit/integration_guide_diffusion_transformer_quantization.md) |
| Weight Conversion | Purely configuration-driven, with no model adapter code required; configure YAML rules for the new model's weight structure to complete integration | [Weight Conversion Usage Guide](convert/usage_weight_conversion.md) |
