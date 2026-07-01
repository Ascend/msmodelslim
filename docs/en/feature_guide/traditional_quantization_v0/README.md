# V0 Framework and Traditional Model Documentation Navigation (Evolution Stopped)

The documents in this directory are organized by model type and task scenario to facilitate quick scanning.

## 1. Traditional Model Quantization and Calibration

- [Traditional Model Quantization and Calibration](traditional_model_quantization_and_calibration.md)
  - Includes post-training quantization (PTQ) and quantization-aware training (QAT) for PyTorch, ONNX, and MindSpore.

## 2. Foundation Model Quantization and Compression

- [Compression and Structure Optimization (Mainly for Foundation Models)](foundation_model_compression.md)
  - Includes low-memory quantization, mixed calibration datasets, and FA3 quantization.
- Compression and Structure Optimization (Mainly for Foundation Models)(foundation_model_compression.md)
  - Includes sparse quantization, weight compression, long-sequence compression, and low-rank decomposition.

## 3. Training Acceleration and Model Reconstruction

- [Training Acceleration and Model Reconstruction](pruning_and_distillation.md)
  - Includes importance-based pruning, transformer model pruning, Sparse Tool description, and model distillation.
- [Sparse Training Acceleration](sparse_acceleration_training.md)
  - Includes sparse training acceleration workflows for width-expanded and depth-expanded models.

## 4. Tool and Ecosystem Adaptation

- [Quantized Weight Format Description](quantized_weight_format.md)
  - Includes descriptions of the quantized weight file and the weight description file, alongside dequantization formulas and KV Cache quantization specifications.
- [MindSpeed Adapter](mindspeed_adapter.md)
  - Includes MindSpeed-LLM model quantization adaptation workflows and examples.
- [Fake Quantization Accuracy Testing Tool](fake_quantization_accuracy_testing_tool.md)
  - Includes the usage and testing process of the Precision Tool.
- [Inference Optimization for Multimodal Generative Models](inference_optimization_for_multimodal_generative_model.md)
  - Includes Diffusion Transformer (DiT) cache optimization and adaptive sampling optimization workflows.
- [Quantization Code Samples](quantization_and_sparse_quantization_scenario_import_code_examples.md)
  - Includes code samples for common quantization and sparse quantization scenarios.
