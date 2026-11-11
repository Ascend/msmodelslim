# V0框架与传统模型文档导航（已停止演进）

本目录文档按模型类型与任务场景梳理，便于按需求快速定位。

## 1. 传统模型量化与校准

- [传统模型量化与校准](traditional_model_quantization_and_calibration.md)
  - 包含 PyTorch/ONNX/MindSpore 训练后量化与 QAT。

## 2. 大模型量化与压缩

- [大模型量化与校准](foundation_model_quantization_and_calibration.md)
  - 包含低显存量化、混合校准数据集、FA3 量化。
- [压缩与结构优化（大模型为主）](foundation_model_compression.md)
  - 包含稀疏量化与权重压缩、长序列压缩、低秩分解。

## 3. 训练加速与模型改造

- [训练加速与模型改造](pruning_and_distillation.md)
  - 包含重要性剪枝、Transformer 剪枝、Sparse tool、模型蒸馏。
- [稀疏加速训练](sparse_acceleration_training.md)
  - 包含宽度扩增与深度扩增模型的稀疏训练加速流程。

## 4. 工具与生态适配

- [量化权重格式说明](quantized_weight_format.md)
  - 包含量化权重与描述文件格式、反量化公式及 KV Cache 量化说明。
- [MindSpeed适配器](mindspeed_adapter.md)
  - 包含 MindSpeed-LLM 模型量化适配流程与示例。
- [伪量化精度测试工具](fake_quantization_accuracy_testing_tool.md)
  - 包含 Precision Tool 使用方式与测试流程。
- [多模态生成模型推理优化](inference_optimization_for_multimodal_generative_model.md)
  - 包含 DiT 缓存优化与自适应采样优化流程。
- [量化代码示例](quantization_and_sparse_quantization_scenario_import_code_examples.md)
  - 包含常见量化/稀疏量化场景导入代码样例。
