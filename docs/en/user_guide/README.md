# MindStudio ModelSlim User Guide

<br>

## 1. Functional Overview

**MindStudio ModelSlim (msModelSlim)** is a high-performance model compression tool in the Ascend ecosystem that supports the quantization and compression of dense LLMs, MoE models, and multimodal models. Developers can use the command line and configurations to tune quickly and export models adapted to frameworks such as MindIE and vLLM-Ascend for efficient deployment on Ascend AI processors.

The documents in this directory are divided into **business process guides** and **tool usage guides**; consult them as needed.

### Business Process Guides

| Function Module | Description | Applicable Scenario |
|---------|------|---------|
| [New Model Quantization Tuning Process](#21-new-model-quantization-tuning-process) | Quantization scheme design -> weight quantization -> operator and graph construction -> evaluation -> release; if the target is not met, revise the scheme and close the loop again | A new model, or a target quantization mode that has not been verified, requiring the whole path from scheme design to release |
| [Accuracy Tuning Methods](#22-accuracy-tuning-methods) | Systematic tuning approaches: outlier suppression, quantization strategy, calibration dataset, and fallback | The manual tuning path when the accuracy target is not met |
| [Mainstream Model Quantization Deployment](#23-mainstream-model-quantization-deployment) | End-to-end guidance: obtain quantized weights, deploy the inference service, evaluate accuracy, and perform online monitoring | Scenarios where quantized weights have been obtained and need to be deployed in production |
| [Quantization Inference Accuracy Anomaly Locating](#24-quantization-inference-accuracy-anomaly-locating) | Locate the source of abnormal output by "understanding the input/output format -> logprobs confirmation -> control experiment -> layer-by-layer activation comparison" | More bad cases, abnormal output, or inability to align with the floating-point output after quantization |

### Tool Usage Guides

| Function Module | Description | Applicable Scenario |
|---------|------|---------|
| [One-Click Quantization](#31-one-click-quantization) | Integrates the best practices for quantizing mainstream large models, supports multiple quantization algorithms and export formats, and is ready to use out of the box; covers the complete flow from model support confirmation to deliverable verification, and explains each parameter of the `msmodelslim quant` command step by step | Recommended first choice; applicable to models that support one-click quantization |
| [Weight Quantization](#32-weight-quantization) | From downloading the floating-point model, model adaptation, and writing the configuration to running quantization and verifying the deliverables | LLM / multimodal understanding / multimodal generation models that are not included in the support matrix, or whose target quantization mode has not been verified |
| [Sensitive Layer Analysis](#33-sensitive-layer-analysis) | Analyze sensitive layers by `linear` / `layer` / `attn` scope and rewrite the quantization configuration | Linear layer fallback, whole-block fallback, FA quantization scheme design, and accuracy tuning |
| [Auto Tuning](#34-auto-tuning) | Automatically and iteratively search for the quantization configuration according to the accuracy target, with the whole flow automated | When the accuracy target is not met and repeated manual parameter tuning is not desired |
| [Debug Mode](#35-debug-mode) | Saves the context of the quantization process for troubleshooting and algorithm analysis | Issue diagnosis and algorithm research |
| [Fake Quantization Inference](#36-fake-quantization-inference) | Use small samples such as bad cases to verify whether the quantization deliverables can run inference correctly, and locate the stage where the accuracy problem occurs | Accuracy verification based on small-sample data |

## 2. Business Process Guides

### 2.1 New Model Quantization Tuning Process

When a new model or a target quantization mode has not been verified, follow the main path of "quantization scheme design -> weight quantization -> operator development and graph construction -> accuracy/performance evaluation -> weight release". If the evaluation does not meet the target, revise the quantization scheme and then return to weight quantization to continue the loop until release.

For the detailed process, see the [New Model Quantization Tuning Process](process_new_model_quantization_tuning.md).

### 2.2 Accuracy Tuning Methods

When the quantization accuracy is lower than expected, perform systematic tuning along the progressive path of "confirm that the accuracy issue is genuine -> adjust outlier suppression -> adjust the quantization strategy -> adjust the calibration dataset -> quantization fallback".

For the detailed methods, see the [Quantization Accuracy Tuning Guide](process_quantization_precision_tuning.md).

### 2.3 Mainstream Model Quantization Deployment

Starting from the quantized weights, complete the production deployment of the quantized model by following the steps "obtain quantized weights -> deploy the inference service -> evaluate accuracy -> perform online monitoring", establish an evaluation baseline, and continuously observe resource usage.

For the detailed method, see the [Mainstream Model Quantization Deployment Process Guide](process_mainstream_model_deployment.md).

### 2.4 Quantization Inference Accuracy Anomaly Locating

When the quantized model produces bad cases, abnormal output, or output that cannot be aligned with the floating-point output, narrow down the problem from the macro level to the micro level by following "understanding the input/output format -> quick logprobs confirmation -> Completion API control experiment -> layer-by-layer activation comparison". This process locates the source of the abnormal output and provides handling suggestions.

For the detailed method, see the [Quantization Inference Accuracy Anomaly Locating Process Guide](process_quantization_accuracy_anomaly_locating.md).

## 3. Tool Usage Guides

### 3.1 One-click Quantization

One-click quantization is based on the msModelSlim best practice library, automatically matches the optimal configuration, and freely combines quantization pipelines through YAML. In addition to command-line parameters, it also supports:

- **Multiple quantization algorithms**: outlier suppression, linear quantization, KV Cache, FA3, and others. For the algorithm overview and configuration fields, see the [Quantization Algorithm Overview](../knowledge_base/quantization_algorithms/README.md).
- **Multiple export formats**: AscendV1, compressed-tensors, MindIE SD, and others. For format selection and compatibility, see [Quantization Formats](../knowledge_base/quantization_format/README.md).

For detailed usage, see the [One-Click Quantization Usage Guide](usage_one_click_quantization.md), which explains each parameter of the `msmodelslim quant` command step by step.

### 3.2 Weight Quantization

When the target model (large language model, multimodal understanding, or multimodal generation) is not included in the support matrix, or is included but its target quantization mode has not been verified, complete the integration by "downloading the floating-point model -> completing model adaptation -> writing the quantization configuration -> running quantization -> verifying the deliverables"; for the adaptation details of each category, see the corresponding integration guide. This process uses `msmodelslim quant` as the unified entry point to produce deployable quantized weights; deployment and accuracy/performance evaluation are follow-up operations.

For the detailed method, see the [Weight Quantization Usage Guide](usage_weight_quantization.md).

### 3.3 Sensitive Layer Analysis

Sensitive layer analysis is used to evaluate how sensitive each layer is to quantization. Advance by analysis scope: "confirm the recommended metric -> obtain the floating-point weights -> complete model adaptation as needed -> run the analysis command -> interpret the results and rewrite the YAML"; for the few metrics that require additional analysis interfaces, implement the required interfaces as described in the corresponding algorithm documentation.

For the procedure of each scope, see:

- [Sensitive Linear Layer Analysis Usage Guide](usage_sensitive_linear_analysis.md) (`scope=linear`)
- [Layer-wise Sensitive Layer Analysis Usage Guide](usage_sensitive_layer_analysis.md) (`scope=layer`)
- [Attention Sensitive Layer Analysis Usage Guide](usage_sensitive_attn_analysis.md) (`scope=attn`)

### 3.4 Auto Tuning

The auto tuning feature can automatically and iteratively search for the optimal quantization configuration according to the accuracy target, with the whole quantization and evaluation flow automated and no repeated manual parameter tuning.

For detailed usage, see the [Auto Tuning Usage Guide](usage_auto_precision_tuning.md).

### 3.5 Debug Mode

Debug mode is an advanced feature of one-click quantization. Once enabled, it saves the process context (intermediate results, statistics, tensor information, and more) when quantization completes, which facilitates troubleshooting and algorithm analysis.

For detailed usage, see the [Debug Mode User Guide](usage_debug_mode.md).

### 3.6 Fake Quantization Inference

After the quantized weights are generated, fake quantization inference can be used to verify the accuracy of the quantization deliverables based on small-sample data.

For detailed usage, see the [Fake Quantization Inference Usage Guide](usage_fake_quant_evaluation.md).
