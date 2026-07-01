# **msModelSlim Rich Quantization Model Support Feature Design Specifications**

|                                           |                  |
| ----------------------------------------- | ---------------- |
| SIG group:                                | sig-msit         |
| Incorporated into the following versions: | To be determined |
| Designer:                                 | panyj1993        |
| Date:                                     | 2026             |

**Copyright © 2026 msModelSlim Community**

Your reproduction, use, modification and distribution of this document is subject to the Creative Commons Attribution-ShareAlike 4.0 International Public License ("CC BY-SA 4.0"). For ease of understanding, you can visit thehttps://creativecommons.org/licenses/by-sa/4.0/Get an overview (but not an alternative) of CC BY-SA 4.0. You can obtain the complete agreement of CC BY-SA 4.0 from the following website:https://creativecommons.org/licenses/by-sa/4.0/legalcode.

**Revision records**

| Date       | Revised version | Revision Description                               | Authors   | Audited |
| ---------- | --------------- | -------------------------------------------------- | --------- | ------- |
| 2026.1. 22 | v1              | Typical Model Quantification Design Specifications | panyj1993 | xxx     |

**The Table of Contents**

1. Feature Overview

     1.1 Scope

     1.2 Feature Requirement List

2. Requirement Scenario Analysis

     2.1 Feature Requirement Source and Value Overview

     2.2 Feature Scenario Analysis

     2.3 Feature Impact Analysis

     2.3.1 Hardware Limitations

     2.3.2 Technical Limitations

     2.3.3 Impact on the License

     2.3.4 Analysis of Impact on System Performance Specifications

     2.3.5 Analysis of Impact on System Reliability Specifications

     2.3.6 Impact on System Compatibility

     2.3.7 Impact Analysis on Interaction and Conflicts with Other Key Features

     2.4 Analysis on the Implementation Solution of Similar Community/Commercial Software

3. Feature/Function implementation principles (multiple use cases can be broken down)

     3.1 Objectives

     3.2 Overall Solution

4. Use Case 1 Implementation

     4.1 Design Idea

     4.2 Constraints

     4.3 Detailed implementation (module-level or process-level message sequence diagram from user entry)

     4.4 Interfaces Between Subsystems (Mainly Covering the Definition of Module Interfaces)

     4.5 Detailed Design of Subsystems

     4.6 DFX Attribute Design

     4.6.1 Performance Design

     4.6.2 Upgrade and Capacity Expansion Design

     4.6.3 Exception Handling Design

     4.6.4 Resource Management Design

     4.6.5 Miniaturized Design

     4.6.6 Testability Design

     4.6.7 Security Design

     4.7 External Interfaces

     4.8 Self-Test Case Design

5. Use Case 2 Implementation

6. Reliability and availability design

     6.1 Redundancy Design

     6.2 Fault Management

     6.3 Overload control design

     6.4 Upgrade Without Service Interruption

     6.5 Design for human error

     6.6 Fault prediction and prevention design

7. Security Design

     7.1 Low Level Threat Analysis

     7.1.1 Layer 2 Data Flow Diagram

     7.1.2 Service Scenarios and Trust Boundary Description

     7.1.3 Analysis of External Interaction Parties

     7.1.4 Data Stream Analysis

     7.1.5 Process Analysis

     7.1.6 Data Storage Analysis

     7.1.7 Defect List

     7.2 Sensitive Data Analysis

     7.2.1 Sensitive Data List

     7.2.2 Sensitive Operation Check

     7.3 Use Case Implementation

     7.3.1 Design Idea

     7.3.2 Detailed Implementation

8. Design for features and non-functional quality attributes

     8.1 Testability

     8.2 Serviceability

     8.3 Evolvability

     8.4 Openness

     8.5 Compatibility

     8.6 Scalability/Scalability

     8.7 Maintainability

     8.8 Documentation

9. (Optional) Data Structure Design

10. List of references

**Table Catalogue**

Table X: Feature scenario correlation analysis

Table X: List of feature requirements

**Figure Catalogue**

Figure X: Overall implementation principle

Figure X: Sample Diagram: Process Flow Diagram

**List of abbreviations:**

| Abbreviations Abbreviations | Full spelling                     | Chinese explanation Chinese explanation                                                                        |
| --------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| LLM                         | Large Language Model              | large language model                                                                                           |
| VLM                         | Vision-Language Model             | Visual Language Model (Multimodal Large Model)                                                                 |
| MoE                         | Mixture of Experts                | Hybrid Expert Architecture                                                                                     |
| W8A8                        | Weight 8-bit Activation 8-bit     | Both the weight and activation values are quantized as INT8.                                                   |
| W4A8                        | Weight 4-bit Activation 8-bit     | The weight is changed to INT4, and the activation value is quantized to INT8.                                  |
| W4A16                       | Weight 4-bit Activation 16-bit    | The weight is changed to INT4, and the activation value remains FP16.                                          |
| W8A16                       | Weight 8-bit Activation 16-bit    | The weight is changed to INT8, and the activation value remains FP16.                                          |
| W4A4                        | Weight 4-bit Activation 4-bit     | Both the weight and activation values are quantized as INT4.                                                   |
| KV Cache                    | Key-Value Cache                   | Key-value pair caching in attention mechanism                                                                  |
| PDMIX                       | Prefill-Decode Mixed Quantization | Hybrid strategy for dynamic quantization in the prefilling phase and static quantization in the decoding phase |
| FA3                         | Flash Attention 3                 | An INT8 Quantization Algorithm for Attention Activation Value Based on Per-head Granularity                    |
| NPU                         | Neural Processing Unit            | Neural network processor (Ascend AI processor)                                                                 |
| ViT                         | Vision Transformer                | Vision Transformer, for image feature extraction                                                               |
| MLA                         | Multi-head Latent Attention       | Potential attention mechanism of multiple heads (used by models such as DeepSeek)                              |
| YAML                        | YAML Ain't Markup Language        | A human-readable data serialization format for quantifying profiles                                            |
| OOM                         | Out of Memory                     | Video buffer/memory overflow                                                                                   |

## 1. Feature Overview

With the rapid development of large language model and multimodal model, the demand for model inference acceleration is increasingly urgent. Quantification technology,as an important means of model compression,can significantly reduce model storage and computing overhead and improve inference speed. This feature enables the msModelSlim tool to support typical models in the industry and provides multiple quantitative configuration solutions to accelerate inference and ensure precision compliance.

This feature adds quantification for model series such as GLM-4.7, Qwen2.5-VL, Qwen3-VL, GLM4.6V, HunyuanVideo, Flux.1-dev, Wan2.2, Qwen2.5-Omni, and Qwen3-Omni. Supported. Multiple quantization precision configurations, such as W8A8 and W4A4, are supported to balance performance and precision in different scenarios.

### 1.1 Scope

This feature provides the following functions:

1. **Quantification support for GLM series models: W8A8 quantization for GLM-4.7 and W8A8 quantization for GLM4.6V models**
2. **Qwen2.5-VL series model quantification support: supports W8A8 quantification of 7B, 32B, and 72B.**
3. **Qwen3-VL series model quantization support: supports W8A8 quantization of 30B-A3B-Instruct and 235B-A22B-Instruct models.**
4. **HunyuanVideo model quantization support: supports W8A8 and W4A4 quantization configurations.**
5. **Flux.1-dev model quantization support: supports W8A8 and W4A4 quantization configurations.**
6. **Wan2.2 model quantization support: supports W8A8 and W4A4 quantization configurations.**
7. **Qwen2.5-Omni model quantization support: 7B model W8A8 quantization**
8. **Qwen3-Omni series model quantization support: supports W8A8 quantization of 30B-A3B-Thinking and 30B-A3B-Instruct models.**

### 1.2 Feature Requirement List

Table 1 List of feature requirements

| Requirement No. | Requirement name                       | Feature Description                                                                                                                    | Remarks              |
| --------------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| 1               | GLM-4.7 Model W8A8 Quantification      | Supports INT8 quantification of the weight and activation value of the GLM-4.7 model, providing the inference acceleration capability. | large language model |
| 2               | Qwen2.5-VL Model W8A8 Quantification   | Supports W8A8 quantization of three Qwen2.5-VL 7B/32B/72B scales and multi-modal scenarios of visual languages.                        | multimodal model     |
| 3               | Qwen3-VL Model W8A8 Quantification     | W8A8 quantization for Qwen3-VL-30B-A3B-Instruct and 235B-A22B-Instruct models                                                          | multimodal model     |
| 4               | GLM4.6V Model W8A8 Quantification      | W8A8 Quantization for GLM4.6V Vision Model                                                                                             | multimodal model     |
| 5               | Quantification of HunyuanVideo Model   | Supports W8A8 and W4A4 quantization of HunyuanVideo models to meet different precision requirements.                                   | multimodal model     |
| 6               | Flux.1-dev model quantization          | W8A8 and W4A4 quantization for Flux.1-dev models                                                                                       | multimodal model     |
| 7               | Wan2.2 Model Quantification            | W8A8 and W4A4 quantization for Wan2.2 model                                                                                            | multimodal model     |
| 8               | Qwen2.5-Omni Model W8A8 Quantification | W8A8 quantization for Qwen2.5-Omni-7B model                                                                                            | multimodal model     |
| 9               | Qwen3-Omni Model W8A8 Quantification   | W8A8 quantization for Qwen3-Omni-30B-A3B-Thinking and 30B-A3B-Instruct models                                                          | multimodal model     |

## 2. Requirement Scenario Analysis

### 2.1 Feature Requirement Source and Value Overview

With the wide application of large-language model and multi-modal model in the industry, model inference performance has become a key factor restricting large-scale deployment. Currently, mainstream models in the industry, such as GLM, Qwen, HunyuanVideo, and Flux, are slow in the original precision and occupy a large amount of storage resources. Therefore, the real-time requirements of the production environment cannot be met.

The quantization technology can significantly improve the inference speed and reduce the memory usage by reducing the numerical precision of model weights and activation values (for example, from FP16/BF16 to INT8/INT4). This feature provides quantitative support for typical models in the industry and helps users:

1. **Reduced storage costs: Quantification model size can be reduced by 50% to 75%, reducing storage and transport costs**
2. **Expanded deployment scale: More model instances can be deployed with the same hardware resources, improving service throughput.**
3. **Precision assurance: The quantization configuration and calibration policies are carefully designed to ensure that the quantized model precision meets service requirements.**

Without this feature, users cannot use the msModelSlim tool to quantify the preceding models. Instead, they need to implement the quantification process or use other tools, which increases the cost and maintenance burden and reduces the market competitiveness of the tool.

### 2.2 Feature Scenario Analysis

#### Scenario Trigger Conditions and Objects

**Role: AI model developer, model deployment engineer, and algorithm optimization personnel**

**Use the msModelSlim quantization tool (command line tool or Python API).**

**Trigger conditions:**

 * Users need to quantify the supported models to improve the inference performance.
 * Users need to reduce the model storage usage.
 * Users need to deploy models in resource-constrained environments.

**Skill requirements for the target audience:**

 * Familiar with Python programming and deep learning frameworks (PyTorch/MindSpore)
 * Understand the basic concepts of model quantization.
 * Basic command line operations

#### Main Application Scenarios

1. **Model Inference Acceleration Scenario**
    
     * Sub-scenario: online service inference acceleration and batch inference task acceleration
     * Key operations: Load the original model, configure quantization parameters, perform quantization, save the quantization model, and deploy inference.
2. **Resource-limited deployment scenario**
    
     * Sub-scenario: edge device deployment, mobile device deployment, and multi-model concurrent deployment
     * Key operations: Select proper quantization precision (W8A8/W4A4) → Quantization model → Accuracy verification → Deployment
3. **Multi-modal model optimization scenario**
    
     * Sub-scenario: Visual language model optimization, video generation model optimization, and image generation model optimization
     * Key operations: Prepare multimodal calibration data → Configure multimodal quantization parameters → Perform quantization → Verify multimodal task precision

### 2.3 Feature Impact Analysis

As a core function extension of the msModelSlim tool, this feature is located at the model adaptation layer of the quantization processing pipeline. The main impact modules are as follows:

 * **Model loading module: supports loading and parsing of new models.**
 * **Quantification configuration module: A proper quantitative strategy needs to be configured for each model.**
 * **Quantization execution module: needs to adapt to the quantization processing of different model structures.**
 * **Calibration data processing module: supports calibration data formats of different model types.**

#### Interaction Analysis with Other Requirements and Features

 * **Interaction with the existing quantization function: Reuse the existing quantization algorithm and process framework, and add the model adaptation layer.**
 * **Interaction with multi-modal quantization: Some models (such as Qwen2.5-VL and Qwen3-VL) are multi-modal models and need to reuse the multi-modal quantization framework.**
 * **Interaction with the inference framework: The quantified model needs to be verified on the inference framework such as MindIE and vLLM.**

#### Platform Difference Analysis

**Hardware platform: supports the Ascend NPU (Atlas series). Some functions support the CPU.**

**Operating system: Linux, such as Ubuntu and CentOS**

#### Compatibility Analysis

 * **Forward compatibility: New models support the quantization function that does not affect the existing models.**
 * **Configuration compatibility: The new model quantitative configuration complies with the uniform YAML configuration protocol.**
 * **Interface compatibility: Compatibility between Python APIs and command-line interfaces is maintained.**

#### Constraints and Limitations

1. Model needs to be loaded from the HuggingFace format.
2. Quantification requires calibration of datasets
3. Some models require a specific version of the transformers library.

#### 2.3.1 Hardware Limitations

**NPU hardware requirements:**

 * Ascend NPU (Atlas 300I/300T/800)
 * The 7B model requires at least 16 GB memory, and the 72B model requires at least 128 GB memory.
 * Multi-card quantization: supports parallel quantization of multiple cards to accelerate processing.

**Workaround:**

 * Provides the fragment quantification capability for large models.
 * Supports the CPU fallback mode (low performance).
 * Provides the parameter for controlling the size of the quantization weight file.

#### 2.3.2 Technical Limitations

**Operating system: Linux (Ubuntu 18.04+ and CentOS 7+)**

**Programming language: Python 3.7+**

**Deep learning framework:**

 * PyTorch 1.8+ (for model loading and quantification)
 * MindSpore (for partial inference verification)
 * transformers library (version requirements vary by model)

**Workaround:**

 * Environment dependency check scripts are provided.
 * Specify the dependency version requirements of each model in the document.
 * Docker container-based deployment

#### 2.3.3 Impact Analysis on the License

The following table lists the licenses for the models and libraries on which this feature depends.

1. **Model license:**
    
     * GLM series: Apache 2.0
     * Qwen series: Tongyi Qianwen license (partial commercial restrictions)
     * HunyuanVideo: To be confirmed
     * Flux.1-dev:CreativeML Open RAIL-M License
     * Wan2.2: to be confirmed
2. **Dependent library license:**
    
     * transformers:Apache 2.0
     * PyTorch:BSD-style
     * Other dependent libraries need to be confirmed one by one.

**Compliance requirements:**

 * All third-party libraries must pass the license compliance review.
 * Specify the license limit of each model in the document.
 * Provide the license declaration file.

#### 2.3.4 Analysis of Impact on System Performance Specifications

**Memory requirements:**

 * 7B Model Quantization: Minimum 32 GB system memory required
 * 32B model quantization: requires at least 64 GB system memory
 * 72B/235B model quantization: requires at least 128 GB system memory, multi-card recommended

**Storage requirements:**

 * Temp file for quantization process: about 2-3 times the size of the model
 * Quantized model storage: about 50% to 75% of the original model

**Computing resources:**

 * Quantification time: 30 to 60 minutes for the 7B model and 2-4 hours for the 72B model (single SIM card)
 * Supports multi-card parallelism to speed up the quantization process

#### 2.3.5 Analysis of Impact on System Reliability Specifications

**Quantified success rate:**

 * Objective: Quantification success rate ≥ 95% under standard calibration dataset
 * Precision assurance: After quantization, the precision loss of the model is less than or equal to 3% (compared with the original model).

**Troubleshooting:**

 * Provide detailed error logs when quantization process exceptions occur.
 * Resumable transfer after quantization is interrupted
 * Provide a mechanism for verifying quantitative results.

#### 2.3.6 Impact on System Compatibility

**Forward compatibility:**

 * The new model supports the quantization function that does not affect the existing model.
 * Existing quantization configuration and API interface compatibility

**Version compatibility:**

 * The quantified model weight format is compatible with the existing reasoning framework.
 * Compatibility processing during model version upgrade

#### 2.3.7 Impact Analysis on Interaction and Conflicts with Other Key Features

**Interaction with multimodal quantization characteristics:**

 * Multimodal quantization framework for model reuse such as Qwen2.5-VL and Qwen3-VL
 * Shared Multimodal Calibration Data Processing Logic

**Interaction with the Inference Framework:**

 * Quantification models need to be verified on the MindIE and vLLM inference frameworks.
 * Ensure that the quantization format is compatible with the inference framework

**Interaction with the model transformation feature:**

 * Supports quantized model format conversion.
 * Supports model conversion between different inference frameworks.

### 2.4 Analysis of implementation solutions for similar community/commercial software

#### Comparison of similar tools

**1. GPTQ/AWQ (community tool)**

 * **Implementation mechanism: post-training quantization method based on weight quantization**
 * **Advantages: supports multiple models and provides fast quantization speed.**
 * **Disadvantages: This problem mainly focuses on weight quantification, limited support for activation value quantification, and insufficient support for multimodal models.**

**2. msModelSlim (this tool)**

 * **Implementation mechanism: Ascend NPU-based quantitative optimization, supporting multiple quantitative policy combinations**
 * **Advantage:**
    
     * Optimized for Ascend NPU hardware and excellent performance
     * Supports multi-modal model quantization (VL, SD, etc.)
     * Provides unified configuration protocols and is easy to use.
     * Supports multiple quantization precisions, such as W8A8 and W4A4.
 * **Disadvantages: Mainly oriented to the Ascend ecosystem and limited support for other hardware**

#### Competitive advantages of this feature

1. **Wide model coverage: Supports mainstream models in the industry, including language models, multi-modal models, and generation models.**
2. **Various quantization precisions: supports multiple precision configurations, such as W8A8 and W4A4, to meet different scenario requirements.**
3. **Hardware optimization: In-depth optimization for the Ascend NPU and full use of hardware features**
4. **Ease-of-use: Unified YAML configuration protocol, lowering the usage threshold.**

## 3. Feature/Function implementation principles (multiple use cases can be broken down)

### 3.1 Objectives

This feature aims to add the support for 16 model quantization configurations for the msModelSlim tool. The objectives are as follows:

1. **Functional Objectives:**
    
     * W8A8 quantization for GLM-4.7 and GLM4.6V models
     * Supports W8A8 quantization of Qwen2.5-VL 7B/32B/72B model
     * W8A8 quantization for Qwen3-VL-30B-A3B-Instruct and 235B-A22B-Instruct models
     * W8A8 and W4A4 quantization for HunyuanVideo, Flux.1-dev, Wan2.2 models
     * W8A8 quantization for Qwen2.5-Omni-7B, Qwen3-Omni-30B-A3B-Thinking and 30B-A3B-Instruct models
2. **Performance objective:**
    
     * 50% - 75% reduction in model size after quantization
     * Quantization precision loss within 3% (compared with the original model)
3. **Ease-of-use objective:**
    
     * Provides a unified YAML configuration interface.
     * One-click quantification
     * Provide detailed quantitative documentation and examples

### 3.2 Overall Solution

#### Hardware Selection

 * **Main hardware platform: Ascend NPU (Atlas 300I/300T/800 series)**
 * **Auxiliary hardware platform: CPU (for partial preprocessing and postprocessing)**

#### Algorithm selection

1. **Weight quantization algorithm:**
    
     * W8A8: Use the MinMax or AutoRound algorithm.
     * W4A4: Use the Smooth Scale Zero (SSZ) or AutoRound algorithm.
2. **Activated value quantization algorithm:**
    
     * W8A8: Dynamic Quantization Using MinMax
     * W4A4: Dynamic quantization using MinMax
3. **Abnormal value suppression algorithm:**
    
     * SmoothQuant (m1/m2/m4)
     * Flex Smooth Quant
     * QuaRot (for multimodal models)

#### Architecture Layout

The quantification process adopts a hierarchical architecture:

1. **Model adaptation layer: loads different models and parses structures.**
2. **Configuration parsing layer: parses YAML configurations and generates quantification policies.**
3. **Quantization execution layer: Executes specific quantization algorithms.**
4. **Calibration data processing layer: processes calibration data and supports multiple formats such as text, image, and video.**
5. **Result saving layer: saves the quantized model weight.**

#### Use Case Breakdown

Based on the model type and quantitative configuration, the feature implementation is divided into the following use cases:

1. **Use Case 1: GLM-4.7 Model W8A8 Quantification**
2. **Use Case 2: Qwen2.5-VL 7B Model W8A8 Quantization**
3. **Use Case 3: Qwen2.5-VL 32B Model W8A8 Quantization**
4. **Use Case 4: Qwen2.5-VL 72B Model W8A8 Quantization**
5. **Use Case 5: Qwen3-VL-30B-A3B-Instruct Model W8A8 Quantization**
6. **Use Case 6: Qwen3-VL-235B-A22B-Instruct Model W8A8 Quantization**
7. **Use Case 7: GLM4.6V Model W8A8 Quantization**
8. **Use Case 8: Quantization of HunyuanVideo Model W8A8**
9. **Use Case 9: W4A4 Quantization of HunyuanVideo Model**
10. **Use Case 10: Flux.1-dev Model W8A8 Quantization**
11. **Use Case 11: Flux.1-dev Model W4A4 Quantization**
12. **Use Case 12: Wan2.2 Model W8A8 Quantization**
13. **Use Case 13: Wan2.2 Model W4A4 Quantification**
14. **Use Case 14: Qwen2.5-Omni-7B Model W8A8 Quantization**
15. **Use Case 15: Qwen3-Omni-30B-A3B-Thinking Model W8A8 Quantization**
16. **Use Case 16: Qwen3-Omni-30B-A3B-Instruct Model W8A8 Quantization**

#### Interconnection Principles

1. **Unified configuration protocol: All models use the same YAML configuration protocol.**
2. **Interface compatibility: Keep compatibility with existing quantization interfaces**
3. **Modular design: Model adaptation, quantization algorithm, and data processing modules are independent, facilitating expansion.**
4. **Backward compatibility: New model support does not affect existing functions.**

#### Overall solution architecture

```text
┌─────────────────────────────────────────────────────────┐
│               User Interface Layer                      │
│ (Command Line Tool / Python API / YAML Configuration)   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│           Configuration parsing layer                   │
│       (YAML parsing/Quantization policy                 |
|       generation/Parameter verification)                │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                 Model Adaptation Layer                  │
│          (Loading models such as GLM/                   |
|        Qwen/HunyuanVideo/Flux/Wan2.2, etc.)             │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                Calibration data processing layer        │
│  (Text/Image/Video data preprocessing/Data loading)     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Quantization execution layer               │
│ (Weight quantization/Activation quantization/Abnormal   |
|        value suppression/KVCache quantization)          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  Result Storage Layer                   │
│         (Quantized Weight Storage / Format              |
|              Conversion / Verification)                 │
└─────────────────────────────────────────────────────────┘
```

Figure 1: Overall architecture of quantization processing

## 4. Use Case 1 Implementation: GLM-4.7 Model W8A8 Quantification

### 4.1 Design Idea

GLM-4.7 is a large language model developed by AI and uses the Transformer architecture. This use case implements W8A8 quantization of the GLM-4.7 model, that is, both the weight and activation value are quantized to the INT8 precision.

**Design Idea:**

1. **Model adaptation: The GLM-4.7 model loading and parsing logic is added to the model adaptation layer to identify key layers (such as Linear, Embedding, and LayerNorm) in the model structure.**
2. **Quantization strategy: Use the MinMax algorithm to quantize weights, use dynamic quantization to quantize activation values, and use FP16 precision for sensitive layers such as LayerNorm.**
3. **Abnormal value processing: Use the SmoothQuant (m1/m2) algorithm to suppress abnormal activation values and improve the quantization precision.**
4. **Calibration data: Use text datasets such as C4 or WikiText for calibration. The recommended number of calibration samples is 512 to 1024.**
5. **Precision assurance: The hierarchical quantization policy and sensitive layer protection are used to ensure that the precision loss of the model is within 3% after quantization.**

### 4.2 Constraints

**Hardware Constraints:**

 * The Ascend NPU (Atlas 300I/300T/800 series) is required.
 * Minimum 32 GB system memory (for 7B scale models)
 * At least 16 GB video memory

**Software Constraints:**

 * Python 3.7+
 * PyTorch 1.8+
 * The transformers library version must support the GLM-4.7 model (4.30+ recommended).
 * The model must be in the HuggingFace format.

**Data Constraints:**

 * A calibration data set needs to be prepared. (text format, 512 to 1024 samples recommended)
 * The distribution of calibration data must be similar to that of model training data.

**Function Constraints:**

 * Model training is not supported during quantization.
 * Quantized models support only inference and do not support continuous training.

### 4.3 Detailed implementation (module-level or process-level message sequence diagram from user entry)

#### 4.3.1 Quantization Process Sequence Diagram

The quantization process includes five phases: configuration analysis, model loading, calibration data processing, quantization execution and result saving. Each module interacts with each other through a unified interface to ensure the integrity and reliability of the quantization process.

**Main process:**

1. Users submit quantification tasks through command lines or Python APIs.
2. The configuration parsing module parses the YAML configuration and generates a quantization policy.
3. The model mediation loads the GLM-4.7 model and parses the structure.
4. The calibration data processing module loads and preprocesses calibration data.
5. The quantization execution engine performs weight quantization and activation value quantization.
6. The result saving module saves the quantized model weight and configuration.

#### 4.3.2 Module Interaction Description

**Configuration parsing module: parses user configurations and generates quantization policiesConfiguration object model adaptation layer: loads the GLM-4.7 model, parses the model structure, and identifies the layer calibration to be quantizedData processing module: loads text data, performs tokenization, and generates calibration data batch quantization Execution engine: Result saving module for executing core quantification processes, such as weight quantization, activation value quantization, and abnormal value processing: saves the quantification weight and configuration for subsequent inference.**

### 4.4 Interfaces Between Subsystems (Mainly Covering the Definition of Module Interfaces)

#### 4.4.1 Model Adaptation Layer Interfaces

**Added the following APIs:**`GLM47ModelLoader`

 * `load_model(model_path: str, device: str) -> torch.nn.Module`Load the GLM-4.7 model.
 * `analyze_structure(model: torch.nn.Module) -> ModelStructure`\: Analyze the model structure and identify the quantization target layer.

#### 4.4.2 Quantized Execution Engine Interface

**Modify the interface:**`QuantizationEngine`

 * `quantize_glm47_w8a8(model, calib_data, config) -> QuantizedModel`\: GLM-4.7 Model W8A8 Quantification

#### 4.4.3 Configuring the Parsing Interface

**Modify the interface:**`ConfigParser`

 * `parse_glm47_config(config_path: str) -> GLM47QuantConfig`\: Parses the GLM-4.7 quantization configuration.

### 4.5 Subsystem LLD

#### 4.5.1 Detailed Design of the Model Mediation Layer

**GLM-4.7 Model Loader: Using Transformers Library**`AutoModelForCausalLM`Load models from the HuggingFace Hub or local path. Identify the key components of the models, such as the embedding layer, transformer layer, and lm_head layer.

**Model structure analyzer: traverses all layers of the model, identifies the Linear layer (used for weight quantization), identifies the LayerNorm layer (marked as the sensitive layer, and retains the FP16), identifies the location of the activation function (used for activation value quantization), and generates the quantization target layer mapping table.**

#### 4.5.2 Detailed Design of the Quantized Execution Engine

**Weight quantization module: Apply MinMax quantization to the weight matrix at the linear layer, calculate the min/max value of each weight matrix, and use symmetric quantization. The quantization range is \[-128, 127\].**

**Activation value quantization module: uses the SmoothQuant algorithm to suppress abnormal values, collects statistics on activation value distribution on calibration data, calculates the quantization parameters of each activation layer, and applies dynamic quantization (quantification during inference).**

**Sensitive layer protection: The LayerNorm layer maintains the FP16 precision, the Embedding layer can retain the FP16 precision or quantization, and the output layer (lm_head) maintains the FP16 precision.**

#### 4.5.3 Detailed Design of Calibration Data Processing

**Text data loading: supports common text dataset formats such as C4 and WikiText, supports user-defined text files (one sample in each line), and automatically performs tokenization (using the tokenizer corresponding to the model).**

**Data preprocessing: Convert the text to token IDs, generate a fixed-length sequence (based on the max_length model), and generate the sequence in batches (batch_size is configurable).**

### 4.6 DFX Attribute Design

#### 4.6.1 Performance Design

**Quantified performance goals:**

 * Quantification time: 30 to 60 minutes for the 7B model (single NPU card)
 * Quantized model size: reduced by about 50%

**Performance optimization measures:**

 * Supports parallel quantification of multiple cards, accelerating the quantification process of large models.
 * Use asynchronous I/O to load calibration data, reducing wait time
 * Resumable transfer is supported during quantization to avoid repeated calculation.

**Impact on existing performance:**

 * New models do not affect the quantization performance of existing models.
 * The quantization algorithm reuses the existing implementation without extra performance overhead.

#### 4.6.2 Upgrade and Capacity Expansion Design

**Version compatibility:**

 * The quantified model weight format is compatible with the existing inference framework.
 * Compatibility processing during model version upgrade
 * The quantization configuration format is backward compatible.

**Capacity expansion design:**

 * Supports multi-card quantization and linearly expands the quantization speed.
 * Supports model shard quantification and processing of ultra-large models.

#### 4.6.3 Exception Handling Design

**Abnormal Scenarios and Handling:**

1. **If the model fails to be loaded, a detailed error message is returned, prompting the user to check the model path and format, and recording the error model path and error type.**
2. **Insufficient calibration data: Warn the user, but allow the quantification to continue (which may affect the precision), and record the number of calibration data samples.**
3. **Excessive loss of quantization precision: A precision report is provided. It is recommended that the quantization policy be adjusted or a higher precision be used, and the precision comparison before and after quantization is recorded.**
4. **Out of memory: Provide the option of fragment quantization, or recommend a larger memory device to record memory usage and overflow location**
5. **Quantized interrupt: Supports resumable upload, saves intermediate results, and records the interrupt location and completed quantification progress.**

#### 4.6.4 Resource Management Design

**Memory usage:**

 * Model loading: about 14 GB (7B model FP16)
 * Temporary memory for the quantization process: 1.5 times the model size (21 GB)
 * Total: Approximately 35 GB system memory

**Disk I/O:**

 * Model loading: Reads the model weight file (about 14 GB).
 * Calibration Data Load: Read the calibration dataset (depending on the size of the dataset)
 * Save the quantization result: Write the quantization weight (about 7 GB).

**Network I/O: If the model is loaded from the HuggingFace Hub, the network download is required.**

**Processing the resource usage:**

 * Out of memory: Offers the option of fragmentation quantization, or prompts the user to use a larger memory device
 * Insufficient disk space: Check the disk space and notify the user in advance.
 * If the network is abnormal, the offline mode is supported and the local model is used.

#### 4.6.5 Miniaturized Design

**Impact on the installation package size:**

 * Added the GLM-4.7 model adaptation code: about 50 KB.
 * New quantization configuration: about 10 KB
 * The total increase is about 60 KB, and the impact is negligible.

**Runtime Memory Impact:**

 * Memory added to the model adaptation layer: about 10 MB
 * Quantization engine has no extra memory overhead (reuse existing implementation)

**CPU usage: The quantization process is mainly performed on the NPU, and the CPU usage is low (< 10%).**

#### 4.6.6 Testability Design

**Function test: Test the GLM-4.7 model loading function, W8A8 quantification process integrity, quantization model inference function, and quantization configuration parsing function.**

**Performance test: test quantification time (target: 7B model < 60 minutes), inference speed after quantification, and model size after quantification (target: reduced by 50%)**

**Precision test: Accuracy comparison before and after quantification (target: loss < 3%), impact of different calibration data sets, and test boundary scenarios (minimum/maximum calibration data sets)**

**Abnormal test: test model loading failure, test calibration data insufficient, test memory insufficient, and test quantization interrupt and recovery.**

**Compatibility test: Test the compatibility between different transformers versions, PyTorch versions, and NPU hardware.**

#### 4.6.7 Security Design

##### 4.6.7.1 Safety Design Qualification

*Check the security design by referring to the security design checklist.*

| Security attributes                | Check Item                                                                                                                                                   | Check Item Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Involved or Not | Satisfied or not |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ---------------- |
| Access channel control             | Whether to add a listening port                                                                                                                              | The communication matrix needs to be updated for new listening ports.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | N               | Y                |
| Access channel control             | Whether to add new processes or inter-component communication                                                                                                | Added the communication matrix between new processes or components.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | N               | Y                |
| Access channel control             | Whether to add an authentication mode                                                                                                                        | The communication matrix and product documentation must be updated for the new authentication mode.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | N               | Y                |
| Permission control                 | Whether files or directories need to be created                                                                                                              | When creating a file or directory, you must explicitly specify the access permission for the file or directory.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | N               | Y                |
| Permission control                 | Check whether the account permission meets the "minimum permission principle".                                                                               | All accounts in the system must be assigned with the least permission.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | N               | Y                |
| Permission control                 | Whether user privilege escalation exists                                                                                                                     | Illegal user privilege escalation is prohibited.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | N               | Y                |
| Undisclosed Interface              | Whether to add GUC parameters                                                                                                                                | The product documentation needs to be updated when the GUC parameter is added.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | N               | Y                |
| Undisclosed Interface              | Add or modify functions, views, and system tables.                                                                                                           | When adding or modifying functions, views, and system tables, update the product documentation and consider permission control.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Y               | Y                |
| Undisclosed Interface              | Add SQL Syntax                                                                                                                                               | The new SQL syntax needs to be updated in the product documentation to support recording audit logs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | N               | Y                |
| Undisclosed Interface              | Whether to add internal tools                                                                                                                                | Product documentation needs to be updated for new internal tools.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | N               | Y                |
| Undisclosed Interface              | Check whether the script contains comment code.                                                                                                              | Do not comment out code in explanatory languages such as Shell and Python. The comment code must be deleted.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | N               | Y                |
| Undisclosed Interface              | Check whether there are access modes such as hidden commands, parameters, and ports.                                                                         | Access modes, such as commands, parameters, and ports, that are not used during maintenance on the live network (including but not limited to product production, commissioning, and maintenance purposes), must be deleted (e.g. by compiling macros)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | N               | Y                |
| Undisclosed Interface              | Check whether the system has hidden backdoors.                                                                                                               | Do not reserve any undisclosed accounts in the system. All accounts must be managed by the system and must be described in the documentation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | N               | Y                |
| Undisclosed Interface              | It is prohibited to provide cracking and network sniffing tools in the software (including software packages and patch packages) released to external users. | 1. It is prohibited to provide the software (including software packages and patch packages) released to external users that can change any user password or have the "password cracking capability". (Brute force cracking of passwords and malicious cracking of passwords by exploiting system/algorithm vulnerabilities) 2. Functions or tools used to decrypt files that contain sensitive data (such as configuration files and databases that contain keys). 2. Do not retain third-party network sniffing tools, such as tcpdump, gdb, strace, readelf, and process debugging tools, in the system. CPP, GCC, dexdump, mirror, JDK development/compilation tools, and self-developed debugging tools/scripts used only in the commissioning phase (for example, encryption and decryption scripts, commissioning functions, and commands that can be used only in the commissioning phase), which must be retained due to service requirements, and strict access control is required. In addition, describe the reason, application scenario, and risks for the retention. | N               | Y                |
| Sensitive data protection          | Authentication credentials cannot be stored in the system in plaintext and must be encrypted.                                                                | Authentication credentials (such as passwords and private keys) must be encrypted and cannot be stored in the system in plaintext.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | N               | Y                |
| Sensitive data protection          | The key used for encrypting sensitive data transmission cannot be hard-coded.                                                                                | Hard coding of passwords and keys is prohibited.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | N               | Y                |
| Sensitive data protection          | Check whether sensitive information, such as passwords and keys, is printed in plaintext.                                                                    | Do not display sensitive information (passwords, private keys, and pre-shared keys) in plaintext in logs, debugging information, error messages, and ps commands stored in the system.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | N               | Y                |
| Sensitive data protection          | Specifies whether to display the password in plaintext.                                                                                                      | Do not display passwords in plaintext.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | N               | Y                |
| Sensitive data protection          | Whether the default passwords of third-party and open-source software are used                                                                               | Do not use the default passwords of third-party and open-source software. For details, see section 1.5 in the Security Design Guide.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | N               | Y                |
| Sensitive data protection          | Indicates whether to store passwords in plaintext in configuration files.                                                                                    | Plaintext passwords cannot be written into configuration files. (except the scenario where the password must be configured during the installation, deployment, and use of the command-line tool.)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | N               | Y                |
| Sensitive data protection          | Whether to use insecure encryption algorithms                                                                                                                | Do not use proprietary or insecure encryption algorithms. Recommended Encryption Algorithm Security Design Guide.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | N               | Y                |
| Sensitive data protection          | Check whether sensitive information, such as passwords, is transmitted over secure channels.                                                                 | Sensitive information must be transmitted between untrusted networks through secure transmission channels or encrypted transmission. For details, see chapter 10 of the Security Design Guide.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | N               | Y                |
| Sensitive data protection          | Check whether sensitive information such as passwords and keys in the memory is destroyed after being used.                                                  | The passwords or keys in the memory are cleared immediately after being used.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | N               | Y                |
| Sensitive data protection          | The random number used in cryptographic algorithm must be the cryptographically defined secure random number.                                                | The random number used in the cryptographic algorithm must be the cryptographic secure random number. For details, see section 6.3 in the Security Design Guide.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | N               | Y                |
| Sensitive data protection          | Check whether there are insecure examples in the documentation.                                                                                              | The examples in the documentation must be secure and provide correct guidance for users. If the examples contain potential risks, describe the risks in the documentation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | N               | Y                |
| Certification                      | Provide authentication mechanism                                                                                                                             | The new system needs to provide the authentication mechanism and the authentication mechanism is enabled by default.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | N               | Y                |
| Certification                      | Indicates whether authentication is performed on the server.                                                                                                 | The authentication process needs to be performed on the server.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | N               | Y                |
| Certification                      | Indicates whether the server returns valid information after the authentication fails.                                                                       | After the authentication fails, the information returned by the server does not provide detailed information that can be used to locate the error cause.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | N               | Y                |
| External parameter validation      | Indicates whether to verify the validity of external input.                                                                                                  | 1. If external input data is used as the loop termination condition, array subscript, and memory allocation parameter, infinite loop, buffer overflow, memory overwriting, and DoS may occur. 2. Verify the validity of external input, such as file paths, to prevent injection risks.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | N               | Y                |
| Third-party component introduction | Third-party components are introduced.                                                                                                                       | 1. New third-party components must be scanned by using secure compilation options, viruses, vulnerabilities, open source fragment reference, license compliance, and open source components. For details, see the version release cyber security quality requirements. 2. The source of the new third-party components must be trusted.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | N               | Y                |

##### 4.6.7.2 Sensitive Data Analysis

###### 1. Sensitive data list

*The specific scope of sensitive data depends on the specific application scenario of the system. Designers should analyze and judge the sensitive data based on the risks. Typical sensitive data includes authentication credentials (such as passwords) and keys.*

| **Data field**                 | **Remarks/Descriptions**                           | **Data Field Sensitivity** | **Association processing module** | **Forced action**                                      | **Prohibited operations** |
| ------------------------------ | -------------------------------------------------- | -------------------------- | --------------------------------- | ------------------------------------------------------ | ------------------------- |
| Administrator Account/Password | User name and password of the system administrator | High                       | Login/Authentication              | Encrypted transmission/encrypted storage/anonymization | Output and logs           |
| ...                            | ...                                                | ...                        | ...                               | ...                                                    | ...                       |
|                                |                                                    |                            |                                   |                                                        |                           |

###### 2. Check sensitive operations

*1) Lifecycle dimension: For sensitive data identified, we need to identify the lifecycle of the data and identify the process of generation, use, transmission, persistence, and destruction to avoid unintentional omissions in the subsequent risk identification process. 2) High-risk handling process Identify whether sensitive data is handled with high risks. Typical high-risk processing includes printing, echoing, storage, hard coding, and insecure algorithms. From the perspective of information processing, these high-risk processes are prone to security vulnerabilities when sensitive data is processed. Therefore, detailed check is required. All identified sensitive data must be checked. The sensitive data check matrix is as follows:*

For example, in a typical web system, the following table lists the check results of sensitive data (administrator accounts and passwords) in the lifecycle.

 * Generated: The administrator logs in to the system for the first time to set the password.
 * Usage: The administrator uses the password for authentication when logging in to the system.
 * Transmission: After the administrator enters the login password on the client, the password is transmitted to the server through the network.
 * Persistence: After the administrator sets a password for the first time, the server persists the password in the backend database.
 * Destroy: After a specified period, the administrator is forced to change the password and delete the old password.

|                    |                                                               Produced                                                               |                         Use the                          |                                                        Transmission                                                        |                Persistence                 |                                       Destroy                                        |
|:------------------:|:------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------:|:------------------------------------------------------------------------------------:|
|       print        |                                                            Not involved.                                                             | The password will not be printed in any form during use. | No encryption is required in the secure transmission channel. Encrypted transmission over non-secure transmission channels |               Not involved.                | The password is not printed during the destruction, but operation logs are recorded. |
|       Output       |                 In the ciphertext command output on the client, the password is displayed as \*\*\*\*\*\*\*\*\*\*\*.                 |                      Not involved.                       |                                                       Not involved.                                                        |               Not involved.                |                                    Not involved.                                     |
|      Storage       | After a user enters a password, the password is encrypted and saved to the backend database using the security encryption algorithm. |                         congener                         |                                                       Not involved.                                                        | Encrypted storage of the back-end database |          Delete the corresponding password from the backend database table.          |
|     Hard-coded     |                                                            Not involved.                                                             |                      Not involved.                       |                                                       Not involved.                                                        |               Not involved.                |                                    Not involved.                                     |
| Insecure algorithm |                                            Encryption using the AES256 security algorithm                                            |              In-memory decryption when used              |                             Non-secure transmission channels use secure encryption algorithms.                             |                  congener                  |                                    Not involved.                                     |

##### 4.6.7.3 Design Implementation

**File permission control: When creating a quantification result file, the permission is explicitly set to 644 (users can read and write, and other users can read only). When creating a temporary file, the permission is set to 600 (only users can read and write).**

**External input verification: Model path verification: Check path validity to prevent path injection. Configuration file verification: Verify the YAML format to prevent configuration injection. Calibration data check: Check the data format to prevent malicious data.**

**Log security: Logs do not record the model weight content and complete file paths (only relative paths or file names are printed). Sensitive information (such as model paths) is anonymized in logs.**

### 4.7 External Interfaces

**Command line interface: The command line parameter is added to support GLM-4.7 model quantization. The command format is as follows:**`msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type glm-4.7 --quant_type w8a8 --trust_remote_code True`

**Python API: Added**`quantize_glm47_w8a8()`Function, function signature:`quantize_glm47_w8a8(model_path: str, config_path: str, output_path: str) -> None`

**Configuration file format: The configuration file in YAML format is supported. The GLM-4.7 model-specific configuration item is added. (model type, path, quantization precision, algorithm, etc.)**

**N/A: GUC parameters, SQL syntax, network protocols, system table view functions, and drivers (JDBC/ODBC) are not involved.**

### 4.8 Self-Test Case Design

#### 4.8.1 Function Test Cases

**Test case 1: GLM-4.7 model loading test: Prepare the GLM-4.7 model in HuggingFace format, invoke the model loading interface, and verify that the model object is successfully created. Expected result: The model is loaded successfully and the model object is returned.**

**Case 2: W8A8 Quantization Process Test: Prepare the GLM-4.7 model and calibration data, configure quantization parameters (W8A8), perform quantization, and verify the generation of quantization result files. Expected result: The quantization is successful and a weight file is generated.**

**Case 3: Quantized model inference test: Load the quantized model, input the test text, perform inference, and verify the output. Expected result: The inference is successful and a proper result is generated.**

#### 4.8.2 Performance Test Cases

**Case 4: Quantization time test: Record the quantization start time, perform quantization, record the quantization end time, and calculate the quantization time. Expected result: The 7B model quantization time is less than 60 minutes.**

**Case 5: Inference speed test: Test the inference speed of the FP16 model, the inference speed of the quantized model, and the computing speed improvement ratio.**

#### 4.8.3 Precision Test Cases

**Case 6: Quantization precision test: Evaluate the precision of the FP16 model on the test set, evaluate the precision of the quantized model on the same test set, and calculate the precision loss. Expected result: The precision loss is less than 3%.**

#### 4.8.4 Abnormal Test Cases

**Case 7: Model loading failure test: Use a nonexistent model path to load a model and verify the error information. Expected result: A clear error message is returned, prompting the user to check the path.**

**Case 8: Insufficient calibration data test: Use calibration data with less than 512 samples to perform quantization and verify the warning message. Expected result: A warning message is displayed, but the quantification is allowed.**

## 5. Use Case 2: Qwen2.5-VL 7B model W8A8 quantization

### 5.1 Design Idea

Qwen2.5-VL is a multimodal visual language model developed by Alibaba Cloud, which supports the joint understanding of images and texts. This use case implements W8A8 quantization for the Qwen2.5-VL 7B model.

**Design idea:**

1. **Multi-modal adaptation: The Qwen2.5-VL model loading logic is added at the model adaptation layer to identify the vision encoder and language model.**
2. **Hierarchical quantization strategy: W8A8 quantization is used for the visual encoder, W8A8 quantization is used for the language model, and FP16 precision is maintained at the visual-language connection layer.**
3. **Multimodal calibration: Uses image-text pairs as calibration data and supports dataset formats such as COCO and Flickr 30k.**
4. **Outlier processing: Use the QuaRot algorithm to process the outliers of active values in multimodal models, especially for the fusion layer of visual and text features.**
5. **Precision assurance: Use the hierarchical quantization strategy of the visual coder and language model to ensure that the precision loss of multi-modal tasks is within 3%.**

### 5.2 Constraints

**Hardware Constraints:**

 * The Ascend NPU (Atlas 300I/300T/800 series) is required.
 * Minimum 32 GB system memory (for 7B scale models)
 * At least 16 GB video memory

**Software Constraints:**

 * Python 3.7+
 * PyTorch 1.8+
 * The transformers library version must support the Qwen2.5-VL model (4.37+ is recommended).
 * The model must be in the HuggingFace format.

**Data Constraints:**

 * Multimodal calibration dataset required (Image-text pair, 512-1024 samples recommended)
 * The calibration data must contain images and corresponding text descriptions.

**Function Constraints:**

 * Model training is not supported during quantization.
 * Quantized models support only inference and do not support continuous training.

### 5.3 Detailed Implementation

#### 5.3.1 Quantification Process Description

Qwen2.5-VL model quantization process includes configuration analysis, multi-modal model loading, multi-modal calibration data processing, hierarchical quantization execution and result saving. The key points are to deal with the different quantization requirements of visual coders and language models, and the fusion layer quantization of multi-modal features.

#### 5.3.2 Module Interaction Description

**Configuration parsing module: parses user configurations and generates multi-modal quantization policiesConfiguration object model adaptation layer: loads the Qwen2.5-VL model, identifies the visual coder and language modelPartial calibration data processing module: loads image-text pair data, performs image preprocessing, and text tokenization quantizationExecutive engine: Quantize the visual coder and language model respectively, process the multi-modal fusion layer result saving module: save the quantized model weight and configuration**

### 5.4 Inter-Subsystem Interface

#### 5.4.1 Model Adaptation Layer Interface

**Added the following APIs:**`Qwen25VLModelLoader`

 * `load_model(model_path: str, device: str) -> torch.nn.Module`\: Load the Qwen2.5-VL model.
 * `analyze_multimodal_structure(model: torch.nn.Module) -> MultimodalStructure`\: Analyze multimodal model structure

#### 5.4.2 Quantization Execution Engine Interface

**Modify the interface:**`QuantizationEngine`

 * `quantize_qwen25vl_w8a8(model, calib_data, config) -> QuantizedModel`\: Qwen2.5-VL Model W8A8 Quantization

### 5.5 Subsystem LLD

#### 5.5.1 Detailed Design of the Model Mediation Layer

**Qwen2.5-VL Model Loader: Use transformers to load models, identify the visual coder (ViT) and language model (Qwen2), and identify the visual-language connection layer (Projection Layer).**

**Multimodal Structure Analyzer: Analyze the structure of the visual coder and language model, identify the layers to be quantified, and mark the visual-language fusion layer as a sensitive layer (keeping FP16 or using special quantization strategy).**

#### 5.5.2 Detailed Design of the Quantized Execution Engine

**Visual encoder quantization: Use W8A8 quantization for the Linear layer of the ViT, use MinMax algorithm, and maintain FP16 precision for the LayerNorm layer.**

**Language model quantization: Apply W8A8 quantization to the Linear layer of Qwen2, and use the SmoothQuant algorithm to process abnormal activation values.**

**Multimodal fusion layer processing: The visual-language connection layer uses the QuaRot algorithm to quantify the multimodal feature fusion precision.**

#### 5.5.3 Detailed Design of Calibration Data Processing

**Multimodal data loading: supports image-text-to-dataset formats such as COCO and Flickr30k, and supports custom image-to-text-to-data formats.**

**Data preprocessing: image preprocessing (such as resize and normalize), text tokenization, and image-text pair batch generation.**

### 5.6 DFX Attribute Design

#### 5.6.1 Performance Design

**Quantization performance objective: Quantization time: 40 to 70 minutes for a 7B model (single card NPU, including multi-modal processing). After quantization, the model size is reduced by about 50%.**

**Performance optimization measures: Supports parallel quantization of multiple cards, optimizes the multi-modal data processing process, and uses asynchronous I/Os to load calibration data.**

#### 5.6.2 Upgrade and Capacity Expansion Design

**Version compatibility: The quantized model weight format is compatible with the existing inference framework and supports compatibility processing during model version upgrade.**

**Capacity expansion design: supports multi-card quantization and linear expansion of the quantization speed.**

#### 5.6.3 Exception Handling Design

**Abnormal Scenarios and Handling:**

1. **Multimodal data format error: Return detailed error information, prompting the user to check the data format.**
2. **Visual coder load failed: Check model integrity, provide repair suggestions**
3. **Multimodal fusion layer quantization failed: Downgrade to FP16 precision, record warning message**

#### 5.6.4 Resource Management Design

**Memory usage: 14 GB for model loading, 21 GB for quantization, and 5 GB for multimodal data processing. Total system memory is about 40 GB.**

**Disk I/O: 14 GB for model loading and read, 7 GB for calibration data loading (large image data), and 7 GB for quantization results storage**

#### 5.6.5 Miniaturized Design

**Impact on the installation package size: The size of the new Qwen2.5-VL model adaptation code is about 80 KB, and the size of the new multi-modal quantization configuration is about 15 KB. The total size of the new Qwen2.5-VL model adaptation code is about 95 KB.**

**Memory impact during running: The memory of the model adaptation layer increases by about 15 MB, and the memory of the multi-modal data processing module increases by about 20 MB.**

#### 5.6.6 Testability Design

**Function test: Test Qwen2.5-VL model loading, multi-modal data loading, W8A8 quantization process integrity, and multi-modal inference after quantization.**

**Performance test: test quantification time (target: 7B model < 70 minutes), inference speed after quantification, and model size after quantification (target: reduced by 50%)**

**Accuracy test: Compare the accuracy of multimodal tasks before and after the test quantization (target: loss < 3%), and test the impact of different calibration data sets.**

**Abnormality test: Test multi-modal data format error scenario, visual encoder loading failure scenario, and multi-modal fusion layer quantization failure scenario.**

#### 5.6.7 Security Design

##### 5.6.7.1 Safety Design Qualification

Similar to Use Case 1, the security design validation items focus on the security of multi-modal data processing and ensure the security of image and text data during quantification.

##### 5.6.7.2 Sensitive Data Analysis

**List of sensitive data:**

 * Model weight file: quantized model weight, medium sensitivity
 * Quantified configuration information: includes model path information, which is low sensitivity.
 * Calibration data: User-provided image-text is low-sensitive to data

**Sensitive operation check: Similar to Use Case 1, focus on deleting temporary files of multi-modal data.**

##### 5.6.7.3 Design Implementation

**File permission control: 644 for quantified result files and 600 for temporary files**

**External input verification: Model path verification, configuration file verification, and multi-modal data format verification (image format, text encoding, etc.)**

**Log security: Logs do not record the model weight content, complete file paths, and sensitive information is anonymized.**

### 5.7 External Interfaces

**Command line interface: The command line parameter is added to support Qwen2.5-VL model quantization. The command format is as follows:**`msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen2.5-VL-7B-Instruct --quant_type w8a8 --trust_remote_code True`

**Python API: Added**`quantize_qwen25vl_7b_w8a8()`Function

**Configuration file format: The configuration file in YAML format is supported. A configuration item dedicated to the Qwen2.5-VL model is added. (model type, path, quantization precision, multi-modal calibration data path, etc.)**

### 5.8 Self-Test Case Design

#### 5.8.1 Function Test Cases

**Case 1: Qwen2.5-VL model loading test: Prepare the Qwen2.5-VL 7B model, invoke the model loading interface, and verify that the model object is successfully created. Expected result: The model is loaded successfully and the model object is returned.**

**Case 2: Multi-modal data loading test: Prepare image-text pair calibration data, call the data loading interface, and verify that the data loading is successful. Expected result: The data is loaded successfully, and the data batch is returned.**

**Case 3: W8A8 quantization process test: Prepare models and calibration data, configure quantization parameters, perform quantization, and verify the generation of quantization result files. Expected result: The quantization is successful and a weight file is generated.**

**Case 4: Multi-modal inference test for quantized models: Load the quantized models, input images and texts, perform inference, and verify the output results. Expected result: The inference is successful, and a proper result is generated.**

#### 5.8.2 Performance Test Cases

**Case 5: Quantization time test: Record the quantization start time and end time, and calculate the quantization time. Expected result: Model 7B quantization time < 70 minutes.**

**Case 6: Inference speed test: Test the inference speed of the FP16 model and the quantized model, and calculate the speed improvement ratio.**

#### 5.8.3 Precision Test Cases

**Case 7: Multimodal task precision test: Evaluate the multimodal task precision of the FP16 model and the quantized model on the test set, and calculate the precision loss. Expected result: The precision loss is less than 3%.**

#### 5.8.4 Abnormal Test Cases

**Case 8: Multimodal data format error test: Use calibration data in incorrect format, perform quantization, and verify the error information. Expected result: A clear error message is returned, prompting the user to check the data format.**

**Case 9: Visual coder loading failure test: Use an incomplete model file to load a model and verify the error information. Expected result: A clear error message is returned, prompting you to check the model integrity.**

## 6. Reliability and availability design

### 6.1 Redundancy Design

*The system adopts the redundancy design. The mirror backup, configuration parameter backup, and data synchronization between the active/standby redundant systems must be considered.*

*During feature design, you need to provide the list of key configuration parameters for backup, data synchronization time and policies between active/standby redundant systems, key data list, data check mechanism, dirty data processing policy, and backup and restoration policy during active/standby switchover.*

*For mirror backup, such as the snapshot/checkpoint mechanism, the backup period, data check mechanism, dirty data processing policy, and restoration policy must be provided. For features that have obvious impact on system performance, design constraints must be provided.*

### 6.2 Fault Management

*Fault management includes fault detection, fault isolation, fault locating, fault recovery, and correlation design.*

*Feature fault management includes fault detection, alarm/log design, fault recovery, and fault interface design.*

*Common design principles for fault management are as follows:*

1. *Comprehensive and rapid fault detection usually considers the detection scope, backup detection, detection speed, and detection impact.*
2. *To control the impact scope of failure, isolation domains such as multi-plane, multi-granularity, and isolation units are usually considered.*
3. *Fast fault recovery usually takes into account the policies such as automatic recovery, priority recovery, hierarchical reset, uncoupled recovery, and hierarchical protection.*

*Common design modes for fault management include RollBack, Fault Bypass, Circuit Breaker, and Isolation Compartment.*

### 6.3 Overload control design

*The overload control design must consider the traffic detection, detection location, service discard location, response message information when a service is discarded, and invoking, invoking relationship, and interfaces between the overload control mechanism and the unified overload control mechanism.*

*A simple overload control mechanism is usually implemented by limiting the rate. The location, default rate limit, and log alarms must be considered.*

*Common design principles of overload control include dynamic rate limiting, flexible scaling, load balancing before traffic control, early control, priority assurance, and elegant degradation.*

1. *Early control: When the system is overloaded, control service access on the front end of service process processing or the processing module that processes services earlier to avoid unnecessary performance consumption caused by intermediate control.*
2. *Priority guarantee: When the system is overloaded, services with higher priorities are preferentially allocated and processed, thus maximizing social benefits.*
3. *Elegant degrade design: degrades non-core services, bypasses core functions, and degrades experience.*

### 6.4 Upgrade Without Service Interruption

*Services are not interrupted during the upgrade of a feature. The message compatibility, configuration data format compatibility, and interface compatibility of the feature in different software versions, interdependency between the feature and peripheral features, and quick rollback in the case of upgrade failure are considered.*

### 6.5 Design for human-caused errors

*The human-caused errors of the feature mainly take the following aspects into consideration:*

1. *High-risk messages and secondary confirmation must be provided for deletion and destructive modification. The default value of the page focus is Cancel. User-visible interfaces (cli and web pages) must be considered, including command interfaces provided by open-source components.*
2. *Check whether the restart operation affects the running of the customer's VM and provide a clear prompt for the restart operation.*
3. *All high-risk operations must be recorded in audit logs.*
4. *Prevent configuration errors, hardware misoperations, system check before operations, and quick rollback after operations are incorrect.*

*Common design principles for human error include:*

1. *Role constraint: The permission control design is used to prevent the configuration scope of different roles from being restricted, preventing configuration errors caused by unauthorized configuration.*
2. *Configuration verification: The configuration validation mechanism is designed to ensure that necessary verification is performed before the configuration takes effect to prevent incorrect configurations from taking effect.*
3. *Backup and restoration: The configuration data backup and restoration design ensures that the configuration data can be quickly restored to the correct state when a configuration error occurs.*

### 6.6 Fault prediction and prevention design

*This feature should cooperate with the system fault prediction and prevention capability to provide related data collection and statistics interfaces. For example, disk space detection.*

## 7. Design for features and non-functional quality attributes

### 7.1 Testability

*Describe the test direction and specifications of the feature, and describe the aspects that should be tested by the test personnel, and the boundary values, abnormal values, and abnormal scenarios that need to be noted.*

### 7.2 Serviceability

*Provides various maintainable and serviceable measures for features, and provides complete documentation for feature usage, maintenance, and troubleshooting.*

### 7.3 Evolvability

*This document focuses on the evolvability of the feature architecture and functions.*

### 7.4 Openness

*Focus on the openness of external interfaces, including the standardization of the interfaces, for example, compliance with the SQL 2011 standard.*

### 7.5 Compatibility

*Focus on whether the feature affects the forward compatibility of the system, that is, whether the old functions are available after the upgrade and whether the usage behavior is consistent with that of the old version.*

### 7.6 Scalability/Scalability

*This feature effectively meets the requirements of system capacity changes, including scaling of database nodes and database servers.*

### 7.7 Maintainability

*Focus on feature maintainability, such as diagnosis view and log printing.*

### 7.8 Information

*Refer to the following table to evaluate the modification points of various documents involved in the feature evaluation and describe the specific modification points.*

| Category  | Manual Name | Involved or Not (Y/N)                                      | Description of the modified or added content |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | ---------------------------------------------------------- | -------------------------------------------- |
| White Paper                                                                                                                                                               | Technical white paper | N                                                          | Added the XX technology in section XX.       |
| Product Documentation                                                                                                                                                     | Product Description   | N                                                          | Updated the technical specifications to XX.  |
|-| Feature Description                                                                                                                                                       | N                     | Added the XX feature.                                      |
|-| Compilation Guide                                                                                                                                                         | N                     | XXX                                                        |
|-| Installation Guide                                                                                                                                                        | N                     | Updated the XX scenario in section "Installing a Cluster." |
|-| Administrator's Guide                                                                                                                                                     | N                     | XXX                                                        |
|-| Developer guide (including the development tutorial, SQL reference, system tables and system views, GUC parameter description, error code description, and API reference) | N                     | Added the XXX function in section XX.                      |
|-| Tool Reference                                                                                                                                                            | N                     | Added the XX tool.                                         |
|-| Glossary of terms                                                                                                                                                         | N                     | New term XX                                                |
| Getting Started                                                                                                                                                           | Easy tutorial         | N                                                          | XXX                                          |

## 8. (Optional) Data Structure Design

*This section describes how to design the database structure. (Database system table structure, which can be completed by using the Power Designer) (Optional)*

## 9. List of references
