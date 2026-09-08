# **msModelSlim Rich Quantized Model Support Feature Design Specification**

<table>
    <tr>
        <td>SIG group:</td>
        <td>sig-msit</td>
    </tr>
    <tr>
        <td>Target release:</td>
        <td>To be determined</td>
    </tr>
    <tr>
        <td>Designer:</td>
        <td>panyj1993</td>
    </tr>
    <tr>
        <td>Date:</td>
        <td>2026</td>
    </tr>
</table>

**Copyright © 2026 msModelSlim Community**

Your copying, use, modification, and distribution of this document are governed by the Creative Commons Attribution-ShareAlike 4.0 International Public License (referred to as "CC BY-SA 4.0").
For easier understanding, you can visit <https://creativecommons.org/licenses/by-sa/4.0/> to read a summary of CC BY-SA 4.0 (but this summary is not a substitute for the license).
You can obtain the full text of CC BY-SA 4.0 at the following URL: <https://creativecommons.org/licenses/by-sa/4.0/legalcode>.

**Revision History**

<table>
    <tr>
        <th>Date</th>
        <th>Revision</th>
        <th>Description</th>
        <th>Author</th>
        <th>Reviewer</th>
    </tr>
    <tr>
        <td>2026.1.22</td>
        <td>v1</td>
        <td>Typical model quantization design specification</td>
        <td>panyj1993</td>
        <td>xxx</td>
    </tr>
</table>

**Contents**

1. Feature Overview

    1.1 Scope

    1.2 Feature Requirement List

2. Requirement Scenario Analysis

    2.1 Feature Requirement Source and Value Overview

    2.2 Feature Scenario Analysis

    2.3 Feature Impact Analysis

    2.3.1 Hardware Limitations

    2.3.2 Technical Limitations

    2.3.3 License Impact Analysis

    2.3.4 System Performance Specification Impact Analysis

    2.3.5 System Reliability Specification Impact Analysis

    2.3.6 System Compatibility Impact Analysis

    2.3.7 Interaction and Conflict Impact Analysis with Other Major Features

    2.4 Analysis of Similar Community or Commercial Software Implementation

3. Feature or Function Implementation Principles (Can Be Decomposed into Multiple Use Cases)

    3.1 Objectives

    3.2 Overall Solution

4. Use Case One Implementation

    4.1 Design Approach

    4.2 Constraints

    4.3 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from the User Entry Point)

    4.4 Inter-Subsystem Interfaces (Mainly Covers Module Interface Definitions)

    4.5 Detailed Subsystem Design

    4.6 DFX Attribute Design

    4.6.1 Performance Design

    4.6.2 Upgrade and Expansion Design

    4.6.3 Exception Handling Design

    4.6.4 Resource Management Design

    4.6.5 Minimization Design

    4.6.6 Testability Design

    4.6.7 Security Design

    4.7 System External Interfaces

    4.8 Self-Test Case Design

5. Use Case Two Implementation

6. Reliability & Availability Design

    6.1 Redundancy Design

    6.2 Fault Management

    6.3 Overload Control Design

    6.4 Upgrades Without Service Interruption

    6.5 Human Error Design

    6.6 Fault Prediction and Prevention Design

7. Security Design

    7.1 Low-Level Threat Analysis

    7.1.1 Layer-2 Data Flow Diagram

    7.1.2 Business Scenarios and Trust Boundary Description

    7.1.3 External Interaction Party Analysis

    7.1.4 Data Flow Analysis

    7.1.5 Processing Procedure Analysis

    7.1.6 Data Storage Analysis

    7.1.7 Defect List

    7.2 Sensitive Data Analysis

    7.2.1 Sensitive Data List

    7.2.2 Sensitive Operation Check

    7.3 Use Case Implementation

    7.3.1 Design Approach

    7.3.2 Detailed Implementation

8. Non-Functional Quality Attribute Related Design of the Feature

    8.1 Testability

    8.2 Serviceability

    8.3 Evolvability

    8.4 Openness

    8.5 Compatibility

    8.6 Scalability or Extensibility

    8.7 Maintainability

    8.8 Documentation

9. Data Structure Design (Optional)

10. Reference List

**List of Tables**

Table X: Feature Scenario Relevance Analysis

Table X: Feature Requirement List

**List of Figures**

Figure X: Overall Solution Implementation Principle Diagram

Figure X: Sample: Processing Flow Diagram

**List of Abbreviations**:

<table>
    <tr>
        <th>Abbreviations</th>
        <th>Full Spelling</th>
        <th>Chinese Explanation</th>
    </tr>
    <tr>
        <td>LLM</td>
        <td>Large Language Model</td>
        <td>大语言模型</td>
    </tr>
    <tr>
        <td>VLM</td>
        <td>Vision-Language Model</td>
        <td>视觉语言模型（多模态大模型）</td>
    </tr>
    <tr>
        <td>MoE</td>
        <td>Mixture of Experts</td>
        <td>混合专家架构</td>
    </tr>
    <tr>
        <td>W8A8</td>
        <td>Weight 8-bit Activation 8-bit</td>
        <td>权重和激活值均量化为 INT8</td>
    </tr>
    <tr>
        <td>W4A8</td>
        <td>Weight 4-bit Activation 8-bit</td>
        <td>权重量化为 INT4，激活值量化为 INT8</td>
    </tr>
    <tr>
        <td>W4A16</td>
        <td>Weight 4-bit Activation 16-bit</td>
        <td>权重量化为 INT4，激活值保持 FP16</td>
    </tr>
    <tr>
        <td>W8A16</td>
        <td>Weight 8-bit Activation 16-bit</td>
        <td>权重量化为 INT8，激活值保持 FP16</td>
    </tr>
    <tr>
        <td>W4A4</td>
        <td>Weight 4-bit Activation 4-bit</td>
        <td>权重和激活值均量化为 INT4</td>
    </tr>
    <tr>
        <td>KV Cache</td>
        <td>Key-Value Cache</td>
        <td>注意力机制中的键值对缓存</td>
    </tr>
    <tr>
        <td>PDMIX</td>
        <td>Prefill-Decode Mixed Quantization</td>
        <td>Prefilling 阶段动态量化、Decoding 阶段静态量化的混合策略</td>
    </tr>
    <tr>
        <td>FA3</td>
        <td>Flash Attention 3</td>
        <td>基于 per-head 粒度的注意力激活值 INT8 量化算法</td>
    </tr>
    <tr>
        <td>NPU</td>
        <td>Neural Processing Unit</td>
        <td>神经网络处理器（昇腾 AI 处理器）</td>
    </tr>
    <tr>
        <td>ViT</td>
        <td>Vision Transformer</td>
        <td>视觉 Transformer，用于图像特征提取</td>
    </tr>
    <tr>
        <td>MLA</td>
        <td>Multi-head Latent Attention</td>
        <td>多头潜在注意力机制（DeepSeek 等模型采用）</td>
    </tr>
    <tr>
        <td>YAML</td>
        <td>YAML Ain't Markup Language</td>
        <td>人类可读的数据序列化格式，用于量化配置文件</td>
    </tr>
    <tr>
        <td>OOM</td>
        <td>Out of Memory</td>
        <td>显存/内存溢出</td>
    </tr>
</table>

## 1. Feature Overview

With the rapid development of large language models and multimodal models, the demand for accelerating model inference is increasingly urgent. As an important means of model compression, quantization technology can significantly reduce model storage and computation overhead and improve inference speed. This feature aims to enrich the support of the msModelSlim tool for typical industry models, provide multiple quantization configuration solutions, accelerate inference, and ensure acceptable accuracy.

This feature adds quantization support for a series of models, including GLM-4.7, Qwen2.5-VL, Qwen3-VL, GLM4.6V, HunyuanVideo, Flux.1-dev, Wan2.2, Qwen2.5-Omni, and Qwen3-Omni. It covers multiple quantization precision configurations such as W8A8 and W4A4 to meet the requirements of balancing performance and accuracy in different scenarios.

### 1.1 Scope

This feature mainly includes the following function points:

1. **GLM series model quantization support**: Supports W8A8 quantization of the GLM-4.7 model and the GLM4.6V model.
2. **Qwen2.5-VL series model quantization support**: Supports W8A8 quantization of three scales: 7B, 32B, and 72B.
3. **Qwen3-VL series model quantization support**: Supports W8A8 quantization of the 30B-A3B-Instruct and 235B-A22B-Instruct models.
4. **HunyuanVideo model quantization support**: Supports both W8A8 and W4A4 quantization configurations.
5. **Flux.1-dev model quantization support**: Supports both W8A8 and W4A4 quantization configurations.
6. **Wan2.2 model quantization support**: Supports both W8A8 and W4A4 quantization configurations.
7. **Qwen2.5-Omni model quantization support**: Supports W8A8 quantization of the 7B model.
8. **Qwen3-Omni series model quantization support**: Supports W8A8 quantization of the 30B-A3B-Thinking and 30B-A3B-Instruct models.

### 1.2 Feature Requirement List

Table 1: Feature Requirement List

<table>
    <tr>
        <th>Requirement No.</th>
        <th>Requirement Name</th>
        <th>Feature Description</th>
        <th>Remarks</th>
    </tr>
    <tr>
        <td>1</td>
        <td>W8A8 quantization of the GLM-4.7 model</td>
        <td>Supports INT8 quantization of the weights and activations of the GLM-4.7 model and provides inference acceleration.</td>
        <td>Large language model</td>  
    </tr>
    <tr>
        <td>2</td>
        <td>W8A8 quantization of the Qwen2.5-VL model</td>
        <td>Supports W8A8 quantization of the Qwen2.5-VL 7B, 32B, and 72B scales, and supports vision-language multimodal scenarios.</td>
        <td>Multimodal model</td>  
    </tr>
    <tr>
        <td>3</td>
        <td>W8A8 quantization of the Qwen3-VL model</td>
        <td>Supports W8A8 quantization of the Qwen3-VL-30B-A3B-Instruct and 235B-A22B-Instruct models.</td>
        <td>Multimodal model</td>  
    </tr>
    <tr>
        <td>4</td>
        <td>W8A8 quantization of the GLM4.6V model</td>
        <td>Supports W8A8 quantization of the GLM4.6V vision model.</td>
        <td>Multimodal model</td>  
    </tr>
    <tr>
        <td>5</td>
        <td>HunyuanVideo model quantization</td>
        <td>Supports W8A8 and W4A4 quantization of the HunyuanVideo model to meet different precision requirements.</td>
        <td>Multimodal model</td>  
    </tr>
    <tr>
        <td>6</td>
        <td>Flux.1-dev model quantization</td>
        <td>Supports W8A8 and W4A4 quantization of the Flux.1-dev model.</td>
        <td>Multimodal model</td>  
    </tr>
    <tr>
        <td>7</td>
        <td>Wan2.2 model quantization</td>
        <td>Supports W8A8 and W4A4 quantization of the Wan2.2 model.</td>
        <td>Multimodal model</td>  
    </tr>
    <tr>
        <td>8</td>
        <td>W8A8 quantization of the Qwen2.5-Omni model</td>
        <td>Supports W8A8 quantization of the Qwen2.5-Omni-7B model.</td>
        <td>Multimodal model</td>  
    </tr>
    <tr>
        <td>9</td>
        <td>W8A8 quantization of the Qwen3-Omni model</td>
        <td>Supports W8A8 quantization of the Qwen3-Omni-30B-A3B-Thinking and 30B-A3B-Instruct models.</td>
        <td>Multimodal model</td>  
    </tr>
</table>

## 2. Requirement Scenario Analysis

### 2.1 Feature Requirement Source and Value Overview

With the wide application of large language models and multimodal models in the industry, model inference performance has become a key factor that constrains large-scale deployment. Current mainstream industry models, such as GLM, Qwen, HunyuanVideo, and Flux, have slow inference speed and large storage usage at native precision, which makes it difficult to meet the real-time requirements of production environments.

Quantization technology reduces the numerical precision of model weights and activations (for example, from FP16/BF16 to INT8/INT4). It significantly improves inference speed and reduces memory usage while maintaining relatively high accuracy. By providing quantization support for typical industry models, this feature helps users in the following ways:

1. **Reduce storage costs**: The quantized model size can be reduced by 50% to 75%, which lowers storage and transmission costs.
2. **Expand deployment scale**: More model instances can be deployed under the same hardware resources, which improves service throughput.
3. **Ensure acceptable accuracy**: Carefully designed quantization configurations and calibration strategies ensure that the accuracy of the quantized model meets business requirements.

Without this feature, users cannot use the msModelSlim tool to quantize the above models. They must implement the quantization process themselves or use other tools, which increases usage costs and maintenance burden and reduces the market competitiveness of the tool.

### 2.2 Feature Scenario Analysis

#### Scenario Trigger Conditions and Objects

**User roles**: AI model developers, model deployment engineers, and algorithm optimization personnel

**Tools used**: the msModelSlim quantization tool (command-line tool or Python API)

**Trigger conditions**:

- Users need to quantize supported models to improve inference performance.
- Users need to reduce model storage usage.
- Users need to deploy models in resource-constrained environments.

**Required skills for users**:

- Familiarity with Python programming and deep learning frameworks (PyTorch/MindSpore)
- Understanding of basic model quantization concepts
- Basic command-line operation capability

#### Main Application Scenarios

1. **Model inference acceleration scenario**
   - Sub-scenarios: online service inference acceleration and batch inference task acceleration
   - Key operations: load the original model → configure quantization parameters → perform quantization → save the quantized model → deploy for inference

2. **Resource-constrained deployment scenario**
   - Sub-scenarios: edge device deployment, mobile deployment, and concurrent multi-model deployment
   - Key operations: select an appropriate quantization precision (W8A8/W4A4) → quantize the model → verify accuracy → deploy

3. **Multimodal model optimization scenario**
   - Sub-scenarios: vision-language model optimization, video generation model optimization, and image generation model optimization
   - Key operations: prepare multimodal calibration data → configure multimodal quantization parameters → perform quantization → verify multimodal task accuracy

### 2.3 Feature Impact Analysis

This feature, as a core function extension of the msModelSlim tool, is located at the model adaptation layer of the quantization pipeline. The main affected modules include:

- **Model loading module**: Must support loading and parsing new models.
- **Quantization configuration module**: Must configure an appropriate quantization strategy for each model.
- **Quantization execution module**: Must adapt quantization processing for different model structures.
- **Calibration data processing module**: Must support calibration data formats for different model types.

#### Interaction Analysis with Other Requirements and Features

- **Interaction with existing quantization functions**: Reuses the existing quantization algorithms and process framework, and adds a new model adaptation layer.
- **Interaction with multimodal quantization features**: Some models (such as Qwen2.5-VL and Qwen3-VL) are multimodal models, and the multimodal quantization framework must be reused.
- **Interaction with inference frameworks**: Quantized models must be verified on inference frameworks such as MindIE and vLLM.

#### Platform Difference Analysis

**Hardware platform**: Mainly supports Ascend NPUs (Atlas series); some functions support CPUs.

**Operating system**: Supports the Linux operating system (Ubuntu, CentOS, and so on).

#### Compatibility Analysis

- **Forward compatibility**: New model support does not affect the quantization functions of existing models.
- **Configuration compatibility**: The quantization configuration of new models follows the unified YAML configuration protocol.
- **Interface compatibility**: The compatibility of the Python API and the command-line interface is maintained.

#### Constraints and Limitations

1. Models must be loaded in the HuggingFace format.
2. The quantization process requires a calibration dataset.
3. Some models require specific versions of the transformers library.

#### 2.3.1 Hardware Limitations

**NPU hardware requirements**:

- Support Ascend NPUs (Atlas 300I/300T/800, and so on)
- Video memory requirements: Depending on the model scale, a 7B model requires at least 16 GB, and a 72B model requires at least 128 GB.
- Multi-card quantization: Multi-card parallel quantization is supported to accelerate processing.

**Workarounds**:

- Provides sharded quantization for large models.
- Supports CPU fallback mode (with lower performance).
- Provides parameters to control the size of quantized weight files.

#### 2.3.2 Technical Limitations

**Operating system**: Linux (Ubuntu 18.04+, CentOS 7+)

**Programming language**: Python 3.7+

**Deep learning frameworks**:

- PyTorch 1.8+ (for model loading and quantization)
- MindSpore (for partial inference verification)
- The transformers library (version requirements vary by model)

**Workarounds**:

- Provides an environment dependency check script.
- Clearly marks the dependency version requirements of each model in the documentation.
- Supports Docker containerized deployment.

#### 2.3.3 License Impact Analysis

The License status of the models and dependency libraries involved in this feature:

1. **Model licenses**:
   - GLM series: Apache 2.0
   - Qwen series: Tongyi Qianwen LICENSE (with partial commercial restrictions)
   - HunyuanVideo: To be confirmed
   - Flux.1-dev: CreativeML Open RAIL-M License
   - Wan2.2: To be confirmed                                  

2. **Dependency library licenses**:
   - transformers: Apache 2.0
   - PyTorch: BSD-style
   - Other dependency libraries must be confirmed one by one.

**Compliance requirements**:

- All introduced third-party libraries must pass License compliance review.
- Clearly mark the License restrictions of each model in the documentation.
- Provide License statement files.

#### 2.3.4 System Performance Specification Impact Analysis

**Memory requirements**:

- 7B model quantization: at least 32 GB of system memory is required.
- 32B model quantization: at least 64 GB of system memory is required.
- 72B/235B model quantization: at least 128 GB of system memory is required; multi-card is recommended.

**Storage requirements**:

- Temporary files during quantization: about 2 to 3 times the model size.
- Quantized model storage: about 50% to 75% of the original model.

**Compute resources**:

- Quantization time: about 30 to 60 minutes for a 7B model and about 2 to 4 hours for a 72B model (single card).
- Multi-card parallelism is supported to accelerate the quantization process.

#### 2.3.5 System Reliability Specification Impact Analysis

**Quantization success rate**:

- Objective: With a standard calibration dataset, the quantization success rate is greater than or equal to 95%.
- Accuracy assurance: The accuracy loss of the quantized model is less than or equal to 3% (relative to the original model).

**Exception handling**:

- Provides detailed error logs when the quantization process encounters an exception.
- Supports checkpoint resume after quantization interruption.
- Provides a quantization result verification mechanism.

#### 2.3.6 System Compatibility Impact Analysis

**Forward compatibility**:

- New model support does not affect the quantization functions of existing models.
- Existing quantization configurations and API interfaces remain compatible.

**Version compatibility**:

- The quantized model weight format is compatible with existing inference frameworks.
- Compatibility handling is supported when the model version is upgraded.

#### 2.3.7 Interaction and Conflict Impact Analysis with Other Major Features

**Interaction with multimodal quantization features**:

- Models such as Qwen2.5-VL and Qwen3-VL reuse the multimodal quantization framework.
- The multimodal calibration data processing logic is shared.

**Interaction with inference frameworks**:

- Quantized models must be verified on inference frameworks such as MindIE and vLLM.
- The quantization format must be compatible with inference frameworks.

**Interaction with model conversion features**:

- Format conversion of quantized models is supported.
- Model conversion between different inference frameworks is supported.

### 2.4 Analysis of Similar Community or Commercial Software Implementation

#### Comparison with Similar Tools

**1. GPTQ/AWQ (community tools)**

- **Implementation mechanism**: Post-training quantization methods based on weight quantization.
- **Advantages**: Supports multiple models and has fast quantization speed.
- **Disadvantages**: Mainly targets weight quantization, has limited activation quantization support, and has insufficient multimodal model support.

**2. msModelSlim (this tool)**

- **Implementation mechanism**: Quantization optimization based on Ascend NPUs, supporting multiple quantization strategy combinations.
- **Advantages**:
  - Optimized for Ascend NPU hardware with excellent performance.
  - Supports multimodal model quantization (VL, SD, and so on).
  - Provides a unified configuration protocol that is easy to use.
  - Supports multiple quantization precisions such as W8A8 and W4A4.
- **Disadvantages**: Mainly targets the Ascend ecosystem and has limited support for other hardware.

#### Competitive Advantages of This Feature

1. **Wide model coverage**: Supports mainstream industry model series, covering language models, multimodal models, generative models, and so on.
2. **Diverse quantization precisions**: Supports multiple precision configurations such as W8A8 and W4A4 to meet requirements in different scenarios.
3. **Hardware optimization**: Deeply optimized for Ascend NPUs, fully utilizing hardware characteristics.
4. **Strong usability**: Unified YAML configuration protocol lowers the usage threshold.

## 3. Feature or Function Implementation Principles (Can Be Decomposed into Multiple Use Cases)

### 3.1 Objectives

This feature aims to add support for 16 model quantization configurations to the msModelSlim tool. The specific objectives are as follows:

1. **Functional objectives**:
   - Support W8A8 quantization of the GLM-4.7 and GLM4.6V models.
   - Support W8A8 quantization of the Qwen2.5-VL 7B/32B/72B models.
   - Support W8A8 quantization of the Qwen3-VL-30B-A3B-Instruct and 235B-A22B-Instruct models.
   - Support W8A8 and W4A4 quantization of the HunyuanVideo, Flux.1-dev, and Wan2.2 models.
   - Support W8A8 quantization of the Qwen2.5-Omni-7B, Qwen3-Omni-30B-A3B-Thinking, and 30B-A3B-Instruct models.

2. **Performance objectives**:
   - Reduce the quantized model size by 50% to 75%.
   - Keep the quantization accuracy loss within 3% (relative to the original model).

3. **Usability objectives**:
   - Provide a unified YAML configuration interface.
   - Support one-click quantization.
   - Provide detailed quantization documentation and samples.

### 3.2 Overall Solution

#### Hardware Selection

- **Primary hardware platform**: Ascend NPUs (Atlas 300I/300T/800 series)
- **Auxiliary hardware platform**: CPU (for partial preprocessing and postprocessing)

#### Algorithm Selection

1. **Weight quantization algorithms**:
   - W8A8: Uses the MinMax or AutoRound algorithm.
   - W4A4: Uses the SSZ (Smooth Scale Zero) or AutoRound algorithm.

2. **Activation quantization algorithms**:
   - W8A8: Uses MinMax dynamic quantization.
   - W4A4: Uses MinMax dynamic quantization.

3. **Outlier suppression algorithms**:
   - SmoothQuant (m1/m2/m4)
   - Flex Smooth Quant
   - QuaRot (for multimodal models)

#### Architecture Layout

The quantization processing flow uses a layered architecture:

1. **Model adaptation layer**: Responsible for loading and structure parsing of different models.
2. **Configuration parsing layer**: Parses YAML configurations and generates quantization strategies.
3. **Quantization execution layer**: Executes specific quantization algorithms.
4. **Calibration data processing layer**: Processes calibration data and supports multiple formats such as text, image, and video.
5. **Result saving layer**: Saves the quantized model weights.

#### Use Case Decomposition

Based on model types and quantization configurations, the feature implementation is decomposed into the following use cases:

1. **Use Case 1**: W8A8 quantization of the GLM-4.7 model
2. **Use Case 2**: W8A8 quantization of the Qwen2.5-VL 7B model
3. **Use Case 3**: W8A8 quantization of the Qwen2.5-VL 32B model
4. **Use Case 4**: W8A8 quantization of the Qwen2.5-VL 72B model
5. **Use Case 5**: W8A8 quantization of the Qwen3-VL-30B-A3B-Instruct model
6. **Use Case 6**: W8A8 quantization of the Qwen3-VL-235B-A22B-Instruct model
7. **Use Case 7**: W8A8 quantization of the GLM4.6V model
8. **Use Case 8**: W8A8 quantization of the HunyuanVideo model
9. **Use Case 9**: W4A4 quantization of the HunyuanVideo model
10. **Use Case 10**: W8A8 quantization of the Flux.1-dev model
11. **Use Case 11**: W4A4 quantization of the Flux.1-dev model
12. **Use Case 12**: W8A8 quantization of the Wan2.2 model
13. **Use Case 13**: W4A4 quantization of the Wan2.2 model
14. **Use Case 14**: W8A8 quantization of the Qwen2.5-Omni-7B model
15. **Use Case 15**: W8A8 quantization of the Qwen3-Omni-30B-A3B-Thinking model
16. **Use Case 16**: W8A8 quantization of the Qwen3-Omni-30B-A3B-Instruct model

#### Integration Principles

1. **Unified configuration protocol**: All models use the unified YAML configuration protocol.
2. **Interface compatibility**: Compatibility with existing quantization interfaces is maintained.
3. **Modular design**: Modules such as model adaptation, quantization algorithms, and data processing are independent to facilitate extension.
4. **Backward compatibility**: New model support does not affect existing functions.

#### Overall Architecture Diagram of the Solution

```tex
┌─────────────────────────────────────────────────────────┐
│                    User Interface Layer                  │
│  (Command-line tool / Python API / YAML configuration)  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  Configuration Parsing Layer            │
│  (YAML parsing / quantization strategy generation /     │
│   parameter validation)                                 │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  Model Adaptation Layer                 │
│  (GLM/Qwen/HunyuanVideo/Flux/Wan2.2 model loading)      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                Calibration Data Processing Layer        │
│  (text/image/video data preprocessing / data loading)   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  Quantization Execution Layer           │
│  (weight quantization / activation quantization /       │
│   outlier suppression / KVCache quantization)           │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  Result Saving Layer                    │
│  (quantized weight saving / format conversion /         │
│   verification)                                         │
└─────────────────────────────────────────────────────────┘
```

Figure 1: Overall Architecture Diagram of Quantization Processing

## 4. Use Case One Implementation: W8A8 Quantization of the GLM-4.7 Model

### 4.1 Design Approach

GLM-4.7 is a large language model developed by Zhipu AI and adopts the Transformer architecture. This use case implements W8A8 quantization of the GLM-4.7 model, that is, both weights and activations are quantized to INT8 precision.

**Design approach**:

1. **Model adaptation**: Add loading and parsing logic for the GLM-4.7 model in the model adaptation layer, and identify key layers in the model structure (such as Linear, Embedding, and LayerNorm).
2. **Quantization strategy**: Use the MinMax algorithm for weight quantization and dynamic quantization for activation quantization. Keep sensitive layers such as LayerNorm at FP16 precision.
3. **Outlier handling**: Use the SmoothQuant (m1/m2) algorithm to suppress activation outliers and improve quantization accuracy.
4. **Calibration data**: Use text datasets such as C4 or WikiText for calibration. The recommended number of calibration samples is 512 to 1024.
5. **Accuracy assurance**: Through layer-wise quantization strategies and sensitive-layer protection, ensure that the accuracy loss of the quantized model is within 3%.

### 4.2 Constraints

**Hardware constraints**:

- Requires Ascend NPU (Atlas 300I/300T/800 series) support.
- At least 32 GB of system memory (for a 7B-scale model).
- At least 16 GB of video memory.

**Software constraints**:

- Python 3.7+
- PyTorch 1.8+
- The transformers library version must support the GLM-4.7 model (4.30+ recommended).
- The model must be in the HuggingFace format.

**Data constraints**:

- Prepare a calibration dataset (text format; 512 to 1024 samples recommended).
- The calibration data must have a distribution similar to the model training data.

**Functional constraints**:

- The quantization process does not support model training.
- The quantized model supports inference only and does not support further training.

### 4.3 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from the User Entry Point)

#### 4.3.1 Quantization Flow Sequence Diagram

The quantization processing flow includes five main stages: configuration parsing, model loading, calibration data processing, quantization execution, and result saving. The modules interact through unified interfaces to ensure the completeness and reliability of the quantization flow.

**Main flow**:

1. Users submit a quantization task through the command line or the Python API.
2. The configuration parsing module parses the YAML configuration and generates a quantization strategy.
3. The model adaptation layer loads the GLM-4.7 model and parses its structure.
4. The calibration data processing module loads and preprocesses calibration data.
5. The quantization execution engine performs weight quantization and activation quantization.
6. The result saving module saves the quantized model weights and configuration.

#### 4.3.2 Module Interaction Description

**Configuration parsing module**: Parses the user configuration and generates a quantization strategy configuration object.
**Model adaptation layer**: Loads the GLM-4.7 model, parses the model structure, and identifies the layers to be quantized.
**Calibration data processing module**: Loads text data, performs tokenization, and generates calibration data batches.
**Quantization execution engine**: Performs core quantization flows such as weight quantization, activation quantization, and outlier handling.
**Result saving module**: Saves the quantized weights and configuration for subsequent inference.

### 4.4 Inter-Subsystem Interfaces (Mainly Covers Module Interface Definitions)

#### 4.4.1 Model Adaptation Layer Interface

**New interface**: `GLM47ModelLoader`

- `load_model(model_path: str, device: str) -> torch.nn.Module`: Loads the GLM-4.7 model.
- `analyze_structure(model: torch.nn.Module) -> ModelStructure`: Analyzes the model structure and identifies quantization target layers.

#### 4.4.2 Quantization Execution Engine Interface

**Modified interface**: `QuantizationEngine`

- `quantize_glm47_w8a8(model, calib_data, config) -> QuantizedModel`: Performs W8A8 quantization of the GLM-4.7 model.

#### 4.4.3 Configuration Parsing Interface

**Modified interface**: `ConfigParser`

- `parse_glm47_config(config_path: str) -> GLM47QuantConfig`: Parses the GLM-4.7 quantization configuration.

### 4.5 Detailed Subsystem Design

#### 4.5.1 Model Adaptation Layer Detailed Design

**GLM-4.7 model loader**: Uses the `AutoModelForCausalLM` class from the transformers library to load the model. It supports loading from the HuggingFace Hub or a local path and identifies the key components of the model: the embedding layer, the transformer layer, and the lm_head layer.

**Model structure analyzer**: Traverses all layers of the model, identifies the Linear layers (for weight quantization), identifies the LayerNorm layers (marked as sensitive layers and kept at FP16), identifies the activation function positions (for activation quantization), and generates the quantization target layer mapping table.

#### 4.5.2 Quantization Execution Engine Detailed Design

**Weight quantization module**: Applies MinMax quantization to the weight matrix of Linear layers, computes the min/max values of each weight matrix, uses symmetric quantization, with a quantization range of [-128, 127].

**Activation quantization module**: Uses the SmoothQuant algorithm to suppress outliers, collects activation distribution statistics on calibration data, computes quantization parameters for each activation layer, and applies dynamic quantization (quantization at inference time).

**Sensitive-layer protection**: The LayerNorm layers keep FP16 precision, the Embedding layer can keep FP16 precision or be quantized, and the output layer (lm_head) keeps FP16 precision.

#### 4.5.3 Calibration Data Processing Detailed Design

**Text data loading**: Supports common text dataset formats such as C4 and WikiText, supports custom text files (one sample per line), and automatically performs tokenization (using the tokenizer corresponding to the model).

**Data preprocessing**: Converts text into token IDs, generates fixed-length sequences (according to the model max_length), and supports batch generation (batch_size is configurable).

### 4.6 DFX Attribute Design

#### 4.6.1 Performance Design

**Quantization performance objectives**:

- Quantization time: about 30 to 60 minutes for a 7B model (single-card NPU).
- Quantized model size: reduced by about 50%.

**Performance optimization measures**:

- Supports multi-card parallel quantization to accelerate large-model quantization.
- Uses asynchronous I/O to load calibration data and reduce waiting time.
- Supports checkpoint resume during quantization to avoid repeated computation.

**Impact on existing performance**:

- New model support does not affect the quantization performance of existing models.
- The quantization algorithms reuse the existing implementation with no additional performance overhead.

#### 4.6.2 Upgrade and Expansion Design

**Version compatibility**:

- The quantized model weight format is compatible with existing inference frameworks.
- Compatibility handling is supported when the model version is upgraded.
- The quantization configuration format remains backward compatible.

**Expansion design**:

- Supports multi-card quantization with linear scaling of quantization speed.
- Supports sharded quantization of models to handle very large models.

#### 4.6.3 Exception Handling Design

**Exception scenarios and handling**:

1. **Model loading failure**: Returns detailed error information, prompts users to check the model path and format, and records the erroneous model path and error type.
2. **Insufficient calibration data**: Warns users but allows quantization to continue (accuracy may be affected), and records the number of calibration data samples.
3. **Excessive quantization accuracy loss**: Provides an accuracy report, suggests adjusting the quantization strategy or using higher precision, and records the accuracy comparison before and after quantization.
4. **Insufficient memory**: Provides a sharded quantization option, or suggests using a device with more memory, and records memory usage and the overflow location.
5. **Quantization interruption**: Supports checkpoint resume, saves intermediate results, and records the interruption location and the completed quantization progress.

#### 4.6.4 Resource Management Design

**Memory usage**:

- Model loading: about 14 GB (7B model in FP16)
- Temporary memory during quantization: about 1.5 times the model size (21 GB)
- Total: about 35 GB of system memory

**Disk I/O**:

- Model loading: reads the model weight files (about 14 GB)
- Calibration data loading: reads the calibration dataset (depending on the dataset size)
- Quantization result saving: writes the quantized weights (about 7 GB)

**Network I/O**: If the model is loaded from the HuggingFace Hub, network download is required.

**Resource overrun handling**:

- Insufficient memory: Provides a sharded quantization option, or prompts users to use a device with more memory.
- Insufficient disk space: Checks disk space and prompts users in advance.
- Network exception: Supports offline mode using local models.

#### 4.6.5 Minimization Design

**Impact on installation package size**:

- New GLM-4.7 model adaptation code: about 50 KB
- New quantization configuration: about 10 KB
- Total increase: about 60 KB, which is negligible.

**Runtime memory impact**:

- Model adaptation layer memory increase: about 10 MB
- The quantization engine has no additional memory overhead (reuses the existing implementation).

**CPU usage**: The quantization process mainly runs on the NPU, with low CPU usage (less than 10%).

#### 4.6.6 Testability Design

**Functional tests**: Test the GLM-4.7 model loading function, the completeness of the W8A8 quantization flow, the inference function of the quantized model, and the quantization configuration parsing function.

**Performance tests**: Test the quantization time (objective: less than 60 minutes for a 7B model), the inference speed after quantization, and the quantized model size (objective: reduced by 50%).

**Accuracy tests**: Test the accuracy comparison before and after quantization (objective: loss less than 3%), the impact of different calibration datasets, and boundary scenarios (extremely small or large calibration datasets).

**Exception tests**: Test the model loading failure scenario, the insufficient calibration data scenario, the insufficient memory scenario, and the quantization interruption and recovery scenario.

**Compatibility tests**: Test the compatibility of different transformers versions, different PyTorch versions, and different NPU hardware.

#### 4.6.7 Security Design

##### 4.6.7.1 Security Design Confirmation

*Confirm by referring to the security design checklist.*

| Security Attribute     | Check Item                                                       | Detailed Description of the Check Item                                               | Involved | Satisfied |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- | -------- |
| Access channel control | Whether a listening port is newly added                                             | A newly added listening port requires the communication matrix to be refreshed.                                   |     N     |     Y     |
| Access channel control | Whether inter-process or inter-component communication is newly added                                     | Newly added inter-process or inter-component communication requires the communication matrix to be refreshed.                             |     N     |     Y     |
| Access channel control | Whether an authentication method is newly added                                             | A newly added authentication method requires the communication matrix and product documentation to be refreshed.                         |     N     |     Y     |
| Permission control     | Whether file or directory creation is involved                                       | When creating files or directories, access permissions of the files or directories must be explicitly specified.                 |     N     |     Y     |
| Permission control     | Whether account permissions satisfy the principle of least privilege                             | Each account in the system should be granted the minimum permissions.                                   |     N     |     Y     |
| Permission control     | Whether user privilege escalation exists                                         | Illegal user privilege escalation is prohibited.                                     |     N     |     Y     |
| Undocumented interfaces   | Whether a GUC parameter is newly added                                              | A newly added GUC parameter requires the product documentation to be refreshed.                                    |     N     |     Y     |
| Undocumented interfaces   | Whether functions, views, or system tables are newly added or modified                             | Newly added or modified functions, views, or system tables require the product documentation to be refreshed, with permission control considered.     |     Y     |     Y     |
| Undocumented interfaces   | Whether SQL syntax is newly added                                              | Newly added SQL syntax requires the product documentation to be refreshed and audit log recording to be supported.                  |     N     |     Y     |
| Undocumented interfaces   | Whether internal tools are newly added                                             | Newly added internal tools require the product documentation to be refreshed.                                   |     N     |     Y     |
| Undocumented interfaces   | Whether commented-out code exists in scripts                                       | Commented-out code is prohibited in interpreted languages such as Shell and Python; commented-out code must be deleted.       |     N     |     Y     |
| Undocumented interfaces   | Whether hidden commands, parameters, ports, or other access methods exist                                   | Commands/parameters, ports, and other access methods (including but not limited to production, debugging, and maintenance purposes) that are not used during live-network maintenance must be deleted (for example, through compilation macros). |     N     |     Y     |
| Undocumented interfaces   | Whether the system has hidden backdoors                                         | The system is prohibited from reserving any undocumented accounts; all accounts must be manageable by the system and described in the documentation. |     N     |     Y     |
| Undocumented interfaces   | Prohibit providing cracking or network sniffing tools in software released to external users (including software packages and patch packages). | 1. Prohibit providing, in software released to external users (including software packages and patch packages), functions or tools that can modify any user password, have password cracking capability (brute-force cracking or malicious cracking using system/algorithm vulnerabilities), or decrypt files containing sensitive data (such as configuration files containing keys and databases). 2. Prohibit keeping third-party network sniffing tools such as tcpdump, gdb, strace, readelf network and process debugging tools, cpp, gcc, dexdump, mirror, JDK development/compilation tools, and self-developed debugging tools/scripts used only during debugging (for example, encryption/decryption scripts, debugging functions, and privilege-escalation commands used only during debugging). If any must be kept for business needs, strict access control is required, and the reason for keeping, the usage scenario, and the risks must be described in the documentation. |     N     |     Y     |
| Sensitive data protection | Authentication credentials must not be stored in plaintext in the system and should be encrypted.               | Authentication credentials (such as passwords and private keys) must not be stored in plaintext in the system and should be encrypted. |     N     |     Y     |
| Sensitive data protection | Keys used for encrypting sensitive data transmission must not be hardcoded in code.             | Hardcoding of passwords and keys is prohibited.                                       |     N     |     Y     |
| Sensitive data protection | Whether sensitive information such as passwords or keys is printed in plaintext                             | Prohibit printing plaintext sensitive information (passwords/private keys/pre-shared keys) in logs, debugging information, error prompts, and ps command output stored in the system. |     N     |     Y     |
| Sensitive data protection | Whether passwords are echoed in plaintext                                             | Plaintext echoing of passwords is prohibited.                                     |     N     |     Y     |
| Sensitive data protection | Whether default passwords of third-party and open-source software are used                           | The default passwords of third-party and open-source software are prohibited; refer to Section 1.5 of the Security Design Guide. |     N     |     Y     |
| Sensitive data protection | Whether passwords are stored in plaintext in configuration files                               | Plaintext passwords must not be written to configuration files (except for scenarios where passwords must be configured during command-line tool installation, deployment, and usage). |     N     |     Y     |
| Sensitive data protection | Whether insecure encryption algorithms are used                                     | Private or known insecure encryption algorithms are prohibited. Refer to Section 6.2 of the recommended encryption algorithm security design guide. |     N     |     Y     |
| Sensitive data protection | Whether sensitive information such as passwords uses a secure transmission channel                         | Sensitive information transmitted between untrusted networks must use a secure transmission channel or be encrypted before transmission. Refer to Chapter 10 of the Security Design Guide. |     N     |     Y     |
| Sensitive data protection | Whether sensitive information such as passwords or keys in memory is destroyed after use                     | Sensitive information such as passwords or keys in memory must be cleared to zero immediately after use.                  |     N     |     Y     |
| Sensitive data protection | Random numbers used in cryptographic algorithms must be cryptographically secure random numbers.     | Random numbers used in cryptographic algorithms must be cryptographically secure random numbers; refer to Section 6.3 of the Security Design Guide. |     N     |     Y     |
| Sensitive data protection | Whether insecure samples exist in the documentation                                   | Samples in the documentation must be secure and correctly guide users; if potential risks exist in a sample, they must be described in the documentation. |     N     |     Y     |
| Authentication         | Whether an authentication mechanism is provided                                             | New systems must provide an authentication mechanism and enable it by default.                           |     N     |     Y     |
| Authentication         | Whether authentication is performed on the server side                                         | The authentication process must be performed on the server side.                               |     N     |     Y     |
| Authentication         | Whether the server side returns valid information after authentication failure                             | After authentication failure, the server side must not return detailed information that can be used to determine the specific cause. |     N     |     Y     |
| External parameter validation | Whether external inputs are validated for legality                                         | 1. Using external input data as loop termination conditions, array subscripts, memory allocation size parameters, and so on may cause infinite loops, buffer overflows, out-of-bounds memory access, denial of service, and other behaviors. 2. External inputs such as file paths should be validated for legality to prevent injection risks. |     N     |     Y     |
| Third-party component introduction   | Whether third-party components are newly introduced                                           | 1. New third-party components must pass security compilation options, virus, vulnerability, open-source snippet reference, license compliance, and open-source component scans; refer to the version release network security quality requirements. 2. The source of new third-party components must be trusted. |     N     |     Y     |

##### 4.6.7.2 Sensitive Data Analysis

###### 1. Sensitive Data List

*The specific scope of sensitive data depends on the specific application scenario of the system and should be determined by the designer based on risk analysis and judgment. Typical sensitive data includes authentication credentials (such as passwords) and keys.*

| **Data Field**    | **Remarks/Description**          | **Data Field Sensitivity** | **Associated Processing Module** | **Mandatory Operations**             | **Prohibited Operations** |
| --------------- | ---------------------- | ------------------ | ---------------- | -------------------------- | -------------- |
| Administrator account/password | The account and password of the system administrator | High                 | Login/authentication        | Encrypted transmission, encrypted storage, anonymization, and so on | Echoing, logging, and so on |
| ...             | ...                    | ...                | ...              | ...                        | ...            |
|                 |                        |                    |                  |                            |                |

###### 2. Sensitive Operation Check

*1) Lifecycle dimension*
*For the identified sensitive data, fully identify the data lifecycle, including the processes of "generation, use, transmission, persistence, and destruction", to avoid unintentional omissions in subsequent risk identification.*
*2) High-risk processing procedures*
*Identify whether there are high-risk processing procedures for sensitive data. Typical high-risk processing includes "printing", "echoing", "storage", "hardcoding", and "insecure algorithms". From the perspective of information processing, these high-risk processing procedures can easily produce security vulnerabilities when processing sensitive data and require detailed inspection. All identified sensitive data must be checked. The sensitive data check matrix is as follows:*

For example, in a typical web system, the check results of the identified sensitive data (administrator account/password) over its lifecycle are as follows:

- Generation: The administrator sets the password when logging in to the system for the first time.
- Use: The administrator uses the password for authentication when logging in to the system.
- Transmission: After the administrator enters the login password on the client, the password is transmitted to the server over the network.
- Persistence: After the administrator sets the password for the first time, the server persists the password in the backend database.
- Destruction: After a certain period, the administrator is forced to change the password, and the old password is deleted.

|            |                             Generation                             |                  Use                  |                        Transmission                        |       Persistence       |                 Destruction                 |
| :--------: | :----------------------------------------------------------: | :------------------------------------: | :------------------------------------------------: | :----------------: | :----------------------------------: |
|    Printing    |                            Not involved                            | The password is not printed in any form during use | Under a secure transmission channel, encryption is not required; under a non-secure transmission channel, transmission is encrypted |       Not involved       | The destruction process does not print the password, but operation logs must be recorded |
|    Echoing    |            Echoed as ciphertext on the client, with the password displayed as *********             |                 Not involved                 |                       Not involved                       |       Not involved       |                Not involved                |
|    Storage    | After users input and set the password, the password is encrypted with a secure encryption algorithm and saved to the backend database |               Same as [Generation]               |                       Not involved                       | Encrypted storage in the backend database | Delete the corresponding password from the backend database table |
|   Hardcoding   |                            Not involved                            |                 Not involved                 |                       Not involved                       |       Not involved       |                Not involved                |
| Insecure algorithm |                  Encrypt using a secure algorithm (AES256)                  |            Decrypted in memory during use            |           A secure encryption algorithm is used over the non-secure transmission channel           |     Same as [Generation]     |                Not involved                |

##### 4.6.7.3 Design Implementation

**File permission control**: When quantization result files are created, the permission is explicitly set to 644 (user readable and writable, other users read-only). When temporary files are created, the permission is set to 600 (user only readable and writable).

**External input validation**: Model path validation checks path legality to prevent path injection. Configuration file validation verifies the YAML format to prevent configuration injection. Calibration data validation verifies data format to prevent malicious data.

**Log security**: The logs do not print model weight content. The logs do not print complete file paths (only relative paths or file names are printed). Sensitive information (such as model paths) is desensitized in the logs.

### 4.7 System External Interfaces

**Command-line interface**: Adds command-line parameters to support GLM-4.7 model quantization. The command format is: `msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type glm-4.7 --quant_type w8a8 --trust_remote_code True`

**Python API interface**: Adds the `quantize_glm47_w8a8()` function. The function signature is: `quantize_glm47_w8a8(model_path: str, config_path: str, output_path: str) -> None`

**Configuration file format**: Supports YAML configuration files and adds GLM-4.7 model-specific configuration items (model type, path, quantization precision, algorithm, and so on).

**Not involved**: GUC parameters, SQL syntax, network protocols, system table view functions, and drivers (JDBC/ODBC) are not involved.

### 4.8 Self-Test Case Design

#### 4.8.1 Functional Test Cases

**Case 1: GLM-4.7 model loading test**: Prepare the GLM-4.7 model (HuggingFace format), call the model loading interface, and verify that the model object is created successfully. Expected result: The model loads successfully, and the model object is returned.

**Case 2: W8A8 quantization flow test**: Prepare the GLM-4.7 model and calibration data, configure the quantization parameters (W8A8), perform quantization, and verify that the quantization result files are generated. Expected result: Quantization succeeds, and the quantized weight files are generated.

**Case 3: Quantized model inference test**: Load the quantized model, input test text, perform inference, and verify the output results. Expected result: Inference succeeds, and reasonable results are returned.

#### 4.8.2 Performance Test Cases

**Case 4: Quantization time test**: Record the quantization start time, perform quantization, record the quantization end time, and calculate the quantization duration. Expected result: The 7B model quantization time is less than 60 minutes.

**Case 5: Inference speed test**: Test the inference speed of the FP16 model, test the inference speed of the quantized model, and calculate the speed improvement ratio.

#### 4.8.3 Accuracy Test Cases

**Case 6: Quantization accuracy test**: Evaluate the FP16 model accuracy on the test set, evaluate the quantized model accuracy on the same test set, and calculate the accuracy loss. Expected result: The accuracy loss is less than 3%.

#### 4.8.4 Exception Test Cases

**Case 7: Model loading failure test**: Use a non-existent model path, attempt to load the model, and verify the error information. Expected result: Clear error information is returned, prompting users to check the path.

**Case 8: Insufficient calibration data test**: Use calibration data with fewer than 512 samples, perform quantization, and verify the warning information. Expected result: Warning information is output, but quantization is allowed to continue.

## 5. Use Case Two Implementation: W8A8 Quantization of the Qwen2.5-VL 7B Model

### 5.1 Design Approach

Qwen2.5-VL is a multimodal vision-language model developed by Alibaba Cloud and supports the joint understanding of images and text. This use case implements W8A8 quantization of the Qwen2.5-VL 7B model.

**Design approach**:

1. **Multimodal adaptation**: Add loading logic for the Qwen2.5-VL model in the model adaptation layer, and identify the vision encoder and the language model (LLM) components.
2. **Layer-wise quantization strategy**: The vision encoder and the language model both use W8A8 quantization, and the vision-language connection layer keeps FP16 precision.
3. **Multimodal calibration**: Uses image-text pairs as calibration data and supports dataset formats such as COCO and Flickr30k.
4. **Outlier handling**: Uses the QuaRot algorithm to handle activation outliers in multimodal models, especially for the fusion layer of visual features and text features.
5. **Accuracy assurance**: Through the layer-wise quantization strategy of the vision encoder and the language model, ensure that the accuracy loss of multimodal tasks is within 3%.

### 5.2 Constraints

**Hardware constraints**:

- Requires Ascend NPU (Atlas 300I/300T/800 series) support.
- At least 32 GB of system memory (for a 7B-scale model).
- At least 16 GB of video memory.

**Software constraints**:

- Python 3.7+
- PyTorch 1.8+
- The transformers library version must support the Qwen2.5-VL model (4.37+ recommended).
- The model must be in the HuggingFace format.

**Data constraints**:

- Prepare a multimodal calibration dataset (image-text pairs; 512 to 1024 samples recommended).
- The calibration data must contain images and the corresponding text descriptions.

**Functional constraints**:

- The quantization process does not support model training.
- The quantized model supports inference only and does not support further training.

### 5.3 Detailed Implementation

#### 5.3.1 Quantization Flow Description

The quantization flow of the Qwen2.5-VL model includes configuration parsing, multimodal model loading, multimodal calibration data processing, layer-wise quantization execution, and result saving. The key point is to handle the different quantization requirements of the vision encoder and the language model, as well as the quantization of the multimodal feature fusion layer.

#### 5.3.2 Module Interaction Description

**Configuration parsing module**: Parses the user configuration and generates a multimodal quantization strategy configuration object.
**Model adaptation layer**: Loads the Qwen2.5-VL model and identifies the vision encoder and the language model components.
**Calibration data processing module**: Loads image-text pair data and performs image preprocessing and text tokenization.
**Quantization execution engine**: Performs quantization on the vision encoder and the language model separately and handles the multimodal fusion layer.
**Result saving module**: Saves the quantized model weights and configuration.

### 5.4 Inter-Subsystem Interfaces

#### 5.4.1 Model Adaptation Layer Interface

**New interface**: `Qwen25VLModelLoader`

- `load_model(model_path: str, device: str) -> torch.nn.Module`: Loads the Qwen2.5-VL model.
- `analyze_multimodal_structure(model: torch.nn.Module) -> MultimodalStructure`: Analyzes the multimodal model structure.

#### 5.4.2 Quantization Execution Engine Interface

**Modified interface**: `QuantizationEngine`

- `quantize_qwen25vl_w8a8(model, calib_data, config) -> QuantizedModel`: Performs W8A8 quantization of the Qwen2.5-VL model.

### 5.5 Detailed Subsystem Design

#### 5.5.1 Model Adaptation Layer Detailed Design

**Qwen2.5-VL model loader**: Uses the transformers library to load the model, identifies the vision encoder (ViT) and the language model (Qwen2) components, and identifies the vision-language connection layer (Projection Layer).

**Multimodal structure analyzer**: Analyzes the structures of the vision encoder and the language model separately, identifies the layers to be quantized, and marks the vision-language fusion layer as a sensitive layer (keeps FP16 or uses a special quantization strategy).

#### 5.5.2 Quantization Execution Engine Detailed Design

**Vision encoder quantization**: Applies W8A8 quantization to the Linear layers of ViT using the MinMax algorithm, and keeps the LayerNorm layers at FP16 precision.

**Language model quantization**: Applies W8A8 quantization to the Linear layers of Qwen2 and uses the SmoothQuant algorithm to handle activation outliers.

**Multimodal fusion layer handling**: The vision-language connection layer uses the QuaRot algorithm for quantization to ensure the accuracy of multimodal feature fusion.

#### 5.5.3 Calibration Data Processing Detailed Design

**Multimodal data loading**: Supports image-text pair dataset formats such as COCO and Flickr30k, and supports custom image-text pair data.

**Data preprocessing**: Image preprocessing (resize, normalize, and so on), text tokenization, and generation of image-text pair batches.

### 5.6 DFX Attribute Design

#### 5.6.1 Performance Design

**Quantization performance objectives**: Quantization time: about 40 to 70 minutes for a 7B model (single-card NPU, including multimodal processing); quantized model size: reduced by about 50%.

**Performance optimization measures**: Supports multi-card parallel quantization, optimizes the multimodal data processing flow, and uses asynchronous I/O to load calibration data.

#### 5.6.2 Upgrade and Expansion Design

**Version compatibility**: The quantized model weight format is compatible with existing inference frameworks, and compatibility handling is supported when the model version is upgraded.

**Expansion design**: Supports multi-card quantization with linear scaling of quantization speed.

#### 5.6.3 Exception Handling Design

**Exception scenarios and handling**:

1. **Multimodal data format error**: Returns detailed error information and prompts users to check the data format.
2. **Vision encoder loading failure**: Checks model integrity and provides repair suggestions.
3. **Multimodal fusion layer quantization failure**: Degrades to FP16 precision and records warning information.

#### 5.6.4 Resource Management Design

**Memory usage**: Model loading is about 14 GB, temporary memory during quantization is about 1.5 times the model size (21 GB), additional memory for multimodal data processing is about 5 GB, and the total is about 40 GB of system memory.

**Disk I/O**: Model loading reads about 14 GB, calibration data loading reads large image data, and quantization result saving writes about 7 GB.

#### 5.6.5 Minimization Design

**Impact on installation package size**: New Qwen2.5-VL model adaptation code is about 80 KB, new multimodal quantization configuration is about 15 KB, and the total increase is about 95 KB.

**Runtime memory impact**: The model adaptation layer increases memory by about 15 MB, and the multimodal data processing module increases memory by about 20 MB.

#### 5.6.6 Testability Design

**Functional tests**: Test the Qwen2.5-VL model loading function, the multimodal data loading function, the completeness of the W8A8 quantization flow, and the multimodal inference function of the quantized model.

**Performance tests**: Test the quantization time (objective: less than 70 minutes for a 7B model), the inference speed after quantization, and the quantized model size (objective: reduced by 50%).

**Accuracy tests**: Test the multimodal task accuracy comparison before and after quantization (objective: loss less than 3%) and the impact of different calibration datasets.

**Exception tests**: Test the multimodal data format error scenario, the vision encoder loading failure scenario, and the multimodal fusion layer quantization failure scenario.

#### 5.6.7 Security Design

##### 5.6.7.1 Security Design Confirmation

The security design confirmation items are similar to those of Use Case One, with focus on the processing security of multimodal data to ensure the security of image and text data during quantization.

##### 5.6.7.2 Sensitive Data Analysis

**Sensitive data list**:

- Model weight file: the quantized model weights, with medium sensitivity.
- Quantization configuration information: contains information such as model paths, with low sensitivity.
- Calibration data: the image-text pair data provided by users, with low sensitivity.

**Sensitive operation check**: Similar to Use Case One, with focus on the cleanup of temporary files for multimodal data.

##### 5.6.7.3 Design Implementation

**File permission control**: Quantization result file permission is 644, and temporary file permission is 600.

**External input validation**: Model path validation, configuration file validation, and multimodal data format validation (image format, text encoding, and so on).

**Log security**: The logs do not print model weight content, do not print complete file paths, and desensitize sensitive information.

### 5.7 System External Interfaces

**Command-line interface**: Adds command-line parameters to support Qwen2.5-VL model quantization. The command format is: `msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen2.5-VL-7B-Instruct --quant_type w8a8 --trust_remote_code True`

**Python API interface**: Adds the `quantize_qwen25vl_7b_w8a8()` function.

**Configuration file format**: Supports YAML configuration files and adds Qwen2.5-VL model-specific configuration items (model type, path, quantization precision, multimodal calibration data path, and so on).

### 5.8 Self-Test Case Design

#### 5.8.1 Functional Test Cases

**Case 1: Qwen2.5-VL model loading test**: Prepare the Qwen2.5-VL 7B model, call the model loading interface, and verify that the model object is created successfully. Expected result: The model loads successfully, and the model object is returned.

**Case 2: Multimodal data loading test**: Prepare image-text pair calibration data, call the data loading interface, and verify that data loading succeeds. Expected result: Data loading succeeds, and the data batch is returned.

**Case 3: W8A8 quantization flow test**: Prepare the model and calibration data, configure the quantization parameters, perform quantization, and verify that the quantization result files are generated. Expected result: Quantization succeeds, and the quantized weight files are generated.

**Case 4: Quantized model multimodal inference test**: Load the quantized model, input images and text, perform inference, and verify the output results. Expected result: Inference succeeds, and reasonable results are returned.

#### 5.8.2 Performance Test Cases

**Case 5: Quantization time test**: Record the quantization start and end times and calculate the quantization duration. Expected result: The 7B model quantization time is less than 70 minutes.

**Case 6: Inference speed test**: Test the inference speed of the FP16 model and the quantized model and calculate the speed improvement ratio.

#### 5.8.3 Accuracy Test Cases

**Case 7: Multimodal task accuracy test**: Evaluate the multimodal task accuracy of the FP16 model and the quantized model on the test set and calculate the accuracy loss. Expected result: The accuracy loss is less than 3%.

#### 5.8.4 Exception Test Cases

**Case 8: Multimodal data format error test**: Use calibration data with an incorrect format, perform quantization, and verify the error information. Expected result: Clear error information is returned, prompting users to check the data format.

**Case 9: Vision encoder loading failure test**: Use incomplete model files, attempt to load the model, and verify the error information. Expected result: Clear error information is returned, prompting users to check model integrity.

## 6. Reliability & Availability Design

### 6.1 Redundancy Design

*The redundancy considered in the feature design mainly refers to the redundancy adopted by the system. The feature needs to consider mirror backup, configuration parameter backup, and data synchronization between primary and standby redundant systems.*

*During feature design, provide the backup list of key configuration parameters, the data synchronization time __/__ strategy between primary and standby redundant systems and the key data list, the data verification mechanism __/__ dirty data processing strategy during primary/standby switchover, the backup recovery strategy, and so on.*

*For mirror backup, such as the snapshot __/checkpoint__ mechanism, provide the backup cycle, data verification mechanism __/__ dirty data processing strategy, recovery strategy, and so on. For features that significantly affect system performance, provide design constraints.*

### 6.2 Fault Management

*Fault management includes fault detection, fault isolation, fault location, fault recovery, and their interrelated design.*

*The fault management of a feature mainly refers to the fault detection, alarm __/__ log design, fault recovery, and fault interface design of the feature itself.*

*The common design principles of fault management include:*

1. *Comprehensive and fast fault detection usually considers the detection scope, backup detection, detection speed, and detection impact.*
2. *Controlling the impact scope of failures usually considers isolation domain division such as multi-plane, multi-granularity, and isolation units.*
3. *Fast fault recovery usually considers strategies such as automatic recovery, priority recovery, hierarchical reset, decoupled recovery, and layered protection.*

*Common design patterns of fault management include the __RollBack__ pattern, fault __Bypass__, the circuit breaker pattern, the isolation warehouse pattern, and so on.*

### 6.3 Overload Control Design

*The overload control design of a feature needs to consider the traffic detection of the services processed within the feature, the detection location and service drop location, the service message information returned when the service is dropped, and the calling, called relationship, and interfaces with the unified overload control mechanism.*

*The simple overload control mechanism within a feature generally uses rate limiting, which needs to consider the rate limiting location, the default rate limit value, log alarms, and other information.*

*The common design principles of overload control include dynamic rate limiting, elastic scaling, load balancing before flow control, early control, priority assurance, graceful degradation design, and so on:*

1. *Early control: When the system is overloaded, control service access as early as possible at the front end of the business process or at the earlier processing modules of the business process to avoid unnecessary performance consumption caused by intermediate control.*
2. *Priority assurance: When the system is overloaded, ensure that high-priority services obtain resources first and are processed first, thereby maximizing social benefits.*
3. *Graceful degradation design: Degrade non-core services, allow core functions, degrade experience, and so on.*

### 6.4 Upgrades Without Service Interruption

*Upgrades without service interruption within a feature mainly consider the message compatibility of the feature across different software versions, configuration data format compatibility, interface compatibility, mutual dependencies with surrounding features, and the fast rollback process when an upgrade fails.*

### 6.5 Human Error Design

*The human error design of a feature mainly considers the error prevention of human-machine interfaces such as the commands, operations, configuration files __/__ data involved in the feature. It usually considers the following aspects:*

1. *Deletion and destructive modification require high-risk prompts and secondary confirmation, with the page focus defaulting to "Cancel". All user-visible interfaces (__cli__ and __web__ pages) must be considered, including command interfaces provided by open-source components.*
2. *Before restarting a node, check in advance whether the operation affects customer __VM__ running and provide a clear prompt and suggested operation.*
3. *All high-risk operations must record audit logs.*
4. *Prevent configuration errors, prevent hardware misoperation, perform system checks before operations, and support fast rollback after operation errors.*

*The common design principles of human error design include:*

1. *Role constraint: Use permission control design to constrain the configuration scope of different roles and avoid errors caused by unauthorized configuration.*
2. *Configuration validation: Use the configuration-effective mechanism design to ensure that necessary validation is performed before configuration takes effect and that incorrect configurations do not take effect.*
3. *Backup recovery: Use configuration data backup and recovery design to ensure fast recovery to the correct configuration data state when configuration errors occur.*

### 6.6 Fault Prediction and Prevention Design

*The feature should provide related data collection and statistics interfaces in cooperation with the system fault prediction and prevention capability, such as disk space detection.*

## 7. Non-Functional Quality Attribute Related Design of the Feature

### 7.1 Testability

*Focus on describing the test direction and specifications of the feature, explaining which aspects testers should test and which boundary values, abnormal values, and abnormal scenarios need attention.*

### 7.2 Serviceability

*Provide rich maintainability and serviceability measures for the feature, and provide complete documentation for the usage, maintenance, and problem handling of the feature.*

### 7.3 Evolvability

*Focus on describing the evolvability of the feature architecture and functions.*

### 7.4 Openness

*Focus on describing the external interface openness of the feature, including interface standardization, such as compliance with the __SQL 2011__ standard.*

### 7.5 Compatibility

*Focus on describing whether the feature affects the forward compatibility of the system, that is, whether old functions can still be used after upgrading to a new version and whether the usage behavior remains consistent with the old version.*

### 7.6 Scalability or Extensibility

*Effectively meet the requirements of system capacity changes, including the scaling of database nodes and the scaling of database servers themselves.*

### 7.7 Maintainability

*Focus on describing the maintainability of the feature, such as diagnostic views and __log__ printing.*

### 7.8 Documentation

*Refer to the following table to evaluate the modification points of various documents affected by the feature and describe the specific modification points.*

<table>
    <tr>
        <th>Category</th>
        <th>Manual Name</th>
        <th>Involved (Y/N)</th>
        <th>Brief Description of Specific Modifications or Additions</th>
    </tr>
    <tr>
        <td>White paper</td>
        <td>Technical white paper</td>
        <td>N</td>
        <td>Add XX technology in section XX</td>
    </tr>
    <tr>
        <td rowspan="8">Product documentation</td>
        <td>Product description</td>
        <td>N</td>
        <td>Refresh technical indicators to XX</td>
    </tr>
    <tr>
        <td>Feature description</td>
        <td>N</td>
        <td>Add XX feature</td>
    </tr>
    <tr>
        <td>Compilation guide</td>
        <td>N</td>
        <td>XXX</td>
    </tr>
    <tr>
        <td>Installation guide</td>
        <td>N</td>
        <td>The installation cluster section must refresh the XX scenario</td>
    </tr>
    <tr>
        <td>Administrator guide</td>
        <td>N</td>
        <td>XXX</td>
    </tr>
    <tr>
        <td>Developer guide (including development tutorial, SQL reference, system tables and system views, GUC parameter description, error code description, API reference, and so on)</td>
        <td>N</td>
        <td>Add XX function in section XX</td>
    </tr>
    <tr>
        <td>Tool reference</td>
        <td>N</td>
        <td>Add XX tool</td>
    </tr>
    <tr>
        <td>Glossary</td>
        <td>N</td>
        <td>Add term XX</td>
    </tr>
    <tr>
        <td>Getting started</td>
        <td>Simple tutorial</td>
        <td>N</td>
        <td>XXX</td>
    </tr>
</table>

## 8. Data Structure Design (Optional)

*This chapter completes the design of the database structure (the database system table structure, which can be completed using __Power Designer__). This is an optional chapter.*

## 9. Reference List
